"""
TTS engines — converts text to WAV bytes (one chunk at a time).

Backends:
  elevenlabs  — cloud, best quality, ~200 ms latency (default)
  kokoro      — local, good quality, runs on CPU
  coqui       — local, older but battle-tested
  xtts_local  — local XTTS2 voice clone (fine-tuned or zero-shot)

All backends retry with exponential backoff and validate output audio.
"""
import asyncio
import io
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from brain import _retry
from config import TTSConfig


class TTSBackend(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Return validated WAV bytes for the given text."""

    async def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
#  ElevenLabs                                                                   #
# --------------------------------------------------------------------------- #

class ElevenLabsTTS(TTSBackend):
    _BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, cfg: TTSConfig):
        self._cfg = cfg
        self._http = httpx.AsyncClient(
            base_url=self._BASE,
            headers={"xi-api-key": cfg.elevenlabs_api_key},
            timeout=20,
        )

    async def synthesize(self, text: str) -> bytes:
        return await _retry(
            lambda: self._synthesize_once(text),
            max_retries=self._cfg.max_retries,
            base_delay=self._cfg.retry_base_delay_s,
            label="ElevenLabs",
        )

    async def _synthesize_once(self, text: str) -> bytes:
        url = f"/text-to-speech/{self._cfg.elevenlabs_voice_id}/stream"
        payload = {
            "text": text,
            "model_id": self._cfg.elevenlabs_model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            "output_format": "pcm_44100",
        }
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        wav = _pcm_to_wav(resp.content, sample_rate=44100, channels=1, sample_width=2)
        _validate_wav(wav, label="ElevenLabs")
        return wav

    async def close(self) -> None:
        await self._http.aclose()


# --------------------------------------------------------------------------- #
#  Kokoro (local)                                                               #
# --------------------------------------------------------------------------- #

class KokoroTTS(TTSBackend):
    def __init__(self, cfg: TTSConfig):
        self._voice = cfg.kokoro_voice
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from kokoro import KPipeline  # type: ignore
        self._pipeline = KPipeline(lang_code="a")

    async def synthesize(self, text: str) -> bytes:
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(None, self._synthesize_sync, text)
        _validate_wav(wav, label="Kokoro")
        return wav

    def _synthesize_sync(self, text: str) -> bytes:
        self._load()
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
        samples = []
        for _, _, audio in self._pipeline(text, voice=self._voice, speed=1.0):
            samples.append(audio)
        combined = np.concatenate(samples)
        buf = io.BytesIO()
        sf.write(buf, combined, 24000, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# --------------------------------------------------------------------------- #
#  Coqui TTS (local)                                                            #
# --------------------------------------------------------------------------- #

class CoquiTTS(TTSBackend):
    def __init__(self, cfg: TTSConfig):
        self._model_name = cfg.coqui_model
        self._tts = None

    def _load(self) -> None:
        if self._tts is not None:
            return
        from TTS.api import TTS  # type: ignore
        self._tts = TTS(model_name=self._model_name, progress_bar=False, gpu=False)

    async def synthesize(self, text: str) -> bytes:
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(None, self._synthesize_sync, text)
        _validate_wav(wav, label="Coqui")
        return wav

    def _synthesize_sync(self, text: str) -> bytes:
        self._load()
        with NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            self._tts.tts_to_file(text=text, file_path=path)
            return Path(path).read_bytes()
        finally:
            Path(path).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
#  XTTS2 local voice clone                                                      #
# --------------------------------------------------------------------------- #

class XTTSLocalTTS(TTSBackend):
    """
    XTTS2 inference backend.

    If xtts_checkpoint_dir is set, loads the fine-tuned model from that
    directory (output of finetune/train_voice.py).  Otherwise falls back
    to the base XTTS2 model for zero-shot voice cloning using only the
    reference WAV.

    xtts_reference_wav must always point to a clean speaker sample.
    """

    _ZERO_SHOT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, cfg: TTSConfig):
        self._cfg = cfg
        self._tts = None
        if not cfg.xtts_reference_wav:
            raise ValueError(
                "xtts_local backend requires XTTS_REFERENCE_WAV to be set."
            )

    def _load(self) -> None:
        if self._tts is not None:
            return

        try:
            from TTS.api import TTS  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Install Coqui TTS:\n"
                "  pip install coqui-tts"
            )

        import torch

        device = "cuda" if (self._cfg.xtts_use_gpu and torch.cuda.is_available()) else "cpu"

        if self._cfg.xtts_checkpoint_dir:
            model_path  = Path(self._cfg.xtts_checkpoint_dir)
            config_path = model_path / "config.json"
            self._tts = TTS(
                model_path=str(model_path),
                config_path=str(config_path),
                progress_bar=False,
            ).to(device)
        else:
            self._tts = TTS(self._ZERO_SHOT_MODEL, progress_bar=False).to(device)

    async def synthesize(self, text: str) -> bytes:
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(None, self._synthesize_sync, text)
        _validate_wav(wav, label="XTTS2")
        return wav

    def _synthesize_sync(self, text: str) -> bytes:
        self._load()
        with NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            self._tts.tts_to_file(
                text=text,
                speaker_wav=self._cfg.xtts_reference_wav,
                language="en",
                file_path=out_path,
            )
            return Path(out_path).read_bytes()
        finally:
            Path(out_path).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
#  Factory                                                                      #
# --------------------------------------------------------------------------- #

def build_tts(cfg: TTSConfig) -> TTSBackend:
    if cfg.backend == "elevenlabs":
        return ElevenLabsTTS(cfg)
    if cfg.backend == "kokoro":
        return KokoroTTS(cfg)
    if cfg.backend == "coqui":
        return CoquiTTS(cfg)
    if cfg.backend == "xtts_local":
        return XTTSLocalTTS(cfg)
    raise ValueError(f"Unknown TTS backend: {cfg.backend!r}")


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int, sample_width: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _validate_wav(wav_bytes: bytes, label: str = "TTS") -> None:
    if not wav_bytes:
        raise ValueError(f"{label} returned empty audio")
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            duration = wf.getnframes() / wf.getframerate()
        if duration < 0.05:
            raise ValueError(f"{label} audio too short ({duration:.3f} s) — likely empty response")
    except wave.Error as exc:
        raise ValueError(f"{label} returned invalid WAV: {exc}") from exc
