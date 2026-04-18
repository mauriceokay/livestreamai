"""
TTS engines — converts text to a WAV audio file (one chunk at a time).

Backends:
  elevenlabs  — cloud, best quality, ~200ms latency (default)
  kokoro      — local, good quality, runs on CPU
  coqui       — local, older but battle-tested, CPU/GPU
"""
import asyncio
import io
import struct
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from config import TTSConfig


class TTSBackend(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Return raw WAV bytes for the given text."""


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
        url = f"/text-to-speech/{self._cfg.elevenlabs_voice_id}/stream"
        payload = {
            "text": text,
            "model_id": self._cfg.elevenlabs_model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            "output_format": "pcm_44100",
        }
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        # ElevenLabs returns raw PCM — wrap it in a WAV container
        return _pcm_to_wav(resp.content, sample_rate=44100, channels=1, sample_width=2)

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
        # Import lazily so the module loads even when kokoro isn't installed
        from kokoro import KPipeline  # type: ignore
        self._pipeline = KPipeline(lang_code="a")  # "a" = American English

    async def synthesize(self, text: str) -> bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

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

    async def close(self) -> None:
        pass


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
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        self._load()
        with NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        self._tts.tts_to_file(text=text, file_path=path)
        data = Path(path).read_bytes()
        Path(path).unlink(missing_ok=True)
        return data

    async def close(self) -> None:
        pass


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
