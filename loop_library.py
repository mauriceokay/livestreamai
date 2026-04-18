"""
Pre-rendered video loop library.

Instead of running LatentSync in real-time (5–15 s per chunk), we:
  1. Pre-render a small library of short animated clips once (generate_loops.py)
  2. At stream time, pick the right clip and overlay the live TTS audio onto it
     using ffmpeg — pure demux/mux, takes ~50 ms

The result is indistinguishable from real-time lipsync to most viewers because
human perception locks onto audio; the visual just needs to be "plausibly moving".

Loop categories
───────────────
  idle          — slight natural movement, no talking (plays between speech)
  talking       — mouth moving, neutral expression (plays during any speech)
  react_happy   — smile + energy (follows, chat praise)
  react_gift    — excited/grateful (gifts)
"""
import asyncio
import random
import struct
import wave
from enum import Enum
from pathlib import Path


class LoopCategory(str, Enum):
    IDLE = "idle"
    TALKING = "talking"
    REACT_HAPPY = "react_happy"
    REACT_GIFT = "react_gift"


# Maps speech modes to loop categories
from brain import SpeechMode  # noqa: E402 (imported here to avoid circular)

MODE_TO_CATEGORY: dict[SpeechMode, LoopCategory] = {
    SpeechMode.IDLE: LoopCategory.IDLE,
    SpeechMode.CHAT_REPLY: LoopCategory.TALKING,
    SpeechMode.REACT: LoopCategory.REACT_HAPPY,
}


class LoopLibrary:
    def __init__(self, loops_dir: str = "assets/loops"):
        self._dir = Path(loops_dir)
        self._clips: dict[str, list[Path]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def ready(self) -> bool:
        """Return True if at least the 'talking' category has clips."""
        return bool(self._clips.get(LoopCategory.TALKING.value))

    def missing_categories(self) -> list[str]:
        return [
            cat.value
            for cat in LoopCategory
            if not self._clips.get(cat.value)
        ]

    async def make_chunk(
        self,
        category: LoopCategory,
        audio_wav: bytes,
    ) -> bytes:
        """
        Overlay audio_wav onto a looped video clip from the given category.
        Returns MP4 bytes ready to push to the RTMP streamer.
        """
        clip = self._pick(category)
        duration = _wav_duration(audio_wav)

        import tempfile
        with tempfile.TemporaryDirectory(prefix="loop_") as tmp:
            tmp = Path(tmp)
            audio_path = tmp / "audio.wav"
            out_path = tmp / "out.mp4"
            audio_path.write_bytes(audio_wav)

            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                # Video source — loop indefinitely
                "-stream_loop", "-1",
                "-i", str(clip),
                # Audio source — live TTS
                "-i", str(audio_path),
                # Take video track from clip, audio track from TTS
                "-map", "0:v:0",
                "-map", "1:a:0",
                # Duration = audio length
                "-t", f"{duration:.3f}",
                # Stream copy video (no re-encode), encode audio to AAC
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(out_path),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg loop overlay failed:\n{stderr.decode()}"
                )
            return out_path.read_bytes()

    async def idle_chunk(self, duration_s: float = 1.5) -> bytes:
        """Return a silent idle clip of the given duration."""
        clip = self._pick(LoopCategory.IDLE)

        import tempfile
        with tempfile.TemporaryDirectory(prefix="loop_idle_") as tmp:
            tmp = Path(tmp)
            out_path = tmp / "idle.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-stream_loop", "-1",
                "-i", str(clip),
                "-t", f"{duration_s:.3f}",
                "-c:v", "copy",
                "-an",                   # no audio track
                str(out_path),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg idle clip failed:\n{stderr.decode()}"
                )
            return out_path.read_bytes()

    # ------------------------------------------------------------------ #
    #  Internals                                                            #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self._dir.exists():
            return
        for cat in LoopCategory:
            clips = sorted(self._dir.glob(f"{cat.value}_*.mp4"))
            if clips:
                self._clips[cat.value] = clips

    def _pick(self, category: LoopCategory) -> Path:
        clips = self._clips.get(category.value) or self._clips.get(
            LoopCategory.TALKING.value
        )
        if not clips:
            raise FileNotFoundError(
                f"No video loops found for category '{category.value}'. "
                "Run:  python generate_loops.py"
            )
        return random.choice(clips)


# --------------------------------------------------------------------------- #
#  Helper                                                                       #
# --------------------------------------------------------------------------- #

def _wav_duration(wav_bytes: bytes) -> float:
    import io
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.getnframes() / wf.getframerate()
