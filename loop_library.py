"""
Pre-rendered video loop library.

Instead of running a diffusion model in real-time (5–300 s per chunk), we:
  1. Pre-render a small library of short animated clips once (generate_loops.py)
  2. At stream time, pick the right clip and overlay the live TTS audio onto it
     using ffmpeg — pure demux/mux, takes ~50 ms

Loop categories and fallback chain
───────────────────────────────────
  react_gift   → react_happy → talking
  react_happy  → talking
  look_away    → idle
  drink_water  → idle
  laugh        → react_happy → talking
  thinking     → idle
  idle         → talking
  talking      → (required — no fallback)
"""
import asyncio
import io
import random
import wave
from enum import Enum
from pathlib import Path


class LoopCategory(str, Enum):
    IDLE        = "idle"
    TALKING     = "talking"
    REACT_HAPPY = "react_happy"
    REACT_GIFT  = "react_gift"
    LOOK_AWAY   = "look_away"
    DRINK_WATER = "drink_water"
    LAUGH       = "laugh"
    THINKING    = "thinking"


# Fallback chain: if a category has no clips, try the next in list
_FALLBACK: dict[str, list[str]] = {
    LoopCategory.REACT_GIFT.value:  [LoopCategory.REACT_HAPPY.value, LoopCategory.TALKING.value],
    LoopCategory.REACT_HAPPY.value: [LoopCategory.TALKING.value],
    LoopCategory.LOOK_AWAY.value:   [LoopCategory.IDLE.value],
    LoopCategory.DRINK_WATER.value: [LoopCategory.IDLE.value],
    LoopCategory.LAUGH.value:       [LoopCategory.REACT_HAPPY.value, LoopCategory.TALKING.value],
    LoopCategory.THINKING.value:    [LoopCategory.IDLE.value],
    LoopCategory.IDLE.value:        [LoopCategory.TALKING.value],
    LoopCategory.TALKING.value:     [],
}

# Behavior categories that get injected silently (no TTS overlay needed)
BEHAVIOR_CATEGORIES = frozenset({
    LoopCategory.LOOK_AWAY,
    LoopCategory.DRINK_WATER,
    LoopCategory.THINKING,
})


class LoopLibrary:
    def __init__(self, loops_dir: str = "assets/loops"):
        self._dir = Path(loops_dir)
        self._clips: dict[str, list[Path]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def ready(self) -> bool:
        return bool(self._clips.get(LoopCategory.TALKING.value))

    def missing_categories(self) -> list[str]:
        # Only core categories are required; behavior clips are optional
        required = {LoopCategory.TALKING, LoopCategory.IDLE,
                    LoopCategory.REACT_HAPPY, LoopCategory.REACT_GIFT}
        return [cat.value for cat in required if not self._clips.get(cat.value)]

    def has_behavior_clips(self) -> bool:
        return any(self._clips.get(c.value) for c in BEHAVIOR_CATEGORIES)

    async def make_chunk(self, category: LoopCategory, audio_wav: bytes) -> bytes:
        """
        Overlay audio_wav onto a looped video clip from the given category.
        Falls back to less-specific categories if clips are missing.
        Returns MP4 bytes.
        """
        clip = self._pick(category)
        duration = _wav_duration(audio_wav)

        import tempfile
        with tempfile.TemporaryDirectory(prefix="loop_") as tmp_str:
            tmp = Path(tmp_str)
            audio_path = tmp / "audio.wav"
            out_path = tmp / "out.mp4"
            audio_path.write_bytes(audio_wav)

            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-stream_loop", "-1",
                "-i", str(clip),
                "-i", str(audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-t", f"{duration:.3f}",
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
                    f"ffmpeg loop overlay failed:\n{stderr.decode()[-500:]}"
                )
            return out_path.read_bytes()

    async def idle_chunk(self, duration_s: float = 1.5) -> bytes:
        """Return a silent idle clip of the given duration."""
        clip = self._pick(LoopCategory.IDLE)

        import tempfile
        with tempfile.TemporaryDirectory(prefix="loop_idle_") as tmp_str:
            tmp = Path(tmp_str)
            out_path = tmp / "idle.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-stream_loop", "-1",
                "-i", str(clip),
                "-t", f"{duration_s:.3f}",
                "-c:v", "copy",
                "-an",
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
                    f"ffmpeg idle clip failed:\n{stderr.decode()[-500:]}"
                )
            return out_path.read_bytes()

    async def behavior_chunk(self, category: LoopCategory) -> bytes:
        """
        Return a pre-rendered behavior clip (look_away, drink_water, thinking).
        These are played as-is — they have embedded audio from the lipsync render.
        """
        clip = self._pick(category)
        return clip.read_bytes()

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
        chain = [category.value] + _FALLBACK.get(category.value, [])
        for cat_val in chain:
            clips = self._clips.get(cat_val)
            if clips:
                return random.choice(clips)
        raise FileNotFoundError(
            "No video loops found in any category. "
            "Run:  python generate_loops.py"
        )


# --------------------------------------------------------------------------- #
#  Helper                                                                       #
# --------------------------------------------------------------------------- #

def _wav_duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.getnframes() / wf.getframerate()
