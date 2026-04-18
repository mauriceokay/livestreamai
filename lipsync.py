"""
Lip-sync engine — used by generate_loops.py (one-time pre-rendering).
Not used during live streaming; the loop library handles that.

Supported models:
  latsync  — LatentSync (ByteDance, state-of-the-art, needs CUDA)
  wav2lip  — Wav2Lip GAN (older, CPU-compatible, lower quality)
  emo      — EMO diffusion model (most realistic, needs A100/4090)
  hallo2   — Hallo2 diffusion model (excellent long-form stability)

EMO/Hallo2 produce natural blinks, head sway, and micro-expressions
that make the face look genuinely alive rather than mechanically animated.
Generate loops once offline — the live stream just muxes pre-rendered clips.
"""
import asyncio
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from config import LipSyncConfig


class LipSync:
    def __init__(self, cfg: LipSyncConfig):
        self._cfg = cfg

    async def generate_chunk(self, wav_bytes: bytes) -> bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_generate, wav_bytes)

    def _sync_generate(self, wav_bytes: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav_path = tmp / "chunk.wav"
            out_path = tmp / "chunk.mp4"
            wav_path.write_bytes(wav_bytes)

            if self._cfg.model == "latsync":
                self._run_latsync(wav_path, out_path)
            elif self._cfg.model == "emo":
                self._run_emo(wav_path, out_path)
            elif self._cfg.model == "hallo2":
                self._run_hallo2(wav_path, out_path)
            else:
                self._run_wav2lip(wav_path, out_path)

            if not out_path.exists():
                raise RuntimeError("Lip-sync produced no output file")
            return out_path.read_bytes()

    def _run_latsync(self, wav_path: Path, out_path: Path) -> None:
        cmd = [
            "python", "-m", "latentsync.inference",
            "--video_path", self._cfg.base_face_image,
            "--audio_path", str(wav_path),
            "--output_path", str(out_path),
            "--checkpoint_path", self._cfg.latsync_checkpoint,
            "--device", self._cfg.device,
            "--fps", str(self._cfg.fps),
        ]
        _run(cmd, timeout=self._cfg.subprocess_timeout_s)

    def _run_wav2lip(self, wav_path: Path, out_path: Path) -> None:
        cmd = [
            "python", "Wav2Lip/inference.py",
            "--checkpoint_path", self._cfg.wav2lip_checkpoint,
            "--face", self._cfg.base_face_image,
            "--audio", str(wav_path),
            "--outfile", str(out_path),
            "--fps", str(self._cfg.fps),
            "--face_det_batch_size", str(self._cfg.face_det_batch_size),
            "--wav2lip_batch_size", str(self._cfg.wav2lip_batch_size),
            "--nosmooth",
        ]
        _run(cmd, timeout=self._cfg.subprocess_timeout_s)

    def _run_emo(self, wav_path: Path, out_path: Path) -> None:
        """
        EMO (HumanAIGC/EMO) — diffusion talking-head with natural expressions.
        Repo: https://github.com/HumanAIGC/EMO
        Setup: python setup.py --model emo
        """
        cmd = [
            "python", "EMO/run_demo.py",
            "--image_path",      self._cfg.base_face_image,
            "--audio_path",      str(wav_path),
            "--output_path",     str(out_path),
            "--checkpoint_dir",  self._cfg.emo_checkpoint,
            "--fps",             str(self._cfg.fps),
            "--device",          self._cfg.device,
        ]
        _run(cmd, timeout=self._cfg.subprocess_timeout_s)

    def _run_hallo2(self, wav_path: Path, out_path: Path) -> None:
        """
        Hallo2 (fudan-generative-vision/hallo2) — high-quality long-form talking head.
        Repo: https://github.com/fudan-generative-vision/hallo2
        Setup: python setup.py --model hallo2
        """
        cmd = [
            "python", "hallo2/scripts/inference.py",
            "--source_image",  self._cfg.base_face_image,
            "--driving_audio", str(wav_path),
            "--output",        str(out_path),
            "--config",        "hallo2/configs/inference/long.yaml",
            "--checkpoint",    self._cfg.hallo2_checkpoint,
        ]
        _run(cmd, timeout=self._cfg.subprocess_timeout_s)


def generate_idle_loop(face_image_path: str, fps: int, duration_s: float) -> bytes:
    frame = cv2.imread(face_image_path)
    if frame is None:
        raise FileNotFoundError(f"Face image not found: {face_image_path}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name

    try:
        total_frames = int(fps * duration_s)
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for _ in range(total_frames):
            writer.write(frame)
        writer.release()
        return Path(out_path).read_bytes()
    finally:
        Path(out_path).unlink(missing_ok=True)


def _run(cmd: list[str], timeout: int = 300) -> None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Lip-sync timed out after {timeout} s: {' '.join(cmd)}"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Lip-sync failed:\nCMD: {' '.join(cmd)}\n"
            f"STDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"
        )
