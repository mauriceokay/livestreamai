"""
Lip-sync engine — takes a WAV audio chunk + base face image and returns
a raw H.264 video chunk (bytes) that can be piped straight to ffmpeg.

Supported models:
  latsync  — LatentSync (ByteDance, state-of-the-art, needs CUDA GPU)
  wav2lip  — Wav2Lip GAN (older, runs on CPU, lower quality)
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
        self._face_img: np.ndarray | None = None
        self._model = None

    # ---------------------------------------------------------------------- #
    #  Public API                                                              #
    # ---------------------------------------------------------------------- #

    async def generate_chunk(self, wav_bytes: bytes) -> bytes:
        """
        Given WAV audio bytes, return an MP4 video chunk (bytes) of the face
        speaking those words.  Runs the heavy work in a thread pool so the
        async event loop is never blocked.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_generate, wav_bytes)

    # ---------------------------------------------------------------------- #
    #  Sync implementation (runs in thread pool)                              #
    # ---------------------------------------------------------------------- #

    def _sync_generate(self, wav_bytes: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav_path = tmp / "chunk.wav"
            out_path = tmp / "chunk.mp4"
            wav_path.write_bytes(wav_bytes)

            if self._cfg.model == "latsync":
                self._run_latsync(wav_path, out_path)
            else:
                self._run_wav2lip(wav_path, out_path)

            return out_path.read_bytes()

    # ---------------------------------------------------------------------- #
    #  LatentSync                                                              #
    # ---------------------------------------------------------------------- #

    def _run_latsync(self, wav_path: Path, out_path: Path) -> None:
        """
        Calls the LatentSync inference script.
        Expects the repo cloned to ./LatentSync/ with its conda env active,
        or installed as a package.
        """
        face = self._cfg.base_face_image
        ckpt = self._cfg.latsync_checkpoint
        cmd = [
            "python", "-m", "latentsync.inference",
            "--video_path", face,
            "--audio_path", str(wav_path),
            "--output_path", str(out_path),
            "--checkpoint_path", ckpt,
            "--device", self._cfg.device,
            "--fps", str(self._cfg.fps),
        ]
        _run(cmd)

    # ---------------------------------------------------------------------- #
    #  Wav2Lip                                                                 #
    # ---------------------------------------------------------------------- #

    def _run_wav2lip(self, wav_path: Path, out_path: Path) -> None:
        """
        Calls the Wav2Lip inference script.
        Expects the repo cloned to ./Wav2Lip/ with dependencies installed.
        """
        face = self._cfg.base_face_image
        ckpt = self._cfg.wav2lip_checkpoint
        cmd = [
            "python", "Wav2Lip/inference.py",
            "--checkpoint_path", ckpt,
            "--face", face,
            "--audio", str(wav_path),
            "--outfile", str(out_path),
            "--fps", str(self._cfg.fps),
            "--face_det_batch_size", str(self._cfg.face_det_batch_size),
            "--wav2lip_batch_size", str(self._cfg.wav2lip_batch_size),
            "--nosmooth",
        ]
        _run(cmd)


# ---------------------------------------------------------------------------- #
#  Idle-frame generator                                                          #
# ---------------------------------------------------------------------------- #

def generate_idle_loop(face_image_path: str, fps: int, duration_s: float) -> bytes:
    """
    Creates a short silent MP4 of the face image (no movement) to fill gaps
    when no audio is being generated.  Used as a placeholder between chunks.
    """
    frame = cv2.imread(face_image_path)
    if frame is None:
        raise FileNotFoundError(f"Face image not found: {face_image_path}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name

    total_frames = int(fps * duration_s)
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for _ in range(total_frames):
        writer.write(frame)
    writer.release()

    data = Path(out_path).read_bytes()
    Path(out_path).unlink(missing_ok=True)
    return data


# ---------------------------------------------------------------------------- #
#  Helper                                                                        #
# ---------------------------------------------------------------------------- #

def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Lip-sync command failed:\n{' '.join(cmd)}\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
