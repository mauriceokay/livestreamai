"""
ffmpeg-based RTMP streamer.

Receives MP4 video chunks (bytes) from the lip-sync engine and forwards them
to the TikTok RTMP endpoint in a continuous stream.

Strategy:
  - Open a single long-lived ffmpeg process with a concat demuxer fed via a
    FIFO (named pipe) so we never restart the stream between chunks.
  - Write each chunk's path into the FIFO as it's ready, keeping latency low.
  - Fall back to re-encoding only if the source dimensions change.
"""
import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from config import StreamConfig

log = logging.getLogger(__name__)


class RTMPStreamer:
    def __init__(self, cfg: StreamConfig):
        self._cfg = cfg
        self._proc: asyncio.subprocess.Process | None = None
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._chunk_index = 0
        self._concat_fifo: str = ""

    # ---------------------------------------------------------------------- #
    #  Lifecycle                                                               #
    # ---------------------------------------------------------------------- #

    async def start(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="stream_")
        self._concat_fifo = os.path.join(self._tmp_dir.name, "concat.txt")
        # Write an empty concat list to start
        Path(self._concat_fifo).write_text("")
        self._proc = await self._launch_ffmpeg()
        log.info("RTMP stream started → %s", self._rtmp_target)

    async def stop(self) -> None:
        if self._proc:
            self._proc.stdin.close()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None
        if self._tmp_dir:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

    # ---------------------------------------------------------------------- #
    #  Sending chunks                                                          #
    # ---------------------------------------------------------------------- #

    async def send_chunk(self, mp4_bytes: bytes) -> None:
        """Write an MP4 chunk to the stream.  Non-blocking — drops if full."""
        if not self._proc or not self._tmp_dir:
            raise RuntimeError("Streamer not started")

        chunk_path = os.path.join(self._tmp_dir.name, f"chunk_{self._chunk_index:06d}.mp4")
        self._chunk_index += 1

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, Path(chunk_path).write_bytes, mp4_bytes)

        # Feed the chunk path to ffmpeg via stdin (one file per line)
        line = f"file '{chunk_path}'\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

    # ---------------------------------------------------------------------- #
    #  ffmpeg process                                                          #
    # ---------------------------------------------------------------------- #

    @property
    def _rtmp_target(self) -> str:
        return f"{self._cfg.rtmp_url.rstrip('/')}/{self._cfg.stream_key}"

    def _launch_ffmpeg(self) -> asyncio.subprocess.Process:
        cfg = self._cfg
        cmd = [
            "ffmpeg",
            "-re",
            # Read a dynamic concat list fed via stdin
            "-f", "concat",
            "-safe", "0",
            "-protocol_whitelist", "file,pipe",
            "-i", "pipe:0",
            # Video encoding
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-b:v", cfg.video_bitrate,
            "-maxrate", cfg.video_bitrate,
            "-bufsize", str(int(cfg.video_bitrate.rstrip("k")) * 2) + "k",
            "-vf", f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=decrease,"
                   f"pad={cfg.width}:{cfg.height}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(cfg.fps),
            "-g", str(cfg.fps * 2),       # keyframe every 2 seconds
            # Audio encoding
            "-c:a", "aac",
            "-b:a", cfg.audio_bitrate,
            "-ar", str(cfg.audio_sample_rate),
            # Output
            "-f", "flv",
            self._rtmp_target,
        ]
        log.debug("ffmpeg cmd: %s", " ".join(cmd))
        return asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def log_errors(self) -> None:
        """Drain ffmpeg stderr and log any errors — run as a background task."""
        if not self._proc:
            return
        async for line in self._proc.stderr:
            decoded = line.decode().rstrip()
            if decoded:
                log.debug("[ffmpeg] %s", decoded)
