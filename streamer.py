"""
ffmpeg-based RTMP streamer.

Supports TikTok LIVE and YouTube Live via the same RTMP/FLV pipeline.
Adds an AI disclosure overlay and an optional app-promo lower-third.
"""
import asyncio
import logging
import os
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

    # ---------------------------------------------------------------------- #
    #  Lifecycle                                                               #
    # ---------------------------------------------------------------------- #

    async def start(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="stream_")
        self._proc = await self._launch_ffmpeg()
        log.info("RTMP stream started → %s", self._rtmp_target)

    async def stop(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
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
        if not self._proc or not self._tmp_dir:
            raise RuntimeError("Streamer not started")

        chunk_path = os.path.join(
            self._tmp_dir.name, f"chunk_{self._chunk_index:06d}.mp4"
        )
        self._chunk_index += 1

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, Path(chunk_path).write_bytes, mp4_bytes)

        line = f"file '{chunk_path}'\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

    # ---------------------------------------------------------------------- #
    #  ffmpeg process                                                          #
    # ---------------------------------------------------------------------- #

    @property
    def _rtmp_target(self) -> str:
        url = self._cfg.rtmp_url.rstrip("/")
        key = self._cfg.stream_key
        # YouTube uses ?... query params already embedded in the key
        sep = "/" if self._cfg.platform == "tiktok" else "/"
        return f"{url}{sep}{key}"

    def _build_vf(self) -> str:
        cfg = self._cfg
        # Base scale + pad to portrait 9:16
        filters = [
            f"scale={cfg.width}:{cfg.height}:"
            f"force_original_aspect_ratio=decrease",
            f"pad={cfg.width}:{cfg.height}:(ow-iw)/2:(oh-ih)/2:black",
        ]

        # AI disclosure badge (top-left, semi-transparent)
        if cfg.show_disclosure:
            label = cfg.disclosure_text.replace("'", "\\'")
            filters.append(
                f"drawtext=text='{label}'"
                f":fontsize=28"
                f":fontcolor=white"
                f":alpha=0.75"
                f":box=1:boxcolor=black@0.45:boxborderw=6"
                f":x=16:y=16"
            )

        # Optional lower-third app promo banner (bottom strip)
        if cfg.show_promo_banner and cfg.promo_text:
            promo = cfg.promo_text.replace("'", "\\'")
            filters.append(
                f"drawtext=text='{promo}'"
                f":fontsize=32"
                f":fontcolor=white"
                f":alpha=0.9"
                f":box=1:boxcolor=black@0.55:boxborderw=10"
                f":x=(w-text_w)/2"
                f":y=h-80"
            )

        return ",".join(filters)

    def _launch_ffmpeg(self) -> asyncio.subprocess.Process:
        cfg = self._cfg
        bitrate_k = int(cfg.video_bitrate.rstrip("k"))
        cmd = [
            "ffmpeg",
            "-re",
            "-f", "concat",
            "-safe", "0",
            "-protocol_whitelist", "file,pipe",
            "-i", "pipe:0",
            # Video
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-b:v", cfg.video_bitrate,
            "-maxrate", cfg.video_bitrate,
            "-bufsize", f"{bitrate_k * 2}k",
            "-vf", self._build_vf(),
            "-r", str(cfg.fps),
            "-g", str(cfg.fps * 2),
            "-pix_fmt", "yuv420p",
            # Audio
            "-c:a", "aac",
            "-b:a", cfg.audio_bitrate,
            "-ar", str(cfg.audio_sample_rate),
            # Output
            "-f", "flv",
            self._rtmp_target,
        ]
        log.debug("ffmpeg: %s", " ".join(cmd))
        return asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def log_errors(self) -> None:
        if not self._proc:
            return
        async for line in self._proc.stderr:
            decoded = line.decode().rstrip()
            if decoded:
                log.debug("[ffmpeg] %s", decoded)
