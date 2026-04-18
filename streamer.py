"""
ffmpeg-based RTMP streamer with multi-platform tee output and auto-reconnect.

Multi-platform: uses ffmpeg's tee pseudo-muxer to encode once and send to
TikTok, YouTube, Facebook, and Instagram simultaneously.

Auto-reconnect: if ffmpeg exits unexpectedly (network drop, platform restart),
the watchdog restarts it with exponential backoff, up to max_reconnect_attempts.
"""
import asyncio
import logging
import os
import tempfile
from pathlib import Path

from config import StreamConfig, StreamDestination

log = logging.getLogger(__name__)


class RTMPStreamer:
    def __init__(self, cfg: StreamConfig):
        self._cfg = cfg
        self._proc: asyncio.subprocess.Process | None = None
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._chunk_index = 0
        self._running = False
        self._reconnect_count = 0

    # ---------------------------------------------------------------------- #
    #  Lifecycle                                                               #
    # ---------------------------------------------------------------------- #

    async def start(self) -> None:
        if not self._cfg.destinations:
            raise RuntimeError("No stream destinations configured — check .env")
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="stream_")
        self._running = True
        self._proc = await self._launch_ffmpeg()
        names = ", ".join(d.name.upper() for d in self._cfg.destinations if d.enabled)
        log.info("Streaming to: %s", names)

    async def stop(self) -> None:
        self._running = False
        await self._kill_proc()
        if self._tmp_dir:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

    # ---------------------------------------------------------------------- #
    #  Sending chunks                                                          #
    # ---------------------------------------------------------------------- #

    async def send_chunk(self, mp4_bytes: bytes) -> None:
        if not self._tmp_dir:
            raise RuntimeError("Streamer not started")

        # Restart ffmpeg if it died
        if self._proc is None or self._proc.returncode is not None:
            await self._reconnect()

        chunk_path = os.path.join(
            self._tmp_dir.name, f"chunk_{self._chunk_index:06d}.mp4"
        )
        self._chunk_index += 1

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, Path(chunk_path).write_bytes, mp4_bytes)

        # Escape single quotes in path for ffmpeg concat format
        safe_path = chunk_path.replace("'", "'\\''")
        line = f"file '{safe_path}'\n"
        try:
            self._proc.stdin.write(line.encode())
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            log.warning("ffmpeg stdin broken — triggering reconnect")
            await self._reconnect()

    # ---------------------------------------------------------------------- #
    #  Auto-reconnect                                                          #
    # ---------------------------------------------------------------------- #

    async def _reconnect(self) -> None:
        if self._reconnect_count >= self._cfg.max_reconnect_attempts:
            raise RuntimeError(
                f"ffmpeg failed {self._reconnect_count} times — giving up"
            )
        delay = min(
            self._cfg.reconnect_delay_s * (2 ** self._reconnect_count), 60.0
        )
        self._reconnect_count += 1
        log.warning(
            "Stream disconnected (attempt %d/%d) — reconnecting in %.0f s …",
            self._reconnect_count,
            self._cfg.max_reconnect_attempts,
            delay,
        )
        await self._kill_proc()
        await asyncio.sleep(delay)
        self._proc = await self._launch_ffmpeg()
        log.info("Stream reconnected (attempt %d)", self._reconnect_count)
        self._reconnect_count = 0   # reset on success

    async def _kill_proc(self) -> None:
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

    # ---------------------------------------------------------------------- #
    #  ffmpeg process                                                          #
    # ---------------------------------------------------------------------- #

    def _build_vf(self) -> str:
        cfg = self._cfg
        filters = [
            f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=decrease",
            f"pad={cfg.width}:{cfg.height}:(ow-iw)/2:(oh-ih)/2:black",
        ]
        if cfg.show_disclosure:
            label = cfg.disclosure_text.replace("'", "\\'").replace(":", "\\:")
            filters.append(
                f"drawtext=text='{label}':fontsize=28:fontcolor=white"
                f":alpha=0.80:box=1:boxcolor=black@0.50:boxborderw=8:x=16:y=16"
            )
        if cfg.show_promo_banner and cfg.promo_text:
            promo = cfg.promo_text.replace("'", "\\'").replace(":", "\\:")
            filters.append(
                f"drawtext=text='{promo}':fontsize=30:fontcolor=white"
                f":alpha=0.90:box=1:boxcolor=black@0.60:boxborderw=10"
                f":x=(w-text_w)/2:y=h-80"
            )
        return ",".join(filters)

    def _build_output(self) -> list[str]:
        active = [d for d in self._cfg.destinations if d.enabled and d.stream_key]
        if not active:
            raise RuntimeError("No enabled stream destinations with stream keys")

        bitrate_k = int(self._cfg.video_bitrate.rstrip("k"))
        common = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-b:v", self._cfg.video_bitrate,
            "-maxrate", self._cfg.video_bitrate,
            "-bufsize", f"{bitrate_k * 2}k",
            "-vf", self._build_vf(),
            "-r", str(self._cfg.fps),
            "-g", str(self._cfg.fps * 2),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", self._cfg.audio_bitrate,
            "-ar", str(self._cfg.audio_sample_rate),
        ]

        if len(active) == 1:
            # Single destination — simple FLV output
            d = active[0]
            return common + ["-f", "flv", d.target]

        # Multiple destinations — tee muxer (one encode, N outputs)
        # onfail=ignore means if one platform drops, others keep going
        tee_parts = "|".join(
            f"[f=flv:onfail=ignore]{d.target}" for d in active
        )
        return common + ["-f", "tee", tee_parts]

    def _launch_ffmpeg(self) -> asyncio.subprocess.Process:
        cmd = [
            "ffmpeg",
            "-re",
            "-f", "concat",
            "-safe", "0",
            "-protocol_whitelist", "file,pipe,rtmp,rtmps,tls,tcp",
            "-i", "pipe:0",
        ] + self._build_output()

        # Log command with keys redacted
        safe_cmd = _redact_keys(cmd, self._cfg.destinations)
        log.debug("ffmpeg: %s", " ".join(safe_cmd))

        return asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def log_errors(self) -> None:
        while self._running:
            if self._proc:
                try:
                    async for line in self._proc.stderr:
                        decoded = line.decode().rstrip()
                        if decoded:
                            log.debug("[ffmpeg] %s", decoded)
                except Exception:
                    pass
            await asyncio.sleep(0.5)


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _redact_keys(cmd: list[str], destinations: list[StreamDestination]) -> list[str]:
    result = []
    for token in cmd:
        redacted = token
        for d in destinations:
            if d.stream_key and d.stream_key in redacted:
                redacted = redacted.replace(d.stream_key, "***")
        result.append(redacted)
    return result
