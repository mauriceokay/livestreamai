"""
Main async pipeline — orchestrates all stages end-to-end.

Flow:
  TikTok chat  ──►  Brain (LLM)  ──►  TTS  ──►  LipSync  ──►  RTMP streamer
       │                                                              ▲
       └── idle timer fires every N seconds ──────────────────────────┘

Run:
  python pipeline.py
"""
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

from config import Config
from brain import Brain, SpeechMode, SpeechRequest
from tts import build_tts
from lipsync import LipSync, generate_idle_loop
from streamer import RTMPStreamer
from chat_reader import ChatReader, EventType, LiveEvent
from persona import idle_seed, react_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("pipeline")


# --------------------------------------------------------------------------- #
#  Pipeline                                                                     #
# --------------------------------------------------------------------------- #

class Pipeline:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._chat_queue: asyncio.Queue[LiveEvent] = asyncio.Queue(maxsize=50)
        self._video_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=cfg.max_queue_depth
        )
        self._brain = Brain(cfg.llm, cfg.persona)
        self._tts = build_tts(cfg.tts)
        self._lipsync = LipSync(cfg.lipsync)
        self._streamer = RTMPStreamer(cfg.stream)
        self._chat_reader = ChatReader(cfg.tiktok_username, self._chat_queue)
        self._running = False
        self._last_speech_at: float = 0.0
        self._viewer_count: int = 0

    # ---------------------------------------------------------------------- #
    #  Start / stop                                                            #
    # ---------------------------------------------------------------------- #

    async def run(self) -> None:
        self._running = True
        await self._streamer.start()

        tasks = [
            asyncio.create_task(self._chat_reader.start(), name="chat-reader"),
            asyncio.create_task(self._speech_loop(), name="speech-loop"),
            asyncio.create_task(self._render_loop(), name="render-loop"),
            asyncio.create_task(self._stream_loop(), name="stream-loop"),
            asyncio.create_task(self._streamer.log_errors(), name="ffmpeg-log"),
        ]

        log.info("Pipeline running. Press Ctrl+C to stop.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown(tasks)

    async def _shutdown(self, tasks: list[asyncio.Task]) -> None:
        self._running = False
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._chat_reader.stop()
        await self._streamer.stop()
        await self._brain.close()
        log.info("Pipeline shut down cleanly.")

    # ---------------------------------------------------------------------- #
    #  Stage 1 — decide what to say and generate TTS audio                    #
    # ---------------------------------------------------------------------- #

    async def _speech_loop(self) -> None:
        """
        Pulls from the chat queue.  If nothing arrives within idle_interval_s,
        generates an idle remark instead.
        """
        while self._running:
            try:
                event: LiveEvent = await asyncio.wait_for(
                    self._chat_queue.get(),
                    timeout=self._cfg.idle_interval_s,
                )
                req = self._event_to_request(event)
            except asyncio.TimeoutError:
                req = SpeechRequest(
                    mode=SpeechMode.IDLE,
                    user_prompt=idle_seed(self._cfg.persona),
                    priority=0,
                )

            if req is None:
                continue

            try:
                text = await self._brain.speak(req)
                log.info("[%s] %s", req.mode.value.upper(), text)
                wav = await self._tts.synthesize(text)
                await self._video_queue.put(wav)   # pass WAV to render stage
                self._last_speech_at = time.monotonic()
            except Exception as exc:
                log.error("Speech stage error: %s", exc)

    def _event_to_request(self, event: LiveEvent) -> SpeechRequest | None:
        if event.type == EventType.VIEWER_COUNT:
            self._viewer_count = event.viewer_count
            # Only remark on round-number milestones
            if self._viewer_count % 100 == 0 and self._viewer_count > 0:
                return SpeechRequest(
                    mode=SpeechMode.REACT,
                    user_prompt=react_prompt(
                        "milestone", self._cfg.persona,
                        count=self._viewer_count,
                    ),
                    priority=2,
                )
            return None

        if event.type == EventType.CHAT:
            return SpeechRequest(
                mode=SpeechMode.CHAT_REPLY,
                user_prompt=(
                    f"Viewer {event.username!r} says in chat: {event.text!r}\n"
                    "Reply to them naturally in one or two sentences."
                ),
                username=event.username,
                priority=1,
            )

        if event.type == EventType.FOLLOW:
            return SpeechRequest(
                mode=SpeechMode.REACT,
                user_prompt=react_prompt("follow", self._cfg.persona, username=event.username),
                username=event.username,
                priority=2,
            )

        if event.type == EventType.GIFT:
            return SpeechRequest(
                mode=SpeechMode.REACT,
                user_prompt=react_prompt("gift", self._cfg.persona, username=event.username),
                username=event.username,
                priority=3,
            )

        if event.type == EventType.SHARE:
            return SpeechRequest(
                mode=SpeechMode.REACT,
                user_prompt=react_prompt("share", self._cfg.persona, username=event.username),
                username=event.username,
                priority=1,
            )

        return None

    # ---------------------------------------------------------------------- #
    #  Stage 2 — lip-sync rendering                                           #
    # ---------------------------------------------------------------------- #

    async def _render_loop(self) -> None:
        """Pulls WAV bytes from the video queue, runs lip-sync, pushes MP4."""
        # Second queue carries rendered MP4 chunks
        self._mp4_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=self._cfg.max_queue_depth
        )

        while self._running:
            try:
                wav_bytes: bytes = await asyncio.wait_for(
                    self._video_queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                # Inject a short idle frame so the stream never freezes
                try:
                    idle = generate_idle_loop(
                        self._cfg.lipsync.base_face_image,
                        fps=self._cfg.lipsync.fps,
                        duration_s=1.0,
                    )
                    await self._mp4_queue.put(idle)
                except Exception as exc:
                    log.warning("Idle frame generation failed: %s", exc)
                continue

            try:
                mp4 = await self._lipsync.generate_chunk(wav_bytes)
                # Drop oldest chunk if queue is full (prefer freshness)
                if self._mp4_queue.full():
                    try:
                        self._mp4_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await self._mp4_queue.put(mp4)
            except Exception as exc:
                log.error("Render stage error: %s", exc)

    # ---------------------------------------------------------------------- #
    #  Stage 3 — stream to RTMP                                               #
    # ---------------------------------------------------------------------- #

    async def _stream_loop(self) -> None:
        """Pulls MP4 chunks and feeds them to ffmpeg."""
        # Wait for the render loop to create _mp4_queue
        while not hasattr(self, "_mp4_queue"):
            await asyncio.sleep(0.05)

        while self._running:
            try:
                mp4: bytes = await asyncio.wait_for(self._mp4_queue.get(), timeout=2.0)
                await self._streamer.send_chunk(mp4)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                log.error("Stream stage error: %s", exc)


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

async def main() -> None:
    cfg = Config().load_from_env()

    # Basic validation
    errors = []
    if not cfg.tiktok_username:
        errors.append("TIKTOK_USERNAME not set")
    if not cfg.stream.rtmp_url or not cfg.stream.stream_key:
        errors.append("RTMP_URL and STREAM_KEY must be set")
    if cfg.llm.backend == "claude" and not cfg.llm.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY not set (required for claude backend)")
    if cfg.tts.backend == "elevenlabs" and not cfg.tts.elevenlabs_api_key:
        errors.append("ELEVENLABS_API_KEY not set (required for elevenlabs backend)")
    if not Path(cfg.lipsync.base_face_image).exists():
        errors.append(f"Face image not found: {cfg.lipsync.base_face_image}")
    if errors:
        for e in errors:
            log.error("Config error: %s", e)
        sys.exit(1)

    pipeline = Pipeline(cfg)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(pipeline._shutdown([])))

    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
