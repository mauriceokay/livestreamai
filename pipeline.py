"""
Main async pipeline — orchestrates all stages end-to-end.

New flow (loop-library approach, ~650 ms end-to-end latency):

  TikTok/YouTube chat
        │
        ▼
  Brain (LLM) — generates spoken text
        │
        ▼
  TTS engine — converts text → WAV bytes  (~200 ms)
        │
        ▼
  LoopLibrary.make_chunk()                (~50 ms)
    • picks a pre-rendered animated video loop
    • ffmpeg overlays the live TTS audio onto it
        │
        ▼
  RTMPStreamer — pushes MP4 chunk to TikTok/YouTube via ffmpeg/RTMP

Run:
  python pipeline.py
"""
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import Config
from brain import Brain, SpeechMode, SpeechRequest
from tts import build_tts
from loop_library import LoopLibrary, LoopCategory, MODE_TO_CATEGORY
from streamer import RTMPStreamer
from chat_reader import ChatReader, EventType, LiveEvent
from persona import idle_seed, react_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("pipeline")


class Pipeline:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._chat_queue: asyncio.Queue[LiveEvent] = asyncio.Queue(maxsize=100)
        self._speech_queue: asyncio.Queue[tuple[SpeechMode, bytes]] = asyncio.Queue(
            maxsize=cfg.max_queue_depth
        )
        self._video_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=cfg.max_queue_depth
        )
        self._brain = Brain(cfg.llm, cfg.persona)
        self._tts = build_tts(cfg.tts)
        self._loops = LoopLibrary(cfg.lipsync.loops_dir)
        self._streamer = RTMPStreamer(cfg.stream)
        self._chat_reader = ChatReader(cfg.tiktok_username, self._chat_queue)
        self._viewer_count: int = 0

    # ---------------------------------------------------------------------- #
    #  Start / stop                                                            #
    # ---------------------------------------------------------------------- #

    async def run(self) -> None:
        self._validate()
        await self._streamer.start()

        tasks = [
            asyncio.create_task(self._chat_reader.start(),    name="chat-reader"),
            asyncio.create_task(self._speech_stage(),          name="speech"),
            asyncio.create_task(self._video_stage(),           name="video"),
            asyncio.create_task(self._stream_stage(),          name="stream"),
            asyncio.create_task(self._streamer.log_errors(),   name="ffmpeg-log"),
        ]

        log.info("Pipeline running on %s. Press Ctrl+C to stop.",
                 self._cfg.stream.platform.upper())
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self._teardown(tasks)

    async def _teardown(self, tasks: list[asyncio.Task]) -> None:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._chat_reader.stop()
        await self._streamer.stop()
        await self._brain.close()
        log.info("Pipeline shut down.")

    # ---------------------------------------------------------------------- #
    #  Stage 1 — Brain + TTS                                                  #
    #  Input:  chat events (or idle timer)                                    #
    #  Output: (SpeechMode, WAV bytes) → _speech_queue                       #
    # ---------------------------------------------------------------------- #

    async def _speech_stage(self) -> None:
        while True:
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
                )

            if req is None:
                continue

            try:
                t0 = time.monotonic()
                text = await self._brain.speak(req)
                log.info("[%s] %s", req.mode.value.upper(), text)

                wav = await self._tts.synthesize(text)
                log.debug("TTS done in %.0f ms", (time.monotonic() - t0) * 1000)

                # Drop oldest if queue is backed up
                if self._speech_queue.full():
                    try:
                        self._speech_queue.get_nowait()
                        log.debug("Speech queue full — dropped oldest chunk")
                    except asyncio.QueueEmpty:
                        pass
                await self._speech_queue.put((req.mode, wav))

            except Exception as exc:
                log.error("Speech stage: %s", exc)

    # ---------------------------------------------------------------------- #
    #  Stage 2 — Loop library: overlay audio on pre-rendered video            #
    #  Input:  (SpeechMode, WAV bytes) from _speech_queue                    #
    #  Output: MP4 bytes → _video_queue                                      #
    # ---------------------------------------------------------------------- #

    async def _video_stage(self) -> None:
        while True:
            try:
                mode, wav = await asyncio.wait_for(
                    self._speech_queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                # No speech queued — push a silent idle frame to keep stream alive
                try:
                    idle_mp4 = await self._loops.idle_chunk(duration_s=1.5)
                    await self._video_queue.put(idle_mp4)
                except Exception as exc:
                    log.warning("Idle chunk failed: %s", exc)
                continue

            try:
                t0 = time.monotonic()
                category = MODE_TO_CATEGORY.get(mode, LoopCategory.TALKING)

                # React events get the higher-energy clips
                mp4 = await self._loops.make_chunk(category, wav)
                log.debug("Video chunk ready in %.0f ms", (time.monotonic() - t0) * 1000)

                if self._video_queue.full():
                    try:
                        self._video_queue.get_nowait()
                        log.debug("Video queue full — dropped oldest chunk")
                    except asyncio.QueueEmpty:
                        pass
                await self._video_queue.put(mp4)

            except Exception as exc:
                log.error("Video stage: %s", exc)

    # ---------------------------------------------------------------------- #
    #  Stage 3 — RTMP stream                                                  #
    #  Input:  MP4 bytes from _video_queue                                   #
    # ---------------------------------------------------------------------- #

    async def _stream_stage(self) -> None:
        while True:
            try:
                mp4 = await asyncio.wait_for(self._video_queue.get(), timeout=5.0)
                await self._streamer.send_chunk(mp4)
            except asyncio.TimeoutError:
                log.warning("Stream stage: no video for 5 s — is the video stage running?")
            except Exception as exc:
                log.error("Stream stage: %s", exc)

    # ---------------------------------------------------------------------- #
    #  Event → SpeechRequest mapping                                          #
    # ---------------------------------------------------------------------- #

    def _event_to_request(self, event: LiveEvent) -> SpeechRequest | None:
        if event.type == EventType.VIEWER_COUNT:
            self._viewer_count = event.viewer_count
            if self._viewer_count > 0 and self._viewer_count % 100 == 0:
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
                    f"Viewer {event.username!r} says: {event.text!r}\n"
                    "Reply naturally in one or two sentences."
                ),
                username=event.username,
                priority=1,
            )

        if event.type == EventType.FOLLOW:
            return SpeechRequest(
                mode=SpeechMode.REACT,
                user_prompt=react_prompt("follow", self._cfg.persona,
                                         username=event.username),
                username=event.username,
                priority=2,
            )

        if event.type == EventType.GIFT:
            return SpeechRequest(
                mode=SpeechMode.REACT,
                user_prompt=react_prompt("gift", self._cfg.persona,
                                         username=event.username),
                username=event.username,
                priority=3,
            )

        if event.type == EventType.SHARE:
            return SpeechRequest(
                mode=SpeechMode.REACT,
                user_prompt=react_prompt("share", self._cfg.persona,
                                         username=event.username),
                username=event.username,
                priority=1,
            )

        return None

    # ---------------------------------------------------------------------- #
    #  Validation                                                              #
    # ---------------------------------------------------------------------- #

    def _validate(self) -> None:
        errors = []
        cfg = self._cfg

        if not cfg.stream.rtmp_url or not cfg.stream.stream_key:
            errors.append("RTMP_URL and STREAM_KEY must be set in .env")
        if cfg.llm.backend == "claude" and not cfg.llm.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY required for claude LLM backend")
        if cfg.tts.backend == "elevenlabs" and not cfg.tts.elevenlabs_api_key:
            errors.append("ELEVENLABS_API_KEY required for elevenlabs TTS backend")
        if not Path(cfg.lipsync.base_face_image).exists():
            errors.append(
                f"Face image not found: {cfg.lipsync.base_face_image}\n"
                "   Put a portrait photo there, then run: python generate_loops.py"
            )
        if not self._loops.ready():
            missing = self._loops.missing_categories()
            errors.append(
                f"Video loop library not ready (missing: {', '.join(missing)})\n"
                "   Run:  python generate_loops.py"
            )
        if cfg.stream.platform == "tiktok" and not cfg.tiktok_username:
            log.warning("TIKTOK_USERNAME not set — chat reading disabled")

        if errors:
            for e in errors:
                log.error("Config error: %s", e)
            sys.exit(1)


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

async def main() -> None:
    cfg = Config().load_from_env()
    pipeline = Pipeline(cfg)

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        log.info("Shutdown signal received …")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    pipeline_task = asyncio.create_task(pipeline.run())
    await stop_event.wait()
    pipeline_task.cancel()
    await asyncio.gather(pipeline_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
