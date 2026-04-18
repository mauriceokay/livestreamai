"""
Main async pipeline — orchestrates all stages end-to-end.

Flow (~650 ms end-to-end latency):

  TikTok / YouTube / Facebook / Instagram chat
        │
        ▼
  ChatModerator — filters spam, profanity, rate-limits
        │
        ▼
  Brain (LLM) — generates spoken text  (Claude Haiku or Ollama)
        │
        ▼
  TTS engine — text → WAV bytes         (~200 ms, ElevenLabs / Kokoro)
        │
        ▼
  LoopLibrary.make_chunk()              (~50 ms, ffmpeg audio overlay)
    • picks pre-rendered animated video loop matching speech mode
    • overlays live TTS audio track
        │
        ▼
  RTMPStreamer — ffmpeg tee → all platforms simultaneously

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
from loop_library import LoopLibrary, LoopCategory
from streamer import RTMPStreamer
from chat_reader import ChatReader, EventType, LiveEvent
from moderator import ChatModerator
from persona import idle_seed, react_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("pipeline")

# Maps SpeechMode to the video loop category to play
_MODE_CATEGORY: dict[SpeechMode, LoopCategory] = {
    SpeechMode.IDLE:       LoopCategory.IDLE,
    SpeechMode.CHAT_REPLY: LoopCategory.TALKING,
    SpeechMode.REACT:      LoopCategory.REACT_HAPPY,
}


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
        self._moderator = ChatModerator(cfg.moderator)
        self._chat_reader = ChatReader(cfg.tiktok_username, self._chat_queue)
        self._viewer_count: int = 0

    # ---------------------------------------------------------------------- #
    #  Start / stop                                                            #
    # ---------------------------------------------------------------------- #

    async def run(self) -> None:
        self._validate()
        await self._streamer.start()

        tasks = [
            asyncio.create_task(self._chat_reader.start(),  name="chat-reader"),
            asyncio.create_task(self._speech_stage(),        name="speech"),
            asyncio.create_task(self._video_stage(),         name="video"),
            asyncio.create_task(self._stream_stage(),        name="stream"),
            asyncio.create_task(self._streamer.log_errors(), name="ffmpeg-log"),
        ]

        platforms = ", ".join(
            d.name.upper() for d in self._cfg.stream.destinations if d.enabled
        )
        log.info("Live on: %s — Press Ctrl+C to stop.", platforms)

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
        await self._tts.close()
        stats = self._moderator.stats()
        log.info("Pipeline shut down. Moderator blocked %d messages.", stats["blocked_total"])

    # ---------------------------------------------------------------------- #
    #  Stage 1 — Brain + TTS                                                  #
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
                log.debug("Speech+TTS: %.0f ms", (time.monotonic() - t0) * 1000)
                _push_dropping(self._speech_queue, (req.mode, wav))
            except Exception as exc:
                log.error("Speech stage: %s", exc)

    # ---------------------------------------------------------------------- #
    #  Stage 2 — Loop library: overlay audio on pre-rendered video            #
    # ---------------------------------------------------------------------- #

    async def _video_stage(self) -> None:
        while True:
            try:
                mode, wav = await asyncio.wait_for(
                    self._speech_queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                try:
                    idle_mp4 = await self._loops.idle_chunk(duration_s=1.5)
                    _push_dropping(self._video_queue, idle_mp4)
                except Exception as exc:
                    log.warning("Idle chunk: %s", exc)
                continue

            try:
                t0 = time.monotonic()
                category = _MODE_CATEGORY.get(mode, LoopCategory.TALKING)
                # Gift reactions get the more excited loop category
                if mode == SpeechMode.REACT:
                    category = LoopCategory.REACT_GIFT
                mp4 = await self._loops.make_chunk(category, wav)
                log.debug("Video chunk: %.0f ms", (time.monotonic() - t0) * 1000)
                _push_dropping(self._video_queue, mp4)
            except Exception as exc:
                log.error("Video stage: %s", exc)

    # ---------------------------------------------------------------------- #
    #  Stage 3 — RTMP stream                                                  #
    # ---------------------------------------------------------------------- #

    async def _stream_stage(self) -> None:
        while True:
            try:
                mp4 = await asyncio.wait_for(self._video_queue.get(), timeout=5.0)
                await self._streamer.send_chunk(mp4)
            except asyncio.TimeoutError:
                log.warning("Stream stage: no video for 5 s")
            except Exception as exc:
                log.error("Stream stage: %s", exc)

    # ---------------------------------------------------------------------- #
    #  Event → SpeechRequest                                                  #
    # ---------------------------------------------------------------------- #

    def _event_to_request(self, event: LiveEvent) -> SpeechRequest | None:
        if event.type == EventType.VIEWER_COUNT:
            self._viewer_count = event.viewer_count
            if self._viewer_count > 0 and self._viewer_count % 100 == 0:
                return SpeechRequest(
                    mode=SpeechMode.REACT,
                    user_prompt=react_prompt(
                        "milestone", self._cfg.persona, count=self._viewer_count
                    ),
                    priority=2,
                )
            return None

        if event.type == EventType.CHAT:
            if not self._moderator.allow(event.username, event.text):
                log.debug("Moderated: [%s] %r", event.username, event.text[:60])
                return None
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

        if not cfg.stream.destinations:
            errors.append(
                "No stream destinations configured.\n"
                "   Set at least one of: TIKTOK_STREAM_KEY, YOUTUBE_STREAM_KEY, "
                "FACEBOOK_STREAM_KEY, INSTAGRAM_STREAM_KEY"
            )
        if cfg.llm.backend == "claude" and not cfg.llm.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY required for claude LLM backend")
        if cfg.tts.backend == "elevenlabs" and not cfg.tts.elevenlabs_api_key:
            errors.append("ELEVENLABS_API_KEY required for elevenlabs TTS backend")
        if not Path(cfg.lipsync.base_face_image).exists():
            errors.append(
                f"Face image not found: {cfg.lipsync.base_face_image}\n"
                "   Add a portrait photo, then run: python generate_loops.py"
            )
        if not self._loops.ready():
            missing = self._loops.missing_categories()
            errors.append(
                f"Video loop library incomplete (missing: {', '.join(missing)})\n"
                "   Run:  python generate_loops.py"
            )
        if not cfg.tiktok_username:
            log.warning("TIKTOK_USERNAME not set — TikTok chat reading disabled")

        if errors:
            for e in errors:
                log.error("  %s", e)
            sys.exit(1)


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _push_dropping(queue: asyncio.Queue, item) -> None:
    """Put item in queue, dropping the oldest entry if full."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(item)


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

async def main() -> None:
    cfg = Config().load_from_env()
    pipeline = Pipeline(cfg)
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    pipeline_task = asyncio.create_task(pipeline.run())
    await stop_event.wait()
    log.info("Shutdown signal received …")
    pipeline_task.cancel()
    await asyncio.gather(pipeline_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
