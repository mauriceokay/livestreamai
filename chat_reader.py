"""
TikTok LIVE chat reader — pushes incoming events into an asyncio Queue.

Uses the TikTokLive library (pip install TikTokLive).
Emits typed ChatEvent objects so the pipeline doesn't need to know the
library internals.
"""
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Coroutine

log = logging.getLogger(__name__)


class EventType(str, Enum):
    CHAT = "chat"
    FOLLOW = "follow"
    GIFT = "gift"
    SHARE = "share"
    VIEWER_COUNT = "viewer_count"


@dataclass
class LiveEvent:
    type: EventType
    username: str = ""
    text: str = ""           # chat message text
    gift_name: str = ""      # gift name if type == GIFT
    viewer_count: int = 0    # populated for VIEWER_COUNT events


EventCallback = Callable[[LiveEvent], Coroutine]


class ChatReader:
    def __init__(self, tiktok_username: str, event_queue: asyncio.Queue[LiveEvent]):
        self._username = tiktok_username
        self._queue = event_queue
        self._client = None

    async def start(self) -> None:
        try:
            from TikTokLive import TikTokLiveClient                     # type: ignore
            from TikTokLive.events import (                              # type: ignore
                CommentEvent, FollowEvent, GiftEvent, ShareEvent,
                ViewerCountUpdateEvent,
            )
        except ImportError:
            log.warning(
                "TikTokLive not installed — chat reading disabled. "
                "Install with: pip install TikTokLive"
            )
            return

        client = TikTokLiveClient(unique_id=self._username)
        self._client = client

        @client.on(CommentEvent)
        async def on_comment(event: CommentEvent) -> None:
            await self._queue.put(LiveEvent(
                type=EventType.CHAT,
                username=event.user.unique_id,
                text=event.comment,
            ))

        @client.on(FollowEvent)
        async def on_follow(event: FollowEvent) -> None:
            await self._queue.put(LiveEvent(
                type=EventType.FOLLOW,
                username=event.user.unique_id,
            ))

        @client.on(GiftEvent)
        async def on_gift(event: GiftEvent) -> None:
            if event.gift.streakable and not event.gift.is_repeating:
                return  # wait for the streak to finish before reacting
            await self._queue.put(LiveEvent(
                type=EventType.GIFT,
                username=event.user.unique_id,
                gift_name=event.gift.name,
            ))

        @client.on(ShareEvent)
        async def on_share(event: ShareEvent) -> None:
            await self._queue.put(LiveEvent(
                type=EventType.SHARE,
                username=event.user.unique_id,
            ))

        @client.on(ViewerCountUpdateEvent)
        async def on_viewers(event: ViewerCountUpdateEvent) -> None:
            await self._queue.put(LiveEvent(
                type=EventType.VIEWER_COUNT,
                viewer_count=event.viewer_count,
            ))

        try:
            log.info("Connecting to TikTok LIVE: @%s", self._username)
            await client.start()
        except Exception as exc:
            log.error("TikTok LIVE connection error: %s", exc)

    async def stop(self) -> None:
        if self._client:
            try:
                await self._client.stop()
            except Exception:
                pass
            self._client = None
