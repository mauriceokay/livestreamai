"""
Chat moderator — filters incoming messages before they reach the LLM.

Checks applied in order:
  1. Length cap (drop extremely long messages)
  2. Per-user rate limit (max N messages per minute)
  3. Exact-spam detection (same message repeated in recent history)
  4. Profanity filter (configurable word list + regex)

All checks are O(1) or O(small constant) — this runs on every chat event
so it must be fast.
"""
import re
import time
from collections import defaultdict, deque

from config import ModeratorConfig

# Basic profanity list — extend or replace with a proper dataset as needed.
# Stored as compiled regex for speed.
_DEFAULT_BLOCKED = [
    r"\bfuck\b", r"\bshit\b", r"\bcunt\b", r"\bn[i1]gg[ae]r\b",
    r"\bfaggot\b", r"\bretard\b", r"\bdickhead\b", r"\basshole\b",
]
_PROFANITY_RE = re.compile(
    "|".join(_DEFAULT_BLOCKED),
    re.IGNORECASE,
)


class ChatModerator:
    def __init__(self, cfg: ModeratorConfig):
        self._cfg = cfg
        # Per-user: deque of message timestamps (last 60 s)
        self._user_times: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=cfg.max_msg_per_minute * 2)
        )
        # Global recent message fingerprints for spam detection
        self._recent: deque[str] = deque(maxlen=cfg.spam_window)
        self._blocked_count = 0

    def allow(self, username: str, text: str) -> bool:
        """Return True if the message should be forwarded to the LLM."""
        if not self._cfg.enabled:
            return True

        # 1. Length
        if len(text) > self._cfg.max_msg_length:
            return False

        # 2. Per-user rate limit
        now = time.monotonic()
        times = self._user_times[username]
        # Expire messages older than 60 s
        while times and now - times[0] > 60.0:
            times.popleft()
        if len(times) >= self._cfg.max_msg_per_minute:
            return False
        times.append(now)

        # 3. Spam detection
        if self._cfg.block_spam:
            fingerprint = text.strip().lower()
            if fingerprint in self._recent:
                return False
            self._recent.append(fingerprint)

        # 4. Profanity
        if _PROFANITY_RE.search(text):
            self._blocked_count += 1
            return False

        return True

    def stats(self) -> dict:
        return {"blocked_total": self._blocked_count}
