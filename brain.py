"""
LLM brain — generates the streamer's next spoken line.
Supports Claude (default, lowest latency) and Ollama (local, free).
Both backends retry with exponential backoff on transient failures.
"""
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

import anthropic
import httpx

from config import LLMConfig
from persona import build_system_prompt, PersonaConfig


class SpeechMode(str, Enum):
    CHAT_REPLY = "chat_reply"
    IDLE = "idle"
    REACT = "react"


@dataclass
class SpeechRequest:
    mode: SpeechMode
    user_prompt: str
    username: str = ""
    priority: int = 0


class Brain:
    def __init__(self, llm_cfg: LLMConfig, persona_cfg: PersonaConfig):
        self._llm = llm_cfg
        self._system = build_system_prompt(persona_cfg)
        self._history: list[dict] = []
        self._max_history = 6

        if llm_cfg.backend == "claude":
            self._client = anthropic.AsyncAnthropic(api_key=llm_cfg.anthropic_api_key)
        else:
            self._http = httpx.AsyncClient(
                base_url=llm_cfg.ollama_url,
                timeout=llm_cfg.ollama_timeout_s,
            )

    async def speak(self, req: SpeechRequest) -> str:
        return await _retry(
            lambda: self._speak_once(req),
            max_retries=self._llm.max_retries,
            base_delay=self._llm.retry_base_delay_s,
            label="LLM",
        )

    async def speak_stream(self, req: SpeechRequest) -> AsyncIterator[str]:
        if self._llm.backend == "claude":
            async for token in self._claude_stream(req):
                yield token
        else:
            yield await self.speak(req)

    # ------------------------------------------------------------------ #
    #  Internal                                                             #
    # ------------------------------------------------------------------ #

    async def _speak_once(self, req: SpeechRequest) -> str:
        if self._llm.backend == "claude":
            return await self._claude(req)
        return await self._ollama(req)

    async def _claude(self, req: SpeechRequest) -> str:
        resp = await self._client.messages.create(
            model=self._llm.claude_model,
            max_tokens=self._llm.max_tokens,
            temperature=self._llm.temperature,
            system=self._system,
            messages=self._build_messages(req),
        )
        if not resp.content or resp.content[0].type != "text":
            raise ValueError(f"Unexpected Claude response structure: {resp.stop_reason}")
        text = resp.content[0].text.strip()
        if not text:
            raise ValueError("Claude returned empty response")
        self._push_history(req.user_prompt, text)
        return text

    async def _claude_stream(self, req: SpeechRequest) -> AsyncIterator[str]:
        full = []
        async with self._client.messages.stream(
            model=self._llm.claude_model,
            max_tokens=self._llm.max_tokens,
            temperature=self._llm.temperature,
            system=self._system,
            messages=self._build_messages(req),
        ) as stream:
            async for token in stream.text_stream:
                full.append(token)
                yield token
        self._push_history(req.user_prompt, "".join(full).strip())

    async def _ollama(self, req: SpeechRequest) -> str:
        payload = {
            "model": self._llm.ollama_model,
            "messages": [{"role": "system", "content": self._system}]
                         + self._build_messages(req),
            "stream": False,
            "options": {
                "temperature": self._llm.temperature,
                "num_predict": self._llm.max_tokens,
            },
        }
        resp = await self._http.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "").strip()
        if not text:
            raise ValueError("Ollama returned empty response")
        self._push_history(req.user_prompt, text)
        return text

    def _build_messages(self, req: SpeechRequest) -> list[dict]:
        return list(self._history) + [{"role": "user", "content": req.user_prompt}]

    def _push_history(self, user_msg: str, assistant_msg: str) -> None:
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2):]

    async def close(self) -> None:
        if self._llm.backend == "ollama":
            await self._http.aclose()


# --------------------------------------------------------------------------- #
#  Shared retry helper                                                          #
# --------------------------------------------------------------------------- #

async def _retry(coro_fn, max_retries: int, base_delay: float, label: str = ""):
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except (
            anthropic.APIError,
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            httpx.HTTPError,
            httpx.TimeoutException,
        ) as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** attempt)
            import logging
            logging.getLogger(__name__).warning(
                "%s transient error (attempt %d/%d), retrying in %.1f s: %s",
                label, attempt + 1, max_retries, delay, exc,
            )
            await asyncio.sleep(delay)
    raise last_exc
