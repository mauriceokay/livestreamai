"""
LLM brain — generates the streamer's next spoken line.
Supports Claude (default, lowest latency) and Ollama (local, free).
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
    user_prompt: str          # the text sent to the LLM
    username: str = ""        # chat username if applicable
    priority: int = 0         # higher = more important, processed first


class Brain:
    def __init__(self, llm_cfg: LLMConfig, persona_cfg: PersonaConfig):
        self._llm = llm_cfg
        self._system = build_system_prompt(persona_cfg)
        self._history: list[dict] = []   # short rolling context window
        self._max_history = 6            # keep last 3 exchanges

        if llm_cfg.backend == "claude":
            self._client = anthropic.AsyncAnthropic(api_key=llm_cfg.anthropic_api_key)
        else:
            self._http = httpx.AsyncClient(base_url=llm_cfg.ollama_url, timeout=30)

    async def speak(self, req: SpeechRequest) -> str:
        """Generate a short spoken response for the given request."""
        if self._llm.backend == "claude":
            return await self._claude(req)
        return await self._ollama(req)

    async def speak_stream(self, req: SpeechRequest) -> AsyncIterator[str]:
        """Stream the response token-by-token (useful for very low-latency TTS chunking)."""
        if self._llm.backend == "claude":
            async for token in self._claude_stream(req):
                yield token
        else:
            text = await self._ollama(req)
            yield text

    # ------------------------------------------------------------------ #
    #  Claude backend                                                       #
    # ------------------------------------------------------------------ #

    async def _claude(self, req: SpeechRequest) -> str:
        messages = self._build_messages(req)
        resp = await self._client.messages.create(
            model=self._llm.claude_model,
            max_tokens=self._llm.max_tokens,
            temperature=self._llm.temperature,
            system=self._system,
            messages=messages,
        )
        text = resp.content[0].text.strip()
        self._push_history(req.user_prompt, text)
        return text

    async def _claude_stream(self, req: SpeechRequest) -> AsyncIterator[str]:
        messages = self._build_messages(req)
        full = []
        async with self._client.messages.stream(
            model=self._llm.claude_model,
            max_tokens=self._llm.max_tokens,
            temperature=self._llm.temperature,
            system=self._system,
            messages=messages,
        ) as stream:
            async for token in stream.text_stream:
                full.append(token)
                yield token
        self._push_history(req.user_prompt, "".join(full).strip())

    # ------------------------------------------------------------------ #
    #  Ollama backend                                                       #
    # ------------------------------------------------------------------ #

    async def _ollama(self, req: SpeechRequest) -> str:
        messages = [{"role": "system", "content": self._system}]
        messages += self._build_messages(req)
        payload = {
            "model": self._llm.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._llm.temperature,
                "num_predict": self._llm.max_tokens,
            },
        }
        resp = await self._http.post("/api/chat", json=payload)
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
        self._push_history(req.user_prompt, text)
        return text

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_messages(self, req: SpeechRequest) -> list[dict]:
        msgs = list(self._history)
        msgs.append({"role": "user", "content": req.user_prompt})
        return msgs

    def _push_history(self, user_msg: str, assistant_msg: str) -> None:
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        # trim to max_history pairs
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2):]

    async def close(self) -> None:
        if self._llm.backend == "ollama":
            await self._http.aclose()
