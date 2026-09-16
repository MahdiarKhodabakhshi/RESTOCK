"""OpenAI-compatible chat client for the student model."""

import itertools
import re
import threading
from collections.abc import Sequence
from typing import Any

import httpx

from restock.agent.prompt import SYSTEM

THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class ModelError(RuntimeError):
    """The student server did not return a usable completion."""


class Student:
    """Chat completions against one or more vLLM replicas."""

    def __init__(
        self,
        urls: Sequence[str],
        model: str = "student",
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 600,
        transport: httpx.BaseTransport | None = None,
    ):
        if not urls:
            raise ValueError("at least one server URL is required")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self._urls = itertools.cycle(url.rstrip("/") for url in urls)
        self._lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return the assistant message dict from one chat completion."""
        with self._lock:
            url = next(self._urls)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages
            or [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools is not None:
            body["tools"] = tools
        try:
            response = self._http.post(f"{url}/chat/completions", json=body)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelError(f"{url}: {error}") from error
        content = message.get("content") or ""
        message = {**message, "content": THINKING.sub("", content).strip()}
        return message

    def close(self) -> None:
        self._http.close()
