"""Thin Groq chat client built on the OpenAI-compatible REST endpoint.

We talk to Groq with plain ``requests`` rather than an SDK so the dependency
surface stays tiny and the wire format is obvious to anyone reading the code.
"""
from __future__ import annotations

import json
import time
from typing import Iterator

import requests

from .config import settings


class GroqError(RuntimeError):
    pass


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """Call Groq chat-completions and return the assistant text.

    Retries on transient 429/5xx with exponential backoff.
    """
    url = f"{settings.groq_base_url}/chat/completions"
    payload = {
        "model": model or settings.chat_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {settings.require_key()}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code in (429, 500, 502, 503):
                raise GroqError(f"transient {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, GroqError) as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    raise GroqError(f"Groq request failed after {max_retries} attempts: {last_err}")


def chat_stream(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """Stream assistant text token-by-token (SSE)."""
    url = f"{settings.groq_base_url}/chat/completions"
    payload = {
        "model": model or settings.chat_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {settings.require_key()}",
        "Content-Type": "application/json",
    }
    with requests.post(url, headers=headers, json=payload, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta
