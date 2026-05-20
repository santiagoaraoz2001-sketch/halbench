"""OpenRouter chat backend.

Covers 50+ models under one API. Use `model_id` exactly as listed on
https://openrouter.ai/models, e.g. "anthropic/claude-sonnet-4.6",
"openai/gpt-5.4", "google/gemini-3.1-pro-preview", "x-ai/grok-4.3".

Auth: set OPENROUTER_API_KEY env var (or pass `api_key=` to constructor).
Get a key at https://openrouter.ai/keys.
"""
from __future__ import annotations
import json
import os
import time

import requests

from halbench.backends.base import ChatBackend, ChatResponse


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterBackend(ChatBackend):
    name = "openrouter"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        referer: str = "https://github.com/santiagoaraoz2001-sketch/halbench",
        title: str = "halbench",
    ):
        self.model_id = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Get a key at https://openrouter.ai/keys "
                "and `export OPENROUTER_API_KEY=sk-or-v1-...`"
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.referer = referer
        self.title = title

    def chat(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> ChatResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

        backoff = 2.0
        t0 = time.time()
        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    OPENROUTER_URL, headers=headers, data=json.dumps(payload),
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    body = r.json()
                    choice = (body.get("choices") or [{}])[0]
                    msg = choice.get("message", {}).get("content", "") or ""
                    usage = body.get("usage", {})
                    return ChatResponse(
                        response_text=msg,
                        finish_reason=choice.get("finish_reason", "stop"),
                        usage_in_tokens=int(usage.get("prompt_tokens", 0)),
                        usage_out_tokens=int(usage.get("completion_tokens", 0)),
                        cost_usd=float(usage.get("cost", 0.0) or 0.0),
                        latency_s=time.time() - t0,
                    )
                if r.status_code in (429, 502, 503, 504):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                break
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(backoff)
                backoff *= 2
        return ChatResponse(
            response_text="", finish_reason="error", latency_s=time.time() - t0,
            error=last_err or "unknown",
        )
