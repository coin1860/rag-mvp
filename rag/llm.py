"""OpenAI-compatible chat client for the agent2api gateway.

Deliberately dependency-free (urllib). A failure here must be loud and specific,
because "the gateway is down" and "RAG retrieval is wrong" look identical from
the outside.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict], *, temperature: float = 0.2) -> str:
        if not self.api_key:
            raise LLMError("no API key: set RAG_API_KEY or OPENAI_API_KEY in .env")
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": temperature, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise LLMError(f"gateway HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"gateway unreachable at {self.base_url}: {exc}") from exc
        try:
            data = json.loads(body)
            return data["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"unexpected gateway response: {body[:200]}") from exc

    def reachable(self) -> tuple[bool, str]:
        try:
            self.chat([{"role": "user", "content": "ping"}], temperature=0.0)
            return True, "ok"
        except LLMError as exc:
            return False, str(exc)
