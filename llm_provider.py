"""Standalone OpenAI-compatible chat provider for the Research Paper Companion.

Compatible with any provider that exposes an OpenAI-style /chat/completions
endpoint (Ollama, Gemini OpenAI-compat, FPT Cloud, OpenAI itself, etc.).
Configure via env vars:
    OPENAI_BASE_URL   e.g. https://generativelanguage.googleapis.com/v1beta/openai/
    OPENAI_API_KEY    your provider key
    MODES_LLM_MODEL   model name (set by caller; not read here)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Protocol


class ChatProvider(Protocol):
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Return assistant message text for the given chat payload."""


class OpenAIChatProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required (pip install openai)") from exc

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._model = model
        self._extra_body = extra_body or {}

    def generate(self, messages: List[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            extra_body=self._extra_body or None,
        )
        content = completion.choices[0].message.content
        return (content or "").strip()


def _extra_body_from_env() -> Optional[Dict[str, Any]]:
    raw = os.getenv("OPENAI_EXTRA_BODY_JSON")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OPENAI_EXTRA_BODY_JSON must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("OPENAI_EXTRA_BODY_JSON must decode to a JSON object")
    return parsed


def build_provider(model: str) -> Optional[ChatProvider]:
    """Build a chat provider from env vars. Returns None if no API key."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAIChatProvider(
            api_key=api_key,
            model=model,
            base_url=os.getenv("OPENAI_BASE_URL"),
            extra_body=_extra_body_from_env(),
        )
    except RuntimeError:
        return None
