"""
Pluggable LLM backend layer.

Abstract base class + two concrete implementations:
  - OllamaBackend  → local inference via Ollama REST API
  - DeepSeekBackend → cloud inference via DeepSeek OpenAI-compatible API

Usage:
    backend = get_backend()
    response = await backend.chat(messages)
"""
import json
from abc import ABC, abstractmethod
from loguru import logger
import httpx

from app.core.config import get_settings


class LLMBackend(ABC):
    """Abstract base for all LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        """Send messages and return the assistant reply as a string."""
        ...

    async def is_available(self) -> bool:
        """Health check — return True if backend is reachable."""
        return True


# ── Ollama ────────────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    """
    Local inference via Ollama.
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(self):
        self._settings = get_settings()
        self._base_url = self._settings.ollama_base_url
        self._model = self._settings.ollama_model

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        logger.debug(f"Ollama request → {url} model={self._model}")
        # 300s timeout: first run loads model into memory (~60-90s on iMac)
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


# ── DeepSeek ──────────────────────────────────────────────────

class DeepSeekBackend(LLMBackend):
    """
    Cloud inference via DeepSeek's OpenAI-compatible API.
    Docs: https://platform.deepseek.com/api-docs/
    """

    def __init__(self):
        self._settings = get_settings()
        self._api_key = self._settings.deepseek_api_key
        self._base_url = self._settings.deepseek_base_url
        self._model = self._settings.deepseek_model

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set in environment")

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.debug(f"DeepSeek request → {url} model={self._model}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def is_available(self) -> bool:
        return bool(self._api_key)


# ── Factory ───────────────────────────────────────────────────

def get_backend(override: str | None = None) -> LLMBackend:
    """
    Return the configured backend.
    If override is provided (from request), use that.
    Falls back to settings default → Ollama → DeepSeek.
    """
    settings = get_settings()
    selected = override or settings.llm_backend

    backends: dict[str, type[LLMBackend]] = {
        "ollama": OllamaBackend,
        "deepseek": DeepSeekBackend,
    }

    backend_class = backends.get(selected)
    if backend_class is None:
        raise ValueError(f"Unknown backend: '{selected}'. Choose 'ollama' or 'deepseek'.")

    logger.info(f"Using LLM backend: {selected}")
    return backend_class()