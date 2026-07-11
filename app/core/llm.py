"""LLM client service for AI-powered scientific discovery.

This module provides a provider-agnostic interface for querying large language
models.  Currently supported providers:

- **OpenAI** - requires ``OPENAI_API_KEY`` to be set.
- **Anthropic** - requires ``ANTHROPIC_API_KEY`` to be set.

The active provider is controlled by the ``LLM_PROVIDER`` setting.

Example::

    from app.core.llm import get_llm_client

    client = get_llm_client()
    response = await client.complete(
        system="You are a prosthetics research assistant.",
        prompt="Summarise the key challenges in myoelectric hand control.",
    )
    print(response.text)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMResponse:
    """Container for a completed LLM response.

    Attributes:
        text: The model's generated text.
        model: The model identifier used for the request.
        provider: The provider that served the request.
        usage: Token-usage metadata returned by the provider (may be empty).
    """

    def __init__(
        self,
        text: str,
        model: str,
        provider: str,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Initialise an LLMResponse.

        Args:
            text: Generated text from the model.
            model: Model identifier (e.g. ``"gpt-4o"``).
            provider: Provider name (``"openai"`` or ``"anthropic"``).
            usage: Optional token-usage statistics.
        """
        self.text = text
        self.model = model
        self.provider = provider
        self.usage: dict[str, int] = usage or {}


class BaseLLMClient(ABC):
    """Abstract base class for LLM provider clients."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return a completion.

        Args:
            prompt: The user's input / question.
            system: Optional system instruction to set assistant behaviour.
            max_tokens: Maximum tokens to generate in the response.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            LLMClientError: If the request fails or the provider returns an error.
        """


class LLMClientError(Exception):
    """Raised when an LLM provider request fails."""


class OpenAIClient(BaseLLMClient):
    """LLM client backed by the OpenAI Chat Completions API.

    Attributes:
        model: OpenAI model identifier.
        api_key: OpenAI API key.
    """

    _API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str) -> None:
        """Initialise the OpenAI client.

        Args:
            api_key: OpenAI API key (starts with ``sk-``).
            model: Model identifier, e.g. ``"gpt-4o"``.
        """
        self.api_key = api_key
        self.model = model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a chat completion request to the OpenAI API.

        Args:
            prompt: User prompt text.
            system: Optional system message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with the generated text.

        Raises:
            LLMClientError: On HTTP or API errors.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

        logger.info(
            "llm.openai.request_started",
            model=self.model,
            prompt_length=len(prompt),
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self._API_URL,
                    headers=headers,
                    content=json.dumps(payload),
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "llm.openai.request_failed",
                status_code=exc.response.status_code,
                exc_info=True,
            )
            raise LLMClientError(
                f"OpenAI API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("llm.openai.request_failed", exc_info=True)
            raise LLMClientError(f"OpenAI request error: {exc}") from exc

        data: dict[str, Any] = response.json()
        text: str = data["choices"][0]["message"]["content"]
        usage: dict[str, int] = data.get("usage", {})

        logger.info(
            "llm.openai.request_completed",
            model=self.model,
            response_length=len(text),
            total_tokens=usage.get("total_tokens", 0),
        )

        return LLMResponse(text=text, model=self.model, provider="openai", usage=usage)


class AnthropicClient(BaseLLMClient):
    """LLM client backed by the Anthropic Messages API.

    Attributes:
        model: Anthropic model identifier.
        api_key: Anthropic API key.
    """

    _API_URL = "https://api.anthropic.com/v1/messages"
    _API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str) -> None:
        """Initialise the Anthropic client.

        Args:
            api_key: Anthropic API key (starts with ``sk-ant-``).
            model: Model identifier, e.g. ``"claude-opus-4-5"``.
        """
        self.api_key = api_key
        self.model = model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a messages request to the Anthropic API.

        Args:
            prompt: User prompt text.
            system: Optional system message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with the generated text.

        Raises:
            LLMClientError: On HTTP or API errors.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self._API_VERSION,
            "Content-Type": "application/json",
        }

        logger.info(
            "llm.anthropic.request_started",
            model=self.model,
            prompt_length=len(prompt),
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self._API_URL,
                    headers=headers,
                    content=json.dumps(payload),
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "llm.anthropic.request_failed",
                status_code=exc.response.status_code,
                exc_info=True,
            )
            raise LLMClientError(
                f"Anthropic API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("llm.anthropic.request_failed", exc_info=True)
            raise LLMClientError(f"Anthropic request error: {exc}") from exc

        data: dict[str, Any] = response.json()
        text: str = data["content"][0]["text"]
        usage: dict[str, int] = data.get("usage", {})

        logger.info(
            "llm.anthropic.request_completed",
            model=self.model,
            response_length=len(text),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

        return LLMResponse(text=text, model=self.model, provider="anthropic", usage=usage)


def get_llm_client() -> BaseLLMClient:
    """Factory function that returns the configured LLM client.

    Reads ``LLM_PROVIDER``, ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``, and
    model settings from application config.

    Returns:
        An instance of :class:`OpenAIClient` or :class:`AnthropicClient`.

    Raises:
        ValueError: If the provider is unknown or the required API key is missing.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Set it in your .env file to enable LLM features."
            )
        return OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_model)

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured. "
                "Set it in your .env file to enable LLM features."
            )
        return AnthropicClient(
            api_key=settings.anthropic_api_key, model=settings.anthropic_model
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Supported providers: 'openai', 'anthropic'."
    )
