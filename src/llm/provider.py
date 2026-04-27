"""Shared LLM provider configuration.

Both OpenAI and DeepSeek expose an OpenAI-compatible chat API, so the project can
use LangChain's ChatOpenAI wrapper for both providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI

from logging_config.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model_name: str
    api_key: Optional[str]
    base_url: Optional[str] = None


def _get_api_key(env_name: str) -> Optional[str]:
    value = os.getenv(env_name, "").strip()
    if not value:
        return None
    if value.lower().startswith("your_") or value.lower() in {"changeme", "replace_me"}:
        return None
    return value


def get_llm_config(provider: Optional[str] = None, model_name: Optional[str] = None) -> LLMConfig:
    """Return the active LLM provider configuration.

    LLM_PROVIDER can be "openai", "deepseek", or "auto" (default). In auto mode
    OpenAI is used when OPENAI_API_KEY is present; otherwise DeepSeek is used when
    DEEPSEEK_API_KEY is present.
    """
    requested_provider = (provider or os.getenv("LLM_PROVIDER", "auto")).strip().lower()
    openai_api_key = _get_api_key("OPENAI_API_KEY")
    deepseek_api_key = _get_api_key("DEEPSEEK_API_KEY")

    if requested_provider == "auto":
        if openai_api_key:
            resolved_provider = "openai"
        elif deepseek_api_key:
            resolved_provider = "deepseek"
        else:
            resolved_provider = "openai"
    else:
        resolved_provider = requested_provider

    if resolved_provider == "deepseek":
        return LLMConfig(
            provider="deepseek",
            model_name=model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=deepseek_api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    if resolved_provider == "openai":
        return LLMConfig(
            provider="openai",
            model_name=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_api_key,
        )

    logger.warning(f"Unsupported LLM_PROVIDER '{requested_provider}'. Falling back to OpenAI.")
    return LLMConfig(
        provider="openai",
        model_name=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=openai_api_key,
    )


def has_llm_credentials(provider: Optional[str] = None) -> bool:
    """Return True when the configured provider has an API key."""
    return bool(get_llm_config(provider).api_key)


def create_chat_llm(
    *,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    request_timeout: int = 120,
    max_retries: int = 3,
):
    """Create a ChatOpenAI-compatible LLM for the configured provider.

    Returns None when no API key is configured, matching the project's previous
    behavior of disabling optional LLM features without failing app startup.
    """
    config = get_llm_config(provider=provider, model_name=model_name)
    if not config.api_key:
        key_name = "DEEPSEEK_API_KEY" if config.provider == "deepseek" else "OPENAI_API_KEY"
        logger.warning(
            f"{key_name} not set. LLM features are disabled."
        )
        return None

    kwargs = {
        "model_name": config.model_name,
        "temperature": temperature,
        "openai_api_key": config.api_key,
        "max_retries": max_retries,
        "request_timeout": request_timeout,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if config.base_url:
        kwargs["base_url"] = config.base_url

    logger.info(f"Using {config.provider} LLM model {config.model_name}")
    return ChatOpenAI(**kwargs)
