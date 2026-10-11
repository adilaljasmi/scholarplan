"""LLM access: interface, live client, deterministic mock and output schemas."""

from __future__ import annotations

from scholarplan.config import Settings
from scholarplan.llm.base import LLMClient, LLMError, Usage, call_json
from scholarplan.llm.mock import MockLLM, ScriptedLLM


def build_llm(settings: Settings) -> LLMClient:
    """Factory: choose the backend from configuration, not from agent code."""
    if settings.llm_backend == "groq":
        from scholarplan.llm.openai_compat import OpenAICompatibleClient
        return OpenAICompatibleClient(settings.groq_base_url, settings.groq_api_key,
                                      fallbacks={settings.models.critic: settings.models.critic_fallback})
    if settings.llm_backend == "mock":
        return MockLLM()
    raise LLMError(f"unknown LLM backend {settings.llm_backend!r}")


__all__ = ["LLMClient", "LLMError", "Usage", "call_json", "MockLLM", "ScriptedLLM", "build_llm"]
