"""
LLM 工厂 — 统一管理 LLM 客户端创建, 支持 cloud / vLLM 双 provider 切换。

vLLM 提供 OpenAI 兼容 API, 因此无论 cloud 还是 vLLM, 底层均使用
langchain_openai.ChatOpenAI, 仅 base_url / api_key / model 不同。

切换方式:
    环境变量 ECOM_LLM_PROVIDER=vllm  → 路由到本地 vLLM 推理服务
    环境变量 ECOM_LLM_PROVIDER=cloud → 路由到云端 API (DeepSeek / MiniMax / OpenAI 等)
"""
from __future__ import annotations

from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from config import get_settings

logger = structlog.get_logger()


def get_llm_provider() -> str:
    """Return current LLM provider name: 'cloud' or 'vllm'."""
    return get_settings().llm_provider


def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance routed to the active provider.

    Args:
        temperature: Sampling temperature; defaults to settings.llm_temperature.
        max_tokens: Max output tokens; defaults to settings.llm_max_tokens.
        **kwargs: Extra kwargs forwarded to ChatOpenAI (e.g. streaming=True).

    Returns:
        ChatOpenAI configured for cloud API or local vLLM server.
    """
    settings = get_settings()
    provider = settings.llm_provider

    if temperature is None:
        temperature = settings.llm_temperature
    if max_tokens is None:
        max_tokens = settings.llm_max_tokens

    if provider == "vllm":
        api_key = settings.vllm_api_key_str or "EMPTY"
        logger.info(
            "llm.factory.vllm",
            base_url=settings.vllm_base_url,
            model=settings.vllm_model,
        )
        return ChatOpenAI(
            api_key=api_key,
            base_url=settings.vllm_base_url,
            model=settings.vllm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    # default: cloud provider
    api_key = settings.llm_api_key_str
    if not api_key:
        logger.warning("llm.factory.no_api_key", msg="ECOM_LLM_API_KEY not set; LLM calls will fail until configured")
        api_key = "not-configured"
    logger.info(
        "llm.factory.cloud",
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    return ChatOpenAI(
        api_key=api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
