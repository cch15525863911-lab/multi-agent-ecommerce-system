"""
多模型路由器 — 按场景智能选择最优模型。

背景:
    不同 Agent 任务对模型能力的要求差异很大:
    - 文案生成 / 推荐重排 → 创意型任务, 用通用大模型 (DeepSeek-V3)
    - 风控推理 / 信用评估 → 逻辑推理型, 用推理模型 (DeepSeek-R1)
    - 意图识别 / 结构化提取 → 简单任务, 用通用模型 (DeepSeek-V3, 低延迟)

多模型路由策略:
    1. 按任务类型路由 (task_type → model)
    2. 按成本敏感度路由 (低成本优先 / 高质量优先)
    3. 降级路由 (主模型不可用时自动切换备用模型)

配置:
    通过环境变量 ECOM_MODEL_ROUTING_ENABLED 启用, 关闭时回退到单模型。
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from config import get_settings

logger = structlog.get_logger()

# 任务类型 → 推荐模型映射
TASK_MODEL_MAP = {
    # 简单任务: 结构化提取、分类、路由 → 用小模型省钱
    "intent_routing": {"model_key": "flash", "reason": "简单分类任务, 小模型足够"},
    "structured_extraction": {"model_key": "flash", "reason": "结构化提取, 小模型足够"},

    # 通用任务: 推荐重排、文案生成 → 用通用大模型
    "recommendation_rerank": {"model_key": "general", "reason": "需要语义理解和个性化"},
    "marketing_copy": {"model_key": "general", "reason": "需要创意和文案能力"},
    "user_profile_analysis": {"model_key": "general", "reason": "需要用户行为理解"},

    # 推理任务: 风控、授信、复杂决策 → 用推理模型
    "fraud_detection": {"model_key": "reasoning", "reason": "风控需要严谨逻辑推理"},
    "credit_assessment": {"model_key": "reasoning", "reason": "授信需要深度分析和推理"},
    "refund_risk": {"model_key": "reasoning", "reason": "退款风控需要多维度推理"},
    "meta_decision": {"model_key": "reasoning", "reason": "Meta-Agent 需要全局决策推理"},
    "supply_chain_planning": {"model_key": "general", "reason": "履约编排需要逻辑但不复杂"},
}


class MultiModelRouter:
    """多模型路由器 — 按任务类型智能选择最优模型。

    支持三种模型档位:
        - flash:     DeepSeek-V3 (简单任务, 低延迟)
        - general:   DeepSeek-V3 (常规任务)
        - reasoning: DeepSeek-R1 (复杂/高风险任务)
    """

    def __init__(self):
        settings = get_settings()
        self.enabled = getattr(settings, "model_routing_enabled", True)
        self.provider = settings.llm_provider

        # 各档位模型配置 (可通过环境变量覆盖)
        self.models = {
            "flash": {
                "model": getattr(settings, "model_flash", "deepseek-chat"),
                "base_url": settings.llm_base_url,
                "api_key": settings.llm_api_key_str,
            },
            "general": {
                "model": getattr(settings, "model_general", settings.llm_model),
                "base_url": settings.llm_base_url,
                "api_key": settings.llm_api_key_str,
            },
            "reasoning": {
                "model": getattr(settings, "model_reasoning", "deepseek-reasoner"),
                "base_url": settings.llm_base_url,
                "api_key": settings.llm_api_key_str,
            },
        }

        # vLLM 模式下, 只有一个本地模型, 路由失效
        if self.provider == "vllm":
            self.enabled = False
            logger.info("model_router.disabled_in_vllm_mode")

        # 健康状态缓存
        self._health_status: dict[str, bool] = {}
        self._last_health_check: float = 0.0

        # 路由决策日志 — 记录每次路由的 task_type → model 映射
        self._routing_log: list[dict[str, Any]] = []

    def get_model_for_task(self, task_type: str) -> tuple[str, dict[str, Any]]:
        """获取指定任务类型的推荐模型。

        Args:
            task_type: 任务类型 (参见 TASK_MODEL_MAP)

        Returns:
            (model_key, model_config_dict)
        """
        if not self.enabled:
            return "general", self.models["general"]

        task_config = TASK_MODEL_MAP.get(task_type, {"model_key": "general"})
        model_key = task_config["model_key"]

        # 如果首选模型不健康, 降级到通用模型
        if not self._is_healthy(model_key) and model_key != "general":
            logger.warning(
                "model_router.fallback",
                task=task_type,
                primary=model_key,
                fallback="general",
            )
            model_key = "general"

        return model_key, self.models[model_key]

    def create_llm(
        self,
        task_type: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatOpenAI:
        """为指定任务创建 LLM 实例。

        这是对外的主入口, 替代原来的 get_llm()。
        """
        settings = get_settings()
        if temperature is None:
            temperature = settings.llm_temperature
        if max_tokens is None:
            max_tokens = settings.llm_max_tokens

        if not self.enabled:
            # 关闭路由时回退到原始行为
            from llm.factory import get_llm
            return get_llm(temperature=temperature, max_tokens=max_tokens, **kwargs)

        model_key, config = self.get_model_for_task(task_type)
        model_name = config["model"]
        base_url = config["base_url"]
        api_key = config["api_key"] or "not-configured"

        self._routing_log.append({
            "task_type": task_type,
            "model_key": model_key,
            "model": model_name,
            "timestamp": time.time(),
        })
        logger.debug(
            "model_router.route",
            task=task_type,
            model_key=model_key,
            model=model_name,
        )

        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _is_healthy(self, model_key: str) -> bool:
        """检查模型健康状态 (带缓存, 避免频繁检查)。"""
        now = time.time()
        if now - self._last_health_check < 60:  # 60秒缓存
            return self._health_status.get(model_key, True)
        # 简化: 默认都健康; 生产环境可接入真实健康检查
        self._last_health_check = now
        return True

    def get_route_info(self) -> dict[str, Any]:
        """获取路由配置信息 (用于 /api/v1/llm/status 展示)。"""
        from collections import Counter

        routing_stats: dict[str, dict[str, Any]] = {}
        if self._routing_log:
            by_task: dict[str, Counter] = {}
            for entry in self._routing_log:
                task = entry["task_type"]
                by_task.setdefault(task, Counter())[entry["model_key"]] += 1
            for task, counter in by_task.items():
                total = sum(counter.values())
                routing_stats[task] = {
                    "total_calls": total,
                    "model_distribution": dict(counter),
                }

        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "models": {
                key: {"model": val["model"]}
                for key, val in self.models.items()
            },
            "task_routes": {
                task: info["model_key"]
                for task, info in TASK_MODEL_MAP.items()
            },
            "routing_stats": routing_stats,
        }


# 单例
_router: MultiModelRouter | None = None


def get_model_router() -> MultiModelRouter:
    global _router
    if _router is None:
        _router = MultiModelRouter()
    return _router
