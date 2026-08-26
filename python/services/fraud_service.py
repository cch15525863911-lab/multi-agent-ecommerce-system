"""
反欺诈 Service — 基于确定性规则引擎的交易欺诈检测。

传统业务 Service, 非 LLM Agent:
    交易请求 → IP/设备黑名单 → 行为规则匹配 → 历史欺诈查询 → 综合风险评分 → 建议动作

主链路: 确定性规则引擎 (risk_tools.check_fraud)
"""

from __future__ import annotations

from typing import Any

import structlog

from config import get_settings
from models.schemas import FraudCheckResult, FraudRiskLevel, FraudRuleHit
from services import risk_tools as rt
from services.base_service import BaseProtectedService

logger = structlog.get_logger()


class FraudService(BaseProtectedService):
    """反欺诈规则引擎 Service — 确定性检测, 亚毫秒级响应。"""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            name="fraud_detection",
            timeout=getattr(settings, "agent_timeout_fraud", 8.0),
            max_retries=1,
        )

    async def execute(self, **kwargs: Any) -> FraudCheckResult:
        user_id = kwargs["user_id"]
        amount = kwargs.get("amount", 0.0)
        payment_method = kwargs.get("payment_method", "alipay")
        device_id = kwargs.get("device_id")
        ip_address = kwargs.get("ip_address")
        order_id = kwargs.get("order_id")

        result = await rt.check_fraud(
            user_id, amount, payment_method, device_id, ip_address, order_id
        )
        return FraudCheckResult(
            agent_name=self.name,
            success=True,
            risk_level=FraudRiskLevel(result["risk_level"]),
            risk_score=result["risk_score"],
            rules_hit=[FraudRuleHit(**r) for r in result["rules_hit"]],
            recommended_action=result["recommended_action"],
            needs_human_review=result["needs_human_review"],
        )

    def _fallback(self, latency_ms: float, exc: Exception) -> FraudCheckResult:
        return FraudCheckResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            risk_level=FraudRiskLevel.HIGH,
            risk_score=0.0,
            recommended_action="review",
            needs_human_review=True,
            confidence=0.0,
        )
