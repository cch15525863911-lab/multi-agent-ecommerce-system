"""
退款风控 Service — 基于确定性规则引擎的退款风险评估。

传统业务 Service, 非 LLM Agent:
    退款申请 → 风险规则评估 (频率/金额/历史/理由) → 风险等级判定 → 处理策略

主链路: 确定性退款规则引擎 (risk_tools.assess_refund_risk)
"""

from __future__ import annotations

from typing import Any

import structlog

from config import get_settings
from models.schemas import FraudRiskLevel, RefundRiskResult, RefundStatus
from services import risk_tools as rt
from services.base_service import BaseProtectedService

logger = structlog.get_logger()


class RefundRiskService(BaseProtectedService):
    """退款风控规则引擎 Service — 确定性检测, 分级处理策略。"""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            name="refund_risk",
            timeout=getattr(settings, "agent_timeout_refund", 8.0),
            max_retries=1,
        )

    async def execute(self, **kwargs: Any) -> RefundRiskResult:
        user_id = kwargs["user_id"]
        order_id = kwargs["order_id"]
        product_id = kwargs.get("product_id", "")
        refund_amount = kwargs.get("refund_amount", 0.0)
        refund_reason = kwargs.get("refund_reason", "")

        result = await rt.assess_refund_risk(
            user_id, order_id, product_id, refund_amount, refund_reason
        )
        return RefundRiskResult(
            agent_name=self.name,
            success=True,
            risk_level=FraudRiskLevel(result["risk_level"]),
            risk_score=result["risk_score"],
            refund_status=RefundStatus(result["refund_status"]),
            rejection_reason=result["rejection_reason"],
            flash_refund_eligible=result["flash_refund_eligible"],
            needs_human_review=result["needs_human_review"],
        )

    def _fallback(self, latency_ms: float, exc: Exception) -> RefundRiskResult:
        return RefundRiskResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            risk_level=FraudRiskLevel.HIGH,
            risk_score=0.0,
            refund_status=RefundStatus.MANUAL_REVIEW,
            flash_refund_eligible=False,
            needs_human_review=True,
            confidence=0.0,
        )
