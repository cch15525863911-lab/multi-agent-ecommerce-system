"""
信用授信 Service — 基于确定性评分卡的授信决策。

传统业务 Service, 非 LLM Agent:
    授信申请 → 查询信用档案 → 评估可用额度 → 计算利率 → 批准/拒绝 → 扣减额度

主链路: 确定性评分卡评估 (risk_tools.assess_credit)
"""

from __future__ import annotations

from typing import Any

import structlog

from config import get_settings
from models.schemas import CreditAssessmentResult, CreditStatus
from services import risk_tools as rt
from services.base_service import BaseProtectedService

logger = structlog.get_logger()


class CreditService(BaseProtectedService):
    """信用评分引擎 Service — 确定性评分卡, 可审计可解释。"""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            name="credit_assessment",
            timeout=getattr(settings, "agent_timeout_credit", 8.0),
            max_retries=1,
        )

    async def execute(self, **kwargs: Any) -> CreditAssessmentResult:
        user_id = kwargs["user_id"]
        requested_amount = kwargs.get("requested_amount", 0.0)
        order_id = kwargs.get("order_id")

        result = await rt.assess_credit(user_id, requested_amount, order_id)
        return CreditAssessmentResult(
            agent_name=self.name,
            success=True,
            credit_score=result["credit_score"],
            credit_limit=result["credit_limit"],
            available_limit=result["available_limit"],
            credit_status=CreditStatus(result["status"]),
            approved=result["approved"],
            approved_amount=result["approved_amount"],
            interest_rate=result["interest_rate"],
            tenure_days=result["tenure_days"],
            data={"reason": result.get("reason", "")},
        )

    def _fallback(self, latency_ms: float, exc: Exception) -> CreditAssessmentResult:
        return CreditAssessmentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            credit_score=0,
            credit_limit=0.0,
            available_limit=0.0,
            credit_status=CreditStatus.NONE,
            approved=False,
            approved_amount=0.0,
            confidence=0.0,
        )
