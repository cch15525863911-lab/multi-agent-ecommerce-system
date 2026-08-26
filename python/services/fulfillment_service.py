"""
供应链履约 Service — 基于 Saga 事务编排的高价值商品履约。

传统业务 Service, 非 LLM Agent:
    推荐结果(高价值商品) → 库存校验/选仓 → 分布式预占(防超卖)
        → 加密保价物流匹配 → 订单自动创建

主链路: Saga 事务编排 (check_inventory → reserve → match_logistics → create_order)
        每步有独立熔断器, 失败时逆序自动补偿, 保证最终一致性。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from config import get_settings
from models.schemas import FulfillmentRequest, FulfillmentResult, LogisticsRoute, Order, Reservation
from services.base_service import BaseProtectedService
from services.saga import run_fulfillment_saga

logger = structlog.get_logger()


class FulfillmentService(BaseProtectedService):
    """供应链履约 Saga 编排 Service — 确定性事务编排 + 四层防护补偿。"""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            name="fulfillment",
            timeout=getattr(settings, "agent_timeout_supply_chain", 12.0),
            max_retries=1,
        )

    async def execute(self, **kwargs: Any) -> FulfillmentResult:
        request: FulfillmentRequest = kwargs["request"]

        result, records = await run_fulfillment_saga(request)
        if hasattr(result, "data") and result.data is not None:
            result.data["saga_audit"] = [
                {"step": r.name, "state": r.state.value, "error": r.error}
                for r in records
            ]
        return result

    def _fallback(self, latency_ms: float, exc: Exception) -> FulfillmentResult:
        return FulfillmentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
        )
