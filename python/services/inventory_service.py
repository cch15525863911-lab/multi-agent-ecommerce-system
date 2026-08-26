"""
库存决策 Service — 实时库存查询与预警。

传统业务 Service, 非 LLM Agent:
    实时库存查询 → 库存预警 (安全库存阈值) → 限购策略 (动态调整)
"""

from __future__ import annotations

from typing import Any

from config import get_settings
from models.schemas import InventoryResult, Product
from services.base_service import BaseProtectedService

SAFETY_STOCK_THRESHOLD = 50
LOW_STOCK_THRESHOLD = 100
HOT_ITEM_PURCHASE_LIMIT = 2


class InventoryService(BaseProtectedService):
    """库存查询与预警 Service — 确定性库存校验。"""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            name="inventory",
            timeout=settings.agent_timeout_inventory,
            max_retries=1,
        )

    async def execute(self, **kwargs: Any) -> InventoryResult:
        products: list[Product] = kwargs.get("products", [])

        available: list[str] = []
        low_stock_alerts: list[dict[str, Any]] = []
        purchase_limits: dict[str, int] = {}

        for product in products:
            stock = await self._check_stock(product.product_id, product.stock)

            if stock <= 0:
                continue

            available.append(product.product_id)

            if stock <= SAFETY_STOCK_THRESHOLD:
                low_stock_alerts.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "current_stock": stock,
                    "level": "critical",
                    "action": "urgent_restock",
                })
            elif stock <= LOW_STOCK_THRESHOLD:
                low_stock_alerts.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "current_stock": stock,
                    "level": "warning",
                    "action": "plan_restock",
                })

            limit = self._calc_purchase_limit(product, stock)
            if limit is not None:
                purchase_limits[product.product_id] = limit

        return InventoryResult(
            success=True,
            available_products=available,
            low_stock_alerts=low_stock_alerts,
            purchase_limits=purchase_limits,
            data={
                "total_checked": len(products),
                "available_count": len(available),
                "alert_count": len(low_stock_alerts),
            },
            confidence=0.95,
        )

    async def _check_stock(self, product_id: str, fallback_stock: int) -> int:
        return fallback_stock

    def _calc_purchase_limit(self, product: Product, stock: int) -> int | None:
        is_hot = "新品" in product.tags or "旗舰" in product.tags
        if stock <= SAFETY_STOCK_THRESHOLD:
            return 1
        if stock <= LOW_STOCK_THRESHOLD and is_hot:
            return HOT_ITEM_PURCHASE_LIMIT
        if is_hot and stock <= 300:
            return 3
        return None
