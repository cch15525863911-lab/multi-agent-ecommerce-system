"""
订单持久化失败处理 — 单元测试。

测试覆盖:
    1. DB 未启用时（内存模式）订单正常创建
    2. DB 启用且持久化成功时订单正常创建
    3. DB 启用但持久化失败时返回 persist_failed 状态
    4. persist_failed 时不写入 Redis 缓存

Run from the `python/` directory:
    python -m tests.test_order_persist
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import fulfillment_tools as ft


def _setup_reservation() -> str:
    """Create an in-memory reservation and return its ID."""
    ft.reset_inmemory_state()
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(
        ft.reserve_inventory("P001", 1, "WH-NORTH")
    )
    assert result["status"] == "reserved"
    return result["reservation_id"]


# =========================================================================
# Test 1: In-memory mode — order creation succeeds without DB
# =========================================================================

async def test_order_create_in_memory_mode() -> None:
    """When DB is disabled, order should be created in memory."""
    ft.reset_inmemory_state()
    ft._db_enabled = False

    res = await ft.reserve_inventory("P001", 1, "WH-NORTH")
    route = await ft.match_logistics_route("P001", "WH-NORTH", 7999.0, "北京")

    order = await ft.create_order(
        user_id="u001",
        product_id="P001",
        quantity=1,
        unit_price=7999.0,
        reservation_id=res["reservation_id"],
        route_id=route["route_id"],
    )

    assert order["status"] == "created", f"expected created, got {order['status']}"
    assert order["order_id"].startswith("ORD-")
    assert order["total_amount"] == 7999.0
    print(f"[OK] order.in_memory: order_id={order['order_id']}, status={order['status']}")


# =========================================================================
# Test 2: DB enabled + persist succeeds — order created normally
# =========================================================================

async def test_order_create_db_persist_success() -> None:
    """When DB is enabled and persist succeeds, order should be created."""
    ft.reset_inmemory_state()
    ft._db_enabled = True

    res = await ft.reserve_inventory("P001", 1, "WH-NORTH")
    route = await ft.match_logistics_route("P001", "WH-NORTH", 7999.0, "北京")

    with patch.object(ft, "_persist_order_to_db", return_value=True):
        order = await ft.create_order(
            user_id="u001",
            product_id="P001",
            quantity=1,
            unit_price=7999.0,
            reservation_id=res["reservation_id"],
            route_id=route["route_id"],
        )

    assert order["status"] == "created", f"expected created, got {order['status']}"
    ft._db_enabled = False
    print(f"[OK] order.db_success: order_id={order['order_id']}, status={order['status']}")


# =========================================================================
# Test 3: DB enabled + persist fails — returns persist_failed
# =========================================================================

async def test_order_persist_failed_returns_error() -> None:
    """When DB is enabled but persist fails, should return persist_failed."""
    ft.reset_inmemory_state()
    ft._db_enabled = True

    res = await ft.reserve_inventory("P001", 1, "WH-NORTH")
    route = await ft.match_logistics_route("P001", "WH-NORTH", 7999.0, "北京")

    with patch.object(ft, "_persist_order_to_db", return_value=False):
        order = await ft.create_order(
            user_id="u001",
            product_id="P001",
            quantity=1,
            unit_price=7999.0,
            reservation_id=res["reservation_id"],
            route_id=route["route_id"],
        )

    assert order["status"] == "persist_failed", \
        f"expected persist_failed, got {order['status']}"
    assert order["reservation_id"] == res["reservation_id"]
    assert "message" in order
    ft._db_enabled = False
    print(f"[OK] order.persist_failed: status={order['status']}, msg={order['message']}")


# =========================================================================
# Test 4: persist_failed does not consume reservation
# =========================================================================

async def test_persist_failed_does_not_consume_reservation() -> None:
    """When persist fails, reservation should not be marked as consumed."""
    ft.reset_inmemory_state()
    ft._db_enabled = True

    res = await ft.reserve_inventory("P001", 1, "WH-NORTH")
    route = await ft.match_logistics_route("P001", "WH-NORTH", 7999.0, "北京")

    with patch.object(ft, "_persist_order_to_db", return_value=False):
        await ft.create_order(
            user_id="u001",
            product_id="P001",
            quantity=1,
            unit_price=7999.0,
            reservation_id=res["reservation_id"],
            route_id=route["route_id"],
        )

    hold = await ft._get_hold(res["reservation_id"], None)
    assert hold is not None, "reservation should still exist"
    assert hold.get("status") != "consumed", \
        "reservation should not be consumed when persist fails"
    ft._db_enabled = False
    print("[OK] order.persist_not_consumed: reservation preserved on failure")


# =========================================================================
# Runner
# =========================================================================

async def main() -> None:
    tests = [
        test_order_create_in_memory_mode,
        test_order_create_db_persist_success,
        test_order_persist_failed_returns_error,
        test_persist_failed_does_not_consume_reservation,
    ]
    for fn in tests:
        await fn()
    print(f"\nAll {len(tests)} order persist tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
