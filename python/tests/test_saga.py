"""
Saga 事务编排与补偿机制 — 单元测试。

测试覆盖:
    1. 正常 Saga: 4 步全部成功, 订单创建
    2. 补偿函数 release_inventory: 预占后释放, 库存恢复
    3. 补偿函数 cancel_order: 创建订单后取消, 预占释放
    4. Saga 失败补偿: 在 CreateOrder 步骤注入失败, 验证 ReserveInventory 被补偿
    5. Saga 审计轨迹: 验证各步骤状态记录

Run from the `python/` directory:
    python -m tests.test_saga
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.schemas import FulfillmentRequest, Product
from services import fulfillment_tools as ft
from services.saga import (
    SagaContext,
    SagaOrchestrator,
    SagaStep,
    StepState,
    build_fulfillment_saga,
    run_fulfillment_saga,
)


def _product(pid: str = "P001", price: float = 7999.0) -> Product:
    return Product(
        product_id=pid, name=f"Test-{pid}", category="手机", price=price, brand="Apple"
    )


def _request(pid: str = "P001", price: float = 7999.0, qty: int = 1) -> FulfillmentRequest:
    return FulfillmentRequest(
        user_id="u001", product=_product(pid, price), quantity=qty, destination="北京"
    )


# =========================================================================
# Test 1: Normal Saga — all 4 steps succeed
# =========================================================================


async def test_saga_success() -> None:
    """Full successful Saga: check → reserve → logistics → order."""
    ft.reset_inmemory_state()

    result, records = await run_fulfillment_saga(_request())

    assert result.success, f"saga should succeed: {result.error}"
    assert result.order is not None
    assert result.order.order_id.startswith("ORD-")
    assert result.reservation is not None
    assert result.logistics_route is not None
    assert len(records) == 4
    assert all(r.state == StepState.COMPLETED for r in records)
    print(
        f"[OK] saga success: order={result.order.order_id} "
        f"steps={[r.name for r in records]}"
    )


# =========================================================================
# Test 2: Compensation function — release_inventory
# =========================================================================


async def test_release_inventory() -> None:
    """Reserve then release: free stock should be restored."""
    ft.reset_inmemory_state()

    inv0 = await ft.check_inventory("P003")
    north0 = next(w for w in inv0["warehouses"] if w["warehouse_id"] == "WH-NORTH")
    free_before = north0["free"]

    r = await ft.reserve_inventory("P003", 5, "WH-NORTH")
    assert r["status"] == "reserved"

    inv1 = await ft.check_inventory("P003")
    north1 = next(w for w in inv1["warehouses"] if w["warehouse_id"] == "WH-NORTH")
    assert north1["free"] == free_before - 5

    rel = await ft.release_inventory(r["reservation_id"])
    assert rel["status"] == "released"

    inv2 = await ft.check_inventory("P003")
    north2 = next(w for w in inv2["warehouses"] if w["warehouse_id"] == "WH-NORTH")
    assert north2["free"] == free_before, "free stock must be restored after release"
    print(f"[OK] release_inventory: free {free_before} -> {north1['free']} -> {north2['free']}")


# =========================================================================
# Test 3: Compensation function — cancel_order
# =========================================================================


async def test_cancel_order() -> None:
    """Create order then cancel: order status=cancelled, reservation released."""
    ft.reset_inmemory_state()

    r = await ft.reserve_inventory("P001", 1, "WH-NORTH")
    route = await ft.match_logistics_route("P001", "WH-NORTH", 7999.0, "北京")
    order = await ft.create_order("u001", "P001", 1, 7999.0, r["reservation_id"], route["route_id"])
    assert order["status"] == "created"

    cancelled = await ft.cancel_order(order["order_id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["reservation_id"] == r["reservation_id"]

    # verify reservation is released (hold is gone)
    hold = await ft._get_hold(r["reservation_id"], None)
    assert hold is None, "reservation hold must be removed after cancel"

    print(f"[OK] cancel_order: order={order['order_id']} cancelled, reservation released")


# =========================================================================
# Test 4: Saga compensation — CreateOrder fails, ReserveInventory compensated
# =========================================================================


class _FailingCreateOrderStep(SagaStep):
    """A CreateOrder step that always fails, to trigger Saga compensation."""

    name = "create_order_fail"

    async def execute(self, ctx: SagaContext) -> bool:
        return False

    async def compensate(self, ctx: SagaContext) -> None:
        pass


async def test_saga_compensation_on_failure() -> None:
    """Saga fails at CreateOrder → ReserveInventory must be compensated."""
    ft.reset_inmemory_state()

    inv0 = await ft.check_inventory("P001")
    north0 = next(w for w in inv0["warehouses"] if w["warehouse_id"] == "WH-NORTH")
    free_before = north0["free"]

    # Build a saga with a failing 4th step
    from services.saga import (
        CheckInventoryStep,
        ReserveInventoryStep,
        MatchLogisticsStep,
    )

    saga = SagaOrchestrator(
        [
            CheckInventoryStep(),
            ReserveInventoryStep(),
            MatchLogisticsStep(),
            _FailingCreateOrderStep(),
        ]
    )

    ctx = SagaContext(request=_request())
    success, records = await saga.execute(ctx)

    assert not success, "saga should fail"
    assert len(records) == 4

    # First 3 steps completed then compensated, 4th failed
    assert records[0].state == StepState.COMPENSATED
    assert records[1].state == StepState.COMPENSATED, (
        "reserve_inventory must be compensated after create_order failure"
    )
    assert records[2].state == StepState.COMPENSATED
    assert records[3].state == StepState.FAILED

    # Verify stock was restored
    inv1 = await ft.check_inventory("P001")
    north1 = next(w for w in inv1["warehouses"] if w["warehouse_id"] == "WH-NORTH")
    assert north1["free"] == free_before, (
        "free stock must be restored after compensation"
    )

    print(
        f"[OK] saga compensation: reserve compensated, "
        f"free restored {free_before} -> {north1['free']}"
    )


# =========================================================================
# Test 5: Saga audit trail — verify step records
# =========================================================================


async def test_saga_audit_trail() -> None:
    """Verify SagaStepRecord fields are populated correctly."""
    ft.reset_inmemory_state()

    _, records = await run_fulfillment_saga(_request())

    assert len(records) == 4
    expected_names = ["check_inventory", "reserve_inventory", "match_logistics", "create_order"]
    actual_names = [r.name for r in records]
    assert actual_names == expected_names, f"step order mismatch: {actual_names}"

    for r in records:
        assert r.state == StepState.COMPLETED
        assert r.started_at is not None
        assert r.completed_at is not None
        assert r.error is None

    print(f"[OK] audit trail: {actual_names}, all COMPLETED")


# =========================================================================
# Main
# =========================================================================


async def main() -> int:
    print("=" * 60)
    print("Saga 事务编排与补偿机制 — 单元测试")
    print("=" * 60)
    await test_saga_success()
    await test_release_inventory()
    await test_cancel_order()
    await test_saga_compensation_on_failure()
    await test_saga_audit_trail()
    print("=" * 60)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
