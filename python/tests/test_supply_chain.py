"""
Supply-chain agent smoke test — verifies the deterministic fulfillment pipeline
(check_inventory → reserve_inventory → match_logistics_route → create_order).

Runs WITHOUT an LLM or a live MCP server: it exercises the in-process fallback
path of FulfillmentService, which calls the same business functions the MCP server

Run from the `python/` directory:
    python -m tests.test_supply_chain
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import FulfillmentService
from models.schemas import FulfillmentRequest, Product
from services import fulfillment_tools as ft


def _product(pid: str, name: str, price: float, category: str = "手机") -> Product:
    return Product(
        product_id=pid, name=name, category=category, price=price, brand="Apple"
    )


async def test_high_value_fulfillment() -> None:
    """High-value item (iPhone, 7999) → insured + encrypted logistics + order."""
    ft.reset_inmemory_state()
    service = FulfillmentService()

    p = _product("P001", "iPhone 16 Pro", 7999.0)
    req = FulfillmentRequest(user_id="u001", product=p, quantity=1, destination="北京")

    result = await service.run(request=req)
    assert result.success, f"fulfillment failed: {result.error}"
    assert result.order is not None and result.order.order_id.startswith("ORD-")
    assert result.reservation is not None
    assert result.logistics_route.insured, "high-value item must be insured"
    assert result.logistics_route.encrypted, "high-value item must be encrypted"
    assert result.logistics_route.insured_amount == 7999.0
    assert result.logistics_route.carrier in {"SF", "JD"}, "high-value uses insured carrier"
    print(f"[OK] high-value: order={result.order.order_id} "
          f"carrier={result.logistics_route.carrier} insured={result.logistics_route.insured}")


async def test_distributed_reservation_decrements_stock() -> None:
    """Two sequential reservations at the same warehouse must decrement free stock."""
    ft.reset_inmemory_state()
    inv0 = await ft.check_inventory("P003")  # WH-NORTH physical=200
    north0 = next(w for w in inv0["warehouses"] if w["warehouse_id"] == "WH-NORTH")
    free0 = north0["free"]

    r1 = await ft.reserve_inventory("P003", 5, "WH-NORTH")
    assert r1["status"] == "reserved"
    inv1 = await ft.check_inventory("P003")
    north1 = next(w for w in inv1["warehouses"] if w["warehouse_id"] == "WH-NORTH")
    assert north1["free"] == free0 - 5, "reservation must decrement free stock"

    r2 = await ft.reserve_inventory("P003", 3, "WH-NORTH")
    assert r2["status"] == "reserved"
    inv2 = await ft.check_inventory("P003")
    north2 = next(w for w in inv2["warehouses"] if w["warehouse_id"] == "WH-NORTH")
    assert north2["free"] == free0 - 8, "second reservation must further decrement stock"
    print(f"[OK] distributed reservation: free {free0} -> {north1['free']} -> {north2['free']}")


async def test_insufficient_stock_rejected() -> None:
    """Reserving more than free stock must be rejected without over-selling."""
    ft.reset_inmemory_state()
    inv = await ft.check_inventory("P014")  # WH-EAST physical=5
    east = next(w for w in inv["warehouses"] if w["warehouse_id"] == "WH-EAST")
    r = await ft.reserve_inventory("P014", east["free"] + 10, "WH-EAST")
    assert r["status"] == "insufficient", "must reject over-allocation"
    print(f"[OK] insufficient rejected: free={east['free']} need={east['free'] + 10}")


async def test_create_order_consumes_reservation() -> None:
    """Creating an order must consume the reservation (cannot reuse)."""
    ft.reset_inmemory_state()
    r = await ft.reserve_inventory("P001", 1, "WH-NORTH")
    route = await ft.match_logistics_route("P001", "WH-NORTH", 7999.0, "北京")
    order = await ft.create_order("u001", "P001", 1, 7999.0, r["reservation_id"], route["route_id"])
    assert order["status"] == "created"

    # reuse same reservation → must be rejected
    dup = await ft.create_order("u001", "P001", 1, 7999.0, r["reservation_id"], route["route_id"])
    assert dup["status"] == "reservation_consumed", "reservation must not be reusable"
    print(f"[OK] order created={order['order_id']}, reuse rejected={dup['status']}")


async def main() -> int:
    print("=" * 60)
    print("Supply-Chain Agent — deterministic fulfillment smoke test")
    print("=" * 60)
    await test_high_value_fulfillment()
    await test_distributed_reservation_decrements_stock()
    await test_insufficient_stock_rejected()
    await test_create_order_consumes_reservation()
    print("=" * 60)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
