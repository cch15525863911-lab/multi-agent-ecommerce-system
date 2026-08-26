"""
履约业务工具 — 供应链与履约调度的核心业务逻辑。

四个工具构成"高价值商品从推荐到履约"的完整链路：
    check_inventory        → 多仓实时库存查询
    reserve_inventory      → 分布式库存预占(Redis SETNX 锁 + 预占池, TTL 自动释放)
    match_logistics_route  → 物流路线匹配 + 高价值商品加密保价
    create_order           → 订单自动创建(绑定预占 + 物流), 持久化到 PostgreSQL

仓库主数据和库存基线从 PostgreSQL 加载(init_db_connection),
订单持久化到 PostgreSQL orders 表, Redis 用于预占分布式锁与缓存。
这些函数由 services/saga.py 的 Saga 步骤直接调用，构成确定性履约事务链路。
services/mcp_fulfillment_server.py 通过 MCP 协议将相同函数封装为标准 MCP 工具，
供外部系统（MCP 客户端）调用；内部履约主链路不经过 LLM，由 Saga 编排直接驱动。
当 PostgreSQL/Redis 不可用时，自动降级到进程内实现，保证链路可运行、可测试。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

logger = structlog.get_logger()

# 高价值商品阈值：超过该金额需加密保价物流
HIGH_VALUE_THRESHOLD = 3000.0

# 预占持有时长(秒)：预占后 15 分钟内未下单则自动释放
RESERVATION_HOLD_TTL = 900
# 预占分布式锁时长(秒)
RESERVATION_LOCK_TTL = 30

# ---------- 仓库主数据(默认值, 启动时从 PostgreSQL 覆盖) ----------
# 真实环境从 WMS / PostgreSQL 加载; 以下为无 DB 时的降级数据
WAREHOUSES: dict[str, dict[str, Any]] = {
    "WH-NORTH": {"region": "华北", "city": "北京", "carriers": ["SF", "JD"]},
    "WH-SOUTH": {"region": "华南", "city": "深圳", "carriers": ["SF", "YTO"]},
    "WH-EAST": {"region": "华东", "city": "上海", "carriers": ["SF", "JD", "YTO"]},
}

# (product_id, warehouse_id) -> 物理库存
WAREHOUSE_STOCK: dict[tuple[str, str], int] = {
    ("P001", "WH-NORTH"): 30,
    ("P001", "WH-SOUTH"): 8,
    ("P001", "WH-EAST"): 120,
    ("P003", "WH-NORTH"): 200,
    ("P003", "WH-EAST"): 60,
    ("P013", "WH-NORTH"): 15,
    ("P014", "WH-EAST"): 5,
    ("P015", "WH-SOUTH"): 50,
}

# 加密保价承运商(高价值商品优先)
INSURED_CARRIERS = {"SF", "JD"}

# 进程内状态(无 Redis 时降级使用)
_inmemory_locks: dict[str, str] = {}
_inmemory_reserved: dict[tuple[str, str], int] = {}
_inmemory_holds: dict[str, dict[str, Any]] = {}
_inmemory_orders: dict[str, dict[str, Any]] = {}
_mem_guard = asyncio.Lock()

_redis_client: Any = None
_db_enabled: bool = False


def set_redis_client(client: Any) -> None:
    """Inject an async Redis client (e.g. redis.asyncio.Redis)."""
    global _redis_client
    _redis_client = client


def init_db_connection() -> None:
    """Initialize PostgreSQL — load warehouse master data into module-level dicts.

    Called at application startup. Falls back to hardcoded data on failure.
    """
    global _db_enabled
    try:
        from database.engine import get_session
        from database.models import Warehouse, WarehouseStock
        from sqlalchemy import select

        with get_session() as session:
            warehouses = session.execute(select(Warehouse)).scalars().all()
            for wh in warehouses:
                WAREHOUSES[wh.warehouse_id] = {
                    "region": wh.region,
                    "city": wh.city,
                    "carriers": wh.carriers or [],
                }

            stocks = session.execute(select(WarehouseStock)).scalars().all()
            for s in stocks:
                WAREHOUSE_STOCK[(s.product_id, s.warehouse_id)] = s.physical_stock

        _db_enabled = True
        logger.info(
            "fulfillment.db.loaded",
            warehouses=len(WAREHOUSES),
            stock_records=len(WAREHOUSE_STOCK),
        )
    except Exception as exc:
        logger.warning("fulfillment.db.unavailable", error=str(exc))
        _db_enabled = False


def _get_redis() -> Any:
    return _redis_client


# =========================================================================
# Tool 1: check_inventory
# =========================================================================


async def check_inventory(
    product_id: str, warehouse_id: str | None = None
) -> dict[str, Any]:
    """查询商品在多个仓库的实时可用库存。

    参数:
        product_id: 商品ID
        warehouse_id: 可选, 指定仓库; 为空则返回全部仓库

    返回:
        {warehouses: [{warehouse_id, region, available, reserved, free}], total_free}
    """
    redis = _get_redis()
    result: list[dict[str, Any]] = []

    target_wh = [warehouse_id] if warehouse_id else list(WAREHOUSES)
    for wh_id in target_wh:
        physical = WAREHOUSE_STOCK.get((product_id, wh_id), 0)
        reserved = await _get_reserved(product_id, wh_id, redis)
        wh_info = WAREHOUSES.get(wh_id, {})
        result.append(
            {
                "warehouse_id": wh_id,
                "region": wh_info.get("region", ""),
                "physical_stock": physical,
                "reserved": reserved,
                "free": max(0, physical - reserved),
            }
        )

    total_free = sum(r["free"] for r in result)
    logger.info(
        "fulfillment.check_inventory",
        product_id=product_id,
        warehouses=len(result),
        total_free=total_free,
    )
    return {"warehouses": result, "total_free": total_free}


# =========================================================================
# Tool 2: reserve_inventory (分布式预占)
# =========================================================================


async def reserve_inventory(
    product_id: str, quantity: int, warehouse_id: str
) -> dict[str, Any]:
    """分布式库存预占: 锁定某仓库指定数量库存, 返回预占单号与过期时间。

    采用 Redis SETNX 分布式锁 + 预占计数池实现, 防止并发超卖;
    预占持有 TTL(默认15分钟)内未创建订单则自动释放。无 Redis 时降级为
    进程内互斥锁实现。

    返回:
        {status, reservation_id, expires_at} 或 {status:"insufficient"/"locked"}
    """
    redis = _get_redis()
    res_id = f"RSV-{uuid.uuid4().hex[:8].upper()}"
    lock_key = f"lock:reserve:{product_id}:{warehouse_id}"
    reserved_key = f"reserve:qty:{product_id}:{warehouse_id}"
    hold_key = f"reserve:hold:{res_id}"

    if redis:
        # 分布式锁: 同一 (商品,仓库) 预占串行化, 避免预占计数竞态
        acquired = await redis.set(lock_key, res_id, nx=True, ex=RESERVATION_LOCK_TTL)
        if not acquired:
            return {"status": "locked", "message": "另一个预占正在进行, 请重试"}
        try:
            physical = int(
                await redis.get(f"stock:{product_id}:{warehouse_id}")
                or WAREHOUSE_STOCK.get((product_id, warehouse_id), 0)
            )
            reserved = int(await redis.get(reserved_key) or 0)
            free = physical - reserved
            if free < quantity:
                return {
                    "status": "insufficient",
                    "product_id": product_id,
                    "warehouse_id": warehouse_id,
                    "free": free,
                    "need": quantity,
                }
            await redis.incrby(reserved_key, quantity)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=RESERVATION_HOLD_TTL)
            await redis.set(
                hold_key,
                json.dumps(
                    {
                        "reservation_id": res_id,
                        "product_id": product_id,
                        "warehouse_id": warehouse_id,
                        "quantity": quantity,
                        "expires_at": expires_at.isoformat(),
                    }
                ),
                ex=RESERVATION_HOLD_TTL,
            )
            logger.info("fulfillment.reserve.redis", reservation_id=res_id, quantity=quantity)
            return {
                "status": "reserved",
                "reservation_id": res_id,
                "warehouse_id": warehouse_id,
                "quantity": quantity,
                "expires_at": expires_at.isoformat(),
            }
        finally:
            await redis.delete(lock_key)

    # ---- 进程内降级实现 ----
    async with _mem_guard:
        physical = WAREHOUSE_STOCK.get((product_id, warehouse_id), 0)
        reserved = _inmemory_reserved.get((product_id, warehouse_id), 0)
        free = physical - reserved
        if free < quantity:
            return {
                "status": "insufficient",
                "product_id": product_id,
                "warehouse_id": warehouse_id,
                "free": free,
                "need": quantity,
            }
        _inmemory_reserved[(product_id, warehouse_id)] = reserved + quantity
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=RESERVATION_HOLD_TTL)
        _inmemory_holds[res_id] = {
            "product_id": product_id,
            "warehouse_id": warehouse_id,
            "quantity": quantity,
            "expires_at": expires_at,
        }
    logger.info("fulfillment.reserve.memory", reservation_id=res_id, quantity=quantity)
    return {
        "status": "reserved",
        "reservation_id": res_id,
        "warehouse_id": warehouse_id,
        "quantity": quantity,
        "expires_at": expires_at.isoformat(),
    }


# =========================================================================
# Tool 3: match_logistics_route (含高价值加密保价)
# =========================================================================


async def match_logistics_route(
    product_id: str,
    warehouse_id: str,
    product_value: float,
    destination: str = "北京",
) -> dict[str, Any]:
    """为商品匹配物流路线; 高价值商品强制加密保价。

    高价值(product_value > 阈值)时: 选择支持保价的承运商(SF/JD),
    开启保价(insured_amount=商品价值)与订单数据加密(encrypted=True)。

    返回: {route_id, carrier, warehouse_id, destination, insured, insured_amount, encrypted, eta_hours}
    """
    wh = WAREHOUSES.get(warehouse_id, {})
    carriers = wh.get("carriers", ["SF"])
    is_high_value = product_value >= HIGH_VALUE_THRESHOLD

    carrier = carriers[0]
    if is_high_value:
        # 高价值优先保价承运商
        carrier = next((c for c in carriers if c in INSURED_CARRIERS), carriers[0])

    route_id = f"RT-{uuid.uuid4().hex[:8].upper()}"
    insured = is_high_value
    insured_amount = product_value if is_high_value else 0.0
    encrypted = is_high_value  # 高价值订单数据加密存储

    # 京津冀/同区域次日达, 跨区 2-3 日
    same_region = wh.get("city") == destination
    eta_hours = 24 if same_region else 48

    logger.info(
        "fulfillment.match_logistics",
        route_id=route_id,
        carrier=carrier,
        high_value=is_high_value,
        insured=insured,
    )
    return {
        "route_id": route_id,
        "carrier": carrier,
        "warehouse_id": warehouse_id,
        "destination": destination,
        "insured": insured,
        "insured_amount": insured_amount,
        "encrypted": encrypted,
        "eta_hours": eta_hours,
    }


# =========================================================================
# Tool 4: create_order
# =========================================================================


async def create_order(
    user_id: str,
    product_id: str,
    quantity: int,
    unit_price: float,
    reservation_id: str,
    route_id: str,
) -> dict[str, Any]:
    """基于已生效的预占单与物流路线自动创建订单。

    校验预占单存在且未过期, 绑定预占单号与物流路线, 落库生成订单号。
    返回: {order_id, status, total_amount, reservation_id, route_id}
    """
    redis = _get_redis()
    hold = await _get_hold(reservation_id, redis)
    if not hold:
        return {"status": "invalid_reservation", "reservation_id": reservation_id}
    if hold.get("status") == "consumed":
        return {"status": "reservation_consumed", "reservation_id": reservation_id}

    order_id = f"ORD-{uuid.uuid4().hex[:10].upper()}"
    total_amount = round(unit_price * quantity, 2)
    created_at = datetime.now(timezone.utc)

    order = {
        "order_id": order_id,
        "user_id": user_id,
        "product_id": product_id,
        "quantity": quantity,
        "reservation_id": reservation_id,
        "logistics_route_id": route_id,
        "status": "created",
        "total_amount": total_amount,
        "created_at": created_at.isoformat(),
    }

    persist_ok = _persist_order_to_db(order)
    if _db_enabled and not persist_ok:
        logger.error(
            "fulfillment.create_order.persist_failed",
            order_id=order_id,
            reservation_id=reservation_id,
        )
        return {
            "status": "persist_failed",
            "order_id": order_id,
            "reservation_id": reservation_id,
            "message": "Order created in cache but failed to persist to database",
        }

    if redis:
        await redis.set(f"order:{order_id}", json.dumps(order))
        await redis.set(
            f"reserve:hold:{reservation_id}",
            json.dumps({**hold, "status": "consumed"}),
            ex=RESERVATION_HOLD_TTL,
        )
    else:
        _inmemory_orders[order_id] = order
        hold["status"] = "consumed"
        _inmemory_holds[reservation_id] = hold

    logger.info("fulfillment.create_order", order_id=order_id, total=total_amount)
    return {
        "order_id": order_id,
        "status": "created",
        "total_amount": total_amount,
        "reservation_id": reservation_id,
        "logistics_route_id": route_id,
    }


# =========================================================================
# Tool 5: release_inventory (Saga 补偿 — 释放预占)
# =========================================================================


async def release_inventory(reservation_id: str) -> dict[str, Any]:
    """释放指定预占单, 归还预占数量到可用库存池。

    用于 Saga 补偿: 当后续步骤(如 create_order)失败时, 回滚已生效的预占。
    无 Redis 时降级为进程内操作。
    """
    redis = _get_redis()
    hold = await _get_hold(reservation_id, redis)
    if not hold:
        return {"status": "not_found", "reservation_id": reservation_id}
    if hold.get("status") == "consumed":
        return {"status": "already_consumed", "reservation_id": reservation_id}

    product_id = hold.get("product_id", "")
    warehouse_id = hold.get("warehouse_id", "")
    quantity = hold.get("quantity", 0)

    if redis:
        reserved_key = f"reserve:qty:{product_id}:{warehouse_id}"
        hold_key = f"reserve:hold:{reservation_id}"
        await redis.decrby(reserved_key, quantity)
        await redis.delete(hold_key)
    else:
        current = _inmemory_reserved.get((product_id, warehouse_id), 0)
        _inmemory_reserved[(product_id, warehouse_id)] = max(0, current - quantity)
        _inmemory_holds.pop(reservation_id, None)

    logger.info(
        "fulfillment.release_inventory",
        reservation_id=reservation_id,
        product_id=product_id,
        quantity=quantity,
    )
    return {
        "status": "released",
        "reservation_id": reservation_id,
        "product_id": product_id,
        "warehouse_id": warehouse_id,
        "quantity": quantity,
    }


# =========================================================================
# Tool 6: cancel_order (Saga 补偿 — 取消订单 + 释放预占)
# =========================================================================


async def cancel_order(order_id: str) -> dict[str, Any]:
    """取消订单并释放关联的预占, 用于 Saga 补偿。

    将订单状态改为 cancelled, 释放绑定的预占库存(即使预占已被 consume)。
    无 Redis 时降级为进程内操作。
    """
    redis = _get_redis()
    order: dict[str, Any] | None = None
    reservation_id = ""

    if redis:
        raw = await redis.get(f"order:{order_id}")
        if raw:
            order = json.loads(raw)
    else:
        order = _inmemory_orders.get(order_id)

    if not order:
        return {"status": "not_found", "order_id": order_id}
    if order.get("status") == "cancelled":
        return {"status": "already_cancelled", "order_id": order_id}

    reservation_id = order.get("reservation_id", "")
    order["status"] = "cancelled"

    if redis:
        await redis.set(f"order:{order_id}", json.dumps(order))
    else:
        _inmemory_orders[order_id] = order

    _update_order_status_in_db(order_id, "cancelled")

    # Release the reserved stock (handles both active and consumed reservations)
    if reservation_id:
        hold = await _get_hold(reservation_id, redis)
        if hold:
            product_id = hold.get("product_id", "")
            warehouse_id = hold.get("warehouse_id", "")
            quantity = hold.get("quantity", 0)

            if redis:
                reserved_key = f"reserve:qty:{product_id}:{warehouse_id}"
                hold_key = f"reserve:hold:{reservation_id}"
                await redis.decrby(reserved_key, quantity)
                await redis.delete(hold_key)
            else:
                current = _inmemory_reserved.get((product_id, warehouse_id), 0)
                _inmemory_reserved[(product_id, warehouse_id)] = max(0, current - quantity)
                _inmemory_holds.pop(reservation_id, None)

            logger.info(
                "fulfillment.cancel_order.released",
                reservation_id=reservation_id,
                product_id=product_id,
                quantity=quantity,
            )

    logger.info(
        "fulfillment.cancel_order",
        order_id=order_id,
        reservation_id=reservation_id,
    )
    return {
        "status": "cancelled",
        "order_id": order_id,
        "reservation_id": reservation_id,
    }


# =========================================================================
# PostgreSQL order persistence
# =========================================================================


def _persist_order_to_db(order: dict[str, Any]) -> bool:
    """Persist order to PostgreSQL. Returns True on success."""
    if not _db_enabled:
        return False
    try:
        from database.engine import get_session
        from database.models import Order as OrderModel

        with get_session() as session:
            session.add(
                OrderModel(
                    order_id=order["order_id"],
                    user_id=order["user_id"],
                    product_id=order["product_id"],
                    quantity=order["quantity"],
                    reservation_id=order["reservation_id"],
                    logistics_route_id=order["logistics_route_id"],
                    status=order["status"],
                    total_amount=order["total_amount"],
                    created_at=datetime.fromisoformat(order["created_at"]),
                )
            )
        logger.info("fulfillment.order.persisted", order_id=order["order_id"])
        return True
    except Exception as exc:
        logger.warning("fulfillment.order.persist_failed", error=str(exc))
        return False


def _update_order_status_in_db(order_id: str, status: str) -> bool:
    """Update order status in PostgreSQL (for Saga compensation)."""
    if not _db_enabled:
        return False
    try:
        from database.engine import get_session
        from database.models import Order as OrderModel
        from sqlalchemy import update

        with get_session() as session:
            session.execute(
                update(OrderModel)
                .where(OrderModel.order_id == order_id)
                .values(status=status)
            )
        logger.info("fulfillment.order.status_updated", order_id=order_id, status=status)
        return True
    except Exception as exc:
        logger.warning("fulfillment.order.update_failed", error=str(exc))
        return False


async def _get_reserved(product_id: str, warehouse_id: str, redis: Any) -> int:
    if redis:
        return int(await redis.get(f"reserve:qty:{product_id}:{warehouse_id}") or 0)
    return _inmemory_reserved.get((product_id, warehouse_id), 0)


async def _get_hold(reservation_id: str, redis: Any) -> dict[str, Any] | None:
    if redis:
        raw = await redis.get(f"reserve:hold:{reservation_id}")
        if not raw:
            return None
        return json.loads(raw)
    hold = _inmemory_holds.get(reservation_id)
    if not hold:
        return None
    # 过期清理
    if hold.get("expires_at") and hold["expires_at"] < datetime.now(timezone.utc):
        _inmemory_holds.pop(reservation_id, None)
        return None
    return hold


def reset_inmemory_state() -> None:
    """Test helper: clear process-local state."""
    _inmemory_locks.clear()
    _inmemory_reserved.clear()
    _inmemory_holds.clear()
    _inmemory_orders.clear()
