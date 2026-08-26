"""
Saga 事务编排 — 履约链路的分布式事务保障, 集成四层防护。

在 Orchestration-based Saga 基础上, 为每个步骤的 execute 和 compensate
分别接入四层防护(重试/独立超时/降级/熔断), 确保补偿机制也具备熔断能力。

四层防护按顺序生效:
    Layer 1  重试        tenacity 指数退避(500ms→1s→2s, 最多2次), 覆盖瞬时抖动
    Layer 2  独立超时    asyncio.wait_for 强制超时, 每次尝试独立计时
    Layer 3  降级        execute 失败返回 False(触发补偿), compensate 失败记录状态
    Layer 4  熔断        滑动窗口错误率≥阈值→OPEN(直接跳过), 恢复期后→HALF_OPEN探测

execute 与 compensate 拥有独立的熔断器, 互不影响:
    - execute 熔断: 步骤连续失败 → 跳过执行 → 直接触发补偿
    - compensate 熔断: 补偿连续失败 → 记录 COMPENSATE_FAILED, 不阻断后续补偿

履约 Saga 的 4 个步骤与补偿:
    ┌─────────────────────┬──────────────────────────┬──────────────────────┐
    │ Step                │ Execute                  │ Compensate           │
    ├─────────────────────┼──────────────────────────┼──────────────────────┤
    │ CheckInventory      │ 查询库存 + 选仓           │ (只读, 无需补偿)     │
    │ ReserveInventory    │ 分布式预占库存           │ release_inventory    │
    │ MatchLogistics      │ 物流路线匹配             │ (无副作用, 无需补偿) │
    │ CreateOrder         │ 订单创建 + 落库          │ cancel_order         │
    └─────────────────────┴──────────────────────────┴──────────────────────┘
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import (
    FulfillmentRequest,
    FulfillmentResult,
    LogisticsRoute,
    Order,
    Reservation,
)
from services import fulfillment_tools as ft
from services.circuit_breaker import CircuitBreaker

logger = structlog.get_logger()


class StepState(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"
    COMPENSATE_FAILED = "compensate_failed"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class SagaStepRecord:
    """Record of a single step's execution state for audit trail."""
    name: str
    state: StepState = StepState.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    execute_circuit: str = "closed"
    compensate_circuit: str = "closed"


@dataclass
class SagaContext:
    """Shared mutable state passed between Saga steps."""
    request: FulfillmentRequest
    warehouse_id: str | None = None
    inventory_result: dict[str, Any] = field(default_factory=dict)
    reservation: dict[str, Any] = field(default_factory=dict)
    route: dict[str, Any] = field(default_factory=dict)
    order: dict[str, Any] = field(default_factory=dict)


# =========================================================================
# SagaStep — four-layer protection for execute & compensate
# =========================================================================


class SagaStep(ABC):
    """Abstract base for a Saga step — execute + compensate, each with four-layer protection.

    Subclasses set ``name``, ``timeout``, ``max_retries`` as class attributes
    and implement :meth:`execute` and :meth:`compensate`.

    The orchestrator calls :meth:`_protected_execute` / :meth:`_protected_compensate`
    which wrap the raw methods with:
        L4: circuit breaker check (execute & compensate have independent breakers)
        L1: tenacity exponential backoff retry
        L2: asyncio.wait_for independent timeout per attempt
    """

    name: str = ""
    timeout: float = 10.0
    max_retries: int = 2

    def __init__(self) -> None:
        self._execute_circuit = CircuitBreaker(
            agent_name=f"saga.{self.name}.execute",
        )
        self._compensate_circuit = CircuitBreaker(
            agent_name=f"saga.{self.name}.compensate",
        )

    # -- abstract business logic (implemented by subclasses) --

    @abstractmethod
    async def execute(self, ctx: SagaContext) -> bool:
        """Execute the step. Return True on success, False on business failure."""
        ...

    @abstractmethod
    async def compensate(self, ctx: SagaContext) -> None:
        """Rollback the step's side effects. Called in reverse order on failure."""
        ...

    # -- four-layer protection wrappers --

    async def _protected_execute(
        self, ctx: SagaContext
    ) -> tuple[bool, str | None]:
        """Wrap execute with L4(circuit) → L1(retry) + L2(timeout).

        Returns ``(success, error_message)``.
        - (True, None): step succeeded
        - (False, None): business failure (execute returned False)
        - (False, msg): infrastructure failure or circuit open
        """
        # L4: circuit breaker — skip if OPEN
        if not self._execute_circuit.allow_request():
            logger.warning(
                "saga.execute.circuit_open",
                step=self.name,
            )
            return False, f"circuit_open:{self.name}.execute"

        try:
            success = await self._retry_call(self.execute, ctx)
            if success:
                self._execute_circuit.record_success()
            # Business failure (False) does NOT trip the circuit — only exceptions do
            return success, None

        except Exception as exc:
            # L4: record infrastructure failure
            self._execute_circuit.record_failure()
            logger.error(
                "saga.execute.failed",
                step=self.name,
                error=str(exc),
                circuit_state=self._execute_circuit.state.value,
            )
            return False, str(exc)

    async def _protected_compensate(
        self, ctx: SagaContext
    ) -> tuple[bool, str | None]:
        """Wrap compensate with L4(circuit) → L1(retry) + L2(timeout).

        Returns ``(success, error_message)``.
        - (True, None): compensation succeeded
        - (False, msg): compensation failed or circuit open
        """
        # L4: circuit breaker — skip if OPEN
        if not self._compensate_circuit.allow_request():
            logger.warning(
                "saga.compensate.circuit_open",
                step=self.name,
            )
            return False, f"circuit_open:{self.name}.compensate"

        try:
            await self._retry_call(self.compensate, ctx)
            self._compensate_circuit.record_success()
            return True, None

        except Exception as exc:
            # L4: record infrastructure failure
            self._compensate_circuit.record_failure()
            logger.error(
                "saga.compensate.failed",
                step=self.name,
                error=str(exc),
                circuit_state=self._compensate_circuit.state.value,
            )
            return False, str(exc)

    # -- L1 (retry) + L2 (independent timeout) --

    async def _retry_call(
        self,
        fn: Callable[[SagaContext], Awaitable[Any]],
        ctx: SagaContext,
    ) -> Any:
        """tenacity exponential backoff + asyncio.wait_for independent timeout."""

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        )
        async def _single_attempt() -> Any:
            return await asyncio.wait_for(
                fn(ctx),
                timeout=self.timeout,
            )

        return await _single_attempt()

    # -- monitoring --

    @property
    def circuit_states(self) -> dict[str, str]:
        """Current circuit breaker states for health monitoring."""
        return {
            "execute": self._execute_circuit.state.value,
            "compensate": self._compensate_circuit.state.value,
        }


# =========================================================================
# Concrete steps
# =========================================================================


class CheckInventoryStep(SagaStep):
    """Step 1: 查询多仓库存, 选可用库存最多的仓库。"""

    name = "check_inventory"
    timeout = 5.0
    max_retries = 2

    async def execute(self, ctx: SagaContext) -> bool:
        req = ctx.request
        inv = await ft.check_inventory(req.product.product_id)
        ctx.inventory_result = inv

        warehouses = inv.get("warehouses", [])
        candidate = max(
            (w for w in warehouses if w["free"] >= req.quantity),
            key=lambda w: w["free"],
            default=None,
        )
        if candidate is None:
            logger.warning(
                "saga.check_inventory.insufficient",
                product_id=req.product.product_id,
                total_free=inv.get("total_free", 0),
            )
            return False

        ctx.warehouse_id = candidate["warehouse_id"]
        logger.info(
            "saga.check_inventory.done",
            warehouse_id=ctx.warehouse_id,
            free=candidate["free"],
        )
        return True

    async def compensate(self, ctx: SagaContext) -> None:
        pass  # read-only, no side effects


class ReserveInventoryStep(SagaStep):
    """Step 2: 分布式库存预占(Redis SETNX 锁 + 预占池)。"""

    name = "reserve_inventory"
    timeout = 5.0
    max_retries = 2

    async def execute(self, ctx: SagaContext) -> bool:
        req = ctx.request
        resv = await ft.reserve_inventory(
            req.product.product_id, req.quantity, ctx.warehouse_id
        )
        ctx.reservation = resv

        if resv.get("status") != "reserved":
            logger.warning(
                "saga.reserve_inventory.failed",
                status=resv.get("status"),
                warehouse_id=ctx.warehouse_id,
            )
            return False

        logger.info(
            "saga.reserve_inventory.done",
            reservation_id=resv.get("reservation_id"),
        )
        return True

    async def compensate(self, ctx: SagaContext) -> None:
        resv = ctx.reservation
        reservation_id = resv.get("reservation_id")
        if reservation_id:
            await ft.release_inventory(reservation_id)
            logger.info(
                "saga.reserve_inventory.compensated",
                reservation_id=reservation_id,
            )


class MatchLogisticsStep(SagaStep):
    """Step 3: 物流路线匹配 + 高价值商品加密保价。"""

    name = "match_logistics"
    timeout = 5.0
    max_retries = 2

    async def execute(self, ctx: SagaContext) -> bool:
        req = ctx.request
        route = await ft.match_logistics_route(
            req.product.product_id,
            ctx.warehouse_id,
            req.product.price,
            req.destination,
        )
        ctx.route = route

        logger.info(
            "saga.match_logistics.done",
            route_id=route.get("route_id"),
            carrier=route.get("carrier"),
            insured=route.get("insured"),
        )
        return True

    async def compensate(self, ctx: SagaContext) -> None:
        pass  # no side effects (route is not persisted at this stage)


class CreateOrderStep(SagaStep):
    """Step 4: 订单创建 + 持久化到 PostgreSQL。"""

    name = "create_order"
    timeout = 8.0
    max_retries = 2

    async def execute(self, ctx: SagaContext) -> bool:
        req = ctx.request
        resv = ctx.reservation
        route = ctx.route

        order = await ft.create_order(
            req.user_id,
            req.product.product_id,
            req.quantity,
            req.product.price,
            resv["reservation_id"],
            route["route_id"],
        )
        ctx.order = order

        if order.get("status") != "created":
            logger.warning(
                "saga.create_order.failed",
                status=order.get("status"),
            )
            return False

        logger.info(
            "saga.create_order.done",
            order_id=order.get("order_id"),
            total_amount=order.get("total_amount"),
        )
        return True

    async def compensate(self, ctx: SagaContext) -> None:
        order = ctx.order
        order_id = order.get("order_id")
        if order_id:
            await ft.cancel_order(order_id)
            logger.info(
                "saga.create_order.compensated",
                order_id=order_id,
            )


# =========================================================================
# Saga Orchestrator
# =========================================================================


class SagaOrchestrator:
    """Orchestration-based Saga: 顺序执行步骤, 失败时逆序补偿。

    Each step's execute and compensate are wrapped with four-layer protection
    (circuit breaker → retry + timeout).  Execute and compensate have
    **independent** circuit breakers so that a failing execute does not
    block compensation, and vice-versa.

    Usage:
        saga = SagaOrchestrator([step1, step2, step3, step4])
        success, records = await saga.execute(ctx)
        if not success:
            # compensation has already been executed
    """

    def __init__(self, steps: list[SagaStep]):
        self.steps = steps
        self._completed: list[SagaStep] = []
        self._records: list[SagaStepRecord] = []

    async def execute(
        self, ctx: SagaContext
    ) -> tuple[bool, list[SagaStepRecord]]:
        """Execute all steps sequentially. On failure, compensate in reverse.

        Returns (success, step_records) where step_records is the audit trail.
        """
        for step in self.steps:
            record = SagaStepRecord(
                name=step.name, started_at=datetime.now()
            )
            try:
                # L4→L1→L2 protected execute
                success, err = await step._protected_execute(ctx)
                record.completed_at = datetime.now()
                record.execute_circuit = step._execute_circuit.state.value

                if not success:
                    # Distinguish circuit-open from regular failure
                    if err and "circuit_open:" in err:
                        record.state = StepState.CIRCUIT_OPEN
                    else:
                        record.state = StepState.FAILED
                    record.error = err
                    self._records.append(record)
                    logger.warning(
                        "saga.step.failed",
                        step=step.name,
                        state=record.state.value,
                        compensating=len(self._completed),
                    )
                    await self._compensate(ctx)
                    return False, self._records

                record.state = StepState.COMPLETED
                record.result = _extract_result(ctx, step.name)
                self._completed.append(step)
                self._records.append(record)

            except Exception as exc:
                # Safety net — _protected_execute already catches,
                # but guard against bugs in record/result extraction
                record.state = StepState.FAILED
                record.error = str(exc)
                record.completed_at = datetime.now()
                record.execute_circuit = step._execute_circuit.state.value
                self._records.append(record)
                logger.warning(
                    "saga.step.exception",
                    step=step.name,
                    error=str(exc),
                    compensating=len(self._completed),
                )
                await self._compensate(ctx)
                return False, self._records

        logger.info("saga.completed", steps=len(self._records))
        return True, self._records

    async def _compensate(self, ctx: SagaContext) -> None:
        """Execute compensation for all completed steps in reverse order.

        Each compensation call is protected by its own circuit breaker.
        A compensation failure does NOT block subsequent compensations.
        """
        for step in reversed(self._completed):
            record = next(
                r for r in reversed(self._records)
                if r.name == step.name and r.state == StepState.COMPLETED
            )
            try:
                # L4→L1→L2 protected compensate
                ok, err = await step._protected_compensate(ctx)
                record.compensate_circuit = (
                    step._compensate_circuit.state.value
                )

                if ok:
                    record.state = StepState.COMPENSATED
                    logger.info("saga.compensated", step=step.name)
                else:
                    record.state = StepState.COMPENSATE_FAILED
                    record.error = err
                    logger.error(
                        "saga.compensation_failed",
                        step=step.name,
                        error=err or "",
                    )
            except Exception as exc:
                # Safety net
                record.state = StepState.COMPENSATE_FAILED
                record.error = str(exc)
                record.compensate_circuit = (
                    step._compensate_circuit.state.value
                )
                logger.error(
                    "saga.compensation_failed",
                    step=step.name,
                    error=str(exc),
                )


# =========================================================================
# Fulfillment Saga — assembled facade
# =========================================================================


def build_fulfillment_saga() -> SagaOrchestrator:
    """Assemble the 4-step fulfillment Saga with four-layer protection."""
    return SagaOrchestrator(
        [
            CheckInventoryStep(),
            ReserveInventoryStep(),
            MatchLogisticsStep(),
            CreateOrderStep(),
        ]
    )


async def run_fulfillment_saga(
    request: FulfillmentRequest,
) -> tuple[FulfillmentResult, list[SagaStepRecord]]:
    """Run the fulfillment Saga and return result + audit trail.

    On success: FulfillmentResult with order/reservation/route.
    On failure: FulfillmentResult with error, compensations already executed.
    """
    ctx = SagaContext(request=request)
    saga = build_fulfillment_saga()
    success, records = await saga.execute(ctx)

    if not success:
        failed_step = next(
            (r for r in records if r.state in (StepState.FAILED, StepState.CIRCUIT_OPEN)),
            None,
        )
        error_msg = (
            f"saga failed at '{failed_step.name}': {failed_step.error or 'business failure'}"
            if failed_step
            else "saga failed"
        )
        return (
            FulfillmentResult(
                success=False,
                error=error_msg,
                data={"saga_steps": [r.name for r in records]},
            ),
            records,
        )

    req = ctx.request
    resv = ctx.reservation
    route = ctx.route
    order = ctx.order

    reservation = Reservation(
        reservation_id=resv["reservation_id"],
        product_id=req.product.product_id,
        warehouse_id=resv.get("warehouse_id", ctx.warehouse_id or ""),
        quantity=req.quantity,
        expires_at=datetime.fromisoformat(resv["expires_at"]),
    )
    logistics_route = LogisticsRoute(
        route_id=route["route_id"],
        carrier=route.get("carrier", ""),
        warehouse_id=route.get("warehouse_id", ctx.warehouse_id or ""),
        destination=route.get("destination", req.destination),
        insured=route.get("insured", False),
        insured_amount=route.get("insured_amount", 0.0),
        encrypted=route.get("encrypted", False),
        eta_hours=route.get("eta_hours", 48),
    )
    placed = Order(
        order_id=order["order_id"],
        user_id=req.user_id,
        product_id=req.product.product_id,
        quantity=req.quantity,
        reservation_id=reservation.reservation_id,
        logistics_route_id=logistics_route.route_id,
        status=order.get("status", "created"),
        total_amount=order.get("total_amount", 0.0),
    )
    return (
        FulfillmentResult(
            success=True,
            order=placed,
            reservation=reservation,
            logistics_route=logistics_route,
            data={
                "carrier": logistics_route.carrier,
                "insured": logistics_route.insured,
                "encrypted": logistics_route.encrypted,
                "saga_steps": [r.name for r in records],
            },
        ),
        records,
    )


# =========================================================================
# helpers
# =========================================================================


def _extract_result(ctx: SagaContext, step_name: str) -> dict[str, Any]:
    """Extract a snapshot of the step's output for the audit trail."""
    mapping = {
        "check_inventory": ctx.inventory_result,
        "reserve_inventory": ctx.reservation,
        "match_logistics": ctx.route,
        "create_order": ctx.order,
    }
    return mapping.get(step_name, {})
