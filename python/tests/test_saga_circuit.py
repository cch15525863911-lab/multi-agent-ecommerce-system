"""
Saga 四层防护集成测试 — execute 与 compensate 独立熔断。

测试覆盖:
    1. 正常 Saga: 四层防护不干预, 步骤全部成功
    2. execute 重试: execute 前几次抛异常, 重试后成功
    3. execute 熔断: execute 连续失败 → OPEN → 步骤跳过(CIRCUIT_OPEN)
    4. compensate 重试: 补偿前几次抛异常, 重试后成功
    5. compensate 熔断: 补偿连续失败 → OPEN → COMPENSATE_FAILED
    6. execute/compensate 独立熔断: execute 熔断不影响 compensate
    7. 审计轨迹: execute_circuit/compensate_circuit 字段正确记录

Run from the `python/` directory:
    python -m tests.test_saga_circuit
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.schemas import FulfillmentRequest, Product
from services import fulfillment_tools as ft
from services.saga import (
    SagaContext,
    SagaOrchestrator,
    SagaStep,
    SagaStepRecord,
    StepState,
)
from services.circuit_breaker import CircuitState


# =========================================================================
# Helpers
# =========================================================================


def _product(pid: str = "P001", price: float = 7999.0) -> Product:
    return Product(
        product_id=pid, name=f"Test-{pid}", category="手机", price=price, brand="Apple"
    )


def _request(pid: str = "P001", price: float = 7999.0, qty: int = 1) -> FulfillmentRequest:
    return FulfillmentRequest(
        user_id="u001", product=_product(pid, price), quantity=qty, destination="北京"
    )


# =========================================================================
# Test fixtures — controllable Saga steps
# =========================================================================


class _ControllableStep(SagaStep):
    """A Saga step with controllable execute/compensate behavior for testing."""

    name = "controllable"
    timeout = 2.0
    max_retries = 2

    def __init__(
        self,
        name: str = "controllable",
        execute_fail_times: int = 0,
        compensate_fail_times: int = 0,
        execute_delay: float = 0.0,
        step_timeout: float | None = None,
        step_max_retries: int | None = None,
        failure_threshold: float = 0.5,
        window_size: int = 4,
        recovery_timeout: float = 999,
    ):
        self.name = name
        if step_timeout is not None:
            self.timeout = step_timeout
        if step_max_retries is not None:
            self.max_retries = step_max_retries
        self._execute_fail_times = execute_fail_times
        self._compensate_fail_times = compensate_fail_times
        self._execute_delay = execute_delay
        self._execute_call_index = 0
        self._compensate_call_index = 0
        # Initialize circuit breakers manually (can't call super().__init__()
        # because we override name after class-level default)
        from services.circuit_breaker import CircuitBreaker
        self._execute_circuit = CircuitBreaker(
            agent_name=f"saga.{name}.execute",
            failure_threshold=failure_threshold,
            window_size=window_size,
            recovery_timeout=recovery_timeout,
        )
        self._compensate_circuit = CircuitBreaker(
            agent_name=f"saga.{name}.compensate",
            failure_threshold=failure_threshold,
            window_size=window_size,
            recovery_timeout=recovery_timeout,
        )

    async def execute(self, ctx: SagaContext) -> bool:
        self._execute_call_index += 1
        if self._execute_delay > 0:
            await asyncio.sleep(self._execute_delay)
        if self._execute_call_index <= self._execute_fail_times:
            raise RuntimeError(f"execute failure #{self._execute_call_index}")
        return True

    async def compensate(self, ctx: SagaContext) -> None:
        self._compensate_call_index += 1
        if self._compensate_call_index <= self._compensate_fail_times:
            raise RuntimeError(f"compensate failure #{self._compensate_call_index}")


class _BusinessFailStep(SagaStep):
    """A step that always returns False (business failure, not exception)."""

    name = "business_fail"
    timeout = 2.0
    max_retries = 0

    async def execute(self, ctx: SagaContext) -> bool:
        return False

    async def compensate(self, ctx: SagaContext) -> None:
        pass


class _SuccessStep(SagaStep):
    """A step that always succeeds."""

    name = "success"
    timeout = 2.0
    max_retries = 0

    async def execute(self, ctx: SagaContext) -> bool:
        return True

    async def compensate(self, ctx: SagaContext) -> None:
        pass


# =========================================================================
# Test 1: Normal Saga — four-layer protection does not interfere
# =========================================================================


async def test_normal_saga() -> None:
    """All steps succeed, circuit breakers stay CLOSED."""
    ft.reset_inmemory_state()
    step = _ControllableStep(name="normal_step", execute_fail_times=0)
    saga = SagaOrchestrator([step])
    ctx = SagaContext(request=_request())

    success, records = await saga.execute(ctx)

    assert success, "saga should succeed"
    assert len(records) == 1
    assert records[0].state == StepState.COMPLETED
    assert records[0].execute_circuit == CircuitState.CLOSED.value
    print("[OK] normal saga: protection does not interfere, circuit=CLOSED")


# =========================================================================
# Test 2: execute retry — transient failure then success
# =========================================================================


async def test_execute_retry() -> None:
    """execute fails once (exception), retry succeeds, circuit stays CLOSED."""
    step = _ControllableStep(
        name="retry_step",
        execute_fail_times=1,
        step_max_retries=2,
    )
    saga = SagaOrchestrator([step])
    ctx = SagaContext(request=_request())

    success, records = await saga.execute(ctx)

    assert success, f"saga should succeed after retry: {records[0].error}"
    assert records[0].state == StepState.COMPLETED
    assert records[0].execute_circuit == CircuitState.CLOSED.value
    assert step._execute_call_index == 2, "should have retried once"
    print("[OK] execute retry: failed once, retried, succeeded, circuit=CLOSED")


# =========================================================================
# Test 3: execute circuit opens after consecutive failures
# =========================================================================


async def test_execute_circuit_opens() -> None:
    """execute keeps failing → circuit OPEN → step skipped (CIRCUIT_OPEN)."""
    step = _ControllableStep(
        name="circuit_step",
        execute_fail_times=99,
        step_max_retries=0,
        failure_threshold=0.5,
        window_size=4,
        recovery_timeout=999,
    )
    saga = SagaOrchestrator([step])

    # Fill the window with failures to trip the circuit
    for _ in range(4):
        ctx = SagaContext(request=_request())
        await saga.execute(ctx)

    # Verify circuit is OPEN
    assert step._execute_circuit.state == CircuitState.OPEN, (
        f"execute circuit should be OPEN after 4 failures, got {step._execute_circuit.state}"
    )

    # Next call should be skipped by circuit
    ctx = SagaContext(request=_request())
    call_before = step._execute_call_index
    success, records = await saga.execute(ctx)
    call_after = step._execute_call_index

    assert not success
    assert records[-1].state == StepState.CIRCUIT_OPEN, (
        f"step should be CIRCUIT_OPEN, got {records[-1].state}"
    )
    assert call_after == call_before, "execute should not be called when circuit OPEN"
    print("[OK] execute circuit OPEN: 4 failures → open → step skipped (CIRCUIT_OPEN)")


# =========================================================================
# Test 4: compensate retry — transient failure then success
# =========================================================================


async def test_compensate_retry() -> None:
    """compensate fails once (exception), retry succeeds."""
    step_ok = _SuccessStep()
    step_fail = _ControllableStep(
        name="fail_step",
        execute_fail_times=0,
        compensate_fail_times=1,
        step_max_retries=2,
    )
    saga = SagaOrchestrator([step_ok, step_fail])
    ctx = SagaContext(request=_request())

    # step_ok succeeds, step_fail succeeds (execute), then we force a failure
    # to trigger compensation. We'll use a business-fail step after them.
    step_business_fail = _BusinessFailStep()
    saga2 = SagaOrchestrator([step_ok, step_fail, step_business_fail])
    ctx2 = SagaContext(request=_request())

    success, records = await saga2.execute(ctx2)

    assert not success, "saga should fail at business_fail step"
    # step_fail's compensate should have been called and retried
    assert step_fail._compensate_call_index == 2, (
        f"compensate should have been retried, got {step_fail._compensate_call_index} calls"
    )
    # step_fail should be COMPENSATED (retry succeeded)
    fail_record = next(r for r in records if r.name == "fail_step")
    assert fail_record.state == StepState.COMPENSATED, (
        f"compensate should succeed after retry, got {fail_record.state}"
    )
    print("[OK] compensate retry: failed once, retried, compensated successfully")


# =========================================================================
# Test 5: compensate circuit opens — COMPENSATE_FAILED
# =========================================================================


async def test_compensate_circuit_opens() -> None:
    """compensate keeps failing → circuit OPEN → COMPENSATE_FAILED."""
    # We need a step that succeeds execute but always fails compensate
    step = _ControllableStep(
        name="comp_circuit_step",
        execute_fail_times=0,
        compensate_fail_times=99,
        step_max_retries=0,
        failure_threshold=0.5,
        window_size=4,
        recovery_timeout=999,
    )
    step_trigger = _BusinessFailStep()

    # Run the saga multiple times to trip the compensate circuit.
    # Each run: step executes OK, then business_fail triggers compensation.
    # step's compensate fails each time.
    for _ in range(4):
        saga = SagaOrchestrator([step, step_trigger])
        ctx = SagaContext(request=_request())
        await saga.execute(ctx)

    # Verify compensate circuit is OPEN
    assert step._compensate_circuit.state == CircuitState.OPEN, (
        f"compensate circuit should be OPEN after 4 failures, "
        f"got {step._compensate_circuit.state}"
    )

    # Next run: compensate should be skipped by circuit
    comp_before = step._compensate_call_index
    saga = SagaOrchestrator([step, step_trigger])
    ctx = SagaContext(request=_request())
    _, records = await saga.execute(ctx)
    comp_after = step._compensate_call_index

    assert comp_after == comp_before, (
        "compensate should not be called when circuit OPEN"
    )
    # The step's record should show COMPENSATE_FAILED
    step_record = next(r for r in records if r.name == "comp_circuit_step")
    assert step_record.state == StepState.COMPENSATE_FAILED, (
        f"step should be COMPENSATE_FAILED, got {step_record.state}"
    )
    assert "circuit_open" in (step_record.error or ""), (
        f"error should mention circuit_open, got: {step_record.error}"
    )
    print("[OK] compensate circuit OPEN: 4 failures → open → COMPENSATE_FAILED")


# =========================================================================
# Test 6: execute/compensate independent breakers
# =========================================================================


async def test_independent_breakers() -> None:
    """execute circuit trips but compensate circuit stays CLOSED."""
    step = _ControllableStep(
        name="indep_step",
        execute_fail_times=99,
        compensate_fail_times=0,
        step_max_retries=0,
        failure_threshold=0.5,
        window_size=4,
        recovery_timeout=999,
    )
    step_trigger = _BusinessFailStep()

    # Trip the execute circuit (4 runs with execute always failing)
    for _ in range(4):
        saga = SagaOrchestrator([step, step_trigger])
        ctx = SagaContext(request=_request())
        await saga.execute(ctx)

    assert step._execute_circuit.state == CircuitState.OPEN
    assert step._compensate_circuit.state == CircuitState.CLOSED, (
        "compensate circuit should remain CLOSED when only execute fails"
    )
    print(
        f"[OK] independent breakers: execute={step._execute_circuit.state.value}, "
        f"compensate={step._compensate_circuit.state.value}"
    )


# =========================================================================
# Test 7: audit trail — circuit states recorded
# =========================================================================


async def test_audit_trail_circuit_fields() -> None:
    """SagaStepRecord contains execute_circuit and compensate_circuit fields."""
    ft.reset_inmemory_state()
    from services.saga import (
        CheckInventoryStep,
        ReserveInventoryStep,
        MatchLogisticsStep,
        CreateOrderStep,
    )

    saga = SagaOrchestrator([
        CheckInventoryStep(),
        ReserveInventoryStep(),
        MatchLogisticsStep(),
        CreateOrderStep(),
    ])
    ctx = SagaContext(request=_request())

    success, records = await saga.execute(ctx)

    assert success
    assert len(records) == 4

    for r in records:
        assert hasattr(r, "execute_circuit"), "record must have execute_circuit"
        assert hasattr(r, "compensate_circuit"), "record must have compensate_circuit"
        assert r.execute_circuit == CircuitState.CLOSED.value, (
            f"execute circuit should be CLOSED for {r.name}"
        )
        assert r.compensate_circuit == CircuitState.CLOSED.value, (
            f"compensate circuit should be CLOSED for {r.name}"
        )

    print(
        f"[OK] audit trail: all records have circuit fields, "
        f"steps={[r.name for r in records]}"
    )


# =========================================================================
# Test 8: execute timeout triggers fallback
# =========================================================================


async def test_execute_timeout() -> None:
    """execute delays beyond timeout → exception → fallback after retries."""
    step = _ControllableStep(
        name="timeout_step",
        execute_delay=0.5,
        step_timeout=0.1,
        step_max_retries=1,
    )
    saga = SagaOrchestrator([step])
    ctx = SagaContext(request=_request())

    success, records = await saga.execute(ctx)

    assert not success, "saga should fail due to timeout"
    assert records[0].state in (StepState.FAILED, StepState.CIRCUIT_OPEN)
    print(f"[OK] execute timeout: delayed 0.5s > timeout 0.1s → {records[0].state.value}")


# =========================================================================
# Main
# =========================================================================


async def main() -> int:
    print("=" * 60)
    print("Saga 四层防护集成测试 — execute/compensate 独立熔断")
    print("=" * 60)
    await test_normal_saga()
    await test_execute_retry()
    await test_execute_circuit_opens()
    await test_compensate_retry()
    await test_compensate_circuit_opens()
    await test_independent_breakers()
    await test_audit_trail_circuit_fields()
    await test_execute_timeout()
    print("=" * 60)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
