"""
四层防护测试 — 重试 / 独立超时 / 降级 / 熔断。

测试覆盖:
    1. 正常调用: 四层全部放行, 返回结果
    2. Layer 1 重试: 瞬时失败后重试成功
    3. Layer 2 超时: _execute 阻塞超过 timeout → 降级
    4. Layer 3 降级: 所有重试失败后返回 fallback 结果
    5. Layer 4 熔断: 滑动窗口错误率≥阈值 → OPEN → 直接降级
    6. Layer 4 半开探测: 恢复期后 HALF_OPEN, 探测成功 → CLOSED
    7. Layer 4 重开: 半开探测失败 → 重新 OPEN
    8. 熔断器独立计数: 调用计数与错误率

Run from the `python/` directory:
    python -m tests.test_circuit_breaker
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import BaseAgent
from models.schemas import AgentResult
from services.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError


# =========================================================================
# Test fixture — a configurable agent for testing each layer
# =========================================================================


class _TestAgent(BaseAgent):
    """Controllable agent: can fail, delay, or succeed on demand."""

    def __init__(
        self,
        fail_times: int = 0,
        delay: float = 0.0,
        timeout: float = 10.0,
        max_retries: int = 2,
        **cb_kwargs: Any,
    ):
        super().__init__(
            name="test_agent",
            timeout=timeout,
            max_retries=max_retries,
            circuit_breaker=CircuitBreaker(
                agent_name="test_agent", **cb_kwargs
            ),
        )
        self._fail_times = fail_times
        self._delay = delay
        self._call_index = 0

    async def _execute(self, **kwargs: Any) -> AgentResult:
        self._call_index += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._call_index <= self._fail_times:
            raise RuntimeError(f"simulated failure #{self._call_index}")
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={"call_index": self._call_index},
            confidence=1.0,
        )

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            data={"fallback": True},
            confidence=0.0,
        )

    def reset_circuit(self) -> None:
        self._circuit.reset()


# =========================================================================
# Test 1: Normal operation — all layers pass through
# =========================================================================


async def test_normal_operation() -> None:
    """Healthy agent: request succeeds, circuit stays CLOSED."""
    agent = _TestAgent(fail_times=0)
    result = await agent.run()

    assert result.success
    assert result.data["call_index"] == 1
    assert agent.circuit_state == CircuitState.CLOSED.value
    print("[OK] normal: success, circuit=CLOSED")


# =========================================================================
# Test 2: Layer 1 — retry succeeds after transient failure
# =========================================================================


async def test_retry_succeeds() -> None:
    """First attempt fails, second succeeds (retry works)."""
    agent = _TestAgent(fail_times=1, max_retries=2)
    result = await agent.run()

    assert result.success, f"should succeed after retry: {result.error}"
    assert result.data["call_index"] == 2, "second attempt should succeed"
    assert agent.circuit_state == CircuitState.CLOSED.value
    print("[OK] retry: failed once, succeeded on retry, circuit=CLOSED")


# =========================================================================
# Test 3: Layer 2 — independent timeout triggers fallback
# =========================================================================


async def test_timeout_triggers_fallback() -> None:
    """_execute delays beyond timeout → fallback after all retries."""
    agent = _TestAgent(delay=0.5, timeout=0.1, max_retries=1)
    result = await agent.run()

    assert not result.success, "should fail due to timeout"
    assert result.data.get("fallback") is True, "should return fallback"
    # asyncio.TimeoutError str() is empty in Python 3.11+, check latency instead
    assert result.latency_ms > 200, "should have taken multiple timeout periods"
    print(f"[OK] timeout: delayed 0.5s > timeout 0.1s → fallback, latency={result.latency_ms:.0f}ms")


# =========================================================================
# Test 4: Layer 3 — degradation fallback after all retries exhausted
# =========================================================================


async def test_fallback_on_all_failures() -> None:
    """Agent always fails → all retries exhausted → fallback returned."""
    agent = _TestAgent(fail_times=99, max_retries=2)
    result = await agent.run()

    assert not result.success
    assert result.data.get("fallback") is True
    assert result.confidence == 0.0
    print("[OK] fallback: all retries exhausted, fallback returned")


# =========================================================================
# Test 5: Layer 4 — circuit breaker opens after threshold failures
# =========================================================================


async def test_circuit_opens() -> None:
    """Enough failures → circuit opens → next request gets fallback immediately."""
    # Use small window for fast testing
    agent = _TestAgent(
        fail_times=99, max_retries=0,
        failure_threshold=0.5, window_size=4, recovery_timeout=999,
    )

    # Fill the window with failures (4 calls, all fail)
    for i in range(4):
        await agent.run()

    state = agent.circuit_state
    assert state == CircuitState.OPEN.value, (
        f"circuit should be OPEN after 4 failures in window of 4, got {state}"
    )

    # Next call should be rejected immediately by circuit (no _execute call)
    call_before = agent._call_index
    result = await agent.run()
    call_after = agent._call_index

    assert not result.success
    assert call_after == call_before, "circuit should not call _execute when OPEN"
    print("[OK] circuit OPEN: 4 failures → open → immediate fallback (no execute)")


# =========================================================================
# Test 6: Layer 4 — half-open probe after recovery period succeeds
# =========================================================================


async def test_circuit_half_open_recovery() -> None:
    """Circuit opens → recovery timeout → HALF_OPEN → probe succeeds → CLOSED."""
    agent = _TestAgent(
        fail_times=0,  # will fail first 4 times, then succeed
        max_retries=0,
        failure_threshold=0.5, window_size=4, recovery_timeout=0.2,
    )

    # Make it fail 4 times to open the circuit
    agent._fail_times = 99
    for _ in range(4):
        await agent.run()
    assert agent.circuit_state == CircuitState.OPEN.value

    # Wait for recovery timeout
    await asyncio.sleep(0.3)

    # Now make it succeed
    agent._fail_times = 0
    result = await agent.run()

    assert result.success, "half-open probe should succeed"
    assert agent.circuit_state == CircuitState.CLOSED.value, (
        "circuit should close after successful probe"
    )
    print("[OK] half-open → CLOSED: recovery timeout → probe → success → closed")


# =========================================================================
# Test 7: Layer 4 — half-open probe fails → circuit reopens
# =========================================================================


async def test_circuit_half_open_failure_reopens() -> None:
    """Circuit opens → recovery → HALF_OPEN → probe fails → back to OPEN."""
    agent = _TestAgent(
        fail_times=99,
        max_retries=0,
        failure_threshold=0.5, window_size=4, recovery_timeout=0.2,
    )

    # Open the circuit
    for _ in range(4):
        await agent.run()
    assert agent.circuit_state == CircuitState.OPEN.value

    # Wait for recovery
    await asyncio.sleep(0.3)

    # Probe should fail (still failing)
    result = await agent.run()
    assert not result.success
    assert agent.circuit_state == CircuitState.OPEN.value, (
        "circuit should reopen after failed probe"
    )
    print("[OK] half-open → OPEN: probe failed → reopened")


# =========================================================================
# Test 8: Circuit breaker metrics — call count and error rate
# =========================================================================


async def test_agent_metrics() -> None:
    """Verify call_count and error_rate are tracked correctly."""
    agent = _TestAgent(fail_times=1, max_retries=2)

    # First run: 1 fail + 1 success = 1 total call, 0 errors
    await agent.run()
    assert agent._call_count == 1
    assert agent._error_count == 0
    assert agent.error_rate == 0.0

    # Second run: succeeds immediately
    await agent.run()
    assert agent._call_count == 2
    assert agent._error_count == 0
    assert agent.error_rate == 0.0

    print(f"[OK] metrics: calls={agent._call_count}, errors={agent._error_count}, rate={agent.error_rate}")


# =========================================================================
# Test 9: Circuit breaker isolation — each agent has independent breaker
# =========================================================================


async def test_circuit_isolation() -> None:
    """Two agents have independent circuit breakers."""
    agent_a = _TestAgent(
        fail_times=99, max_retries=0,
        failure_threshold=0.5, window_size=4, recovery_timeout=999,
    )
    agent_b = _TestAgent(fail_times=0, max_retries=0)

    # Trip agent_a's circuit
    for _ in range(4):
        await agent_a.run()
    assert agent_a.circuit_state == CircuitState.OPEN.value

    # Agent_b should be unaffected
    result = await agent_b.run()
    assert result.success
    assert agent_b.circuit_state == CircuitState.CLOSED.value

    print(f"[OK] isolation: agent_a={agent_a.circuit_state}, agent_b={agent_b.circuit_state}")


# =========================================================================
# Main
# =========================================================================


async def main() -> int:
    print("=" * 60)
    print("四层防护测试 — 重试 / 独立超时 / 降级 / 熔断")
    print("=" * 60)
    await test_normal_operation()
    await test_retry_succeeds()
    await test_timeout_triggers_fallback()
    await test_fallback_on_all_failures()
    await test_circuit_opens()
    await test_circuit_half_open_recovery()
    await test_circuit_half_open_failure_reopens()
    await test_agent_metrics()
    await test_circuit_isolation()
    print("=" * 60)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
