"""
熔断器 (Circuit Breaker) — 四层防护第四层。

三态有限状态机:
    CLOSED     → 正常放行, 记录调用结果
    OPEN       → 熔断打开, 请求直接降级(跳过执行)
    HALF_OPEN  → 半开探测, 允许 1 次试探请求

状态转换:
    CLOSED  --(错误率≥阈值)-->  OPEN
    OPEN    --(超过恢复时间)-->  HALF_OPEN
    HALF_OPEN --(探测成功)-->   CLOSED
    HALF_OPEN --(探测失败)-->   OPEN

滑动窗口: 记录最近 N 次调用的成功/失败, 计算错误率。
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Deque

import structlog

logger = structlog.get_logger()


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is OPEN and rejects a request."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        super().__init__(f"circuit breaker open for agent '{agent_name}'")


class CircuitBreaker:
    """Per-agent circuit breaker with sliding-window failure detection.

    Args:
        agent_name: Agent name for logging.
        failure_threshold: Error rate (0-1) that trips the breaker.
        window_size: Number of recent calls in the sliding window.
        recovery_timeout: Seconds before OPEN transitions to HALF_OPEN.
    """

    def __init__(
        self,
        agent_name: str = "",
        failure_threshold: float = 0.5,
        window_size: int = 10,
        recovery_timeout: float = 30.0,
    ):
        self._agent_name = agent_name
        self._state: CircuitState = CircuitState.CLOSED
        self._window: Deque[bool] = deque(maxlen=window_size)
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time: datetime | None = None

    @property
    def state(self) -> CircuitState:
        """Current state, with lazy OPEN→HALF_OPEN transition."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    "circuit.half_open",
                    agent=self._agent_name,
                    elapsed=round(elapsed, 1),
                )
        return self._state

    def allow_request(self) -> bool:
        """Return True if the request should proceed (CLOSED or HALF_OPEN)."""
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        self._window.append(True)
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._window.clear()
            logger.info(
                "circuit.closed",
                agent=self._agent_name,
                reason="half_open probe succeeded",
            )
        self._evaluate()

    def record_failure(self) -> None:
        """Record a failed call."""
        self._window.append(False)
        self._last_failure_time = datetime.now()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit.reopened",
                agent=self._agent_name,
                reason="half_open probe failed",
            )
        self._evaluate()

    def _evaluate(self) -> None:
        """Trip the breaker if the sliding-window error rate exceeds threshold."""
        if len(self._window) < self._window.maxlen:
            return
        success_count = sum(self._window)
        failure_rate = 1.0 - (success_count / len(self._window))
        if (
            self._state == CircuitState.CLOSED
            and failure_rate >= self._failure_threshold
        ):
            self._state = CircuitState.OPEN
            self._last_failure_time = datetime.now()
            logger.warning(
                "circuit.opened",
                agent=self._agent_name,
                failure_rate=round(failure_rate, 2),
                window=len(self._window),
            )

    def reset(self) -> None:
        """Reset to CLOSED and clear the window (for testing)."""
        self._state = CircuitState.CLOSED
        self._window.clear()
        self._last_failure_time = None
