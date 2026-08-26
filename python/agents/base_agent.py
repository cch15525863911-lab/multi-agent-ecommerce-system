"""
Agent 基类 — 重试/独立超时/降级/熔断 四层防护。

四层防护按顺序生效:
    Layer 1  重试        指数退避重试(500ms→1s→2s, 最多2次), 覆盖瞬时抖动
    Layer 2  独立超时    asyncio.wait_for 强制超时, 每次尝试独立计时, 防止长尾阻塞
    Layer 3  降级        重试/超时/熔断均失败后返回降级结果, 保证链路不中断
    Layer 4  熔断        滑动窗口错误率≥阈值→OPEN(直接降级), 恢复期后→HALF_OPEN探测

执行流程:
    run() → [L4: 熔断检查] → [L1: 重试+L2: 超时] → _execute()
                ↓ OPEN              ↓ 失败
            [L3: 降级]         [L4: 记录失败] → [L3: 降级]
                                    ↓ 成功
                              [L4: 记录成功] → 返回结果
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import AgentResult
from services.circuit_breaker import CircuitBreaker, CircuitOpenError

logger = structlog.get_logger()


class BaseAgent(ABC):
    """All agents inherit from this base class with four-layer protection."""

    def __init__(
        self,
        name: str,
        timeout: float = 10.0,
        max_retries: int = 2,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
        self._error_count = 0
        self._circuit = circuit_breaker or CircuitBreaker(
            agent_name=name,
            failure_threshold=0.5,
            window_size=10,
            recovery_timeout=30.0,
        )

    # ------------------------------------------------------------------
    # public entry — four-layer protection pipeline
    # ------------------------------------------------------------------

    async def run(self, **kwargs: Any) -> AgentResult:
        """Public entry: wraps _execute with four layers of protection."""
        start = time.perf_counter()
        self._call_count += 1

        # Layer 4: Circuit breaker — reject immediately if OPEN
        if not self._circuit.allow_request():
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "agent.circuit_open",
                agent=self.name,
                state=self._circuit.state.value,
            )
            return self._fallback(latency_ms, CircuitOpenError(self.name))

        try:
            # Layer 1 (retry) + Layer 2 (timeout) in _retry_execute
            result = await self._retry_execute(**kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000

            # Layer 4: record success
            self._circuit.record_success()
            logger.info(
                "agent.success",
                agent=self.name,
                latency_ms=round(result.latency_ms, 1),
            )
            return result

        except Exception as exc:
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000

            # Layer 4: record failure
            self._circuit.record_failure()
            logger.error(
                "agent.failed",
                agent=self.name,
                error=str(exc),
                latency_ms=round(latency_ms, 1),
                retries=self.max_retries,
            )

            # Layer 3: degradation / fallback
            return self._fallback(latency_ms, exc)

    # ------------------------------------------------------------------
    # Layer 1 (retry) + Layer 2 (independent timeout)
    # ------------------------------------------------------------------

    async def _retry_execute(self, **kwargs: Any) -> AgentResult:
        """Retry with exponential backoff; each attempt has its own timeout."""

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        )
        async def _single_attempt() -> AgentResult:
            # Layer 2: independent timeout per attempt
            return await asyncio.wait_for(
                self._execute(**kwargs),
                timeout=self.timeout,
            )

        return await _single_attempt()

    # ------------------------------------------------------------------
    # Layer 3: fallback / degradation
    # ------------------------------------------------------------------

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        """Return a degraded but valid result when the agent fails.

        Override in subclasses to provide agent-specific fallback data
        (e.g. default user profile, hot product list, template copy).
        """
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
        )

    # ------------------------------------------------------------------
    # abstract — subclass implements business logic
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Core logic implemented by each concrete agent."""
        ...

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------

    @property
    def error_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count

    @property
    def circuit_state(self) -> str:
        """Current circuit breaker state for health monitoring."""
        return self._circuit.state.value
