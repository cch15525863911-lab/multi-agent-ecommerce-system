"""
传统业务 Service 基类 — 熔断/超时/降级 三层防护。

与 BaseAgent 的区别:
    - 不依赖 LLM, 不做 ReAct 工具选择
    - 去掉重试层 (业务操作多为非幂等, 重试可能导致重复扣减)
    - 保留熔断 (L4) + 超时 (L2) + 降级 (L3)

执行流程:
    run() → [L4: 熔断检查] → [L2: 超时] → execute()
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


class BaseProtectedService(ABC):
    """传统业务 Service 的保护基类 — 熔断/超时/降级。"""

    def __init__(
        self,
        name: str,
        timeout: float = 10.0,
        max_retries: int = 1,
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

    async def run(self, **kwargs: Any) -> AgentResult:
        """公共入口: 熔断检查 → 超时执行 → 降级。"""
        start = time.perf_counter()
        self._call_count += 1

        if not self._circuit.allow_request():
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "service.circuit_open",
                service=self.name,
                state=self._circuit.state.value,
            )
            return self._fallback(latency_ms, CircuitOpenError(self.name))

        try:
            result = await self._retry_execute(**kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000
            self._circuit.record_success()
            logger.info(
                "service.success",
                service=self.name,
                latency_ms=round(result.latency_ms, 1),
            )
            return result
        except Exception as exc:
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            self._circuit.record_failure()
            logger.error(
                "service.failed",
                service=self.name,
                error=str(exc),
                latency_ms=round(latency_ms, 1),
            )
            return self._fallback(latency_ms, exc)

    async def _retry_execute(self, **kwargs: Any) -> AgentResult:
        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        )
        async def _single_attempt() -> AgentResult:
            return await asyncio.wait_for(
                self.execute(**kwargs),
                timeout=self.timeout,
            )

        return await _single_attempt()

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
        )

    @abstractmethod
    async def execute(self, **kwargs: Any) -> AgentResult:
        """子类实现的具体业务逻辑。"""
        ...

    @property
    def error_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count

    @property
    def circuit_state(self) -> str:
        return self._circuit.state.value
