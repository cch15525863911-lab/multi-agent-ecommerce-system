"""
Agent 链路追踪 (Tracing / Observability) — 类 LangSmith / Phoenix 风格。

功能:
    - 记录每次 Agent 调用的完整链路 (输入/输出/耗时/Token/错误)
    - 支持按 request_id / user_id / agent_name 查询追踪记录
    - 统计指标: 调用次数、成功率、平均延迟、Token 消耗
    - 提供 Web 可视化接口 (简化版, 生产环境建议接入 LangSmith / Phoenix)

存储:
    - 默认内存存储 (演示/开发用)
    - 可扩展到 Redis / PostgreSQL / 对接 LangSmith API
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class Span:
    """单个追踪 span — 对应一次 Agent 调用或工具调用。"""
    span_id: str
    parent_span_id: str | None
    trace_id: str
    name: str
    span_type: str  # "agent" | "tool" | "llm" | "http"
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "running"  # "running" | "success" | "error"
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)  # prompt/completion/total


@dataclass
class Trace:
    """完整追踪链路 — 对应一次用户请求的全链路。"""
    trace_id: str
    user_id: str
    intent: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "running"
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentTracer:
    """Agent 链路追踪器 — 记录、查询、统计 Agent 调用链路。"""

    def __init__(self, max_traces: int = 1000):
        self._traces: deque[Trace] = deque(maxlen=max_traces)
        self._trace_index: dict[str, Trace] = {}  # trace_id → Trace
        self._span_index: dict[str, Span] = {}     # span_id → Span
        self._agent_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "successes": 0, "errors": 0,
                     "total_latency_ms": 0, "total_tokens": 0}
        )

    # ------------------------------------------------------------------
    # Trace 管理
    # ------------------------------------------------------------------

    def start_trace(self, user_id: str, intent: str = "") -> str:
        """开始一条追踪链路, 返回 trace_id。"""
        trace_id = str(uuid.uuid4())
        trace = Trace(
            trace_id=trace_id,
            user_id=user_id,
            intent=intent,
            start_time=time.perf_counter(),
        )
        self._traces.append(trace)
        self._trace_index[trace_id] = trace
        return trace_id

    def end_trace(self, trace_id: str, status: str = "success") -> None:
        """结束追踪链路。"""
        trace = self._trace_index.get(trace_id)
        if trace:
            trace.end_time = time.perf_counter()
            trace.duration_ms = (trace.end_time - trace.start_time) * 1000
            trace.status = status

    # ------------------------------------------------------------------
    # Span 管理
    # ------------------------------------------------------------------

    def start_span(
        self,
        trace_id: str,
        name: str,
        span_type: str = "agent",
        parent_span_id: str | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> str:
        """开始一个 span, 返回 span_id。"""
        span_id = str(uuid.uuid4())
        span = Span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            name=name,
            span_type=span_type,
            start_time=time.perf_counter(),
            input_data=input_data or {},
        )
        trace = self._trace_index.get(trace_id)
        if trace:
            trace.spans.append(span)
        self._span_index[span_id] = span
        return span_id

    def end_span(
        self,
        span_id: str,
        status: str = "success",
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """结束一个 span。"""
        span = self._span_index.get(span_id)
        if not span:
            return
        span.end_time = time.perf_counter()
        span.duration_ms = (span.end_time - span.start_time) * 1000
        span.status = status
        span.output_data = output_data or {}
        span.error = error
        span.token_usage = token_usage or {}
        span.metadata = metadata or {}

        # 更新 agent 统计
        if span.span_type == "agent":
            stats = self._agent_stats[span.name]
            stats["calls"] += 1
            stats["total_latency_ms"] += span.duration_ms
            if status == "success":
                stats["successes"] += 1
            else:
                stats["errors"] += 1
            stats["total_tokens"] += span.token_usage.get("total", 0)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """获取单条 trace 详情 (含所有 spans)。"""
        trace = self._trace_index.get(trace_id)
        if not trace:
            return None
        return {
            "trace_id": trace.trace_id,
            "user_id": trace.user_id,
            "intent": trace.intent,
            "duration_ms": round(trace.duration_ms, 1),
            "status": trace.status,
            "span_count": len(trace.spans),
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent_span_id": s.parent_span_id,
                    "name": s.name,
                    "type": s.span_type,
                    "duration_ms": round(s.duration_ms, 1),
                    "status": s.status,
                    "error": s.error,
                    "token_usage": s.token_usage,
                }
                for s in trace.spans
            ],
        }

    def get_recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取最近的追踪记录。"""
        traces = list(self._traces)[-limit:]
        return [
            {
                "trace_id": t.trace_id,
                "user_id": t.user_id,
                "intent": t.intent,
                "duration_ms": round(t.duration_ms, 1),
                "status": t.status,
                "span_count": len(t.spans),
            }
            for t in reversed(traces)
        ]

    def get_agent_stats(self) -> dict[str, Any]:
        """获取各 Agent 的统计指标。"""
        result = {}
        for name, stats in self._agent_stats.items():
            calls = stats["calls"]
            avg_latency = stats["total_latency_ms"] / calls if calls > 0 else 0
            success_rate = stats["successes"] / calls if calls > 0 else 0
            result[name] = {
                "calls": calls,
                "successes": stats["successes"],
                "errors": stats["errors"],
                "success_rate": round(success_rate, 4),
                "avg_latency_ms": round(avg_latency, 1),
                "total_tokens": stats["total_tokens"],
            }
        return result

    def get_summary(self) -> dict[str, Any]:
        """获取追踪系统总览。"""
        total_traces = len(self._traces)
        completed = sum(1 for t in self._traces if t.status != "running")
        avg_duration = (
            sum(t.duration_ms for t in self._traces if t.status != "running") / completed
            if completed > 0 else 0
        )
        return {
            "total_traces": total_traces,
            "completed_traces": completed,
            "avg_duration_ms": round(avg_duration, 1),
            "agent_count": len(self._agent_stats),
        }


# 单例
_tracer: AgentTracer | None = None


def get_tracer() -> AgentTracer:
    global _tracer
    if _tracer is None:
        _tracer = AgentTracer()
    return _tracer
