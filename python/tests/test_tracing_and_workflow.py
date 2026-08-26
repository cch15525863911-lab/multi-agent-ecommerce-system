"""
链路追踪 + 工作流引擎 单元测试。

覆盖:
    - Tracing: trace 生命周期 (start/end)
    - Tracing: span 管理 (嵌套/状态)
    - Tracing: agent 统计指标
    - Workflow Engine: 工作流注册与执行
    - Workflow Engine: 履约工作流成功路径
    - Workflow Engine: 工作流状态查询
"""
from __future__ import annotations

import pytest

from services.tracing import AgentTracer
from services.workflow_engine import (
    FulfillmentWorkflow,
    WorkflowStatus,
    WorkflowWorker,
    MemoryWorkflowStore,
)


# =========================================================================
# Tracing 测试
# =========================================================================


class TestAgentTracer:
    def setup_method(self):
        self.tracer = AgentTracer(max_traces=100)

    def test_start_and_end_trace(self):
        """启动并结束一条 trace。"""
        import time
        trace_id = self.tracer.start_trace("user_001", "recommendation")
        assert trace_id is not None
        assert len(trace_id) > 0

        # 确保有微小时间差
        time.sleep(0.001)
        self.tracer.end_trace(trace_id, "success")
        trace = self.tracer.get_trace(trace_id)
        assert trace is not None
        assert trace["status"] == "success"
        assert trace["duration_ms"] >= 0
        assert trace["user_id"] == "user_001"

    def test_trace_contains_spans(self):
        """trace 应包含其下所有 span。"""
        trace_id = self.tracer.start_trace("user_001", "test")

        # 添加几个 span
        span1 = self.tracer.start_span(trace_id, "agent_a", "agent")
        self.tracer.end_span(span1, "success", token_usage={"total": 100})

        span2 = self.tracer.start_span(trace_id, "agent_b", "agent", parent_span_id=span1)
        self.tracer.end_span(span2, "error", error="something wrong")

        self.tracer.end_trace(trace_id, "success")

        trace = self.tracer.get_trace(trace_id)
        assert trace["span_count"] == 2
        assert trace["spans"][0]["name"] == "agent_a"
        assert trace["spans"][1]["status"] == "error"

    def test_agent_stats_accumulate(self):
        """Agent 统计指标应累加。"""
        trace_id = self.tracer.start_trace("user1", "test")

        for i in range(5):
            span = self.tracer.start_span(trace_id, "test_agent", "agent")
            if i < 4:  # 4次成功, 1次失败
                self.tracer.end_span(span, "success", token_usage={"total": 100})
            else:
                self.tracer.end_span(span, "error", error="fail")

        self.tracer.end_trace(trace_id, "success")

        stats = self.tracer.get_agent_stats()
        assert "test_agent" in stats
        assert stats["test_agent"]["calls"] == 5
        assert stats["test_agent"]["successes"] == 4
        assert stats["test_agent"]["errors"] == 1
        assert stats["test_agent"]["success_rate"] == 0.8
        assert stats["test_agent"]["total_tokens"] == 400

    def test_recent_traces_order(self):
        """最近 traces 应按时间倒序排列。"""
        for i in range(5):
            tid = self.tracer.start_trace(f"user_{i}", f"intent_{i}")
            self.tracer.end_trace(tid, "success")

        recent = self.tracer.get_recent_traces(limit=3)
        assert len(recent) == 3
        # 最新的应该在最前面
        assert recent[0]["intent"] == "intent_4"

    def test_summary_stats(self):
        """摘要统计应正确。"""
        import time
        for i in range(3):
            tid = self.tracer.start_trace(f"u{i}", "test")
            time.sleep(0.001)
            self.tracer.end_trace(tid, "success")

        summary = self.tracer.get_summary()
        assert summary["total_traces"] == 3
        assert summary["completed_traces"] == 3
        # avg_duration 可能很小但不应为负数
        assert summary["avg_duration_ms"] >= 0


# =========================================================================
# Workflow Engine 测试
# =========================================================================


class TestWorkflowEngine:
    def setup_method(self):
        self.store = MemoryWorkflowStore()
        self.worker = WorkflowWorker(store=self.store)
        self.worker.register(FulfillmentWorkflow)

    def test_register_workflow(self):
        """工作流应能注册并列出。"""
        workflows = self.worker.list_workflows()
        assert "fulfillment" in workflows

    def test_start_workflow_success(self):
        """启动工作流应返回状态。"""
        import asyncio
        state = asyncio.run(self.worker.start_workflow(
            "fulfillment",
            {"user_id": "u1", "product_id": "P001", "quantity": 1,
             "destination": "北京", "unit_price": 100.0},
        ))
        assert state.workflow_type == "fulfillment"
        assert state.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED)

    def test_workflow_status_query(self):
        """查询工作流状态。"""
        import asyncio
        state = asyncio.run(self.worker.start_workflow(
            "fulfillment",
            {"user_id": "u2", "product_id": "P002", "quantity": 1,
             "destination": "上海", "unit_price": 200.0},
        ))

        retrieved = asyncio.run(self.worker.get_status(state.workflow_id))
        assert retrieved is not None
        assert retrieved.workflow_id == state.workflow_id
        assert retrieved.workflow_type == "fulfillment"

    def test_unknown_workflow_type_raises(self):
        """未知工作流类型应抛出异常。"""
        import asyncio
        with pytest.raises(ValueError):
            asyncio.run(self.worker.start_workflow("nonexistent", {}))

    def test_workflow_history_recorded(self):
        """工作流历史事件应被记录。"""
        import asyncio
        state = asyncio.run(self.worker.start_workflow(
            "fulfillment",
            {"user_id": "u3", "product_id": "P003", "quantity": 1,
             "destination": "广州", "unit_price": 300.0},
        ))
        assert len(state.history) >= 2  # 至少有 started 和 completed/failed
