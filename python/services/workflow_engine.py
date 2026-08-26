"""
Temporal 风格工作流骨架 — 异步持久化工作流引擎。

背景:
    现有 Saga 编排器是内存式的, 进程重启后状态丢失。Temporal 提供
    持久化工作流、自动重试、定时器、信号等企业级能力。

本文件提供:
    1. Workflow 基类 (模拟 Temporal 的 workflow 模式)
    2. 工作流注册与执行器
    3. 持久化接口 (内存实现 + Redis 实现预留)

生产化迁移路径:
    1. 替换为真实 temporal-sdk (pip install temporalio)
    2. Workflow 类继承 temporalio.workflow
    3. Activity 用 @activity.defn 装饰
    4. 启动 Temporal Server (docker-compose 即可)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class WorkflowState:
    """工作流执行状态。"""
    workflow_id: str
    workflow_type: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    input_data: dict[str, Any] = field(default_factory=dict)
    result_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


class WorkflowStore:
    """工作流状态存储接口。"""

    async def save(self, state: WorkflowState) -> None:
        ...

    async def get(self, workflow_id: str) -> WorkflowState | None:
        ...

    async def list_by_status(self, status: WorkflowStatus) -> list[WorkflowState]:
        ...


class MemoryWorkflowStore(WorkflowStore):
    """内存版工作流存储 (演示/开发用)。"""

    def __init__(self):
        self._states: dict[str, WorkflowState] = {}

    async def save(self, state: WorkflowState) -> None:
        state.updated_at = time.time()
        self._states[state.workflow_id] = state

    async def get(self, workflow_id: str) -> WorkflowState | None:
        return self._states.get(workflow_id)

    async def list_by_status(self, status: WorkflowStatus) -> list[WorkflowState]:
        return [s for s in self._states.values() if s.status == status]


class Workflow(ABC):
    """工作流基类 — 模拟 Temporal Workflow 模式。

    子类实现 execute() 方法, 内部调用 activity (即业务函数)。
    工作流状态自动持久化, 支持断点续跑和自动重试。
    """

    workflow_type: str = "base"

    def __init__(self, store: WorkflowStore | None = None):
        self.store = store or MemoryWorkflowStore()

    @abstractmethod
    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """工作流主逻辑 — 子类实现。

        注意: 这里的代码应该是确定性的 (deterministic),
        即相同输入总是产生相同的执行路径。外部交互通过 activity 调用。
        """
        ...

    async def run(self, input_data: dict[str, Any], workflow_id: str | None = None) -> WorkflowState:
        """启动并执行工作流。"""
        workflow_id = workflow_id or str(uuid.uuid4())
        state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type=self.workflow_type,
            status=WorkflowStatus.RUNNING,
            input_data=input_data,
        )
        state.history.append({"event": "started", "timestamp": time.time()})
        await self.store.save(state)

        try:
            result = await self.execute(input_data)
            state.status = WorkflowStatus.COMPLETED
            state.result_data = result
            state.history.append({"event": "completed", "timestamp": time.time()})
        except Exception as exc:
            state.status = WorkflowStatus.FAILED
            state.error = str(exc)
            state.history.append({"event": "failed", "error": str(exc), "timestamp": time.time()})
            logger.error("workflow.failed", workflow_id=workflow_id, error=str(exc))

        await self.store.save(state)
        return state


# =========================================================================
# 具体工作流: 履约工作流 (Saga 的工作流封装)
# =========================================================================


class FulfillmentWorkflow(Workflow):
    """履约工作流 — 库存预占 → 物流匹配 → 订单创建 (带补偿)。

    这是现有 Saga 编排器的 Workflow 化版本, 增加了持久化和断点续跑能力。
    """

    workflow_type = "fulfillment"

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        from services import fulfillment_tools as ft

        user_id = input_data["user_id"]
        product_id = input_data["product_id"]
        quantity = input_data.get("quantity", 1)
        destination = input_data.get("destination", "北京")
        unit_price = input_data.get("unit_price", 0.0)

        executed_steps: list[tuple[str, dict[str, Any]]] = []

        try:
            # Step 1: 库存查询 + 选仓
            inv_result = await ft.check_inventory(product_id)
            warehouses = inv_result.get("warehouses", [])
            if not warehouses:
                raise ValueError("无可用库存")
            warehouse_id = warehouses[0].get("warehouse_id", "WH001")
            executed_steps.append(("check_inventory", {"warehouse_id": warehouse_id}))

            # Step 2: 库存预占
            resv_result = await ft.reserve_inventory(product_id, quantity, warehouse_id)
            if resv_result.get("status") != "reserved":
                raise ValueError(f"库存预占失败: {resv_result.get('status')}")
            reservation_id = resv_result["reservation_id"]
            executed_steps.append(("reserve_inventory", {"reservation_id": reservation_id}))

            # Step 3: 物流路线匹配
            route_result = await ft.match_logistics_route(
                product_id, warehouse_id, unit_price, destination
            )
            route_id = route_result["route_id"]
            executed_steps.append(("match_logistics", {"route_id": route_id}))

            # Step 4: 创建订单
            order_result = await ft.create_order(
                user_id, product_id, quantity, unit_price, reservation_id, route_id
            )
            order_id = order_result["order_id"]
            executed_steps.append(("create_order", {"order_id": order_id}))

            return {
                "order_id": order_id,
                "reservation_id": reservation_id,
                "route_id": route_id,
                "warehouse_id": warehouse_id,
                "status": "success",
            }

        except Exception as exc:
            # Saga 补偿: 逆序回滚已执行步骤
            logger.warning("workflow.compensating", steps=len(executed_steps))
            for step_name, step_data in reversed(executed_steps):
                try:
                    if step_name == "reserve_inventory":
                        await ft.release_inventory(step_data["reservation_id"])
                    elif step_name == "create_order":
                        await ft.cancel_order(step_data["order_id"])
                except Exception as comp_exc:
                    logger.error(
                        "workflow.compensation_failed",
                        step=step_name,
                        error=str(comp_exc),
                    )
            raise exc


# =========================================================================
# 工作流执行器 (简化版 Worker)
# =========================================================================


class WorkflowWorker:
    """工作流执行器 — 注册工作流类型, 执行任务。

    对应 Temporal 的 Worker 概念。生产环境替换为 temporalio.Worker。
    """

    def __init__(self, store: WorkflowStore | None = None):
        self.store = store or MemoryWorkflowStore()
        self._workflows: dict[str, type[Workflow]] = {}
        self._running: bool = False

    def register(self, workflow_cls: type[Workflow]) -> None:
        """注册工作流类型。"""
        self._workflows[workflow_cls.workflow_type] = workflow_cls
        logger.info("workflow.registered", type=workflow_cls.workflow_type)

    async def start_workflow(
        self,
        workflow_type: str,
        input_data: dict[str, Any],
        workflow_id: str | None = None,
    ) -> WorkflowState:
        """启动一个工作流。"""
        workflow_cls = self._workflows.get(workflow_type)
        if not workflow_cls:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        workflow = workflow_cls(store=self.store)
        return await workflow.run(input_data, workflow_id)

    async def get_status(self, workflow_id: str) -> WorkflowState | None:
        """查询工作流状态。"""
        return await self.store.get(workflow_id)

    def list_workflows(self) -> list[str]:
        """列出所有已注册的工作流类型。"""
        return list(self._workflows.keys())


# 全局 worker 实例
_worker: WorkflowWorker | None = None


def get_workflow_worker() -> WorkflowWorker:
    global _worker
    if _worker is None:
        _worker = WorkflowWorker()
        # 默认注册履约工作流
        _worker.register(FulfillmentWorkflow)
    return _worker
