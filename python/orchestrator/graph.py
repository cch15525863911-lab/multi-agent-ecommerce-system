"""
LangGraph 三阶段拓扑图 — 推荐主链路编排

使用 LangGraph 原生 fan-out / fan-in 边实现并行拓扑, 而非 asyncio.gather 包裹单节点。

拓扑结构 (DAG):
    START → init
        ┬→ user_profile     ┐  Stage 1: 画像‖召回 (并行 fan-out)
        └→ product_recall    ┘
        ┬→ rerank            ┐  Stage 2: 重排‖库存 (并行 fan-out, 依赖 Stage 1 全部完成)
        └→ inventory         ┘
    → filter → marketing_copy → aggregate → END  Stage 3: 文案 (串行)

    fan-out: 一个节点多条出边 → 目标节点并行执行
    fan-in:  多个节点指向同一节点 → 等待所有入边完成才执行 (join)

状态合并:
    并行分支各自返回 partial state dict, LangGraph 通过 channel reducer 合并。
    agent_results 字段使用 _merge_agent_results reducer 合并各分支的子字典;
    其余字段 (user_profile, raw_products, ranked_products, available_ids)
    各自只由单一节点写入, 无合并冲突。
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    MarketingCopyAgent,
    ProductRecAgent,
)
from config import get_settings
from models.schemas import Product, UserProfile
from services.ab_test import ABTestEngine
from services.inventory_service import InventoryService
from services.kg_store import KGStore
from services.profile_service import ProfileService


# =========================================================================
# State — TypedDict with custom reducer for parallel-merged fields
# =========================================================================


def _merge_agent_results(
    old: dict[str, Any] | None, new: dict[str, Any] | None
) -> dict[str, Any]:
    """Reducer: merge agent_results dicts from parallel branches.

    Called by LangGraph when multiple fan-out nodes return updates to the
    same ``agent_results`` channel.  Each branch writes a different sub-key
    (e.g. ``{"user_profile": result}``), so a shallow merge preserves all.
    """
    merged = dict(old or {})
    merged.update(new or {})
    return merged


class PipelineState(TypedDict, total=False):
    """State flowing through the recommendation pipeline.

    Fields annotated with ``Annotated[type, reducer]`` use a custom merge
    function when multiple parallel branches write to the same key.
    """

    request_id: str
    user_id: str
    scene: str
    num_items: int
    context: dict[str, Any]
    experiment_group: str

    user_profile: UserProfile | None
    raw_products: list[Product]
    ranked_products: list[Product]
    available_ids: set[str]
    final_products: list[Product]
    marketing_copies: list[dict[str, str]]

    agent_results: Annotated[dict[str, Any], _merge_agent_results]
    total_latency_ms: float
    _start_time: float


# =========================================================================
# Agent instances — created once at module import
# =========================================================================


def _build_profile_components() -> tuple[ProfileService, KGStore | None]:
    """Build the profile service and shared KG store."""
    kg_store = KGStore()
    return ProfileService(kg_store=kg_store), kg_store


profile_service, kg_store = _build_profile_components()
product_rec_agent = ProductRecAgent(kg_store=kg_store)
marketing_copy_agent = MarketingCopyAgent()
inventory_service = InventoryService()
ab_engine = ABTestEngine()


# =========================================================================
# Graph nodes — each returns a partial state dict (LangGraph convention)
# =========================================================================


async def init_node(state: PipelineState) -> dict[str, Any]:
    """Entry: assign request ID, record start time, assign A/B experiment."""
    exp = ab_engine.assign(state["user_id"])
    return {
        "request_id": str(uuid.uuid4()),
        "_start_time": time.perf_counter(),
        "agent_results": {},
        "experiment_group": exp.get("group", "control"),
    }


# ---- Stage 1: user_profile ‖ product_recall (parallel) ----


async def user_profile_node(state: PipelineState) -> dict[str, Any]:
    """Build user profile from behavior data (Neo4j KG + deterministic rules)."""
    result = await profile_service.run(
        user_id=state["user_id"],
        context=state.get("context", {}),
    )
    return {
        "user_profile": getattr(result, "profile", None),
        "agent_results": {"user_profile": result},
    }


async def product_recall_node(state: PipelineState) -> dict[str, Any]:
    """Multi-strategy recall: collaborative filtering + vector + popularity."""
    result = await product_rec_agent.run(
        user_profile=None,
        user_id=state.get("user_id", ""),
        num_items=state.get("num_items", 10) * 2,
    )
    return {
        "raw_products": getattr(result, "products", []),
        "agent_results": {"product_recall": result},
    }


# ---- Stage 2: rerank ‖ inventory (parallel, after Stage 1 join) ----


async def rerank_node(state: PipelineState) -> dict[str, Any]:
    """LLM re-rank: semantic-level sorting using user profile."""
    result = await product_rec_agent.run(
        user_profile=state.get("user_profile"),
        user_id=state.get("user_id", ""),
        num_items=state.get("num_items", 10),
    )
    return {
        "ranked_products": getattr(
            result, "products", state.get("raw_products", [])
        ),
        "agent_results": {"rerank": result},
    }


async def inventory_node(state: PipelineState) -> dict[str, Any]:
    """Real-time inventory check: filter out out-of-stock products."""
    result = await inventory_service.run(
        products=state.get("raw_products", []),
    )
    return {
        "available_ids": set(getattr(result, "available_products", [])),
        "agent_results": {"inventory": result},
    }


# ---- Stage 3: filter → marketing_copy → aggregate (serial) ----


async def filter_node(state: PipelineState) -> dict[str, Any]:
    """Intersect ranked products with available inventory."""
    ranked = state.get("ranked_products", [])
    avail = state.get("available_ids", set())
    num = state.get("num_items", 10)
    final = [p for p in ranked if p.product_id in avail]
    if not final:
        final = ranked
    return {"final_products": final[:num]}


async def marketing_copy_node(state: PipelineState) -> dict[str, Any]:
    """Generate personalized marketing copy for final product list."""
    graph_context = ""
    if kg_store is not None:
        graph_context = await product_rec_agent.graph_rag.build_user_context(
            state.get("user_id", ""),
            seed_product_ids=[
                p.product_id for p in state.get("final_products", [])
            ],
        )
    result = await marketing_copy_agent.run(
        user_profile=state.get("user_profile"),
        products=state.get("final_products", []),
        graph_context=graph_context,
    )
    return {
        "marketing_copies": getattr(result, "copies", []),
        "agent_results": {"marketing_copy": result},
    }


async def aggregate_node(state: PipelineState) -> dict[str, Any]:
    """Compute total latency and finalize."""
    return {
        "total_latency_ms": (time.perf_counter() - state.get("_start_time", 0))
        * 1000,
    }


# =========================================================================
# Graph builder — native fan-out / fan-in topology
# =========================================================================


def build_recommendation_graph() -> StateGraph:
    """Build and compile the LangGraph three-stage topology graph.

    Topology (fan-out = parallel, fan-in = join):

        START
          │
          ▼
        init ──────────┬──────────────┐
          ▼            ▼              │        Stage 1: 画像‖召回
        user_profile   product_recall │        (fan-out from init)
          │            │              │
          └────────────┴──────────────┘
                       │ (join: both must complete)
                       ▼
                  ┌────┴────┐
                  ▼          ▼                           Stage 2: 重排‖库存
                rerank    inventory                      (fan-out after Stage 1 join)
                  │          │
                  └──────────┘
                       │ (join: both must complete)
                       ▼
                    filter                                  Stage 3: 文案
                       │                                    (serial)
                       ▼
                marketing_copy
                       │
                       ▼
                   aggregate
                       │
                       ▼
                      END
    """
    graph = StateGraph(PipelineState)

    # -- register nodes --
    graph.add_node("init", init_node)
    graph.add_node("user_profile", user_profile_node)
    graph.add_node("product_recall", product_recall_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("inventory", inventory_node)
    graph.add_node("filter", filter_node)
    graph.add_node("marketing_copy", marketing_copy_node)
    graph.add_node("aggregate", aggregate_node)

    # -- entry --
    graph.set_entry_point("init")

    # -- Stage 1: fan-out from init to user_profile + product_recall --
    graph.add_edge("init", "user_profile")
    graph.add_edge("init", "product_recall")

    # -- Stage 1 → Stage 2: fan-in (join) then fan-out --
    # Both user_profile and product_recall point to both rerank and inventory.
    # LangGraph waits for ALL incoming edges before executing a node,
    # so rerank/inventory start only after Stage 1 fully completes.
    graph.add_edge("user_profile", "rerank")
    graph.add_edge("product_recall", "rerank")
    graph.add_edge("user_profile", "inventory")
    graph.add_edge("product_recall", "inventory")

    # -- Stage 2 → Stage 3: fan-in (join) at filter --
    graph.add_edge("rerank", "filter")
    graph.add_edge("inventory", "filter")

    # -- Stage 3: serial --
    graph.add_edge("filter", "marketing_copy")
    graph.add_edge("marketing_copy", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()
