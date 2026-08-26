"""
LangGraph 拓扑图测试 — 验证三阶段 fan-out / fan-in 结构与状态合并。

测试覆盖:
    1. 拓扑结构: 节点数量、fan-out 边、fan-in (join) 边
    2. Reducer: _merge_agent_results 正确合并并行分支的 agent_results
    3. 端到端: mock agents 后图能正常执行, 状态正确流转
    4. 并行合并: agent_results 包含所有分支的结果

Run from the `python/` directory:
    python -m tests.test_graph_topology
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.schemas import (
    Product,
    UserProfile,
)
from models.schemas import (
    ProductRecResult as _ProductRecResult,
)
from models.schemas import (
    UserProfileResult as _UserProfileResult,
)
from models.schemas import (
    InventoryResult as _InventoryResult,
)
from models.schemas import (
    MarketingCopyResult as _MarketingCopyResult,
)
from orchestrator import graph as graph_mod
from orchestrator.graph import (
    PipelineState,
    _merge_agent_results,
    build_recommendation_graph,
)


# =========================================================================
# Helpers — mock agent results using correct schema subclasses
# =========================================================================


def _mock_user_profile() -> UserProfile:
    return UserProfile(
        user_id="u001",
        segments=["active"],
        rfm_score={"recency": 0.8, "frequency": 0.6, "monetary": 0.9},
    )


def _mock_products() -> list[Product]:
    return [
        Product(product_id="P001", name="Phone", category="手机", price=7999.0, brand="Apple"),
        Product(product_id="P002", name="Tablet", category="平板", price=3999.0, brand="Apple"),
    ]


def _profile_result() -> _UserProfileResult:
    return _UserProfileResult(
        success=True,
        confidence=0.9,
        profile=_mock_user_profile(),
    )


def _recall_result() -> _ProductRecResult:
    return _ProductRecResult(
        success=True,
        confidence=0.85,
        products=_mock_products(),
        recall_strategy="hybrid",
    )


def _rerank_result() -> _ProductRecResult:
    return _ProductRecResult(
        success=True,
        confidence=0.92,
        products=list(reversed(_mock_products())),
        recall_strategy="llm_rerank",
    )


def _inventory_result() -> _InventoryResult:
    return _InventoryResult(
        success=True,
        confidence=1.0,
        available_products=["P001", "P002"],
    )


def _copy_result() -> _MarketingCopyResult:
    return _MarketingCopyResult(
        success=True,
        confidence=0.88,
        copies=[
            {"product_id": "P001", "copy": "限时特惠!"},
            {"product_id": "P002", "copy": "平板上新!"},
        ],
    )


# =========================================================================
# Test 1: Topology — correct nodes and fan-out / fan-in edges
# =========================================================================


def test_topology_structure() -> None:
    """Graph has 8 nodes + START/END, with correct fan-out and fan-in edges."""
    compiled = build_recommendation_graph()
    g = compiled.get_graph()
    node_ids = set(g.nodes.keys())

    expected_nodes = {
        "__start__", "__end__",
        "init", "user_profile", "product_recall",
        "rerank", "inventory", "filter",
        "marketing_copy", "aggregate",
    }
    assert node_ids == expected_nodes, (
        f"node mismatch: got {node_ids - expected_nodes} extra, "
        f"{expected_nodes - node_ids} missing"
    )

    edges = {(e.source, e.target) for e in g.edges}

    # Stage 1 fan-out: init → user_profile + product_recall
    assert ("init", "user_profile") in edges, "missing init→user_profile"
    assert ("init", "product_recall") in edges, "missing init→product_recall"

    # Stage 1→2 join: both feed into rerank and inventory
    assert ("user_profile", "rerank") in edges
    assert ("product_recall", "rerank") in edges
    assert ("user_profile", "inventory") in edges
    assert ("product_recall", "inventory") in edges

    # Stage 2→3 join: both feed into filter
    assert ("rerank", "filter") in edges
    assert ("inventory", "filter") in edges

    # Stage 3 serial
    assert ("filter", "marketing_copy") in edges
    assert ("marketing_copy", "aggregate") in edges
    assert ("aggregate", "__end__") in edges
    assert ("__start__", "init") in edges

    print("[OK] topology: 8 nodes, fan-out/fan-in edges verified")


def test_fan_out_count() -> None:
    """init has exactly 2 outgoing edges (fan-out), filter has 2 incoming (join)."""
    compiled = build_recommendation_graph()
    edges = [(e.source, e.target) for e in compiled.get_graph().edges]

    init_outgoing = [e for e in edges if e[0] == "init"]
    assert len(init_outgoing) == 2, (
        f"init should fan-out to 2 nodes, got {len(init_outgoing)}"
    )

    filter_incoming = [e for e in edges if e[1] == "filter"]
    assert len(filter_incoming) == 2, (
        f"filter should have 2 incoming edges (join), got {len(filter_incoming)}"
    )

    rerank_incoming = [e for e in edges if e[1] == "rerank"]
    assert len(rerank_incoming) == 2, (
        f"rerank should have 2 incoming edges (join), got {len(rerank_incoming)}"
    )

    print("[OK] fan-out: init→2 nodes, fan-in: rerank/filter←2 nodes each")


# =========================================================================
# Test 2: Reducer — _merge_agent_results
# =========================================================================


def test_reducer_basic() -> None:
    """Reducer merges two dicts with different keys."""
    result = _merge_agent_results(
        {"user_profile": "r1"},
        {"product_recall": "r2"},
    )
    assert result == {"user_profile": "r1", "product_recall": "r2"}
    print("[OK] reducer: merges different keys")


def test_reorder_independent() -> None:
    """Reducer is order-independent."""
    r1 = _merge_agent_results({"a": 1}, {"b": 2})
    r2 = _merge_agent_results({"b": 2}, {"a": 1})
    assert r1 == r2 == {"a": 1, "b": 2}
    print("[OK] reducer: order-independent merge")


def test_reducer_none() -> None:
    """Reducer handles None inputs gracefully."""
    assert _merge_agent_results(None, {"a": 1}) == {"a": 1}
    assert _merge_agent_results({"a": 1}, None) == {"a": 1}
    assert _merge_agent_results(None, None) == {}
    print("[OK] reducer: handles None inputs")


def test_reducer_overwrite_same_key() -> None:
    """When same key exists in both, new value wins (last write)."""
    result = _merge_agent_results({"a": 1}, {"a": 2})
    assert result == {"a": 2}
    print("[OK] reducer: same key → new value wins")


# =========================================================================
# Test 3: End-to-end with mocked agents
# =========================================================================


async def test_end_to_end_mocked() -> None:
    """Graph executes end-to-end with mocked agents; state flows correctly."""
    compiled = build_recommendation_graph()

    profile_r = _profile_result()
    recall_r = _recall_result()
    rerank_r = _rerank_result()
    inventory_r = _inventory_result()
    copy_r = _copy_result()

    with (
        patch.object(graph_mod.profile_service, "run", new=AsyncMock(return_value=profile_r)),
        patch.object(graph_mod.product_rec_agent, "run", new=AsyncMock(side_effect=[
            recall_r,   # first call: recall (user_profile=None)
            rerank_r,   # second call: rerank (user_profile=profile)
        ])),
        patch.object(graph_mod.inventory_service, "run", new=AsyncMock(return_value=inventory_r)),
        patch.object(graph_mod.marketing_copy_agent, "run", new=AsyncMock(return_value=copy_r)),
        patch.object(graph_mod, "kg_store", None),
    ):
        state: PipelineState = {
            "user_id": "u001",
            "scene": "homepage",
            "num_items": 2,
            "context": {},
        }
        result = await compiled.ainvoke(state)

    # Verify state flow
    assert result.get("request_id"), "request_id should be set"
    assert result.get("user_profile") is not None, "user_profile should be set"
    assert len(result.get("raw_products", [])) == 2, "raw_products should have 2 items"
    assert len(result.get("ranked_products", [])) == 2, "ranked_products should have 2 items"
    assert result.get("available_ids") == {"P001", "P002"}, "available_ids should match"
    assert len(result.get("final_products", [])) == 2, "final_products should have 2 items"
    assert len(result.get("marketing_copies", [])) == 2, "marketing_copies should have 2 items"
    assert result.get("total_latency_ms", 0) > 0, "total_latency_ms should be positive"

    # Verify agent_results merge — all 5 branches present
    agent_results = result.get("agent_results", {})
    expected_keys = {"user_profile", "product_recall", "rerank", "inventory", "marketing_copy"}
    assert set(agent_results.keys()) == expected_keys, (
        f"agent_results should have all 5 keys, got: {set(agent_results.keys())}"
    )

    print(
        f"[OK] end-to-end: products={len(result['final_products'])}, "
        f"copies={len(result['marketing_copies'])}, "
        f"agent_results_keys={list(agent_results.keys())}"
    )


# =========================================================================
# Test 4: Parallel merge — agent_results from fan-out branches
# =========================================================================


async def test_parallel_agent_results_merge() -> None:
    """After Stage 1 fan-out, agent_results contains BOTH user_profile and product_recall."""
    compiled = build_recommendation_graph()

    call_log: list[str] = []

    profile_r = _profile_result()
    recall_r = _recall_result()
    rerank_r = _rerank_result()
    inventory_r = _inventory_result()
    copy_r = _copy_result()

    async def mock_profile_run(**kwargs):
        call_log.append("user_profile")
        return profile_r

    async def mock_inventory_run(**kwargs):
        call_log.append("inventory")
        return inventory_r

    async def mock_copy_run(**kwargs):
        call_log.append("marketing_copy")
        return copy_r

    original_run = graph_mod.product_rec_agent.run

    async def side_effect_run(**kwargs):
        if kwargs.get("user_profile") is None:
            call_log.append("product_recall")
            return recall_r
        call_log.append("rerank")
        return rerank_r

    with (
        patch.object(graph_mod.profile_service, "run", new=mock_profile_run),
        patch.object(graph_mod.inventory_service, "run", new=mock_inventory_run),
        patch.object(graph_mod.marketing_copy_agent, "run", new=mock_copy_run),
        patch.object(graph_mod, "kg_store", None),
    ):
        graph_mod.product_rec_agent.run = side_effect_run

        try:
            state: PipelineState = {
                "user_id": "u001",
                "scene": "homepage",
                "num_items": 2,
                "context": {},
            }
            result = await compiled.ainvoke(state)
        finally:
            graph_mod.product_rec_agent.run = original_run

    agent_results = result.get("agent_results", {})

    # Both Stage 1 branches should be in agent_results
    assert "user_profile" in agent_results, "user_profile result missing from merge"
    assert "product_recall" in agent_results, "product_recall result missing from merge"
    # Both Stage 2 branches should be in agent_results
    assert "rerank" in agent_results, "rerank result missing from merge"
    assert "inventory" in agent_results, "inventory result missing from merge"
    # Stage 3
    assert "marketing_copy" in agent_results, "marketing_copy result missing from merge"

    # Verify execution order: Stage 1 before Stage 2
    stage1_idx = max(
        call_log.index("user_profile"), call_log.index("product_recall")
    )
    stage2_start = min(
        call_log.index("rerank"), call_log.index("inventory")
    )
    assert stage1_idx < stage2_start, (
        f"Stage 1 should complete before Stage 2 starts: "
        f"stage1_last={stage1_idx}, stage2_first={stage2_start}, log={call_log}"
    )

    print(
        f"[OK] parallel merge: agent_results has all 5 keys, "
        f"execution order: {call_log}"
    )


# =========================================================================
# Main
# =========================================================================


async def main() -> int:
    print("=" * 60)
    print("LangGraph 拓扑图测试 — 三阶段 fan-out / fan-in")
    print("=" * 60)

    test_topology_structure()
    test_fan_out_count()
    test_reducer_basic()
    test_reorder_independent()
    test_reducer_none()
    test_reducer_overwrite_same_key()
    await test_end_to_end_mocked()
    await test_parallel_agent_results_merge()

    print("=" * 60)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
