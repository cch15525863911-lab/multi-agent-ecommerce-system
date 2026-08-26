"""Tests for GraphRAGService using an in-memory fake KGStore."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.graph_rag import GraphRAGService


class FakeKGStore:
    async def get_related_products(self, product_id: str, limit: int = 5) -> list[dict]:
        return [
            {
                "product_id": "P-APP",
                "name": "AirPods Pro 2",
                "relation": "bought_together",
                "weight": 0.95,
                "graph_score": 0.95,
            }
        ]

    async def get_multi_hop_candidates(self, user_id: str, max_hops: int = 2, limit: int = 5) -> list[dict]:
        return [
            {
                "product_id": "P-HUAMI",
                "name": "小米手环9 Pro",
                "hops": 2,
                "graph_score": 0.7,
            }
        ]

    async def get_similar_users(self, user_id: str, limit: int = 3) -> list[dict]:
        return [{"user_id": "u_similar", "jaccard": 0.6, "shared_products": 3}]

    async def get_degree_central_products(self, limit: int = 3) -> list[dict]:
        return [{"product_id": "P-IP16", "name": "iPhone 16 Pro", "graph_score": 12}]

    async def get_explanation_paths(self, user_id: str, target_product_id: str, max_hops: int = 2, limit: int = 3) -> list[dict]:
        return [
            {
                "seed_product_id": "P-IP16",
                "seed_name": "iPhone 16 Pro",
                "relations": ["bought_together"],
                "hops": 1,
            }
        ]


class FakeRelationKGStore(FakeKGStore):
    async def get_multi_hop_candidates(self, user_id: str, max_hops: int = 2, limit: int = 5) -> list[dict]:
        return []

    async def get_degree_central_products(self, limit: int = 3) -> list[dict]:
        return []


def test_build_user_context_contains_graph_sections():
    service = GraphRAGService(FakeKGStore())
    context = asyncio.run(
        service.build_user_context("u1", seed_product_ids=["P-IP16"])
    )

    assert "商品关系证据" in context
    assert "多跳候选" in context
    assert "相似用户" in context
    assert "图谱热点商品" in context
    assert "AirPods Pro 2" in context


def test_explain_target_formats_paths():
    service = GraphRAGService(FakeKGStore())
    explanation = asyncio.run(service.explain_target("u1", "P-APP"))

    assert "推荐路径" in explanation
    assert "iPhone 16 Pro" in explanation
    assert "bought_together" in explanation


def test_build_user_context_without_store_is_empty():
    service = GraphRAGService(kg_store=None)
    assert asyncio.run(service.build_user_context("u1")) == ""


def test_product_rec_graph_recall_converts_rows():
    from agents.product_rec_agent import ProductRecAgent
    from models.schemas import UserProfile

    agent = ProductRecAgent(kg_store=FakeRelationKGStore())
    profile = UserProfile(user_id="u1", recent_purchases=["P-IP16"])
    rows = asyncio.run(agent._graph_recall(profile, "u1", 10))
    product = agent._to_product(rows[0])

    assert product.product_id == "P-APP"
    assert product.score == 0.95


if __name__ == "__main__":
    test_build_user_context_contains_graph_sections()
    test_explain_target_formats_paths()
    test_build_user_context_without_store_is_empty()
    test_product_rec_graph_recall_converts_rows()
    print("All GraphRAG service tests passed!")
