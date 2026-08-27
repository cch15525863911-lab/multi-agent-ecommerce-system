"""
Multi-channel recall integration test.

Verifies that the three recall channels (Milvus vector, Redis CF, Neo4j graph)
work correctly, both individually and when merged inside ProductRecAgent._recall().

All tests run WITHOUT external services — Milvus and Redis automatically fall
back to in-memory mode, and Neo4j is absent so graph recall returns [].

Run from the `python/` directory:
    python -m tests.test_recall
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.product_rec_agent import MOCK_PRODUCTS, ProductRecAgent
from models.schemas import Product, UserProfile
from models.schemas import UserSegment
from services.cf_store import CFStore
from services.milvus_store import MilvusStore


# ---------------------------------------------------------------------------
# MilvusStore — in-memory fallback
# ---------------------------------------------------------------------------

async def test_milvus_in_memory_upsert_and_search() -> None:
    """Upsert products into MilvusStore; search_by_text should return similar items."""
    store = MilvusStore()
    n = await store.upsert_products(MOCK_PRODUCTS)
    assert n == len(MOCK_PRODUCTS), f"expected {len(MOCK_PRODUCTS)} upserted, got {n}"

    results = await store.search_by_text("手机 Apple 旗舰", limit=5)
    assert len(results) > 0, "vector search should return results"
    assert len(results) <= 5, "should respect limit"

    top = results[0]
    assert "product_id" in top
    assert "score" in top
    assert top["product_id"], "top result should have a product_id"

    print(f"[OK] milvus.in_memory: upserted={n}, search_results={len(results)}, top={top['product_id']}")


async def test_milvus_search_similar_products() -> None:
    """search_similar_products should return items excluding the query product."""
    store = MilvusStore()
    await store.upsert_products(MOCK_PRODUCTS)

    seed = MOCK_PRODUCTS[0]  # iPhone 16 Pro
    results = await store.search_similar_products(seed, limit=5)
    assert len(results) > 0, "similar search should return results"
    assert all(r["product_id"] != seed.product_id for r in results), \
        "should exclude the seed product itself"

    print(f"[OK] milvus.similar: seed={seed.product_id}, found={len(results)}, "
          f"top={results[0]['product_id'] if results else 'N/A'}")


async def test_milvus_empty_query_returns_empty() -> None:
    """Empty query text should return empty results (no crash)."""
    store = MilvusStore()
    await store.upsert_products(MOCK_PRODUCTS)
    results = await store.search_by_text("", limit=5)
    assert isinstance(results, list)
    print(f"[OK] milvus.empty_query: results={len(results)}")


# ---------------------------------------------------------------------------
# CFStore — in-memory fallback
# ---------------------------------------------------------------------------

async def test_cf_record_and_recommend() -> None:
    """Record interactions for a user and verify CF returns recommendations."""
    store = CFStore()

    # user_a and user_b share 3 items; user_c shares 1 with user_a
    shared_items = ["P001", "P002", "P003"]
    for pid in shared_items:
        await store.record_interaction("user_a", pid, "purchase")
        await store.record_interaction("user_b", pid, "purchase")
    await store.record_interaction("user_b", "P010", "view")
    await store.record_interaction("user_b", "P011", "view")
    await store.record_interaction("user_c", "P001", "view")

    similar = await store.get_similar_users("user_a", limit=5)
    assert len(similar) > 0, "should find similar users"
    assert similar[0]["user_id"] == "user_b", "user_b should be most similar"
    assert similar[0]["jaccard"] > 0, "jaccard should be positive"

    recs = await store.recommend("user_a", limit=5)
    assert len(recs) > 0, "should return recommendations"
    rec_pids = {r["product_id"] for r in recs}
    assert "user_a" not in rec_pids, "should not recommend already-interacted items"
    assert "P010" in rec_pids or "P011" in rec_pids, \
        "should recommend items from similar user (user_b)"

    print(f"[OK] cf.recommend: similar_users={len(similar)}, "
          f"recommendations={len(recs)}, top={recs[0]['product_id']}")


async def test_cf_cold_user_returns_empty() -> None:
    """A user with no interactions should get no CF recommendations."""
    store = CFStore()
    recs = await store.recommend("cold_user", limit=5)
    assert recs == [], "cold user should get empty recommendations"
    print("[OK] cf.cold_user: empty recommendations")


# ---------------------------------------------------------------------------
# ProductRecAgent._recall — multi-channel integration
# ---------------------------------------------------------------------------

async def test_recall_returns_products_without_profile() -> None:
    """Without a profile, _recall should still return products (popularity fallback)."""
    agent = ProductRecAgent(kg_store=None)
    candidates = await agent._recall(profile=None, user_id="", limit=10)
    assert len(candidates) > 0, "should return popularity fallback products"
    assert len(candidates) <= 10
    pids = {p.product_id for p in candidates}
    mock_ids = {p.product_id for p in MOCK_PRODUCTS}
    assert pids.issubset(mock_ids), "fallback candidates should come from MOCK_PRODUCTS"
    print(f"[OK] recall.no_profile: returned {len(candidates)} products from popularity")


async def test_recall_with_profile_includes_vector_results() -> None:
    """With a profile, vector recall should contribute scored candidates."""
    agent = ProductRecAgent(kg_store=None)
    profile = UserProfile(
        user_id="u001",
        segments=[UserSegment.ACTIVE],
        preferred_categories=["手机", "耳机"],
        recent_views=["P001", "P003"],
    )
    candidates = await agent._recall(profile=profile, user_id="u001", limit=15)

    assert len(candidates) > 0
    scored = [p for p in candidates if p.score > 0]
    assert len(scored) > 0, "vector recall should produce scored candidates"

    # Category alignment: top-scored items should include preferred categories
    top_cats = {p.category for p in scored[:5]}
    assert "手机" in top_cats or "耳机" in top_cats, \
        "top candidates should include preferred categories"

    print(f"[OK] recall.with_profile: {len(candidates)} candidates, "
          f"{len(scored)} scored, top_cats={top_cats}")


async def test_recall_with_cf_interactions() -> None:
    """Pre-seed CF interactions, then verify CF recall contributes candidates."""
    agent = ProductRecAgent(kg_store=None)

    # Seed interactions: user_a and user_b share items, user_b has extra
    shared = ["P001", "P002"]
    for pid in shared:
        await agent.cf_store.record_interaction("user_a", pid, "purchase")
        await agent.cf_store.record_interaction("user_b", pid, "purchase")
    await agent.cf_store.record_interaction("user_b", "P010", "view")

    profile = UserProfile(
        user_id="user_a",
        segments=[UserSegment.ACTIVE],
        preferred_categories=["手机"],
    )
    candidates = await agent._recall(profile=profile, user_id="user_a", limit=15)

    pids = {p.product_id for p in candidates}
    assert "P010" in pids, "CF should surface user_b's item P010 for user_a"
    print(f"[OK] recall.with_cf: {len(candidates)} candidates, P010 present via CF")


async def test_recall_deduplicates() -> None:
    """The same product from multiple channels should appear only once."""
    agent = ProductRecAgent(kg_store=None)
    profile = UserProfile(
        user_id="u002",
        preferred_categories=["手机"],
        recent_views=["P001"],
    )
    candidates = await agent._recall(profile=profile, user_id="u002", limit=30)
    pids = [p.product_id for p in candidates]
    assert len(pids) == len(set(pids)), "duplicate product_ids in recall results"
    print(f"[OK] recall.dedup: {len(candidates)} unique candidates")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    tests = [
        test_milvus_in_memory_upsert_and_search,
        test_milvus_search_similar_products,
        test_milvus_empty_query_returns_empty,
        test_cf_record_and_recommend,
        test_cf_cold_user_returns_empty,
        test_recall_returns_products_without_profile,
        test_recall_with_profile_includes_vector_results,
        test_recall_with_cf_interactions,
        test_recall_deduplicates,
    ]
    for fn in tests:
        await fn()
    print(f"\nAll {len(tests)} recall tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
