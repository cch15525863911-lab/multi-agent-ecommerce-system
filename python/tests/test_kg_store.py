"""Unit tests for KGStore that do not require a live Neo4j instance."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.kg_store import KGStore


def test_build_features_filters_empty_optional_matches():
    kg = KGStore()
    features = kg._build_features(
        "u1",
        [{"at": None, "product_id": None, "category": None}],
        [{"at": None, "product_id": None, "category": None}],
        [{"at": None, "product_id": None, "category": None}],
    )

    assert features["recent_views"] == []
    assert features["recent_purchases"] == []
    assert features["recent_favorites"] == []
    assert features["rfm"] == {
        "recency": 0.0,
        "frequency": 0.0,
        "monetary": 0.0,
    }


def test_build_features_weights_favorites():
    kg = KGStore()
    features = kg._build_features(
        "u1",
        [{"at": 1, "product_id": "P1", "category": "手机", "price": 100}],
        [],
        [{"at": 2, "product_id": "P2", "category": "手机", "price": 50}],
    )

    assert features["preferred_categories"] == ["手机"]


def test_record_behavior_rejects_unknown_action():
    kg = KGStore()

    async def run():
        try:
            await kg.record_behavior("u1", "P1", "click")
        except ValueError:
            return True
        return False

    assert asyncio.run(run())


def test_ensure_schema_without_driver_is_noop():
    kg = KGStore()
    asyncio.run(kg.ensure_schema())


def test_upsert_product_relation_rejects_unknown_relation():
    kg = KGStore()

    async def run():
        try:
            await kg.upsert_product_relation("P1", "P2", "unknown")
        except ValueError:
            return True
        return False

    assert asyncio.run(run())


def test_build_co_purchase_relations_without_connection_returns_zero():
    kg = KGStore()
    assert asyncio.run(kg.build_co_purchase_relations(min_support=2)) == 0


if __name__ == "__main__":
    test_build_features_filters_empty_optional_matches()
    test_build_features_weights_favorites()
    test_record_behavior_rejects_unknown_action()
    test_ensure_schema_without_driver_is_noop()
    test_upsert_product_relation_rejects_unknown_relation()
    test_build_co_purchase_relations_without_connection_returns_zero()
    print("All KGStore unit tests passed!")
