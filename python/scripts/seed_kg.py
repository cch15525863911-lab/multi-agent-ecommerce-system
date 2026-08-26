"""Seed demo users, products, and behaviors into Neo4j for the KG profile path.

Usage (from the python/ directory):
    python scripts/seed_kg.py

Requires a reachable Neo4j instance; set ECOM_NEO4J_URI / credentials first.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.kg_store import DAY, KGStore


PRODUCTS = [
    ("P-IP16", "iPhone 16 Pro", 7999.0, "手机", "Apple", "S-APPLE"),
    ("P-APP", "AirPods Pro 2", 1899.0, "耳机", "Apple", "S-APPLE"),
    ("P-XIA14", "小米14", 3999.0, "手机", "Xiaomi", "S-MI"),
    ("P-CHA20", "20W 快充充电器", 49.0, "配件", "Anker", "S-ANKER"),
    ("P-IPAD", "iPad Air", 4799.0, "平板", "Apple", "S-APPLE"),
    ("P-HUAMI", "小米手环9 Pro", 449.0, "数码配件", "Xiaomi", "S-MI"),
]

RELATIONS = [
    ("P-IP16", "P-APP", "bought_together", 0.95),
    ("P-IP16", "P-IPAD", "complementary", 0.90),
    ("P-XIA14", "P-HUAMI", "complementary", 0.80),
    ("P-CHA20", "P-HUAMI", "bought_together", 0.85),
]

USERS_BEHAVIOR = {
    "u_vip": [
        ("view", "P-IP16", 1, 0),
        ("purchase", "P-IP16", 1, 7999),
        ("purchase", "P-IPAD", 2, 4799),
        ("purchase", "P-APP", 3, 1899),
    ],
    "u_new": [
        ("view", "P-CHA20", 0, 0),
        ("view", "P-HUAMI", 0, 0),
        ("view", "P-XIA14", 1, 0),
    ],
}


async def main() -> int:
    kg = KGStore()
    if not await kg.connect():
        print("Neo4j is not reachable; seed skipped.")
        print("Start it with: docker compose up -d neo4j")
        return 1

    now = time.time()
    try:
        for product in PRODUCTS:
            await kg.upsert_product(*product)
        for product_a, product_b, relation, weight in RELATIONS:
            await kg.upsert_product_relation(
                product_a, product_b, relation, weight=weight
            )
        for user_id, events in USERS_BEHAVIOR.items():
            await kg.upsert_user(user_id, city="Shanghai" if user_id == "u_vip" else None)
            for action, product_id, days_ago, amount in events:
                await kg.record_behavior(
                    user_id,
                    product_id,
                    action,
                    at=now - days_ago * DAY,
                    amount=amount,
                )
    finally:
        await kg.close()

    print(
        f"Seeded {len(PRODUCTS)} products, {len(RELATIONS)} relations, "
        f"and {len(USERS_BEHAVIOR)} users."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
