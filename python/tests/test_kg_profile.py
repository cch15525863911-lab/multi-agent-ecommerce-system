"""
Smoke test + demo for the Knowledge-Graph-backed User Profile pipeline.

Run from the `python/` directory:
    python -m tests.test_kg_profile

The test works both WITH and WITHOUT a running Neo4j instance:
  * If Neo4j (bolt://localhost:7687, neo4j/ecommerce123) is reachable, it
    writes real nodes/edges and queries back via Cypher.
  * Otherwise it transparently falls back to the in-memory mock mode of
    KGStore, so you can validate the rule engine & agent contract on a
    laptop without Docker.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# make `python -m tests.test_kg_profile` work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import ProfileService
from models.schemas import UserSegment
from services import KGStore

NOW = time.time()
HOUR = 3600
DAY = 86400
WEEK = 7 * DAY


# ------------------------------------------------------------------
# Scenario seed data — 4 archetypal users so you can visually verify
# each segment rule fires correctly.
# ------------------------------------------------------------------
PRODUCTS = [
    ("P-IP16", "iPhone 16 Pro",       7999.0, "手机",   "Apple",   "S-APPLE"),
    ("P-APP",  "AirPods Pro 2",       1899.0, "耳机",   "Apple",   "S-APPLE"),
    ("P-XIA14","小米14",              3999.0, "手机",   "Xiaomi",  "S-MI"),
    ("P-BT40", "红米 Buds 4 活力版",    199.0,  "耳机",   "Xiaomi",  "S-MI"),
    ("P-CHA20","20W 快充充电器",        49.0,   "配件",   "Anker",   "S-ANKER"),
    ("P-IPAD", "iPad Air",            4799.0, "平板",   "Apple",   "S-APPLE"),
    ("P-SOCKS","5双装棉袜",             59.0,   "服饰",   "UNIQLO",  "S-UNIQ"),
    ("P-TEA",  "云南普洱熟茶饼",        388.0,  "食品",   "大益",     "S-TEA"),
    ("P-SHIRT","纯棉圆领T恤",          129.0,  "服饰",   "UNIQLO",  "S-UNIQ"),
    ("P-HUAMI","小米手环9 Pro",        449.0,  "数码配件","Xiaomi",  "S-MI"),
]

RELATIONS = [
    ("P-IP16", "P-APP", "bought_together", 0.95),
    ("P-IP16", "P-IPAD", "complementary", 0.90),
    ("P-XIA14", "P-HUAMI", "complementary", 0.80),
    ("P-CHA20", "P-HUAMI", "bought_together", 0.85),
]

# user_id -> list of (action, product_id, days_ago, amount_or_0)
USERS_BEHAVIOR = {
    # --- HIGH_VALUE (monetary > 0.7, recency > 0.5) ---
    "u_vip": [
        ("view",     "P-IP16", 1, 0),
        ("purchase", "P-IP16", 1, 7999),
        ("view",     "P-IPAD", 2, 0),
        ("purchase", "P-IPAD", 2, 4799),
        ("view",     "P-APP",  3, 0),
        ("purchase", "P-APP",  3, 1899),
    ],
    # --- PRICE_SENSITIVE (freq > 0.6, monetary < 0.45) — buys cheap, buys often ---
    "u_deal": [
        ("purchase", "P-SOCKS",  1, 59),
        ("purchase", "P-TEA",    2, 388),
        ("purchase", "P-SHIRT",  3, 129),
        ("purchase", "P-CHA20",  4, 49),
        ("purchase", "P-HUAMI",  5, 449),
        ("view",     "P-BT40",   1, 0),
        ("view",     "P-BT40",   2, 0),
    ],
    # --- CHURN_RISK (recency < 0.3 — last activity was 40 days ago) ---
    "u_gone": [
        ("view",     "P-XIA14", 40, 0),
        ("purchase", "P-XIA14", 40, 3999),
        ("view",     "P-BT40",  41, 0),
    ],
    # --- NEW_USER (history_buys < 2, freq < 0.35) + ACTIVE fallback via context ---
    "u_new": [
        ("view", "P-BT40",   0, 0),
        ("view", "P-CHA20",  0, 0),
        ("view", "P-XIA14",  1, 0),
    ],
}


async def seed_graph(kg: KGStore) -> None:
    """Upsert products then append behaviour edges."""
    for pid, name, price, cat, brand, seller in PRODUCTS:
        await kg.upsert_product(pid, name, price, category=cat, brand=brand, seller_id=seller)
    for product_a, product_b, relation, weight in RELATIONS:
        await kg.upsert_product_relation(
            product_a, product_b, relation, weight=weight
        )

    for uid, events in USERS_BEHAVIOR.items():
        await kg.upsert_user(uid, city="Shanghai" if uid == "u_vip" else None)
        for action, pid, days_ago, amount in events:
            at = NOW - days_ago * DAY
            if action == "view":
                await kg.record_view(uid, pid, at=at, duration_sec=45)
            elif action == "purchase":
                await kg.record_purchase(uid, pid, amount=amount or 100, at=at, order_id=f"ORD-{uid}-{pid}-{days_ago}")
            elif action == "favorite":
                await kg.record_favorite(uid, pid, at=at)


EXPECTED_SEGMENTS = {
    "u_vip":  UserSegment.HIGH_VALUE,
    "u_deal": UserSegment.PRICE_SENSITIVE,
    "u_gone": UserSegment.CHURN_RISK,
    "u_new":  UserSegment.NEW_USER,
}


async def main() -> int:
    kg = KGStore()
    await kg.connect()
    mode = "NEO4J" if kg.connected else "MOCK (in-memory)"
    print(f"=== KGStore running in {mode} mode ===\n")

    if not kg.connected:
        # Mock mode: inject pre-computed features that mirror what the graph
        # would produce, so the demo still exercises the rule engine & agent.
        for uid, events in USERS_BEHAVIOR.items():
            views = [e for e in events if e[0] == "view"]
            purchases = [e for e in events if e[0] == "purchase"]
            recent = {
                "u_vip":  ["Apple", "Xiaomi"],
                "u_deal": ["服饰", "食品", "配件", "耳机", "数码配件"],
                "u_gone": ["手机", "耳机"],
                "u_new":  ["耳机", "配件", "手机"],
            }[uid]
            rfm = _mock_rfm(purchases)
            pr = _mock_price_range(purchases + views)
            act = _mock_active_hours(events)
            kg.seed_mock_profile(uid, {
                "user_id": uid,
                "view_count_1h": len([e for e in views if e[2] == 0]),
                "view_count_24h": len([e for e in views if e[2] <= 1]),
                "purchase_count_7d": len([e for e in purchases if e[2] <= 7]),
                "recent_views": [e[1] for e in views[-20:]],
                "recent_purchases": [e[1] for e in purchases[-10:]],
                "preferred_categories": recent,
                "rfm": rfm,
                "price_range": pr,
                "active_hours": act,
            })
    else:
        await seed_graph(kg)

    agent = ProfileService(kg_store=kg)
    contexts = {
        "u_new": {
            "recent_views": ["耳机", "手机", "数码配件"],
            "avg_order_amount": 300,
        },
    }

    passed = 0
    failed = 0
    for uid in USERS_BEHAVIOR:
        ctx = contexts.get(uid, {})
        result = await agent.run(user_id=uid, context=ctx)
        profile = result.profile
        reasons = result.data.get("segment_reasons", []) if result.data else []

        expected = EXPECTED_SEGMENTS[uid]
        ok = expected in profile.segments if profile else False
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {uid}")
        print(f"    segments = {[s.value for s in (profile.segments if profile else [])]}")
        print(f"    expected contains: {expected.value}")
        if profile:
            print(f"    rfm      = {profile.rfm_score}")
            print(f"    price    = ¥{profile.price_range[0]:.0f} – ¥{profile.price_range[1]:.0f}")
            print(f"    categs   = {profile.preferred_categories[:5]}")
            print(f"    tags     = {profile.real_time_tags}")
        for r in reasons:
            print(f"      · {r}")
        print()

    print(f"Summary: {passed} passed, {failed} failed")
    await kg.close()
    return 0 if failed == 0 else 1


# ------------------------------------------------------------------
# Mock helpers (mirror KGStore._build_features / _compute_rfm)
# ------------------------------------------------------------------

def _mock_rfm(purchases: list) -> dict:
    if not purchases:
        return {"recency": 0.1, "frequency": 0.1, "monetary": 0.1}
    now = NOW
    latest = min(e[2] for e in purchases)  # days_ago
    days_since = latest
    recency = max(0.0, 1.0 - days_since / 30.0)
    frequency = min(1.0, len(purchases) / 10.0)
    amounts = [e[3] for e in purchases if e[3]] or [100]
    avg = sum(amounts) / len(amounts)
    monetary = min(1.0, avg / 1000.0)
    return {
        "recency": round(recency, 3),
        "frequency": round(frequency, 3),
        "monetary": round(monetary, 3),
    }


def _mock_price_range(events: list) -> tuple:
    prices = []
    for action, pid, _d, amount in events:
        pmap = {p[0]: p[2] for p in PRODUCTS}
        if pid in pmap:
            prices.append(pmap[pid])
        elif amount:
            prices.append(amount)
    s = sorted(prices)
    if not s:
        return (0.0, 10000.0)
    lo = s[max(0, int(0.1 * (len(s) - 1)))]
    hi = s[min(len(s) - 1, int(0.9 * (len(s) - 1)))]
    return (float(lo), float(hi))


def _mock_active_hours(events: list) -> list:
    return [20, 21, 22]


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
