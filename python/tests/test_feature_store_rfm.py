"""
FeatureStore RFM amount 字段修复 — 单元测试。

测试覆盖:
    1. record_behavior 存储 amount 字段到 payload
    2. _compute_rfm 正确读取 amount 计算 monetary
    3. record_behavior 无 Redis 时输出 warning 日志
    4. _compute_rfm 无购买记录时返回全零
    5. record_behavior 的 amount 默认值为 0.0
    6. RFM recency 和 frequency 正确计算

Run from the `python/` directory:
    python -m tests.test_feature_store_rfm
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.feature_store import FeatureStore


class FakeRedis:
    """Minimal async Redis mock for FeatureStore testing."""

    def __init__(self):
        self._data: dict[str, dict[str, float]] = {}
        self._kv: dict[str, str] = {}

    async def zadd(self, key: str, mapping: dict[str, float]):
        if key not in self._data:
            self._data[key] = {}
        self._data[key].update(mapping)

    async def expire(self, key: str, ttl: int):
        pass

    async def zrangebyscore(self, key: str, lo: float, hi) -> list[str]:
        if key not in self._data:
            return []
        hi_val = float(hi) if isinstance(hi, str) else hi
        items = [
            (member, score)
            for member, score in self._data[key].items()
            if lo <= score <= hi_val
        ]
        items.sort(key=lambda x: x[1])
        return [m for m, _ in items]

    async def get(self, key: str):
        return self._kv.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self._kv[key] = value


# =========================================================================
# Test 1: record_behavior stores amount in payload
# =========================================================================

async def test_record_behavior_stores_amount() -> None:
    """record_behavior should store the amount field in the behavior payload."""
    redis = FakeRedis()
    store = FeatureStore(redis_client=redis)

    await store.record_behavior(
        user_id="u001",
        behavior_type="purchase",
        item_id="P001",
        amount=5999.0,
    )

    behaviors = await store.get_recent_behaviors("u001", "purchase", 604800)
    assert len(behaviors) == 1
    assert behaviors[0]["amount"] == 5999.0, \
        f"expected amount=5999.0, got {behaviors[0].get('amount')}"
    print(f"[OK] rfm.amount_stored: amount={behaviors[0]['amount']}")


# =========================================================================
# Test 2: _compute_rfm reads amount correctly
# =========================================================================

async def test_rfm_monetary_uses_real_amount() -> None:
    """_compute_rfm should compute monetary from actual purchase amounts."""
    redis = FakeRedis()
    store = FeatureStore(redis_client=redis)

    await store.record_behavior("u001", "purchase", "P001", amount=800.0)
    await store.record_behavior("u001", "purchase", "P002", amount=1200.0)

    features = await store.get_user_features("u001")
    rfm = features["rfm"]

    assert rfm["monetary"] > 0, f"monetary should be > 0, got {rfm['monetary']}"
    expected_avg = (800.0 + 1200.0) / 2
    expected_monetary = min(1.0, expected_avg / 1000.0)
    assert abs(rfm["monetary"] - round(expected_monetary, 3)) < 0.01, \
        f"monetary={rfm['monetary']}, expected≈{round(expected_monetary, 3)}"
    print(f"[OK] rfm.monetary: {rfm['monetary']} (avg_amount={expected_avg})")


# =========================================================================
# Test 3: record_behavior without Redis logs warning
# =========================================================================

async def test_record_behavior_warns_without_redis() -> None:
    """record_behavior should log warning when Redis is not available."""
    store = FeatureStore(redis_client=None)

    import io
    import contextlib

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        await store.record_behavior("u001", "view", "P001")

    output = captured.getvalue()
    assert "redis_missing" in output, \
        f"expected warning about missing Redis, got: {output}"
    print("[OK] rfm.warning_logged: warning emitted for missing Redis")


# =========================================================================
# Test 4: _compute_rfm returns zeros with no purchases
# =========================================================================

async def test_rfm_zeros_without_purchases() -> None:
    """_compute_rfm should return all zeros when there are no purchases."""
    redis = FakeRedis()
    store = FeatureStore(redis_client=redis)

    rfm = await store._compute_rfm("u001", [])

    assert rfm == {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}
    print(f"[OK] rfm.empty: {rfm}")


# =========================================================================
# Test 5: amount defaults to 0.0 when not specified
# =========================================================================

async def test_amount_defaults_zero() -> None:
    """record_behavior without amount should default to 0.0 in payload."""
    redis = FakeRedis()
    store = FeatureStore(redis_client=redis)

    await store.record_behavior("u001", "purchase", "P001")

    behaviors = await store.get_recent_behaviors("u001", "purchase", 604800)
    assert len(behaviors) == 1
    assert behaviors[0]["amount"] == 0.0, \
        f"expected default amount=0.0, got {behaviors[0].get('amount')}"
    print(f"[OK] rfm.default_amount: amount={behaviors[0]['amount']}")


# =========================================================================
# Test 6: RFM recency and frequency computed correctly
# =========================================================================

async def test_rfm_recency_frequency() -> None:
    """RFM recency and frequency should be computed from purchase timestamps."""
    redis = FakeRedis()
    store = FeatureStore(redis_client=redis)

    await store.record_behavior("u001", "purchase", "P001", amount=500.0)
    await store.record_behavior("u001", "purchase", "P002", amount=500.0)
    await store.record_behavior("u001", "purchase", "P003", amount=500.0)

    features = await store.get_user_features("u001")
    rfm = features["rfm"]

    assert rfm["recency"] > 0.9, f"recent purchase should have high recency: {rfm['recency']}"
    assert rfm["frequency"] == 0.3, f"3 purchases / 10 = 0.3: {rfm['frequency']}"
    print(f"[OK] rfm.recency_freq: recency={rfm['recency']}, frequency={rfm['frequency']}")


# =========================================================================
# Runner
# =========================================================================

async def main() -> None:
    tests = [
        test_record_behavior_stores_amount,
        test_rfm_monetary_uses_real_amount,
        test_record_behavior_warns_without_redis,
        test_rfm_zeros_without_purchases,
        test_amount_defaults_zero,
        test_rfm_recency_frequency,
    ]
    for fn in tests:
        await fn()
    print(f"\nAll {len(tests)} FeatureStore RFM tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
