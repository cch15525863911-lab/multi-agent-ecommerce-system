"""
Neo4j 启动初始化 — 单元测试。

测试覆盖:
    1. lifespan 启动时调用 KGStore.connect()
    2. lifespan 关闭时调用 KGStore.close()
    3. 健康检查端点返回 neo4j 连接状态
    4. KGStore.connect() 在 Neo4j 不可用时降级为 mock 模式
    5. KGStore.connect() 使用注入的 driver
    6. kg_store 为 None 时健康检查返回 neo4j=False

Run from the `python/` directory:
    python -m tests.test_neo4j_startup
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock uvicorn before importing main (it may not be installed in test env)
if "uvicorn" not in sys.modules:
    sys.modules["uvicorn"] = MagicMock()


# =========================================================================
# Test 1: lifespan calls KGStore.connect() at startup
# =========================================================================

async def test_lifespan_calls_kg_connect() -> None:
    """lifespan startup should call kg_store.connect()."""
    # 开启鉴权后启动需 JWT 密钥, 注入测试密钥避免启动报错 (fail-fast 校验)
    os.environ["ECOM_JWT_SECRET"] = "test-secret-for-lifespan"
    from config import get_settings

    get_settings.cache_clear()
    from services.kg_store import KGStore

    mock_kg = MagicMock(spec=KGStore)
    mock_kg.connect = AsyncMock(return_value=True)
    mock_kg.close = AsyncMock()

    with patch("main.KGStore", return_value=mock_kg):
        from main import lifespan
        app_mock = MagicMock()
        async with lifespan(app_mock):
            pass

    mock_kg.connect.assert_called_once()
    mock_kg.close.assert_called_once()
    print("[OK] neo4j.lifespan: connect() and close() called")


# =========================================================================
# Test 2: Health check includes neo4j status
# =========================================================================

async def test_health_includes_neo4j() -> None:
    """/health should return neo4j connection status."""
    mock_kg = MagicMock()
    mock_kg.connected = True

    with patch("main.kg_store", mock_kg):
        from main import health
        result = await health()

    assert "neo4j" in result, "health response should include neo4j field"
    assert result["neo4j"] is True, "neo4j should be True when connected"
    print(f"[OK] neo4j.health: neo4j={result['neo4j']}")


# =========================================================================
# Test 3: KGStore.connect() falls back to mock when Neo4j unavailable
# =========================================================================

async def test_kg_connect_fallback_on_unavailable() -> None:
    """KGStore.connect() should return False when Neo4j is unreachable."""
    from services.kg_store import KGStore

    store = KGStore()

    # Patch AsyncGraphDatabase to raise ImportError (no neo4j installed)
    with patch.dict("sys.modules", {"neo4j": None}):
        result = await store.connect()

    assert result is False, "connect should return False when Neo4j unavailable"
    assert store.connected is False
    await store.close()
    print("[OK] neo4j.fallback: returns False when Neo4j unavailable")


# =========================================================================
# Test 4: KGStore.connect() with a pre-injected driver
# =========================================================================

async def test_kg_connect_with_injected_driver() -> None:
    """KGStore should work with a pre-injected driver."""
    from services.kg_store import KGStore

    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()
    store = KGStore(driver=mock_driver)
    result = await store.connect()

    assert result is True
    assert store.connected is True
    await store.close()
    print("[OK] neo4j.injected_driver: connected with pre-injected driver")


# =========================================================================
# Test 5: kg_store is None when health called before lifespan
# =========================================================================

async def test_health_neo4j_false_when_no_kg_store() -> None:
    """Health check should report neo4j=False when kg_store is None."""
    with patch("main.kg_store", None):
        from main import health
        result = await health()

    assert result["neo4j"] is False
    print("[OK] neo4j.health_none: neo4j=False when kg_store is None")


# =========================================================================
# Runner
# =========================================================================

async def main() -> None:
    tests = [
        test_lifespan_calls_kg_connect,
        test_health_includes_neo4j,
        test_kg_connect_fallback_on_unavailable,
        test_kg_connect_with_injected_driver,
        test_health_neo4j_false_when_no_kg_store,
    ]
    for fn in tests:
        await fn()
    print(f"\nAll {len(tests)} Neo4j startup tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
