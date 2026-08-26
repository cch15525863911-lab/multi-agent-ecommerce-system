"""
协同过滤引擎 — 基于用户-商品交互矩阵的推荐。

架构:
  Redis Sorted Set 存储双向索引:
    cf:user:{uid}:items    → 商品集合 (score=时间戳, member=product_id)
    cf:item:{pid}:users    → 用户集合 (score=时间戳, member=user_id)

算法 — User-based CF with Jaccard similarity:
  1. 取当前用户的交互商品集合 A
  2. 对 A 中每个商品，取所有交互过该商品的用户
  3. 计算每个用户与当前用户的 Jaccard 相似度
  4. 取相似度最高的 N 个用户
  5. 聚合这些用户的商品，排除当前用户已有的
  6. 按频次+相似度加权排序，取 top-N

降级:
  Redis 不可用时自动切换到内存字典，保证开发/测试可用。
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from config import get_settings

logger = structlog.get_logger()

# 交互权重：不同行为的贡献度不同
ACTION_WEIGHTS = {"purchase": 3.0, "favorite": 2.0, "view": 1.0}


class CFStore:
    """User-based collaborative filtering with Redis backend."""

    USER_KEY = "cf:user:{uid}:items"
    ITEM_KEY = "cf:item:{pid}:users"
    INTERACTION_TTL = 86400 * 30  # 30 天

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client
        self._connected = False
        self._connect_attempted = False

        # 内存降级
        self._mem_user_items: dict[str, dict[str, float]] = {}
        self._mem_item_users: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> bool:
        if self._connected and self._redis is not None:
            return True
        if self._connect_attempted:
            return False
        if self._redis is not None:
            try:
                await self._redis.ping()
                self._connected = True
                self._connect_attempted = True
                return True
            except Exception:
                self._connect_attempted = True
                return False

        # 尝试创建连接
        try:
            import redis.asyncio as aioredis

            settings = get_settings()
            self._redis = aioredis.from_url(
                settings.redis_url, decode_responses=True, socket_connect_timeout=2,
            )
            await self._redis.ping()
            self._connected = True
            self._connect_attempted = True
            logger.info("cf_store.redis_connected", url=settings.redis_url)
            return True
        except Exception as exc:
            logger.warning("cf_store.redis_unavailable", error=str(exc), fallback="in-memory")
            self._connected = False
            self._connect_attempted = True
            return False

    # ------------------------------------------------------------------
    # Interaction recording
    # ------------------------------------------------------------------

    async def record_interaction(
        self,
        user_id: str,
        product_id: str,
        action: str = "view",
    ) -> None:
        """记录用户-商品交互。"""
        weight = ACTION_WEIGHTS.get(action, 1.0)
        ts = time.time()

        if await self._ensure_connected():
            user_key = self.USER_KEY.format(uid=user_id)
            item_key = self.ITEM_KEY.format(pid=product_id)
            # 用 score = weight * timestamp 保持最近且高权重的交互排前
            score = weight * ts
            pipe = self._redis.pipeline()
            pipe.zadd(user_key, {product_id: score})
            pipe.zadd(item_key, {user_id: score})
            pipe.expire(user_key, self.INTERACTION_TTL)
            pipe.expire(item_key, self.INTERACTION_TTL)
            await pipe.execute()
        else:
            # 内存降级
            self._mem_user_items.setdefault(user_id, {})[product_id] = weight * ts
            self._mem_item_users.setdefault(product_id, {})[user_id] = weight * ts

    # ------------------------------------------------------------------
    # Similar users
    # ------------------------------------------------------------------

    async def get_similar_users(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """基于 Jaccard 相似度找到相似用户。"""
        my_items = await self._get_user_items(user_id)
        if not my_items:
            return []

        # 收集所有与当前用户有交集的用户
        candidate_users: set[str] = set()
        for pid in my_items:
            item_users = await self._get_item_users(pid)
            candidate_users.update(item_users)

        candidate_users.discard(user_id)
        if not candidate_users:
            return []

        my_set = set(my_items)
        scores: list[dict[str, Any]] = []
        for other_uid in candidate_users:
            other_items = await self._get_user_items(other_uid)
            if not other_items:
                continue
            other_set = set(other_items)
            intersection = len(my_set & other_set)
            if intersection == 0:
                continue
            union = len(my_set | other_set)
            jaccard = intersection / union if union > 0 else 0
            scores.append(
                {
                    "user_id": other_uid,
                    "jaccard": round(jaccard, 4),
                    "shared_products": intersection,
                }
            )

        scores.sort(key=lambda x: (-x["jaccard"], -x["shared_products"]))
        return scores[:limit]

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    async def recommend(
        self,
        user_id: str,
        limit: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """协同过滤推荐：找相似用户的商品，排除已有的。"""
        my_items = set(await self._get_user_items(user_id))
        if exclude_ids:
            my_items |= exclude_ids

        similar_users = await self.get_similar_users(user_id, limit=20)
        if not similar_users:
            return []

        # 聚合相似用户的商品
        product_scores: dict[str, float] = {}
        for su in similar_users:
            uid = su["user_id"]
            sim = su["jaccard"]
            other_items = await self._get_user_items(uid)
            for pid in other_items:
                if pid in my_items:
                    continue
                product_scores[pid] = product_scores.get(pid, 0) + sim

        ranked = sorted(product_scores.items(), key=lambda x: -x[1])[:limit]
        return [{"product_id": pid, "cf_score": score} for pid, score in ranked]

    # ------------------------------------------------------------------
    # Internal: data access with fallback
    # ------------------------------------------------------------------

    async def _get_user_items(self, user_id: str) -> list[str]:
        """获取用户交互过的商品列表。"""
        if await self._ensure_connected():
            key = self.USER_KEY.format(uid=user_id)
            items = await self._redis.zrevrange(key, 0, -1)
            return list(items) if items else []
        else:
            return list(self._mem_user_items.get(user_id, {}).keys())

    async def _get_item_users(self, product_id: str) -> list[str]:
        """获取交互过该商品的用户列表。"""
        if await self._ensure_connected():
            key = self.ITEM_KEY.format(pid=product_id)
            users = await self._redis.zrevrange(key, 0, -1)
            return list(users) if users else []
        else:
            return list(self._mem_item_users.get(product_id, {}).keys())

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "backend": "redis" if self._connected else "in-memory",
            "tracked_users": len(self._mem_user_items),
            "tracked_items": len(self._mem_item_users),
        }

    # ------------------------------------------------------------------
    # Seed helper
    # ------------------------------------------------------------------

    async def seed_interactions(
        self, interactions: list[tuple[str, str, str]]
    ) -> int:
        """批量注入交互数据: [(user_id, product_id, action), ...]"""
        for uid, pid, action in interactions:
            await self.record_interaction(uid, pid, action)
        return len(interactions)
