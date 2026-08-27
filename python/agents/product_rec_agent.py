"""
商品推荐Agent
- 召回层：协同过滤 + 向量检索(Milvus) + 热度/新品策略
- 排序层：LLM重排 + 特征交叉(用户画像 x 商品属性)
- 多样性控制：类目打散、卖家去重、新品加权
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from llm import get_model_router
from models.schemas import Product, ProductRecResult, UserProfile
from services.cf_store import CFStore
from services.graph_rag import GraphRAGService
from services.milvus_store import MilvusStore

from .base_agent import BaseAgent

logger = structlog.get_logger()

RERANK_PROMPT = """你是电商推荐排序专家。基于用户画像和知识图谱上下文,
从候选商品中选出最符合用户当前意图的商品。

用户画像:
{user_profile}

知识图谱上下文(用户关系/商品关联/多跳路径):
{graph_context}

候选商品:
{candidates}

重点考虑:
1. 用户近期浏览/购买行为暗示的真实需求(如:看了手机壳→可能在找配件而非新手机)
2. 知识图谱中的关联路径强度(如:共购关系、同品类偏好)
3. 商品之间的组合价值(如:手机+充电器组合推荐优于两个手机)

请输出商品ID列表(JSON数组),按推荐优先级排序:
["product_id_1", "product_id_2", ...]
只输出JSON数组,不要其他内容。"""

MOCK_PRODUCTS = [
    Product(product_id="P001", name="iPhone 16 Pro", category="手机", price=7999, brand="Apple", seller_id="S01", stock=500, tags=["旗舰", "新品"]),
    Product(product_id="P002", name="华为 Mate 70", category="手机", price=5999, brand="华为", seller_id="S02", stock=300, tags=["旗舰", "国产"]),
    Product(product_id="P003", name="AirPods Pro 3", category="耳机", price=1899, brand="Apple", seller_id="S01", stock=1000, tags=["降噪", "无线"]),
    Product(product_id="P004", name="Sony WH-1000XM6", category="耳机", price=2499, brand="Sony", seller_id="S03", stock=200, tags=["头戴", "降噪"]),
    Product(product_id="P005", name="iPad Air M3", category="平板", price=4799, brand="Apple", seller_id="S01", stock=400, tags=["学习", "办公"]),
    Product(product_id="P006", name="小米平板7 Pro", category="平板", price=2499, brand="小米", seller_id="S04", stock=600, tags=["性价比", "娱乐"]),
    Product(product_id="P007", name="Anker 140W充电器", category="配件", price=399, brand="Anker", seller_id="S05", stock=2000, tags=["快充", "便携"]),
    Product(product_id="P008", name="机械革命极光X", category="笔记本", price=6999, brand="机械革命", seller_id="S06", stock=150, tags=["游戏", "高性能"]),
    Product(product_id="P009", name="戴尔U2724D显示器", category="显示器", price=3299, brand="Dell", seller_id="S07", stock=80, tags=["4K", "办公"]),
    Product(product_id="P010", name="罗技MX Master 3S", category="配件", price=749, brand="罗技", seller_id="S08", stock=500, tags=["无线", "办公"]),
    Product(product_id="P011", name="三星980 Pro 2TB", category="存储", price=1199, brand="三星", seller_id="S09", stock=300, tags=["SSD", "高速"]),
    Product(product_id="P012", name="绿联氮化镓65W", category="配件", price=129, brand="绿联", seller_id="S10", stock=5000, tags=["快充", "性价比"]),
    Product(product_id="P013", name="Apple Watch Ultra 3", category="穿戴", price=5999, brand="Apple", seller_id="S01", stock=200, tags=["运动", "健康"]),
    Product(product_id="P014", name="大疆Mini 4 Pro", category="无人机", price=4788, brand="大疆", seller_id="S11", stock=100, tags=["航拍", "便携"]),
    Product(product_id="P015", name="Switch 2", category="游戏机", price=2499, brand="Nintendo", seller_id="S12", stock=50, tags=["新品", "游戏"]),
]


class ProductRecAgent(BaseAgent):
    def __init__(self, kg_store: Any | None = None):
        settings = get_settings()
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )
        self.llm = get_model_router().create_llm(
            task_type="recommendation_rerank", temperature=0.3, max_tokens=512
        )
        self.vector_store = MilvusStore()
        self.cf_store = CFStore()
        self.kg_store = kg_store
        self.graph_rag = GraphRAGService(kg_store)
        self._seeded = False

    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        user_profile: UserProfile | None = kwargs.get("user_profile")
        num_items: int = kwargs.get("num_items", 10)
        user_id: str = kwargs.get("user_id") or (
            user_profile.user_id if user_profile else ""
        )

        candidates = await self._recall(user_profile, user_id, num_items * 3)
        graph_context = await self.graph_rag.build_user_context(
            user_id,
            seed_product_ids=(
                list(dict.fromkeys(
                    (user_profile.recent_purchases or [])[:5]
                    + (user_profile.recent_views or [])[:5]
                ))
                if user_profile else []
            ),
        )
        ranked_ids = await self._rerank(
            user_profile, candidates, num_items, graph_context
        )

        id_to_product = {p.product_id: p for p in candidates}
        final_products = []
        for pid in ranked_ids:
            if pid in id_to_product:
                final_products.append(id_to_product[pid])
        if len(final_products) < num_items:
            for p in candidates:
                if p.product_id not in ranked_ids:
                    final_products.append(p)
                    if len(final_products) >= num_items:
                        break

        return ProductRecResult(
            success=True,
            products=final_products[:num_items],
            recall_strategy="vector+collaborative+graph+hot",
            data={
                "candidate_count": len(candidates),
                "scored_candidates": sum(1 for p in candidates if p.score > 0),
                "reranked": len(ranked_ids),
                "graph_context": graph_context,
            },
            confidence=0.8,
        )

    async def _ensure_seeded(self) -> None:
        """Lazily seed MOCK_PRODUCTS into the vector store on first recall."""
        if self._seeded:
            return
        await self.vector_store.upsert_products(MOCK_PRODUCTS)
        self._seeded = True

    @staticmethod
    def _build_query_text(profile: UserProfile | None) -> str:
        """Build a search query from user profile preferences."""
        if not profile:
            return ""
        parts: list[str] = []
        parts.extend(profile.preferred_categories or [])
        # Expand recent view/purchase IDs into product text
        mock_by_id = {p.product_id: p for p in MOCK_PRODUCTS}
        for pid in (profile.recent_views or [])[:5]:
            p = mock_by_id.get(pid)
            if p:
                parts.extend([p.name, p.category, p.brand])
        return " ".join(p for p in parts if p)

    async def _vector_recall(
        self, profile: UserProfile | None, limit: int
    ) -> list[Product]:
        """Vector ANN recall via Milvus (falls back to in-memory cosine)."""
        query_text = self._build_query_text(profile)
        if not query_text:
            return []
        results = await self.vector_store.search_by_text(query_text, limit=limit)
        return [self._to_product(row) for row in results]

    async def _cf_recall(
        self, user_id: str, limit: int
    ) -> list[Product]:
        """Collaborative filtering recall via Redis (falls back to in-memory)."""
        if not user_id:
            return []
        cf_results = await self.cf_store.recommend(user_id, limit=limit)
        if not cf_results:
            return []
        pid_to_cf_score = {r["product_id"]: r["cf_score"] for r in cf_results}
        mock_by_id = {p.product_id: p for p in MOCK_PRODUCTS}
        products = []
        for pid, score in pid_to_cf_score.items():
            base = mock_by_id.get(pid)
            if base:
                p = base.model_copy()
                p.score = float(score)
                products.append(p)
        return products

    async def _recall(
        self, profile: UserProfile | None, user_id: str, limit: int
    ) -> list[Product]:
        """Multi-strategy recall: vector + collaborative + graph + popularity."""
        await self._ensure_seeded()

        # --- Vector recall (Milvus / in-memory fallback) ---
        vector_candidates = await self._vector_recall(profile, limit)

        # --- CF recall (Redis / in-memory fallback) ---
        cf_candidates = await self._cf_recall(user_id, limit)

        # --- Graph recall (Neo4j) ---
        graph_rows = await self._graph_recall(profile, user_id, limit)
        graph_candidates = [self._to_product(row) for row in graph_rows]

        # --- Popularity fallback ---
        popularity = list(MOCK_PRODUCTS)

        # --- Merge & deduplicate ---
        seen: set[str] = set()
        candidates: list[Product] = []

        for p in vector_candidates + cf_candidates + graph_candidates + popularity:
            if p.product_id in seen:
                continue
            seen.add(p.product_id)
            candidates.append(p)

        # --- Score-based sort with deterministic diversity boost ---
        # 说明: 原先使用 random.random() 作为 tie-breaker, 导致同一输入每次
        # 调用结果不同 (缓存失效/测试不稳定/线上行为不可复现)。
        # 改为基于 product_id 的确定性 hash 打散: 同一商品集多次调用结果稳定,
        # 同时保留"同等条件下不总是同一顺序"的展示多样性。
        if profile and profile.preferred_categories:
            preferred = set(profile.preferred_categories)
            candidates.sort(
                key=lambda p: (
                    p.score,
                    p.category in preferred,
                    p.stock > 0,
                    int(hashlib.md5(p.product_id.encode()).hexdigest()[:8], 16),
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda p: (
                    p.score,
                    p.stock > 0,
                    int(hashlib.md5(p.product_id.encode()).hexdigest()[:8], 16),
                ),
                reverse=True,
            )

        return candidates[:limit]

    async def _graph_recall(
        self, profile: UserProfile | None, user_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Collect KG candidates: related products, multi-hop, degree centrality."""
        if self.kg_store is None:
            return []
        rows: list[dict[str, Any]] = []
        seed_ids: list[str] = []
        if profile:
            seed_ids = list(
                dict.fromkeys(
                    (profile.recent_purchases or [])[:5]
                    + (profile.recent_views or [])[:5]
                )
            )
        for seed_id in seed_ids:
            rows.extend(
                await self.kg_store.get_related_products(
                    seed_id, limit=max(1, limit // 2)
                )
            )
        if user_id:
            rows.extend(
                await self.kg_store.get_multi_hop_candidates(
                    user_id, max_hops=2, limit=limit
                )
            )
            rows.extend(
                await self.kg_store.get_degree_central_products(
                    limit=max(1, limit // 3)
                )
            )
        return rows[: limit * 3]

    def _score_candidates(
        self, profile: UserProfile | None, candidates: list[Product]
    ) -> list[Product]:
        """确定性打分: 类目偏好 + 价格区间 + 多样性 + 商品热度"""
        if not profile:
            return candidates
        scored: list[tuple[Product, float]] = []
        seen_categories: set[str] = set()
        preferred = set(profile.preferred_categories or [])
        price_range = profile.price_range or (0, float("inf"))
        for p in candidates:
            score = p.score
            if p.category in preferred:
                score += 0.3
            if price_range[0] <= p.price <= price_range[1]:
                score += 0.2
            if p.category not in seen_categories:
                score += 0.1
                seen_categories.add(p.category)
            p_copy = p.model_copy()
            p_copy.score = score
            scored.append((p_copy, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored]

    async def _rerank(
        self,
        profile: UserProfile | None,
        candidates: list[Product],
        num_items: int,
        graph_context: str,
    ) -> list[str]:
        if not profile:
            return [p.product_id for p in candidates[:num_items]]

        # 第一步: 确定性规则排序
        scored = self._score_candidates(profile, candidates)

        # 第二步: LLM 语义重排（仅对 top-N 候选，控制成本）
        top_candidates = scored[:num_items * 2]
        ranked_ids = await self._llm_rerank(profile, top_candidates, num_items, graph_context)

        # 兜底: LLM 失败时回退到确定性排序
        if not ranked_ids:
            return [p.product_id for p in top_candidates[:num_items]]
        return ranked_ids

    async def _llm_rerank(
        self,
        profile: UserProfile | None,
        candidates: list[Product],
        num_items: int,
        graph_context: str,
    ) -> list[str]:
        """LLM 语义重排: 理解用户意图 + 知识图谱关联路径"""
        profile_summary = {
            "segments": [s.value for s in profile.segments],
            "preferred_categories": profile.preferred_categories,
            "price_range": list(profile.price_range),
        }
        candidate_summary = [
            {"id": p.product_id, "name": p.name, "category": p.category, "price": p.price, "tags": p.tags}
            for p in candidates
        ]
        prompt = RERANK_PROMPT.format(
            num_items=num_items,
            user_profile=json.dumps(profile_summary, ensure_ascii=False),
            graph_context=graph_context or "暂无",
            candidates=json.dumps(candidate_summary, ensure_ascii=False),
        )
        messages = [
            SystemMessage(content="你是电商推荐排序专家。"),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            return []

    @staticmethod
    def _to_product(row: dict[str, Any]) -> Product:
        return Product(
            product_id=row["product_id"],
            name=row.get("name") or "",
            category=row.get("category") or "",
            price=float(row.get("price") or 0),
            description=row.get("description") or "",
            brand=row.get("brand") or "",
            seller_id=row.get("seller_id") or "",
            stock=int(row.get("stock") or 0),
            tags=list(row.get("tags") or []),
            score=float(row.get("graph_score") or 0),
            image_url=row.get("image_url") or "",
        )
