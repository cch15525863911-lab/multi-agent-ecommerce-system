"""
Milvus 向量存储 — 商品向量索引与相似度检索。

功能:
1. 懒连接 Milvus 客户端（首次查询时才建立连接）
2. 自动创建 product_embeddings 集合（含 HNSW 索引）
3. 商品数据向量化并 upsert
4. 基于向量余弦相似度的 ANN 搜索
5. 支持元数据过滤（类目、价格区间）
6. Milvus 不可用时降级为内存余弦相似度（保证开发/测试可用）

Embedding 策略:
  主链路: sentence-transformers + BAAI/bge-small-zh-v1.5 (512维语义向量)
    - 真实语义编码, 相似商品在向量空间中距离更近
    - 支持 中英文混合, 适合电商场景
  降级链路: signed hashing trick (512维)
    - 当 sentence-transformers 未安装或模型下载失败时自动降级
    - 保留文本相似度近似能力, 但语义表达弱于真实 embedding
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import structlog

from config import get_settings
from models.schemas import Product

logger = structlog.get_logger()

EMBEDDING_DIM = 512  # bge-small-zh-v1.5 output dimension

# ---------------------------------------------------------------------------
# Embedding model — lazy singleton
# ---------------------------------------------------------------------------

_embedding_model: Any = None
_embedding_mode: str = ""  # "semantic" or "hashing"
_init_attempted = False


def _get_embedding_model() -> tuple[Any, str]:
    """Lazy-load the sentence-transformers model.

    Returns (model_or_none, mode) where mode is "semantic" or "hashing".
    On first call, attempts to load BGE model; on failure, falls back to hashing.
    """
    global _embedding_model, _embedding_mode, _init_attempted
    if _init_attempted:
        return _embedding_model, _embedding_mode

    _init_attempted = True
    model_name = get_settings().embedding_model
    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(model_name)
        _embedding_mode = "semantic"
        logger.info(
            "milvus.embedding_loaded",
            mode="semantic",
            model=model_name,
            dim=EMBEDDING_DIM,
        )
    except Exception as exc:
        _embedding_mode = "hashing"
        logger.warning(
            "milvus.embedding_fallback",
            mode="hashing",
            reason=str(exc),
            model=model_name,
        )
    return _embedding_model, _embedding_mode


def _product_to_text(product: Product) -> str:
    parts = [product.name, product.category, product.brand]
    parts.extend(product.tags)
    if product.description:
        parts.append(product.description[:100])
    return " ".join(p for p in parts if p)


def _semantic_encode(texts: str | list[str]) -> np.ndarray:
    """Encode text(s) using BGE model with L2 normalization."""
    model, mode = _get_embedding_model()
    if mode == "semantic" and model is not None:
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
    # hashing fallback
    if isinstance(texts, str):
        return np.array(_hashing_vector(texts), dtype=np.float32)
    return np.array([_hashing_vector(t) for t in texts], dtype=np.float32)


def _text_to_vector(text: str) -> list[float]:
    """Convert text to a dense embedding vector (512-dim).

    Uses BGE semantic encoding when available, hashing trick as fallback.
    """
    vec = _semantic_encode(text)
    if vec.ndim > 1:
        vec = vec[0]
    return vec.tolist()


def _hashing_vector(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Fallback: signed hashing trick when sentence-transformers unavailable."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().replace(",", " ").split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


class MilvusStore:
    """Milvus 向量存储，支持 ANN 搜索与内存降级。"""

    COLLECTION = "product_embeddings"

    def __init__(self) -> None:
        settings = get_settings()
        self._host = settings.milvus_host
        self._port = settings.milvus_port
        self._collection = settings.milvus_collection or self.COLLECTION
        self._client: Any = None
        self._connected = False

        # 内存降级模式：当 Milvus 不可用时使用
        self._fallback_products: list[Product] = []
        self._fallback_vectors: list[list[float]] = []

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> bool:
        """懒连接：首次调用时尝试连接 Milvus。失败则切换到内存降级。"""
        if self._connected:
            return True
        try:
            from pymilvus import MilvusClient, DataType

            uri = f"http://{self._host}:{self._port}"
            self._client = MilvusClient(uri=uri)
            await self._ensure_collection()
            self._connected = True
            logger.info(
                "milvus.connected",
                host=self._host,
                port=self._port,
                collection=self._collection,
            )
            return True
        except Exception as exc:
            logger.warning("milvus.unavailable", error=str(exc), fallback="in-memory")
            self._connected = False
            return False

    async def _ensure_collection(self) -> None:
        """集合不存在则创建，含 HNSW 向量索引。"""
        if self._client is None:
            return
        if self._client.has_collection(self._collection):
            return

        from pymilvus import DataType

        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("product_id", DataType.VARCHAR, max_length=64)
        schema.add_field("name", DataType.VARCHAR, max_length=256)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("brand", DataType.VARCHAR, max_length=64)
        schema.add_field("price", DataType.FLOAT)
        schema.add_field("tags", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info("milvus.collection_created", collection=self._collection)

    # ------------------------------------------------------------------
    # Vector upsert
    # ------------------------------------------------------------------

    async def upsert_products(self, products: list[Product]) -> int:
        """向量化并 upsert 商品列表。返回成功写入的数量。"""
        if not products:
            return 0

        texts = [_product_to_text(p) for p in products]
        vecs = _semantic_encode(texts)  # batch encode for efficiency

        data = []
        vectors = []
        for i, p in enumerate(products):
            vec = vecs[i].tolist() if vecs.ndim > 1 else vecs.tolist()
            vectors.append(vec)
            data.append(
                {
                    "product_id": p.product_id,
                    "name": p.name,
                    "category": p.category,
                    "brand": p.brand,
                    "price": float(p.price),
                    "tags": ",".join(p.tags),
                    "embedding": vec,
                }
            )

        # 内存降级模式也要存储
        self._fallback_products.extend(products)
        self._fallback_vectors.extend(vectors)

        if await self._ensure_connected():
            try:
                self._client.upsert(
                    collection_name=self._collection,
                    data=data,
                )
                logger.info("milvus.upserted", count=len(data))
            except Exception as exc:
                logger.warning("milvus.upsert_failed", error=str(exc))
                return 0

        return len(data)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def search_by_text(
        self,
        query_text: str,
        limit: int = 10,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        """基于文本的向量相似度搜索。返回 [{product_id, name, category, ...}]。"""
        query_vec = _text_to_vector(query_text)
        return await self._search(query_vec, limit, filter_expr)

    async def search_similar_products(
        self,
        product: Product,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """基于商品向量的相似商品搜索（item-to-item）。"""
        query_vec = _text_to_vector(_product_to_text(product))
        filter_expr = f'product_id != "{product.product_id}"'
        return await self._search(query_vec, limit, filter_expr)

    async def _search(
        self,
        query_vec: list[float],
        limit: int,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        # --- Milvus 路径 ---
        if await self._ensure_connected():
            try:
                results = self._client.search(
                    collection_name=self._collection,
                    data=[query_vec],
                    limit=limit,
                    output_fields=["product_id", "name", "category", "brand", "price", "tags"],
                    filter=filter_expr or None,
                )
                if results and len(results) > 0:
                    hits = results[0] if isinstance(results[0], list) else results
                    output = []
                    for hit in hits:
                        entity = hit.get("entity", hit) if isinstance(hit, dict) else hit
                        output.append(
                            {
                                "product_id": entity.get("product_id", ""),
                                "name": entity.get("name", ""),
                                "category": entity.get("category", ""),
                                "brand": entity.get("brand", ""),
                                "price": float(entity.get("price", 0)),
                                "tags": entity.get("tags", "").split(",") if entity.get("tags") else [],
                                "score": float(hit.get("distance", 0.0)) if isinstance(hit, dict) else 0.0,
                                "graph_score": float(hit.get("distance", 0.0)) if isinstance(hit, dict) else 0.0,
                            }
                        )
                    return output
            except Exception as exc:
                logger.warning("milvus.search_failed", error=str(exc))

        # --- 内存降级路径 ---
        return self._fallback_search(query_vec, limit, filter_expr)

    def _fallback_search(
        self,
        query_vec: list[float],
        limit: int,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        """内存余弦相似度搜索（支持简单 product_id != "..." 过滤）。"""
        if not self._fallback_vectors:
            return []

        # Parse simple filter: product_id != "xxx"
        exclude_pids: set[str] = set()
        if filter_expr:
            import re
            for m in re.finditer(r'product_id\s*!=\s*"([^"]+)"', filter_expr):
                exclude_pids.add(m.group(1))

        q = np.array(query_vec, dtype=np.float32)
        scores = []
        for i, vec in enumerate(self._fallback_vectors):
            p = self._fallback_products[i]
            if p.product_id in exclude_pids:
                continue
            v = np.array(vec, dtype=np.float32)
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-8))
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        output = []
        for idx, sim in scores[:limit]:
            p = self._fallback_products[idx]
            output.append(
                {
                    "product_id": p.product_id,
                    "name": p.name,
                    "category": p.category,
                    "brand": p.brand,
                    "price": float(p.price),
                    "tags": p.tags,
                    "score": sim,
                    "graph_score": sim,
                }
            )
        return output

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        _, mode = _get_embedding_model()
        return {
            "connected": self._connected,
            "host": self._host,
            "port": self._port,
            "collection": self._collection,
            "embedding_mode": mode,
            "embedding_dim": EMBEDDING_DIM,
            "fallback_products": len(self._fallback_products),
        }
