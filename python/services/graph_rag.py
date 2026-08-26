"""GraphRAG: turn KG query results into compact, LLM-friendly context."""

from __future__ import annotations

from typing import Any


class GraphRAGService:
    """Build prompt-ready subgraph evidence from a KGStore."""

    def __init__(self, kg_store: Any | None = None):
        self.kg_store = kg_store

    async def build_user_context(
        self,
        user_id: str,
        seed_product_ids: list[str] | None = None,
        limit: int = 5,
    ) -> str:
        """Return a text block describing the user's relevant subgraph."""
        if self.kg_store is None:
            return ""

        sections: list[str] = []
        related = await self._related_from_seeds(seed_product_ids or [], limit)
        if related:
            sections.append("商品关系证据:")
            sections.extend(f"- {line}" for line in related)

        multi_hop = await self.kg_store.get_multi_hop_candidates(
            user_id, max_hops=2, limit=limit
        )
        if multi_hop:
            sections.append("多跳候选:")
            sections.extend(
                f"- {row.get('name', row.get('product_id'))} "
                f"经由 {row.get('hops')} 跳关系，"
                f"图谱得分 {float(row.get('graph_score') or 0):.2f}"
                for row in multi_hop
            )

        similar_users = await self.kg_store.get_similar_users(user_id, limit=3)
        if similar_users:
            sections.append("相似用户:")
            sections.extend(
                f"- 用户 {row['user_id']}，Jaccard {row['jaccard']}，"
                f"共享 {row['shared_products']} 个商品"
                for row in similar_users
            )

        central_products = await self.kg_store.get_degree_central_products(limit=3)
        if central_products:
            sections.append("图谱热点商品:")
            sections.extend(
                f"- {row.get('name', row.get('product_id'))} "
                f"交互度 {row.get('graph_score', 0)}"
                for row in central_products
            )

        return "\n".join(sections) if sections else "图谱上下文：暂无可用关系。"

    async def explain_target(self, user_id: str, target_product_id: str) -> str:
        """Return human-readable graph paths toward one target product."""
        if self.kg_store is None:
            return ""
        paths = await self.kg_store.get_explanation_paths(
            user_id, target_product_id, max_hops=2, limit=3
        )
        if not paths:
            return ""
        lines = ["推荐路径:"]
        for path in paths:
            relations = " -> ".join(path.get("relations") or [])
            lines.append(
                f"- 用户交互 {path.get('seed_name', path.get('seed_product_id'))} "
                f"经 {relations} 到达 {target_product_id}（{path.get('hops')} 跳）"
            )
        return "\n".join(lines)

    async def _related_from_seeds(
        self, seed_product_ids: list[str], limit: int
    ) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for seed_id in list(dict.fromkeys(seed_product_ids))[:5]:
            rows = await self.kg_store.get_related_products(
                seed_id, limit=limit
            )
            for row in rows:
                key = f"{seed_id}:{row.get('product_id')}"
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"交互商品 {seed_id} 与 "
                    f"{row.get('name', row.get('product_id'))} 存在 "
                    f"{row.get('relation', 'related')} 关系，"
                    f"权重 {float(row.get('weight') or 0):.2f}"
                )
        return lines
