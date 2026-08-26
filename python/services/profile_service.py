"""
用户画像 Service — 基于知识图谱 + 确定性规则的用户分群。

传统业务 Service, 非 LLM Agent:
    数据源: Neo4j 知识图谱 (KGStore) 或 Redis FeatureStore
    分群逻辑: 确定性规则引擎 (RFM 模型), 100% 可解释可复现

与旧版 UserProfileAgent 的区别:
    - 不调用 LLM 做分群推断 (改为确定性规则)
    - 10-20x 更快, 100% 可复现
    - 每个 segment 有明确的规则 + 原因
"""

from __future__ import annotations

from typing import Any

import structlog

from config import get_settings
from models.schemas import AgentResult, UserProfile, UserProfileResult, UserSegment
from services.base_service import BaseProtectedService
from services.kg_store import KGStore

logger = structlog.get_logger()


class ProfileService(BaseProtectedService):
    """用户画像 Service — Neo4j 图谱特征 + 确定性规则分群。"""

    def __init__(self, kg_store: KGStore | None = None) -> None:
        settings = get_settings()
        super().__init__(
            name="profile",
            timeout=settings.agent_timeout_user_profile,
            max_retries=2,
        )
        self.kg_store = kg_store or KGStore()
        self._connect_attempted = False

    async def _ensure_store(self) -> None:
        if not self._connect_attempted:
            self._connect_attempted = True
            await self.kg_store.connect()

    async def execute(self, **kwargs: Any) -> UserProfileResult:
        await self._ensure_store()

        user_id: str = kwargs["user_id"]
        context: dict = kwargs.get("context", {})

        features = await self.kg_store.get_profile_features(user_id)
        features = self._overlay_context(features, context)

        segments = self._rule_segments(features)
        real_time_tags = self._rule_tags(features, segments)
        preferred = features.get("preferred_categories", [])
        if not preferred and context.get("recent_views"):
            preferred = list(dict.fromkeys(context["recent_views"]))
        preferred = preferred[:8]

        price_range = features.get("price_range", (0.0, 10000.0))

        profile = UserProfile(
            user_id=user_id,
            segments=segments,
            preferred_categories=preferred,
            price_range=(float(price_range[0]), float(price_range[1])),
            recent_views=features.get("recent_views", []),
            recent_purchases=features.get("recent_purchases", []),
            rfm_score=features.get("rfm", {}),
            real_time_tags=real_time_tags,
        )

        return UserProfileResult(
            success=True,
            profile=profile,
            data={
                "feature_source": "neo4j" if self.kg_store.connected else "mock+context",
                "features": features,
                "segment_reasons": self._segment_reasons(features, segments),
            },
            confidence=0.92,
        )

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        return UserProfileResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
            profile=UserProfile(
                user_id="unknown",
                segments=[UserSegment.ACTIVE],
                preferred_categories=[],
                price_range=(0.0, 10000.0),
                rfm_score={"recency": 0.3, "frequency": 0.3, "monetary": 0.3},
            ),
        )

    # ------------------------------------------------------------------
    # 特征冷启动覆盖
    # ------------------------------------------------------------------

    @staticmethod
    def _overlay_context(features: dict, ctx: dict) -> dict:
        if ctx.get("recent_views") and not features.get("recent_views"):
            features["recent_views"] = list(ctx["recent_views"])
            features["view_count_24h"] = features.get("view_count_24h") or len(ctx["recent_views"])
        if ctx.get("recent_purchases") and not features.get("recent_purchases"):
            features["recent_purchases"] = list(ctx["recent_purchases"])
            features["purchase_count_7d"] = features.get("purchase_count_7d") or len(ctx["recent_purchases"])

        if features.get("price_range", (0, 10000)) == (0.0, 10000.0):
            avg = ctx.get("avg_order_amount")
            if isinstance(avg, (int, float)) and avg > 0:
                lo = max(0.0, float(avg) * 0.3)
                hi = float(avg) * 3.0
                features["price_range"] = (lo, hi)

        if not features.get("rfm") or all(v == 0 for v in features.get("rfm", {}).values()):
            r = 0.5 if features.get("view_count_24h", 0) > 0 else 0.1
            f = min(1.0, (features.get("purchase_count_7d", 0) or 0) / 5.0)
            avg_amount = ctx.get("avg_order_amount", 100) if isinstance(ctx.get("avg_order_amount"), (int, float)) else 100
            m = min(1.0, float(avg_amount) / 1000.0)
            features["rfm"] = {
                "recency": round(r, 3),
                "frequency": round(f, 3),
                "monetary": round(m, 3),
            }
        return features

    # ------------------------------------------------------------------
    # 确定性规则引擎 — 用户分群
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_segments(features: dict) -> list[UserSegment]:
        rfm = features.get("rfm") or {}
        r = rfm.get("recency", 0.0)
        f = rfm.get("frequency", 0.0)
        m = rfm.get("monetary", 0.0)
        total_interactions = (
            features.get("view_count_24h", 0)
            + features.get("purchase_count_7d", 0) * 3
        )
        recent_buys = features.get("purchase_count_7d", 0) or 0
        history_buys = len(features.get("recent_purchases", [])) or recent_buys

        segments: list[UserSegment] = []
        active_now = (features.get("view_count_24h", 0) + features.get("purchase_count_7d", 0)) > 0

        if r < 0.3 and not active_now:
            segments.append(UserSegment.CHURN_RISK)

        if history_buys < 2 and f < 0.35:
            segments.append(UserSegment.NEW_USER)

        if m > 0.7 and r > 0.5:
            segments.append(UserSegment.HIGH_VALUE)

        if f > 0.4 and m < 0.45:
            segments.append(UserSegment.PRICE_SENSITIVE)

        if (total_interactions >= 2 and not segments) or (not segments and r >= 0.3):
            segments.append(UserSegment.ACTIVE)

        if not segments:
            segments.append(UserSegment.ACTIVE)
        return segments

    @staticmethod
    def _segment_reasons(features: dict, segments: list[UserSegment]) -> list[str]:
        rfm = features.get("rfm") or {}
        reasons: list[str] = []
        active_now = (features.get("view_count_24h", 0) + features.get("purchase_count_7d", 0)) > 0
        for seg in segments:
            if seg == UserSegment.CHURN_RISK:
                reasons.append(
                    f"churn_risk: recency={rfm.get('recency'):.2f} < 0.3, no_recent_activity={not active_now}"
                )
            elif seg == UserSegment.NEW_USER:
                n = len(features.get("recent_purchases", []))
                reasons.append(f"new_user: {n} historical purchases, freq={rfm.get('frequency'):.2f}")
            elif seg == UserSegment.HIGH_VALUE:
                reasons.append(
                    f"high_value: monetary={rfm.get('monetary'):.2f}, recency={rfm.get('recency'):.2f}"
                )
            elif seg == UserSegment.PRICE_SENSITIVE:
                reasons.append(
                    f"price_sensitive: freq={rfm.get('frequency'):.2f} (>0.4), monetary={rfm.get('monetary'):.2f} (<0.45)"
                )
            else:
                reasons.append("active: default engagement level")
        return reasons

    @staticmethod
    def _rule_tags(features: dict, segments: list[UserSegment]) -> dict[str, Any]:
        tags: dict[str, Any] = {}

        hours = features.get("active_hours") or []
        if hours:
            tags["活跃时段"] = [f"{h:02d}:00" for h in sorted(hours)]
        else:
            tags["活跃时段"] = "未识别"

        pr = features.get("price_range", (0, 10000))
        if isinstance(pr, tuple) and len(pr) == 2:
            tags["价格偏好区间"] = f"¥{pr[0]:.0f} – ¥{pr[1]:.0f}"

        tags["近24h浏览次数"] = features.get("view_count_24h", 0)
        tags["近7天购买次数"] = features.get("purchase_count_7d", 0)

        if UserSegment.HIGH_VALUE in segments:
            tags["用户价值分层"] = "高价值"
        elif UserSegment.CHURN_RISK in segments:
            tags["用户价值分层"] = "流失预警"
        elif UserSegment.NEW_USER in segments:
            tags["用户价值分层"] = "新客"
        else:
            tags["用户价值分层"] = "普通活跃"

        return tags
