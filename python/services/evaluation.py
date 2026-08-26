"""
评估框架 — 为简历提供真实可量化的业务指标。

覆盖四类评估:
1. 推荐质量: Precision@K / NDCG@K / 类目多样性 / 覆盖率
2. LLM 输出质量: JSON 解析成功率 / 合规通过率 / 文案长度达标率
3. 风控准确性: 风险评分分布 / 决策分布 / 规则命中覆盖
4. 模型路由: 任务-模型映射验证 / 路由决策统计

运行方式: python -m tests.test_evaluation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecommendationMetrics:
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    category_diversity: float = 0.0
    catalog_coverage: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_at_k": round(self.precision_at_k, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "category_diversity": round(self.category_diversity, 4),
            "catalog_coverage": round(self.catalog_coverage, 4),
            "sample_count": self.sample_count,
        }


@dataclass
class LLMOutputMetrics:
    json_parse_success_rate: float = 0.0
    compliance_pass_rate: float = 0.0
    copy_length_adherence: float = 0.0
    total_responses: int = 0
    parse_failures: int = 0
    compliance_violations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "json_parse_success_rate": round(self.json_parse_success_rate, 4),
            "compliance_pass_rate": round(self.compliance_pass_rate, 4),
            "copy_length_adherence": round(self.copy_length_adherence, 4),
            "total_responses": self.total_responses,
            "parse_failures": self.parse_failures,
            "compliance_violations": self.compliance_violations,
        }


@dataclass
class RiskControlMetrics:
    risk_score_distribution: dict[str, int] = field(default_factory=dict)
    decision_distribution: dict[str, int] = field(default_factory=dict)
    rule_hit_coverage: dict[str, int] = field(default_factory=dict)
    total_evaluations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score_distribution": self.risk_score_distribution,
            "decision_distribution": self.decision_distribution,
            "rule_hit_coverage": self.rule_hit_coverage,
            "total_evaluations": self.total_evaluations,
        }


class RecommendationEvaluator:
    """推荐质量评估器 — 基于 Ground Truth 用户偏好计算离线指标。"""

    # 测试数据集: user_id → (preferred_categories, recent_views)
    GROUND_TRUTH = {
        "U001": (["手机", "配件"], ["P001", "P007"]),
        "U002": (["耳机", "平板"], ["P003", "P005"]),
        "U003": (["笔记本", "显示器"], ["P008", "P009"]),
        "U004": (["游戏机", "穿戴"], ["P015", "P013"]),
        "U005": (["配件", "存储"], ["P010", "P011"]),
    }

    # 全量商品目录 (用于覆盖率计算)
    CATALOG_SIZE = 15

    def __init__(self, k: int = 5):
        self.k = k

    def precision_at_k(
        self, recommended_ids: list[str], preferred_categories: list[str], products: list[dict]
    ) -> float:
        """Precision@K: 推荐商品中属于用户偏好类目的比例。"""
        if not recommended_ids:
            return 0.0
        top_k = recommended_ids[: self.k]
        product_by_id = {p["product_id"]: p for p in products}
        relevant = sum(
            1
            for pid in top_k
            if pid in product_by_id and product_by_id[pid]["category"] in preferred_categories
        )
        return relevant / len(top_k)

    def ndcg_at_k(
        self, recommended_ids: list[str], preferred_categories: list[str], products: list[dict]
    ) -> float:
        """NDCG@K: 归一化折损累积增益，位置越靠前的相关推荐贡献越大。"""
        if not recommended_ids:
            return 0.0
        top_k = recommended_ids[: self.k]
        product_by_id = {p["product_id"]: p for p in products}

        dcg = 0.0
        for i, pid in enumerate(top_k):
            if pid in product_by_id and product_by_id[pid]["category"] in preferred_categories:
                dcg += 1.0 / math.log2(i + 2)

        total_relevant = sum(
            1 for p in products if p["category"] in preferred_categories
        )
        ideal_hits = min(self.k, total_relevant)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    def category_diversity(self, recommended_ids: list[str], products: list[dict]) -> float:
        """类目多样性: top-K 推荐中不同类目数 / K。"""
        if not recommended_ids:
            return 0.0
        top_k = recommended_ids[: self.k]
        product_by_id = {p["product_id"]: p for p in products}
        categories = {product_by_id[pid]["category"] for pid in top_k if pid in product_by_id}
        return len(categories) / len(top_k)

    def evaluate_batch(
        self, recommendations: dict[str, list[str]], products: list[dict]
    ) -> RecommendationMetrics:
        """批量评估多个用户的推荐结果。"""
        precisions = []
        ndcgs = []
        diversities = []
        all_recommended: set[str] = set()

        for user_id, rec_ids in recommendations.items():
            if user_id not in self.GROUND_TRUTH:
                continue
            preferred_cats, _ = self.GROUND_TRUTH[user_id]

            precisions.append(self.precision_at_k(rec_ids, preferred_cats, products))
            ndcgs.append(self.ndcg_at_k(rec_ids, preferred_cats, products))
            diversities.append(self.category_diversity(rec_ids, products))
            all_recommended.update(rec_ids[: self.k])

        n = len(precisions)
        return RecommendationMetrics(
            precision_at_k=sum(precisions) / n if n else 0.0,
            ndcg_at_k=sum(ndcgs) / n if n else 0.0,
            category_diversity=sum(diversities) / n if n else 0.0,
            catalog_coverage=len(all_recommended) / self.CATALOG_SIZE,
            sample_count=n,
        )


class LLMOutputEvaluator:
    """LLM 输出质量评估器 — 评估 JSON 可解析性、合规性、长度达标率。"""

    FORBIDDEN_WORDS = [
        "最好", "第一", "国家级", "全球首", "绝对", "100%",
        "永久", "万能", "祖传", "纯天然",
    ]

    TARGET_COPY_MIN = 30
    TARGET_COPY_MAX = 50

    def evaluate_copy_outputs(self, copies: list[dict[str, str]]) -> LLMOutputMetrics:
        """评估营销文案 Agent 的 LLM 输出质量。"""
        total = len(copies)
        if total == 0:
            return LLMOutputMetrics()

        parse_ok = 0
        compliance_ok = 0
        length_ok = 0

        for item in copies:
            if isinstance(item, dict) and "copy" in item and "product_id" in item:
                parse_ok += 1
            else:
                continue

            text = item.get("copy", "")
            has_forbidden = any(w in text for w in self.FORBIDDEN_WORDS)
            if not has_forbidden:
                compliance_ok += 1

            if self.TARGET_COPY_MIN <= len(text) <= self.TARGET_COPY_MAX:
                length_ok += 1

        return LLMOutputMetrics(
            json_parse_success_rate=parse_ok / total,
            compliance_pass_rate=compliance_ok / total if parse_ok else 0.0,
            copy_length_adherence=length_ok / total if parse_ok else 0.0,
            total_responses=total,
            parse_failures=total - parse_ok,
            compliance_violations=parse_ok - compliance_ok,
        )

    def evaluate_rerank_output(self, reranked_ids: list[str], candidate_ids: list[str]) -> dict[str, Any]:
        """评估推荐重排 Agent 的 LLM 输出。"""
        total_candidates = len(candidate_ids)
        returned = len(reranked_ids)
        valid_ids = sum(1 for pid in reranked_ids if pid in candidate_ids)
        return {
            "json_parse_success": returned > 0,
            "valid_id_rate": valid_ids / returned if returned else 0.0,
            "return_rate": returned / total_candidates if total_candidates else 0.0,
        }

    def evaluate_meta_decision(self, raw_response: str) -> dict[str, Any]:
        """评估 MetaAgent 的 LLM 仲裁输出。"""
        import json

        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            decision = data.get("decision", "")
            has_confidence = "confidence" in data
            has_reason = "reason" in data
            valid_decision = decision in ("approve", "escalate", "reject")
            return {
                "json_parse_success": True,
                "valid_decision": valid_decision,
                "has_confidence": has_confidence,
                "has_reason": has_reason,
            }
        except Exception:
            return {
                "json_parse_success": False,
                "valid_decision": False,
                "has_confidence": False,
                "has_reason": False,
            }


class RiskControlEvaluator:
    """风控准确性评估器 — 评估风控/授信/退款规则覆盖与决策分布。"""

    def evaluate_fraud_results(self, results: list[dict[str, Any]]) -> RiskControlMetrics:
        """评估反欺诈检测结果。"""
        metrics = RiskControlMetrics(total_evaluations=len(results))
        for r in results:
            score = r.get("risk_score", 0)
            bucket = "low" if score < 30 else "medium" if score < 60 else "high" if score < 80 else "critical"
            metrics.risk_score_distribution[bucket] = metrics.risk_score_distribution.get(bucket, 0) + 1

            action = r.get("recommended_action", "unknown")
            metrics.decision_distribution[action] = metrics.decision_distribution.get(action, 0) + 1

            for rule in r.get("rules_hit", []):
                name = rule.get("rule_name", "unknown")
                metrics.rule_hit_coverage[name] = metrics.rule_hit_coverage.get(name, 0) + 1
        return metrics

    def evaluate_refund_results(self, results: list[dict[str, Any]]) -> RiskControlMetrics:
        """评估退款风控检测结果。"""
        metrics = RiskControlMetrics(total_evaluations=len(results))
        for r in results:
            score = r.get("risk_score", 0)
            bucket = "low" if score < 30 else "medium" if score < 60 else "high" if score < 80 else "critical"
            metrics.risk_score_distribution[bucket] = metrics.risk_score_distribution.get(bucket, 0) + 1

            status = r.get("refund_status", "unknown")
            metrics.decision_distribution[status] = metrics.decision_distribution.get(status, 0) + 1
        return metrics


class ModelRoutingEvaluator:
    """模型路由评估器 — 验证任务到模型的映射是否生效。"""

    def __init__(self):
        from llm.router import TASK_MODEL_MAP

        self.expected_mapping = {task: info["model_key"] for task, info in TASK_MODEL_MAP.items()}

    def evaluate_routing(self) -> dict[str, Any]:
        """从 MultiModelRouter 获取路由日志并评估。"""
        from llm import get_model_router

        router = get_model_router()
        route_info = router.get_route_info()
        routing_stats = route_info.get("routing_stats", {})

        verified_tasks: list[str] = []
        mismatched: list[dict[str, str]] = []

        for task, stats in routing_stats.items():
            total = stats.get("total_calls", 0)
            if total > 0:
                dist = stats.get("model_distribution", {})
                actual_model = max(dist, key=dist.get)
                expected_model = self.expected_mapping.get(task, "general")
                if actual_model == expected_model:
                    verified_tasks.append(task)
                else:
                    mismatched.append({
                        "task": task,
                        "expected": expected_model,
                        "actual": actual_model,
                    })

        return {
            "routing_enabled": route_info.get("enabled", False),
            "verified_tasks": verified_tasks,
            "mismatched_routes": mismatched,
            "routing_stats": routing_stats,
        }
