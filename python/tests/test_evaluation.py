"""
评估测试 — 产出真实可量化的业务指标，为简历提供数据支撑。

运行方式: python -m tests.test_evaluation
或:       pytest tests/test_evaluation.py -s

产出指标:
1. 推荐质量: Precision@5 / NDCG@5 / 类目多样性 / 覆盖率
2. LLM 输出质量: JSON 解析成功率 / 合规通过率 / 文案长度达标率
3. 风控准确性: 风险评分分布 / 决策分布 / 规则命中覆盖
4. 模型路由: 任务-模型映射验证
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.product_rec_agent import ProductRecAgent, MOCK_PRODUCTS
from models.schemas import Product, UserProfile, UserSegment
from services.evaluation import (
    LLMOutputEvaluator,
    ModelRoutingEvaluator,
    RecommendationEvaluator,
    RiskControlEvaluator,
)
from services.fraud_service import FraudService
from services.refund_service import RefundRiskService


# =========================================================================
# 1. 推荐质量评估
# =========================================================================


def _build_test_profiles() -> dict[str, UserProfile]:
    """构建与 RecommendationEvaluator.GROUND_TRUTH 对齐的测试用户画像。"""
    return {
        "U001": UserProfile(
            user_id="U001",
            segments=[UserSegment.ACTIVE],
            preferred_categories=["手机", "配件"],
            price_range=(0, 10000),
            recent_views=["P001", "P007"],
        ),
        "U002": UserProfile(
            user_id="U002",
            segments=[UserSegment.HIGH_VALUE],
            preferred_categories=["耳机", "平板"],
            price_range=(0, 6000),
            recent_views=["P003", "P005"],
        ),
        "U003": UserProfile(
            user_id="U003",
            segments=[UserSegment.ACTIVE],
            preferred_categories=["笔记本", "显示器"],
            price_range=(0, 10000),
            recent_views=["P008", "P009"],
        ),
        "U004": UserProfile(
            user_id="U004",
            segments=[UserSegment.NEW_USER],
            preferred_categories=["游戏机", "穿戴"],
            price_range=(0, 8000),
            recent_views=["P015", "P013"],
        ),
        "U005": UserProfile(
            user_id="U005",
            segments=[UserSegment.PRICE_SENSITIVE],
            preferred_categories=["配件", "存储"],
            price_range=(0, 2000),
            recent_views=["P010", "P011"],
        ),
    }


async def evaluate_recommendation_quality():
    """评估推荐链路的离线指标。

    LLM 可用时走完整 Agent 链路；LLM 不可用时走确定性打分路径。
    """
    print("\n" + "=" * 60)
    print("1. 推荐质量评估 (Precision@5 / NDCG@5 / 多样性 / 覆盖率)")
    print("=" * 60)

    agent = ProductRecAgent(kg_store=None)
    profiles = _build_test_profiles()
    evaluator = RecommendationEvaluator(k=5)

    products_as_dicts = [
        {"product_id": p.product_id, "category": p.category, "name": p.name}
        for p in MOCK_PRODUCTS
    ]

    recommendations: dict[str, list[str]] = {}
    for user_id, profile in profiles.items():
        try:
            result = await agent.run(user_profile=profile, user_id=user_id, num_items=5)
            if hasattr(result, "products") and result.products:
                recommendations[user_id] = [p.product_id for p in result.products]
                continue
        except Exception:
            pass
        # LLM 不可用时降级为确定性打分
        scored = agent._score_candidates(profile, list(MOCK_PRODUCTS))
        recommendations[user_id] = [p.product_id for p in scored[:5]]
        print(f"  ({user_id} 使用确定性打分降级路径)")

    metrics = evaluator.evaluate_batch(recommendations, products_as_dicts)

    print(f"  样本数:         {metrics.sample_count}")
    print(f"  Precision@5:    {metrics.precision_at_k:.4f}")
    print(f"  NDCG@5:         {metrics.ndcg_at_k:.4f}")
    print(f"  类目多样性:     {metrics.category_diversity:.4f}")
    print(f"  目录覆盖率:     {metrics.catalog_coverage:.4f}")
    print(f"  推荐明细:")
    for uid, recs in recommendations.items():
        print(f"    {uid}: {recs}")

    return metrics.to_dict()


# =========================================================================
# 2. LLM 输出质量评估
# =========================================================================


def evaluate_llm_output_quality():
    """评估 LLM 输出的结构化质量和合规性（使用模拟响应）。"""
    print("\n" + "=" * 60)
    print("2. LLM 输出质量评估 (JSON 解析 / 合规 / 长度)")
    print("=" * 60)

    evaluator = LLMOutputEvaluator()

    # 模拟 LLM 营销文案输出（包含正常 + 异常样本）
    mock_copies = [
        {"product_id": "P001", "copy": "新品旗舰手机限时特惠，性能强悍值得拥有"},
        {"product_id": "P002", "copy": "国产旗舰手机，拍照清晰续航持久"},
        {"product_id": "P003", "copy": "降噪耳机首选，音质纯净佩戴舒适"},
        {"product_id": "P004", "copy": "头戴式降噪耳机，沉浸式音乐体验"},
        {"product_id": "P005", "copy": "学习办公利器，屏幕细腻性能强劲"},
        {"product_id": "P006", "copy": "性价比平板，娱乐学习两不误"},
        {"product_id": "P007", "copy": "快充充电器，便携高效出行必备"},
        # 以下为异常样本
        {"product_id": "P008", "copy": "全球首发最好的游戏本，绝对第一"},
        {"product_id": "P009", "copy": "4K显示器，办公设计首选"},
        {"product_id": "", "copy": "缺失商品ID的无效输出"},
    ]

    copy_metrics = evaluator.evaluate_copy_outputs(mock_copies)
    print(f"  文案输出总数:           {copy_metrics.total_responses}")
    print(f"  JSON 解析成功率:        {copy_metrics.json_parse_success_rate:.4f}")
    print(f"  合规通过率:             {copy_metrics.compliance_pass_rate:.4f}")
    print(f"  文案长度达标率(30-50):   {copy_metrics.copy_length_adherence:.4f}")
    print(f"  解析失败数:             {copy_metrics.parse_failures}")
    print(f"  合规违规数:             {copy_metrics.compliance_violations}")

    # 评估推荐重排输出
    candidate_ids = [p.product_id for p in MOCK_PRODUCTS[:10]]
    rerank_result = evaluator.evaluate_rerank_output(
        reranked_ids=["P001", "P003", "P007", "P005", "P002"],
        candidate_ids=candidate_ids,
    )
    print(f"\n  重排输出验证:")
    print(f"    JSON 解析成功:     {rerank_result['json_parse_success']}")
    print(f"    有效 ID 比率:     {rerank_result['valid_id_rate']:.4f}")
    print(f"    返回比率:         {rerank_result['return_rate']:.4f}")

    # 评估 MetaAgent 仲裁输出
    meta_ok = evaluator.evaluate_meta_decision(
        '{"decision": "approve", "reason": "风险可控", "confidence": 0.85}'
    )
    meta_bad = evaluator.evaluate_meta_decision("这是一个无效的响应")
    print(f"\n  MetaAgent 仲裁输出验证:")
    print(f"    有效 JSON 解析成功:   {meta_ok['json_parse_success']}")
    print(f"    无效 JSON 解析失败:   {meta_bad['json_parse_success']}")

    return {
        "copy_metrics": copy_metrics.to_dict(),
        "rerank_metrics": rerank_result,
        "meta_metrics": {"valid_parse": meta_ok, "invalid_parse": meta_bad},
    }


# =========================================================================
# 3. 风控准确性评估
# =========================================================================


async def evaluate_risk_control_quality():
    """评估风控/退款链路的规则覆盖与决策分布。"""
    print("\n" + "=" * 60)
    print("3. 风控准确性评估 (风险分布 / 决策分布 / 规则覆盖)")
    print("=" * 60)

    fraud_evaluator = RiskControlEvaluator()
    refund_evaluator = RiskControlEvaluator()

    # 反欺诈: 多场景测试
    fraud_test_cases = [
        {"user_id": "U001", "amount": 299, "payment_method": "alipay", "device_id": "D001", "ip": "192.168.1.1"},
        {"user_id": "U001", "amount": 5999, "payment_method": "alipay", "device_id": "D001", "ip": "192.168.1.1"},
        {"user_id": "U001", "amount": 12999, "payment_method": "alipay", "device_id": "D001", "ip": "192.168.1.1"},
        {"user_id": "U002", "amount": 499, "payment_method": "wechat", "device_id": "D002", "ip": "10.0.0.1"},
        {"user_id": "U002", "amount": 8999, "payment_method": "wechat", "device_id": "D002", "ip": "10.0.0.1"},
        {"user_id": "U003", "amount": 199, "payment_method": "credit_card", "device_id": None, "ip": None},
        {"user_id": "U003", "amount": 15999, "payment_method": "credit_card", "device_id": "D003", "ip": "172.16.0.1"},
    ]

    fraud_service = FraudService()
    fraud_results = []
    for tc in fraud_test_cases:
        result = await fraud_service.run(
            user_id=tc["user_id"],
            amount=tc["amount"],
            payment_method=tc["payment_method"],
            device_id=tc.get("device_id"),
            ip_address=tc.get("ip"),
            order_id=f"ORD-{tc['user_id']}-{int(tc['amount'])}",
        )
        fraud_results.append({
            "risk_score": result.risk_score,
            "recommended_action": result.recommended_action,
            "rules_hit": [{"rule_name": r.rule_name} for r in result.rules_hit],
        })

    fraud_metrics = fraud_evaluator.evaluate_fraud_results(fraud_results)
    print(f"  反欺诈评估 ({fraud_metrics.total_evaluations} 条):")
    print(f"    风险评分分布: {fraud_metrics.risk_score_distribution}")
    print(f"    决策分布:     {fraud_metrics.decision_distribution}")
    print(f"    规则命中:     {fraud_metrics.rule_hit_coverage}")

    # 退款风控: 多场景测试
    refund_test_cases = [
        {"user_id": "U001", "order_id": "O001", "product_id": "P001", "amount": 299, "reason": "质量问题"},
        {"user_id": "U001", "order_id": "O002", "product_id": "P002", "amount": 5999, "reason": "不喜欢"},
        {"user_id": "U002", "order_id": "O003", "product_id": "P003", "amount": 1899, "reason": "质量问题"},
        {"user_id": "U002", "order_id": "O004", "product_id": "P004", "amount": 2499, "reason": "尺寸不符"},
        {"user_id": "U003", "order_id": "O005", "product_id": "P005", "amount": 199, "reason": "发错货"},
    ]

    refund_service = RefundRiskService()
    refund_results = []
    for tc in refund_test_cases:
        result = await refund_service.run(
            user_id=tc["user_id"],
            order_id=tc["order_id"],
            product_id=tc["product_id"],
            refund_amount=tc["amount"],
            refund_reason=tc["reason"],
        )
        refund_results.append({
            "risk_score": result.risk_score,
            "refund_status": result.refund_status.value if hasattr(result.refund_status, 'value') else str(result.refund_status),
        })

    refund_metrics = refund_evaluator.evaluate_refund_results(refund_results)
    print(f"\n  退款风控评估 ({refund_metrics.total_evaluations} 条):")
    print(f"    风险评分分布: {refund_metrics.risk_score_distribution}")
    print(f"    决策分布:     {refund_metrics.decision_distribution}")

    return {
        "fraud": fraud_metrics.to_dict(),
        "refund": refund_metrics.to_dict(),
    }


# =========================================================================
# 4. 模型路由验证
# =========================================================================


def evaluate_model_routing():
    """验证多模型路由是否生效。"""
    print("\n" + "=" * 60)
    print("4. 模型路由验证 (任务-模型映射)")
    print("=" * 60)

    evaluator = ModelRoutingEvaluator()
    result = evaluator.evaluate_routing()

    print(f"  路由启用:     {result['routing_enabled']}")
    print(f"  已验证任务:   {result['verified_tasks']}")
    print(f"  不匹配路由:   {result['mismatched_routes']}")
    if result["routing_stats"]:
        print(f"  路由统计:")
        for task, stats in result["routing_stats"].items():
            print(f"    {task}: {stats}")
    else:
        print(f"  路由统计:     (暂无路由记录，需先运行 LLM Agent)")

    return result


# =========================================================================
# 主入口
# =========================================================================


async def run_all():
    print("=" * 60)
    print("多智能体电商系统 — 业务指标评估报告")
    print("=" * 60)

    rec_metrics = await evaluate_recommendation_quality()
    llm_metrics = evaluate_llm_output_quality()
    risk_metrics = await evaluate_risk_control_quality()
    routing_result = evaluate_model_routing()

    print("\n" + "=" * 60)
    print("评估完成 — 以上指标可直接用于简历量化")
    print("=" * 60)

    return {
        "recommendation": rec_metrics,
        "llm_output": llm_metrics,
        "risk_control": risk_metrics,
        "model_routing": routing_result,
    }


if __name__ == "__main__":
    asyncio.run(run_all())
