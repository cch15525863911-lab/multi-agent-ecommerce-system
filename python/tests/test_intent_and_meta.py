"""
意图路由 + Meta-Agent 单元测试。

覆盖:
    - IntentRouter: 关键词规则匹配各意图
    - IntentRouter: 指定意图 hint 时直接返回
    - IntentRouter: 未知意图默认走推荐
    - MetaAgent: 低风险 → approve
    - MetaAgent: 高风险 → reject/escalate
    - MetaAgent: 信用+履约场景决策
"""
from __future__ import annotations

import pytest

from models.schemas import (
    CreditAssessmentResult,
    CreditStatus,
    FraudCheckResult,
    FraudRiskLevel,
    UserIntent,
)
from orchestrator.dynamic_engine import IntentRouter
from agents import MetaAgent


# =========================================================================
# Intent Router 测试
# =========================================================================


class TestIntentRouter:
    def setup_method(self):
        self.router = IntentRouter()
        self.router.use_llm = False  # 测试用规则模式, 不依赖LLM

    def test_recommendation_intent_by_keywords(self):
        """推荐相关关键词应识别为推荐意图。"""
        result = self.router.route_by_rules("给我推荐一下手机", None)
        intent, confidence = result
        assert intent == UserIntent.RECOMMENDATION
        assert confidence > 0.5

    def test_fraud_intent_by_keywords(self):
        """风控相关关键词应识别为反欺诈意图。"""
        result = self.router.route_by_rules("这笔交易有风险吗", None)
        intent, confidence = result
        assert intent == UserIntent.FRAUD_CHECK
        assert confidence > 0.5

    def test_credit_intent_by_keywords(self):
        """贷款/授信关键词应识别为信用意图。"""
        result = self.router.route_by_rules("我想借钱买东西", None)
        intent, confidence = result
        assert intent == UserIntent.CREDIT_ASSESSMENT

    def test_refund_intent_by_keywords(self):
        """退款关键词应识别为退款意图。"""
        result = self.router.route_by_rules("我要申请退货退款", None)
        intent, confidence = result
        assert intent == UserIntent.REFUND_REVIEW

    def test_fulfillment_intent_by_keywords(self):
        """下单关键词应识别为履约意图。"""
        result = self.router.route_by_rules("我要下单购买", None)
        intent, confidence = result
        assert intent == UserIntent.FULFILLMENT

    def test_explicit_intent_hint(self):
        """显式指定意图时应直接返回。"""
        result = self.router.route_by_rules("", UserIntent.CREDIT_ASSESSMENT)
        intent, confidence = result
        assert intent == UserIntent.CREDIT_ASSESSMENT
        assert confidence == 1.0

    def test_unknown_intent_default(self):
        """无法识别时应返回 UNKNOWN。"""
        result = self.router.route_by_rules("今天天气怎么样", None)
        intent, confidence = result
        assert intent == UserIntent.UNKNOWN

    @pytest.mark.asyncio
    async def test_route_method_with_hint(self):
        """完整 route 方法在有 hint 时应正确返回 IntentRouteResult。"""
        result = await self.router.route("", UserIntent.FRAUD_CHECK)
        assert result.detected_intent == UserIntent.FRAUD_CHECK
        assert result.confidence == 1.0
        assert "fraud_branch" in result.routing_path

    @pytest.mark.asyncio
    async def test_route_recommendation_path(self):
        """推荐意图的路由路径应包含 recommendation_branch。"""
        result = await self.router.route("推荐商品", None)
        assert "recommendation_branch" in result.routing_path


# =========================================================================
# Meta-Agent 测试
# =========================================================================


class TestMetaAgent:
    def setup_method(self):
        self.meta = MetaAgent()

    @pytest.mark.asyncio
    async def test_low_risk_approve(self):
        """低风险场景应批准。"""
        result = await self.meta.decide(UserIntent.FRAUD_CHECK, {})
        assert result.final_decision == "approve"
        assert result.escalation_required is False

    @pytest.mark.asyncio
    async def test_high_fraud_risk_reject(self):
        """高欺诈风险应升级/拒绝。"""
        fraud = FraudCheckResult(
            agent_name="fraud_detection",
            risk_level=FraudRiskLevel.CRITICAL,
            risk_score=90.0,
            recommended_action="block",
        )
        result = await self.meta.decide(
            UserIntent.FRAUD_CHECK,
            {"fraud_detection": fraud},
        )
        assert result.final_decision in ("reject", "escalate")
        assert result.escalation_required is True
        assert result.aggregated_risks.get("fraud", 0) >= 80

    @pytest.mark.asyncio
    async def test_credit_approved_fulfillment(self):
        """履约+信用批准场景应通过。"""
        credit = CreditAssessmentResult(
            agent_name="credit_assessment",
            credit_score=780,
            credit_limit=50000,
            available_limit=49000,
            credit_status=CreditStatus.ACTIVE,
            approved=True,
            approved_amount=1000.0,
        )
        result = await self.meta.decide(
            UserIntent.FULFILLMENT,
            {"credit_assessment": credit},
        )
        assert result.final_decision == "approve"
        assert "信用支付可用" in result.decision_reason

    @pytest.mark.asyncio
    async def test_credit_rejected_fulfillment(self):
        """履约+信用拒绝场景应升级。"""
        credit = CreditAssessmentResult(
            agent_name="credit_assessment",
            credit_score=400,
            credit_limit=0,
            available_limit=0,
            credit_status=CreditStatus.FROZEN,
            approved=False,
        )
        result = await self.meta.decide(
            UserIntent.FULFILLMENT,
            {"credit_assessment": credit},
        )
        assert result.final_decision == "escalate"
        assert result.escalation_required is True

    @pytest.mark.asyncio
    async def test_multiple_risk_aggregation(self):
        """多重风险应累加。"""
        fraud = FraudCheckResult(
            agent_name="fraud_detection",
            risk_level=FraudRiskLevel.HIGH,
            risk_score=60.0,
        )
        credit = CreditAssessmentResult(
            agent_name="credit_assessment",
            credit_score=500,
            credit_status=CreditStatus.FROZEN,
            approved=False,
        )
        result = await self.meta.decide(
            UserIntent.FULFILLMENT,
            {"fraud_detection": fraud, "credit_assessment": credit},
        )
        assert len(result.aggregated_risks) >= 2
        # 信用分500 → 风险分约 (900-500)/6 ≈ 66.7, 加上 fraud 60 → 总风险>100
        assert result.final_decision in ("reject", "escalate")

    @pytest.mark.asyncio
    async def test_gray_zone_llm_arbitration(self):
        """灰度区间(30≤risk<100)应调用LLM仲裁。"""
        from unittest.mock import AsyncMock, MagicMock

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content='{"decision": "escalate", "reason": "欺诈风险中等但信用良好，建议人工确认", "confidence": 0.8}'
            )
        )
        meta = MetaAgent(llm=mock_llm)

        fraud = FraudCheckResult(
            agent_name="fraud_detection",
            risk_level=FraudRiskLevel.MEDIUM,
            risk_score=45.0,
        )
        result = await meta.decide(
            UserIntent.FRAUD_CHECK,
            {"fraud_detection": fraud},
        )
        assert result.final_decision == "escalate"
        assert result.arbitration_source == "llm_arbitration"
        assert result.confidence == 0.8
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_gray_zone_fallback_when_llm_fails(self):
        """LLM不可用时灰度区间应降级到规则引擎。"""
        from unittest.mock import AsyncMock, MagicMock

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))
        meta = MetaAgent(llm=mock_llm)

        fraud = FraudCheckResult(
            agent_name="fraud_detection",
            risk_level=FraudRiskLevel.MEDIUM,
            risk_score=45.0,
        )
        result = await meta.decide(
            UserIntent.FRAUD_CHECK,
            {"fraud_detection": fraud},
        )
        assert result.final_decision == "approve"
        assert result.arbitration_source == "rule_fallback"
        assert result.confidence == 0.75

    @pytest.mark.asyncio
    async def test_cross_domain_conflict_triggers_llm(self):
        """跨域信号冲突(如欺诈高但信用良好)应触发LLM仲裁。"""
        from unittest.mock import AsyncMock, MagicMock

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content='{"decision": "approve", "reason": "欺诈风险偏高但信用记录良好，允许交易但标记关注", "confidence": 0.7}'
            )
        )
        meta = MetaAgent(llm=mock_llm)

        fraud = FraudCheckResult(
            agent_name="fraud_detection",
            risk_level=FraudRiskLevel.HIGH,
            risk_score=65.0,
        )
        credit = CreditAssessmentResult(
            agent_name="credit_assessment",
            credit_score=850,
            credit_status=CreditStatus.ACTIVE,
            approved=True,
            approved_amount=5000.0,
        )
        result = await meta.decide(
            UserIntent.CREDIT_ASSESSMENT,
            {"fraud_detection": fraud, "credit_assessment": credit},
        )
        assert result.arbitration_source == "llm_arbitration"
        assert result.final_decision == "approve"
        mock_llm.ainvoke.assert_called_once()


# =========================================================================
# 履约门控 (Meta 决策拦截履约写操作)
# =========================================================================


class TestFulfillmentGating:
    def test_route_by_intent_fulfillment_precheck(self):
        """履约意图应路由到预检节点 (先风控/信用, 再由 Meta 决策)。"""
        from orchestrator.dynamic_engine import route_by_intent

        state = {"intent": UserIntent.FULFILLMENT}
        assert route_by_intent(state) == "fulfillment_precheck"

    def test_route_fulfillment_conditional(self):
        """预检后: 批准走真实下单, 拒绝/升级走待人工预订单。"""
        from orchestrator.dynamic_engine import route_fulfillment

        assert (
            route_fulfillment({"_fulfillment_approved": True}) == "fulfillment_execute"
        )
        assert (
            route_fulfillment({"_fulfillment_approved": False})
            == "fulfillment_pending"
        )

    def test_dynamic_engine_compiles(self):
        """重构后的动态引擎应能正常编译 (拓扑有效)。"""
        from orchestrator.dynamic_engine import build_dynamic_engine

        graph = build_dynamic_engine()
        assert graph is not None
