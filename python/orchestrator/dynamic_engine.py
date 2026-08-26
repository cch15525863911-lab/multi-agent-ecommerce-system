"""
LangGraph Dynamic Engine — 意图路由 + 多领域 Service/Agent 集群 + Meta-Agent 协调决策。

整体架构 (替代单一推荐链路, 支持多场景动态编排):

    START → intent_router (意图识别)
        ├→ recommendation_branch (推荐链路: 画像|召回 → 重排|库存 → 文案)
        ├→ fraud_branch (反欺诈链路: 规则引擎)
        ├→ credit_branch (授信链路: 评分卡)
        ├→ refund_branch (退款链路: 规则引擎)
        └→ fulfillment_precheck (履约前风控预检 → 门控)
        ↓ (fan-in)
    meta_agent (跨领域协调决策, 多Service结果冲突时LLM仲裁)
        ↓
    END

架构说明:
    - 3 个 LLM Agent: ProductRecAgent, MarketingCopyAgent, MetaAgent
    - 传统业务 Service: FraudService, CreditService, RefundRiskService,
                       FulfillmentService, ProfileService, InventoryService
    - LLM 负责语义理解/创意生成/灰度仲裁, 传统系统负责确定性计算
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

import structlog
from langgraph.graph import END, StateGraph

from agents import (
    MarketingCopyAgent,
    MetaAgent,
    ProductRecAgent,
)
from config import get_settings
from llm import get_model_router
from models.schemas import (
    CreditAssessmentResult,
    FraudCheckResult,
    IntentRouteResult,
    MetaDecisionResult,
    RefundRiskResult,
    UserIntent,
)
from services.ab_test import ABTestEngine
from services.credit_service import CreditService
from services.fraud_service import FraudService
from services.fulfillment_service import FulfillmentService
from services.kg_store import KGStore
from services.profile_service import ProfileService
from services.refund_service import RefundRiskService

logger = structlog.get_logger()


# =========================================================================
# State — 统一动态引擎状态
# =========================================================================


def _merge_agent_results(
    old: dict[str, Any] | None, new: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(old or {})
    merged.update(new or {})
    return merged


class DynamicState(dict):
    """Dynamic engine state — 灵活承载各场景数据。"""
    pass


# =========================================================================
# Node 1: Intent Router — 意图识别与路由
# =========================================================================

INTENT_KEYWORDS = {
    UserIntent.RECOMMENDATION: ["推荐", "猜你喜欢", "商品", "买什么", "推荐一下", "recommend"],
    UserIntent.FRAUD_CHECK: ["欺诈", "风险", "风控", "审核", "fraud", "risk"],
    UserIntent.CREDIT_ASSESSMENT: ["授信", "贷款", "借钱", "额度", "信用", "credit", "loan"],
    UserIntent.REFUND_REVIEW: ["退款", "退货", "售后", "refund", "return"],
    UserIntent.FULFILLMENT: ["下单", "购买", "履约", "发货", "order", "buy", "fulfill"],
}


class IntentRouter:
    """意图路由器 — 规则匹配优先, LLM兜底。"""

    def __init__(self, llm: Any | None = None):
        settings = get_settings()
        self.llm = llm or get_model_router().create_llm(
            task_type="intent_routing", temperature=0.0, max_tokens=256
        )
        self.use_llm = getattr(settings, "intent_router_use_llm", False)

    def route_by_rules(self, query: str, intent_hint: UserIntent | None = None) -> tuple[UserIntent, float]:
        """基于关键词规则快速路由。"""
        if intent_hint and intent_hint != UserIntent.UNKNOWN:
            return intent_hint, 1.0

        query_lower = query.lower()
        best_intent = UserIntent.UNKNOWN
        best_score = 0.0

        for intent, keywords in INTENT_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw.lower() in query_lower)
            if matches > best_score:
                best_score = float(matches)
                best_intent = intent

        if best_score > 0:
            confidence = min(0.9, 0.5 + best_score * 0.15)
            return best_intent, confidence
        return UserIntent.UNKNOWN, 0.0

    async def route_by_llm(self, query: str) -> tuple[UserIntent, float]:
        """LLM 意图识别 (兜底, 当规则匹配不上时使用)。"""
        intents_list = [e.value for e in UserIntent]
        prompt = f"""请从以下意图类别中选择最符合用户查询的一个:
类别: {intents_list}
用户查询: "{query}"

只返回JSON格式: {{"intent": "类别名", "confidence": 0.0-1.0}}"""
        try:
            response = await self.llm.ainvoke(prompt)
            content = getattr(response, "content", str(response))
            import json
            data = json.loads(content)
            intent_str = data.get("intent", "unknown")
            confidence = float(data.get("confidence", 0.5))
            for e in UserIntent:
                if e.value == intent_str:
                    return e, confidence
        except Exception as exc:
            logger.warning("intent.llm_route_failed", error=str(exc))
        return UserIntent.UNKNOWN, 0.0

    async def route(self, query: str, intent_hint: UserIntent | None = None) -> IntentRouteResult:
        """完整路由流程: 规则优先, 失败则LLM兜底。"""
        start = time.perf_counter()
        intent, confidence = self.route_by_rules(query, intent_hint)

        if intent == UserIntent.UNKNOWN and self.use_llm and query:
            intent, confidence = await self.route_by_llm(query)

        path_map = {
            UserIntent.RECOMMENDATION: ["recommendation_branch"],
            UserIntent.FRAUD_CHECK: ["fraud_branch"],
            UserIntent.CREDIT_ASSESSMENT: ["credit_branch"],
            UserIntent.REFUND_REVIEW: ["refund_branch"],
            UserIntent.FULFILLMENT: ["fulfillment_precheck"],
            UserIntent.UNKNOWN: ["recommendation_branch"],
        }
        routing_path = path_map.get(intent, ["recommendation_branch"])

        return IntentRouteResult(
            agent_name="intent_router",
            success=True,
            latency_ms=(time.perf_counter() - start) * 1000,
            detected_intent=intent,
            confidence=confidence,
            routing_path=routing_path,
        )


# =========================================================================
# MetaAgent — 已迁移至 agents/meta_agent.py (LLM Agent)
# =========================================================================


# =========================================================================
# Branch Nodes — 各领域链路
# =========================================================================


def _init_components() -> tuple[
    ProfileService,
    ProductRecAgent,
    MarketingCopyAgent,
    FulfillmentService,
    FraudService,
    CreditService,
    RefundRiskService,
    KGStore | None,
]:
    """初始化所有 Service/Agent 实例 (模块加载时执行一次)。

    LLM Agents: ProductRecAgent, MarketingCopyAgent, MetaAgent
    传统 Services: FraudService, CreditService, RefundRiskService, FulfillmentService, ProfileService
    """
    kg_store = KGStore()
    profile_service = ProfileService(kg_store=kg_store)
    product_rec_agent = ProductRecAgent(kg_store=kg_store)
    marketing_copy_agent = MarketingCopyAgent()
    fulfillment_service = FulfillmentService()
    fraud_service = FraudService()
    credit_service = CreditService()
    refund_service = RefundRiskService()
    return (
        profile_service,
        product_rec_agent,
        marketing_copy_agent,
        fulfillment_service,
        fraud_service,
        credit_service,
        refund_service,
        kg_store,
    )


(_profile_service, _product_rec_agent, _marketing_copy_agent,
 _fulfillment_service, _fraud_service, _credit_service, _refund_service,
 _kg_store) = _init_components()

_intent_router = IntentRouter()
_meta_agent = MetaAgent()
_ab_engine = ABTestEngine()


# ---- 推荐链路 ----

async def recommendation_branch(state: DynamicState) -> dict[str, Any]:
    """推荐链路: 简化版 (三阶段并行)。"""
    from orchestrator.graph import build_recommendation_graph
    graph = build_recommendation_graph()
    result = await graph.ainvoke({
        "user_id": state.get("user_id", ""),
        "scene": state.get("scene", "homepage"),
        "num_items": state.get("num_items", 10),
        "context": state.get("context", {}),
    })
    return {
        "products": result.get("final_products", []),
        "marketing_copies": result.get("marketing_copies", []),
        "experiment_group": result.get("experiment_group", "control"),
        "agent_results": {
            "recommendation_graph": {"success": True, "latency_ms": result.get("total_latency_ms", 0)},
        },
    }


# ---- 反欺诈链路 ----

async def fraud_branch(state: DynamicState) -> dict[str, Any]:
    """反欺诈检测链路 (确定性规则引擎)。"""
    result = await _fraud_service.run(
        user_id=state.get("user_id", ""),
        amount=state.get("amount", 0.0),
        payment_method=state.get("payment_method", "alipay"),
        device_id=state.get("device_id"),
        ip_address=state.get("ip_address"),
        order_id=state.get("order_id"),
    )
    return {
        "fraud_result": result,
        "agent_results": {"fraud_detection": result},
    }


# ---- 信用授信链路 ----

async def credit_branch(state: DynamicState) -> dict[str, Any]:
    """信用授信链路 (确定性评分卡)。"""
    result = await _credit_service.run(
        user_id=state.get("user_id", ""),
        requested_amount=state.get("requested_amount", state.get("amount", 0.0)),
        order_id=state.get("order_id"),
    )
    return {
        "credit_result": result,
        "agent_results": {"credit_assessment": result},
    }


# ---- 退款风控链路 ----

async def refund_branch(state: DynamicState) -> dict[str, Any]:
    """退款风控链路 (确定性规则引擎)。"""
    result = await _refund_service.run(
        user_id=state.get("user_id", ""),
        order_id=state.get("order_id", ""),
        product_id=state.get("product_id", ""),
        refund_amount=state.get("refund_amount", state.get("amount", 0.0)),
        refund_reason=state.get("refund_reason", ""),
    )
    return {
        "refund_result": result,
        "agent_results": {"refund_risk": result},
    }


# ---- 履约链路 ----

async def fulfillment_branch(state: DynamicState) -> dict[str, Any]:
    """供应链履约链路 (Saga事务编排)。"""
    from models.schemas import FulfillmentRequest
    product = state.get("product")
    if product is None:
        return {
            "agent_results": {
                "supply_chain": {"success": False, "error": "no product provided", "latency_ms": 0}
            }
        }
    request = FulfillmentRequest(
        user_id=state.get("user_id", ""),
        product=product,
        quantity=state.get("context", {}).get("quantity", 1),
        destination=state.get("context", {}).get("destination", "北京"),
    )
    result = await _fulfillment_service.run(request=request)
    return {
        "order": getattr(result, "order", None),
        "agent_results": {"supply_chain": result},
    }


# ---- 履约前风控预检 + 门控 (修复 P1: Meta 决策必须真正拦截履约写操作) ----


async def fulfillment_precheck(state: DynamicState) -> dict[str, Any]:
    """履约前风控预检: 先反欺诈 + 信用评估, 再经 Meta-Agent 仲裁。

    仅当 final_decision == "approve" 时才允许进入真实履约写操作;
    reject / escalate 仅生成"待人工确认预订单", 不落真实订单、不占库存。
    """
    fraud_result = await _fraud_service.run(
        user_id=state.get("user_id", ""),
        amount=state.get("amount", 0.0),
        payment_method=state.get("payment_method", "alipay"),
        device_id=state.get("device_id"),
        ip_address=state.get("ip_address"),
        order_id=state.get("order_id"),
    )
    credit_result = await _credit_service.run(
        user_id=state.get("user_id", ""),
        requested_amount=state.get("requested_amount", state.get("amount", 0.0)),
        order_id=state.get("order_id"),
    )
    agent_results = {
        "fraud_detection": fraud_result,
        "credit_assessment": credit_result,
    }
    decision = await _meta_agent.decide(UserIntent.FULFILLMENT, agent_results)
    approved = decision.final_decision == "approve"
    return {
        "fraud_result": fraud_result,
        "credit_result": credit_result,
        "agent_results": agent_results,
        "meta_decision": decision,
        "_fulfillment_approved": approved,
        "_fulfillment_reason": decision.decision_reason,
    }


async def fulfillment_execute(state: DynamicState) -> dict[str, Any]:
    """Meta 批准后, 执行真实履约写操作 (创建订单 + 占用库存)。"""
    result = await fulfillment_branch(state)
    # 保留预检阶段的 fraud/credit 结果, 不覆盖
    merged = dict(state.get("agent_results", {}))
    merged.update(result.get("agent_results", {}))
    return {**result, "agent_results": merged}


async def fulfillment_pending(state: DynamicState) -> dict[str, Any]:
    """Meta 拒绝 / 升级: 仅生成"待人工确认预订单", 不落真实订单、不占库存。"""
    reason = state.get("_fulfillment_reason", "风险未通过自动审核, 转人工确认")
    decision = state.get("meta_decision")
    merged = dict(state.get("agent_results", {}))
    merged["supply_chain"] = {
        "success": False,
        "skipped": True,
        "reason": reason,
        "latency_ms": 0,
    }
    return {
        "order": None,
        "pending_order": {
            "status": "pending_human_review",
            "needs_human_review": True,
            "meta_decision": decision.final_decision if decision else "escalate",
            "reason": reason,
        },
        "agent_results": merged,
    }


def route_fulfillment(state: DynamicState) -> str:
    """履约预检后的条件边: 批准后走真实下单, 否则走待人工预订单。"""
    return "fulfillment_execute" if state.get("_fulfillment_approved") else "fulfillment_pending"


# =========================================================================
# Router + Meta Nodes (Graph entry)
# =========================================================================


async def intent_router_node(state: DynamicState) -> dict[str, Any]:
    """入口节点: 意图识别。"""
    intent_hint = state.get("intent")
    query = state.get("query", "")
    route_result = await _intent_router.route(query, intent_hint)
    return {
        "request_id": str(uuid.uuid4()),
        "_start_time": time.perf_counter(),
        "intent": route_result.detected_intent,
        "agent_results": {"intent_router": route_result},
    }


def route_by_intent(state: DynamicState) -> str:
    """条件边: 根据意图路由到对应 branch。"""
    intent = state.get("intent", UserIntent.UNKNOWN)
    route_map = {
        UserIntent.RECOMMENDATION: "recommendation_branch",
        UserIntent.FRAUD_CHECK: "fraud_branch",
        UserIntent.CREDIT_ASSESSMENT: "credit_branch",
        UserIntent.REFUND_REVIEW: "refund_branch",
        UserIntent.FULFILLMENT: "fulfillment_precheck",
        UserIntent.UNKNOWN: "recommendation_branch",
    }
    return route_map.get(intent, "recommendation_branch")


async def meta_agent_node(state: DynamicState) -> dict[str, Any]:
    """Meta-Agent: 汇总各链路结果, 做最终决策。"""
    intent = state.get("intent", UserIntent.UNKNOWN)
    agent_results = state.get("agent_results", {})
    decision = await _meta_agent.decide(intent, agent_results)
    total_ms = (time.perf_counter() - state.get("_start_time", time.perf_counter())) * 1000
    return {
        "meta_decision": decision,
        "total_latency_ms": round(total_ms, 1),
    }


# =========================================================================
# Graph Builder — Dynamic Engine
# =========================================================================


def build_dynamic_engine() -> StateGraph:
    """构建动态编排引擎 — 意图路由 → 领域分支 → Meta-Agent 决策。

    拓扑图:
        START → init (intent_router)
          │
          ├─[recommendation]──→ recommendation_branch ──┐
          ├─[fraud_check]──────→ fraud_branch ──────────┤
          ├─[credit_assessment]→ credit_branch ─────────┤
          ├─[refund_review]────→ refund_branch ─────────┤
          └─[fulfillment]──────→ fulfillment_precheck ─┐
                                  │ (预检: 先风控/信用, Meta 决策)
                                  ▼
                          ┌── approve → fulfillment_execute (真实下单)
                          └── reject/escalate → fulfillment_pending (待人工)
                                  │
                                  ▼
                                 END
    """
    graph = StateGraph(DynamicState)

    graph.add_node("init", intent_router_node)
    graph.add_node("recommendation_branch", recommendation_branch)
    graph.add_node("fraud_branch", fraud_branch)
    graph.add_node("credit_branch", credit_branch)
    graph.add_node("refund_branch", refund_branch)
    graph.add_node("fulfillment_precheck", fulfillment_precheck)
    graph.add_node("fulfillment_execute", fulfillment_execute)
    graph.add_node("fulfillment_pending", fulfillment_pending)
    graph.add_node("meta_agent", meta_agent_node)

    graph.set_entry_point("init")

    graph.add_conditional_edges(
        "init",
        route_by_intent,
        {
            "recommendation_branch": "recommendation_branch",
            "fraud_branch": "fraud_branch",
            "credit_branch": "credit_branch",
            "refund_branch": "refund_branch",
            "fulfillment_precheck": "fulfillment_precheck",
        },
    )

    graph.add_edge("recommendation_branch", "meta_agent")
    graph.add_edge("fraud_branch", "meta_agent")
    graph.add_edge("credit_branch", "meta_agent")
    graph.add_edge("refund_branch", "meta_agent")

    # 履约路径: 预检 → 条件边 (Meta 批准才真实下单, 否则待人工预订单) → END
    graph.add_conditional_edges(
        "fulfillment_precheck",
        route_fulfillment,
        {
            "fulfillment_execute": "fulfillment_execute",
            "fulfillment_pending": "fulfillment_pending",
        },
    )
    graph.add_edge("fulfillment_execute", END)
    graph.add_edge("fulfillment_pending", END)

    graph.add_edge("meta_agent", END)

    return graph.compile()
