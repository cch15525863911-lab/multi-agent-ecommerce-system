"""
Meta-Agent — 跨领域协调决策 Agent (LLM Agent)。

这是项目3个 LLM Agent 之一, 负责灰度区间的模糊决策:
    1. 规则快通道 (确定性, <1ms): total_risk >= 100 → 拒绝, < 30 → 通过
    2. LLM 仲裁灰度通道 (30 <= risk < 100 或跨域信号冲突): LLM 权衡各维度风险后决策
    3. LLM 不可用时降级回规则引擎

典型场景:
    - 推荐链路 + 反欺诈链路: 推荐了高风险用户/商品 → 是否拦截
    - 履约链路 + 信用链路: 信用不足时降级方案
    - 退款链路 + 反欺诈链路: 双重高风险如何处理
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from config import get_settings
from llm import get_model_router
from models.schemas import MetaDecisionResult, UserIntent

logger = structlog.get_logger()


class MetaAgent:
    """跨域风险仲裁 LLM Agent — 规则快通道 + LLM 仲裁灰度双路径。"""

    RULE_REJECT_THRESHOLD = 100.0
    RULE_APPROVE_THRESHOLD = 30.0

    ARBITRATION_PROMPT = """你是电商分期购场景的跨域风险仲裁 Agent。
多个风控 Agent 已完成各自领域的评估，请你综合所有结果做出最终决策。

## 各 Agent 评估结果
{agent_findings}

## 综合风险评分
{risk_summary}

## 决策指南
- approve: 综合判断风险可控，可正常执行
- escalate: 存在不确定因素，需人工审核确认
- reject: 综合风险过高，应直接拒绝

## 输出要求
仅返回 JSON，不要包含其他文字：
{{"decision": "approve|escalate|reject", "reason": "一句话说明决策理由", "confidence": 0.0-1.0}}"""

    def __init__(self, llm: Any | None = None):
        settings = get_settings()
        self.llm = llm or get_model_router().create_llm(
            task_type="meta_decision", temperature=0.1, max_tokens=512
        )

    async def decide(
        self,
        intent: UserIntent,
        agent_results: dict[str, Any],
    ) -> MetaDecisionResult:
        """双路径决策: 规则快通道 → LLM 仲裁灰度 → 规则降级。"""
        start = time.perf_counter()

        risks = self._collect_risk_signals(agent_results)
        total_risk = sum(risks.values())

        fraud = agent_results.get("fraud_detection")
        refund = agent_results.get("refund_risk")
        credit = agent_results.get("credit_assessment")

        if intent == UserIntent.FULFILLMENT and credit:
            decision, reason, escalate = self._fulfillment_credit_rule(credit)
            return MetaDecisionResult(
                agent_name="meta_agent",
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000,
                final_decision=decision,
                decision_reason=reason,
                aggregated_risks=risks,
                escalation_required=escalate,
                confidence=0.9 if not escalate else 0.75,
                arbitration_source="rule",
            )

        if total_risk >= self.RULE_REJECT_THRESHOLD:
            return MetaDecisionResult(
                agent_name="meta_agent",
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000,
                final_decision="reject",
                decision_reason="综合风险过高，系统自动拒绝",
                aggregated_risks=risks,
                escalation_required=True,
                confidence=0.95,
                arbitration_source="rule_fast_reject",
            )
        if total_risk < self.RULE_APPROVE_THRESHOLD:
            return MetaDecisionResult(
                agent_name="meta_agent",
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000,
                final_decision="approve",
                decision_reason="各维度检查通过，正常执行",
                aggregated_risks=risks,
                escalation_required=False,
                confidence=0.9,
                arbitration_source="rule_fast_approve",
            )

        recommendation = agent_results.get("recommendation_graph")
        if (
            intent == UserIntent.RECOMMENDATION
            and recommendation
            and fraud
            and fraud.risk_score > 40
        ):
            products = recommendation.get("products", [])
            max_price = max((p.get("price", 0) for p in products), default=0)
            if max_price > 5000 and fraud.risk_score > 50:
                try:
                    decision = await self._llm_arbitrate(intent, agent_results, risks, total_risk)
                    decision.latency_ms = (time.perf_counter() - start) * 1000
                    return decision
                except Exception as exc:
                    logger.warning("meta_agent.recommendation_conflict_arbitration_failed", error=str(exc))

        if (
            intent == UserIntent.REFUND_REVIEW
            and risks.get("refund", 0) >= 30
            and risks.get("fraud", 0) >= 30
        ):
            try:
                decision = await self._llm_arbitrate(intent, agent_results, risks, total_risk)
                decision.latency_ms = (time.perf_counter() - start) * 1000
                return decision
            except Exception as exc:
                logger.warning("meta_agent.refund_fraud_conflict_arbitration_failed", error=str(exc))

        has_conflict = self._has_cross_domain_conflict(risks, agent_results)
        if has_conflict or self.RULE_APPROVE_THRESHOLD <= total_risk < self.RULE_REJECT_THRESHOLD:
            try:
                decision = await self._llm_arbitrate(intent, agent_results, risks, total_risk)
                decision.latency_ms = (time.perf_counter() - start) * 1000
                return decision
            except Exception as exc:
                logger.warning("meta_agent.llm_arbitration_failed", error=str(exc))

        return self._rule_fallback(intent, risks, total_risk, start)

    def _collect_risk_signals(
        self, agent_results: dict[str, Any]
    ) -> dict[str, float]:
        risks: dict[str, float] = {}
        fraud = agent_results.get("fraud_detection")
        if fraud and hasattr(fraud, "risk_score"):
            risks["fraud"] = fraud.risk_score
        refund = agent_results.get("refund_risk")
        if refund and hasattr(refund, "risk_score"):
            risks["refund"] = refund.risk_score
        credit = agent_results.get("credit_assessment")
        if credit and hasattr(credit, "credit_score"):
            risks["credit"] = max(0.0, (900 - credit.credit_score) / 6)
        return risks

    def _has_cross_domain_conflict(
        self, risks: dict[str, float], agent_results: dict[str, Any]
    ) -> bool:
        if len(risks) < 2:
            return False
        scores = list(risks.values())
        return max(scores) >= 60 and min(scores) <= 20

    @staticmethod
    def _fulfillment_credit_rule(credit: Any) -> tuple[str, str, bool]:
        if credit.approved and credit.approved_amount > 0:
            return (
                "approve",
                f"信用支付可用，批准额度{credit.approved_amount}元",
                False,
            )
        if not credit.approved:
            return (
                "escalate",
                f"信用支付不可用: {credit.data.get('reason', '额度不足')}",
                True,
            )
        return "escalate", "信用评估结果不确定，需人工确认", True

    async def _llm_arbitrate(
        self,
        intent: UserIntent,
        agent_results: dict[str, Any],
        risks: dict[str, float],
        total_risk: float,
    ) -> MetaDecisionResult:
        findings = self._build_agent_findings(agent_results)
        risk_summary = self._build_risk_summary(risks, total_risk)

        prompt = self.ARBITRATION_PROMPT.format(
            agent_findings=findings,
            risk_summary=risk_summary,
        )

        response = await self.llm.ainvoke(prompt)
        content = getattr(response, "content", str(response))

        parsed = self._parse_llm_decision(content)

        return MetaDecisionResult(
            agent_name="meta_agent",
            success=True,
            latency_ms=0,
            final_decision=parsed["decision"],
            decision_reason=parsed["reason"],
            aggregated_risks=risks,
            escalation_required=parsed["decision"] in ("escalate", "reject"),
            confidence=parsed["confidence"],
            arbitration_source="llm_arbitration",
        )

    @staticmethod
    def _build_agent_findings(agent_results: dict[str, Any]) -> str:
        lines: list[str] = []

        fraud = agent_results.get("fraud_detection")
        if fraud and hasattr(fraud, "risk_score"):
            lines.append(
                f"- 反欺诈Service: 风险评分={fraud.risk_score}, "
                f"风险等级={getattr(fraud, 'risk_level', 'unknown')}, "
                f"建议动作={getattr(fraud, 'recommended_action', 'unknown')}, "
                f"命中规则={[r.get('rule_name', '') for r in getattr(fraud, 'rules_hit', []) if isinstance(r, dict)]}"
            )

        credit = agent_results.get("credit_assessment")
        if credit and hasattr(credit, "credit_score"):
            lines.append(
                f"- 信用评估Service: 信用分={credit.credit_score}, "
                f"是否授信通过={getattr(credit, 'approved', False)}, "
                f"授信额度={getattr(credit, 'approved_amount', 0)}"
            )

        refund = agent_results.get("refund_risk")
        if refund and hasattr(refund, "risk_score"):
            lines.append(
                f"- 退款风控Service: 风险评分={refund.risk_score}, "
                f"建议动作={getattr(refund, 'recommended_action', 'unknown')}"
            )

        supply = agent_results.get("supply_chain")
        if supply:
            success = getattr(supply, "success", None)
            if success is not None:
                lines.append(f"- 履约Service: 执行{'成功' if success else '失败'}")

        return "\n".join(lines) if lines else "- 无可用 Service 结果"

    @staticmethod
    def _build_risk_summary(risks: dict[str, float], total_risk: float) -> str:
        lines = [f"综合风险总分: {total_risk:.1f}"]
        for domain, score in risks.items():
            level = "高" if score >= 60 else "中" if score >= 30 else "低"
            lines.append(f"- {domain}: {score:.1f} ({level})")
        return "\n".join(lines)

    @staticmethod
    def _parse_llm_decision(content: str) -> dict[str, Any]:
        import json as _json

        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            if text.startswith("json"):
                text = text[4:]

        data = _json.loads(text)
        decision = data.get("decision", "escalate")
        if decision not in ("approve", "reject", "escalate"):
            decision = "escalate"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "decision": decision,
            "reason": data.get("reason", "LLM 仲裁决策"),
            "confidence": confidence,
        }

    def _rule_fallback(
        self,
        intent: UserIntent,
        risks: dict[str, float],
        total_risk: float,
        start: float,
    ) -> MetaDecisionResult:
        if total_risk >= 60:
            decision, reason, escalate = "escalate", "存在一定风险，需人工审核确认", True
        else:
            decision, reason, escalate = "approve", "各维度检查通过，正常执行", False

        return MetaDecisionResult(
            agent_name="meta_agent",
            success=True,
            latency_ms=(time.perf_counter() - start) * 1000,
            final_decision=decision,
            decision_reason=reason,
            aggregated_risks=risks,
            escalation_required=escalate,
            confidence=0.75,
            arbitration_source="rule_fallback",
        )
