"""
风控业务工具集 — 反欺诈 / 信用授信 / 售后退款风控。

提供三类核心工具:
    1. 实时反欺诈 (fraud detection): 设备/IP/行为/名单规则引擎
    2. 信用授信 (credit assessment): 信用评分 / 额度管理 / 授信决策
    3. 售后退款风控 (refund risk): 退款审核 / 极速退款资格 / 恶意退款识别

所有工具均为纯函数实现, 可直接被 Agent 调用, 也通过 MCP Server 暴露。
生产环境可替换为对接第三方征信 API、Flink 实时特征、风控规则引擎等。
"""

from __future__ import annotations

import hashlib
import random
import time
from datetime import datetime, timedelta
from typing import Any

import structlog

from models.schemas import (
    CreditStatus,
    FraudRiskLevel,
    FraudRuleHit,
    RefundStatus,
)

logger = structlog.get_logger()


# =========================================================================
# 内存存储 — 演示用 (生产环境替换为 PostgreSQL / Redis)
# =========================================================================

_user_credit_profile: dict[str, dict[str, Any]] = {}
_user_fraud_history: dict[str, list[dict[str, Any]]] = {}
_user_refund_history: dict[str, list[dict[str, Any]]] = {}
_blacklist_ips: set[str] = set()
_blacklist_devices: set[str] = set()


def _seed_demo_data() -> None:
    """初始化演示数据 — 几个测试用户的信用/欺诈历史."""
    if _user_credit_profile:
        return
    for uid, score, limit in [
        ("user_001", 780, 50000.0),
        ("user_002", 650, 10000.0),
        ("user_003", 520, 2000.0),
        ("user_risk", 350, 0.0),
    ]:
        _user_credit_profile[uid] = {
            "credit_score": score,
            "credit_limit": limit,
            "available_limit": limit,
            "status": CreditStatus.ACTIVE if score >= 600 else CreditStatus.FROZEN,
            "overdue_count": 0 if score >= 600 else 2,
        }
    # 黑名单
    _blacklist_ips.add("192.168.1.100")
    _blacklist_devices.add("risky_device_001")
    # 退款历史
    _user_refund_history["user_003"] = [
        {"order_id": f"refund_test_{i}", "amount": 100.0 * i, "approved": True,
         "timestamp": (datetime.now() - timedelta(days=i)).isoformat()}
        for i in range(1, 6)
    ]


_seed_demo_data()


# =========================================================================
# 1. 实时反欺诈工具
# =========================================================================

FRAUD_RULES = [
    {
        "rule_id": "R001",
        "rule_name": "IP黑名单",
        "weight": 40,
        "check": lambda ctx: ctx.get("ip_address") in _blacklist_ips,
        "description": "请求IP命中风险黑名单",
    },
    {
        "rule_id": "R002",
        "rule_name": "设备黑名单",
        "weight": 45,
        "check": lambda ctx: ctx.get("device_id") in _blacklist_devices,
        "description": "设备ID命中风险黑名单",
    },
    {
        "rule_id": "R003",
        "rule_name": "高频小额交易",
        "weight": 25,
        "check": lambda ctx: ctx.get("order_count_1h", 0) > 10,
        "description": "1小时内下单次数异常偏高",
    },
    {
        "rule_id": "R004",
        "rule_name": "异地登录",
        "weight": 20,
        "check": lambda ctx: ctx.get("city_mismatch", False),
        "description": "下单城市与常用收货地不一致",
    },
    {
        "rule_id": "R005",
        "rule_name": "新用户大额首单",
        "weight": 30,
        "check": lambda ctx: ctx.get("is_new_user") and ctx.get("amount", 0) > 5000,
        "description": "新用户首单金额超过5000元",
    },
    {
        "rule_id": "R006",
        "rule_name": "历史欺诈记录",
        "weight": 50,
        "check": lambda ctx: ctx.get("has_fraud_history", False),
        "description": "用户存在历史欺诈交易记录",
    },
    {
        "rule_id": "R007",
        "rule_name": "支付方式异常",
        "weight": 15,
        "check": lambda ctx: ctx.get("payment_method") == "crypto",
        "description": "使用高风险支付方式",
    },
]


def _calculate_risk_level(total_score: float) -> tuple[FraudRiskLevel, str]:
    """根据累计风险分数判定风险等级和建议动作."""
    if total_score >= 80:
        return FraudRiskLevel.CRITICAL, "block"
    if total_score >= 50:
        return FraudRiskLevel.HIGH, "review"
    if total_score >= 20:
        return FraudRiskLevel.MEDIUM, "allow"
    return FraudRiskLevel.LOW, "allow"


async def check_fraud(
    user_id: str,
    amount: float,
    payment_method: str = "alipay",
    device_id: str | None = None,
    ip_address: str | None = None,
    order_id: str | None = None,
) -> dict[str, Any]:
    """实时反欺诈检测 — 规则引擎 + 风险评分。

    Args:
        user_id: 用户ID
        amount: 交易金额
        payment_method: 支付方式
        device_id: 设备指纹
        ip_address: 客户端IP
        order_id: 订单号 (可选)

    Returns:
        {risk_level, risk_score, rules_hit, recommended_action, needs_human_review}
    """
    ctx = {
        "user_id": user_id,
        "amount": amount,
        "payment_method": payment_method,
        "device_id": device_id,
        "ip_address": ip_address,
        "order_id": order_id,
        "is_new_user": user_id.startswith("new_"),
        "has_fraud_history": len(_user_fraud_history.get(user_id, [])) > 0,
        "order_count_1h": random.randint(0, 3),  # 演示: 模拟实时特征
        "city_mismatch": random.random() < 0.1,  # 演示: 10%概率异地
    }

    rules_hit: list[FraudRuleHit] = []
    total_score = 0.0

    for rule in FRAUD_RULES:
        try:
            if rule["check"](ctx):
                rules_hit.append(FraudRuleHit(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    risk_score=rule["weight"],
                    description=rule["description"],
                ))
                total_score += rule["weight"]
        except Exception as exc:
            logger.warning("fraud.rule_check_error", rule=rule["rule_id"], error=str(exc))

    risk_level, action = _calculate_risk_level(total_score)
    needs_review = risk_level in (FraudRiskLevel.HIGH, FraudRiskLevel.CRITICAL)

    # 记录到历史
    if total_score > 0:
        _user_fraud_history.setdefault(user_id, []).append({
            "order_id": order_id,
            "risk_score": total_score,
            "risk_level": risk_level.value,
            "timestamp": datetime.now().isoformat(),
        })

    logger.info(
        "fraud.check_complete",
        user_id=user_id,
        risk_level=risk_level.value,
        risk_score=round(total_score, 1),
        rules_hit_count=len(rules_hit),
        action=action,
    )

    return {
        "risk_level": risk_level.value,
        "risk_score": round(total_score, 1),
        "rules_hit": [r.model_dump() for r in rules_hit],
        "recommended_action": action,
        "needs_human_review": needs_review,
    }


async def get_fraud_history(user_id: str, limit: int = 20) -> dict[str, Any]:
    """查询用户欺诈历史记录."""
    records = _user_fraud_history.get(user_id, [])[-limit:]
    return {
        "user_id": user_id,
        "total_records": len(records),
        "records": list(reversed(records)),
    }


async def add_to_blacklist(
    item_type: str,  # "ip" | "device"
    value: str,
    reason: str = "",
) -> dict[str, Any]:
    """将IP或设备加入黑名单."""
    if item_type == "ip":
        _blacklist_ips.add(value)
    elif item_type == "device":
        _blacklist_devices.add(value)
    else:
        return {"status": "error", "message": f"unknown type: {item_type}"}
    return {"status": "added", "type": item_type, "value": value, "reason": reason}


# =========================================================================
# 2. 信用授信工具
# =========================================================================


async def get_credit_profile(user_id: str) -> dict[str, Any]:
    """查询用户信用档案.

    Args:
        user_id: 用户ID

    Returns:
        {user_id, credit_score, credit_limit, available_limit, status, overdue_count}
    """
    profile = _user_credit_profile.get(user_id)
    if profile is None:
        # 新用户: 基于ID hash生成一个稳定的初始分数
        score = 550 + int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 150
        limit = 5000.0 if score >= 600 else 1000.0
        profile = {
            "credit_score": score,
            "credit_limit": limit,
            "available_limit": limit,
            "status": CreditStatus.ACTIVE.value if score >= 600 else CreditStatus.NONE.value,
            "overdue_count": 0,
        }
        _user_credit_profile[user_id] = profile

    logger.info(
        "credit.profile_query",
        user_id=user_id,
        credit_score=profile["credit_score"],
        status=profile["status"],
    )
    return {"user_id": user_id, **profile}


async def assess_credit(
    user_id: str,
    requested_amount: float,
    order_id: str | None = None,
) -> dict[str, Any]:
    """信用授信评估 — 判断是否可放款及放款额度.

    Args:
        user_id: 用户ID
        requested_amount: 申请金额
        order_id: 关联订单号

    Returns:
        {approved, approved_amount, credit_score, interest_rate, tenure_days, reason}
    """
    profile_result = await get_credit_profile(user_id)
    score = profile_result["credit_score"]
    available = profile_result["available_limit"]
    status = profile_result["status"]

    approved = False
    approved_amount = 0.0
    interest_rate = 0.0
    reason = ""

    if status != CreditStatus.ACTIVE.value:
        reason = f"账户状态异常: {status}"
    elif requested_amount > available:
        reason = f"可用额度不足: 可用{available:.2f}, 申请{requested_amount:.2f}"
        approved_amount = available
        # 如果可用额度>0, 可以部分批准
        approved = available > 0
    else:
        approved = True
        approved_amount = requested_amount
        # 扣减可用额度
        _user_credit_profile[user_id]["available_limit"] -= requested_amount

    # 利率根据信用分浮动 (日息万3 到 万8)
    if approved and approved_amount > 0:
        if score >= 750:
            interest_rate = 0.0003  # 日息万3
        elif score >= 650:
            interest_rate = 0.0005  # 日息万5
        else:
            interest_rate = 0.0008  # 日息万8

    logger.info(
        "credit.assessment_complete",
        user_id=user_id,
        requested=requested_amount,
        approved=approved,
        approved_amount=round(approved_amount, 2),
        score=score,
    )

    return {
        "user_id": user_id,
        "approved": approved,
        "approved_amount": round(approved_amount, 2),
        "credit_score": score,
        "credit_limit": profile_result["credit_limit"],
        "available_limit": profile_result["available_limit"],
        "interest_rate": interest_rate,
        "tenure_days": 30,
        "status": status,
        "reason": reason,
        "order_id": order_id,
    }


async def repay_credit(
    user_id: str,
    amount: float,
    order_id: str | None = None,
) -> dict[str, Any]:
    """还款 — 恢复可用额度."""
    profile = _user_credit_profile.get(user_id)
    if profile is None:
        return {"status": "error", "message": "用户不存在"}

    limit = profile["credit_limit"]
    current = profile["available_limit"]
    new_available = min(limit, current + amount)
    profile["available_limit"] = new_available

    return {
        "status": "success",
        "user_id": user_id,
        "repayment_amount": amount,
        "available_limit_before": current,
        "available_limit_after": new_available,
    }


# =========================================================================
# 3. 售后退款风控工具
# =========================================================================

REFUND_RISK_RULES = [
    {
        "rule_id": "RF001",
        "rule_name": "高退款频率用户",
        "weight": 35,
        "description": "近30天退款次数超过5次",
    },
    {
        "rule_id": "RF002",
        "rule_name": "退款金额过大",
        "weight": 25,
        "description": "单笔退款金额超过用户历史平均订单金额的3倍",
    },
    {
        "rule_id": "RF003",
        "rule_name": "收货即退",
        "weight": 20,
        "description": "签收后24小时内申请退款",
    },
    {
        "rule_id": "RF004",
        "rule_name": "历史欺诈退款",
        "weight": 50,
        "description": "用户存在恶意退款历史记录",
    },
    {
        "rule_id": "RF005",
        "rule_name": "理由可疑",
        "weight": 15,
        "description": "退款理由模糊或多次变更",
    },
]


async def assess_refund_risk(
    user_id: str,
    order_id: str,
    product_id: str,
    refund_amount: float,
    refund_reason: str = "",
) -> dict[str, Any]:
    """退款风控审核 — 评估退款风险并决定处理策略.

    Args:
        user_id: 用户ID
        order_id: 订单号
        product_id: 商品ID
        refund_amount: 退款金额
        refund_reason: 退款理由

    Returns:
        {risk_level, risk_score, refund_status, flash_refund_eligible,
         needs_human_review, rejection_reason}
    """
    refund_history = _user_refund_history.get(user_id, [])
    recent_refunds = [
        r for r in refund_history
        if (datetime.now() - datetime.fromisoformat(r["timestamp"])).days <= 30
    ]

    risk_score = 0.0
    rules_hit: list[dict[str, Any]] = []

    # RF001: 高退款频率
    if len(recent_refunds) > 5:
        risk_score += 35
        rules_hit.append({"rule_id": "RF001", "rule_name": "高退款频率用户",
                          "risk_score": 35, "description": f"近30天退款{len(recent_refunds)}次"})

    # RF002: 金额过大 (简化: 超过2000元即触发)
    if refund_amount > 2000:
        risk_score += 25
        rules_hit.append({"rule_id": "RF002", "rule_name": "退款金额过大",
                          "risk_score": 25, "description": f"退款金额{refund_amount}元"})

    # RF005: 理由可疑
    if not refund_reason or len(refund_reason) < 5:
        risk_score += 15
        rules_hit.append({"rule_id": "RF005", "rule_name": "理由可疑",
                          "risk_score": 15, "description": "退款理由不充分"})

    # 风险等级判定
    if risk_score >= 60:
        risk_level = FraudRiskLevel.HIGH
        refund_status = RefundStatus.MANUAL_REVIEW
        flash_refund = False
        needs_review = True
        rejection_reason = "高风险退款，需人工审核"
    elif risk_score >= 30:
        risk_level = FraudRiskLevel.MEDIUM
        refund_status = RefundStatus.PENDING
        flash_refund = False
        needs_review = False
        rejection_reason = ""
    else:
        risk_level = FraudRiskLevel.LOW
        refund_status = RefundStatus.FLASH_REFUND
        flash_refund = True
        needs_review = False
        rejection_reason = ""

    # 记录退款历史
    _user_refund_history.setdefault(user_id, []).append({
        "order_id": order_id,
        "product_id": product_id,
        "amount": refund_amount,
        "risk_score": risk_score,
        "approved": risk_score < 60,
        "timestamp": datetime.now().isoformat(),
    })

    logger.info(
        "refund.assessment_complete",
        user_id=user_id,
        order_id=order_id,
        risk_level=risk_level.value,
        risk_score=round(risk_score, 1),
        status=refund_status.value,
        flash_refund=flash_refund,
    )

    return {
        "user_id": user_id,
        "order_id": order_id,
        "product_id": product_id,
        "risk_level": risk_level.value,
        "risk_score": round(risk_score, 1),
        "refund_status": refund_status.value,
        "flash_refund_eligible": flash_refund,
        "needs_human_review": needs_review,
        "rejection_reason": rejection_reason,
        "rules_hit": rules_hit,
    }


async def approve_refund(order_id: str, user_id: str, amount: float) -> dict[str, Any]:
    """[审核动作] 批准退款."""
    return {
        "status": "approved",
        "order_id": order_id,
        "user_id": user_id,
        "refund_amount": amount,
        "refund_id": f"RF_{int(time.time())}",
        "processed_at": datetime.now().isoformat(),
    }


async def reject_refund(order_id: str, user_id: str, reason: str) -> dict[str, Any]:
    """[审核动作] 拒绝退款."""
    return {
        "status": "rejected",
        "order_id": order_id,
        "user_id": user_id,
        "rejection_reason": reason,
        "processed_at": datetime.now().isoformat(),
    }
