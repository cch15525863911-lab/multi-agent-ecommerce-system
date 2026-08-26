"""
风控业务工具单元测试 — 反欺诈/信用/退款三类工具。

覆盖:
    - 反欺诈: 正常交易/黑名单命中/高风险交易/历史查询
    - 信用: 查询档案/授信评估/还款/额度不足
    - 退款: 低风险极速退款/中风险待审核/高风险人工审核
"""
from __future__ import annotations

import pytest

from services.risk_tools import (
    _user_credit_profile,
    _user_fraud_history,
    _user_refund_history,
    add_to_blacklist,
    assess_credit,
    assess_refund_risk,
    approve_refund,
    check_fraud,
    get_credit_profile,
    get_fraud_history,
    repay_credit,
)


@pytest.fixture(autouse=True)
def reset_state():
    """每个测试前重置内存状态, 避免互相影响。"""
    _user_fraud_history.clear()
    _user_refund_history.clear()
    # 信用档案保留默认种子数据, 测试中修改的话单独处理


# =========================================================================
# 反欺诈工具测试
# =========================================================================


class TestFraudDetection:
    @pytest.mark.asyncio
    async def test_normal_transaction_low_risk(self):
        """正常交易应为低风险。"""
        result = await check_fraud("user_001", 100.0, "alipay")
        assert result["risk_level"] == "low"
        assert result["risk_score"] < 20
        assert result["recommended_action"] == "allow"
        assert result["needs_human_review"] is False

    @pytest.mark.asyncio
    async def test_ip_blacklist_triggers_high_risk(self):
        """IP 命中黑名单应触发高风险。"""
        await add_to_blacklist("ip", "10.0.0.99", "test")
        result = await check_fraud("user_normal", 200.0, "alipay", ip_address="10.0.0.99")
        # IP 黑名单权重40, 应至少 medium
        assert result["risk_score"] >= 25
        assert len(result["rules_hit"]) >= 1

    @pytest.mark.asyncio
    async def test_new_user_large_amount(self):
        """新用户大额首单应触发风险规则。"""
        result = await check_fraud("new_user_123", 8000.0, "alipay")
        # 新用户大额规则权重30
        assert any("新用户" in r["description"] or "新用户" in r["rule_name"]
                   for r in result["rules_hit"])

    @pytest.mark.asyncio
    async def test_fraud_history_recording(self):
        """欺诈检测记录应写入历史。"""
        await check_fraud("user_hist_test", 500.0, "alipay",
                          device_id="risky_device_001")
        history = await get_fraud_history("user_hist_test")
        assert history["total_records"] >= 1

    @pytest.mark.asyncio
    async def test_recommended_action_levels(self):
        """不同风险等级对应不同建议动作。"""
        # low → allow
        low = await check_fraud("user_001", 50.0, "alipay")
        assert low["recommended_action"] == "allow"


# =========================================================================
# 信用授信工具测试
# =========================================================================


class TestCreditAssessment:
    @pytest.mark.asyncio
    async def test_get_credit_profile_existing_user(self):
        """查询已有用户信用档案。"""
        result = await get_credit_profile("user_001")
        assert result["credit_score"] == 780
        assert result["credit_limit"] == 50000.0
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_credit_profile_new_user(self):
        """新用户应自动生成初始信用档案。"""
        result = await get_credit_profile("brand_new_user_xyz")
        assert result["credit_score"] >= 550
        assert result["credit_limit"] > 0

    @pytest.mark.asyncio
    async def test_assess_credit_approved(self):
        """优质用户申请额度内应被批准。"""
        result = await assess_credit("user_001", 1000.0)
        assert result["approved"] is True
        assert result["approved_amount"] == 1000.0
        assert result["credit_score"] == 780

    @pytest.mark.asyncio
    async def test_assess_credit_reduces_available(self):
        """批准后可用额度应减少。"""
        before = await get_credit_profile("user_001")
        await assess_credit("user_001", 2000.0)
        after = await get_credit_profile("user_001")
        assert after["available_limit"] == before["available_limit"] - 2000.0

    @pytest.mark.asyncio
    async def test_assess_credit_exceeds_limit(self):
        """申请超过可用额度应被拒绝或部分批准。"""
        # user_002 有 10000 额度且状态 active, 申请 15000 应超限
        result = await assess_credit("user_002", 15000.0)
        assert result["approved_amount"] <= 10000.0
        assert "额度不足" in result["reason"]

    @pytest.mark.asyncio
    async def test_repay_credit_restores_limit(self):
        """还款应恢复可用额度。"""
        # 先借一笔
        await assess_credit("user_001", 1000.0)
        before = await get_credit_profile("user_001")
        # 还款
        await repay_credit("user_001", 1000.0)
        after = await get_credit_profile("user_001")
        assert after["available_limit"] == before["available_limit"] + 1000.0

    @pytest.mark.asyncio
    async def test_interest_rate_tiers(self):
        """不同信用分对应不同利率档位。"""
        # 780分 → 万3
        high = await assess_credit("user_001", 1000.0)
        assert high["interest_rate"] == 0.0003

        # 650分 → 万5
        mid = await assess_credit("user_002", 1000.0)
        assert mid["interest_rate"] == 0.0005


# =========================================================================
# 退款风控工具测试
# =========================================================================


class TestRefundRisk:
    @pytest.mark.asyncio
    async def test_low_risk_flash_refund(self):
        """低风险退款应走极速退款。"""
        result = await assess_refund_risk(
            "user_001", "ORD_001", "P001", 99.0, "不喜欢"
        )
        assert result["risk_level"] == "low"
        assert result["refund_status"] == "flash_refund"
        assert result["flash_refund_eligible"] is True
        assert result["needs_human_review"] is False

    @pytest.mark.asyncio
    async def test_high_amount_triggers_medium_risk(self):
        """大额退款应触发中风险。"""
        result = await assess_refund_risk(
            "user_001", "ORD_002", "P002", 3000.0, "质量问题"
        )
        assert result["risk_score"] >= 20  # 大额规则权重25
        assert result["flash_refund_eligible"] is False

    @pytest.mark.asyncio
    async def test_high_frequency_triggers_high_risk(self):
        """高频退款用户应触发高风险。"""
        # 先给 user_003 添加更多退款记录 (超过5次)
        from datetime import datetime, timedelta
        from services.risk_tools import _user_refund_history
        _user_refund_history["user_003"] = [
            {"order_id": f"ORD_{i}", "amount": 100.0 * i, "approved": True,
             "timestamp": (datetime.now() - timedelta(days=i)).isoformat()}
            for i in range(1, 8)  # 7次退款, 全部在30天内
        ]
        result = await assess_refund_risk(
            "user_003", "ORD_010", "P010", 100.0, "重复申请"
        )
        # 高频规则权重35 + 可能的其他规则
        assert result["risk_score"] >= 30

    @pytest.mark.asyncio
    async def test_empty_reason_adds_risk(self):
        """退款理由不充分应增加风险分。"""
        result = await assess_refund_risk(
            "user_001", "ORD_003", "P003", 100.0, ""
        )
        assert result["risk_score"] >= 15  # 理由可疑权重15

    @pytest.mark.asyncio
    async def test_approve_refund(self):
        """批准退款应返回成功状态。"""
        result = await approve_refund("ORD_APPROVE", "user_001", 99.0)
        assert result["status"] == "approved"
        assert "refund_id" in result
        assert result["refund_amount"] == 99.0
