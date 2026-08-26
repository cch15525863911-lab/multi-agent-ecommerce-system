"""
MCP 风控工具服务器 — 用 MCP 协议封装反欺诈/信用/退款三类风控工具。

通过 FastMCP 将 services/risk_tools.py 中的业务函数封装为标准 MCP 工具,
暴露给 Agent (MCP 客户端) 经 LLM 以 ReAct 方式调用:

反欺诈工具:
    check_fraud              → 实时反欺诈检测 (规则引擎 + 风险评分)
    get_fraud_history        → 查询用户欺诈历史
    add_to_blacklist         → 将IP/设备加入黑名单

信用授信工具:
    get_credit_profile       → 查询用户信用档案
    assess_credit            → 信用授信评估 (是否可放款及额度)
    repay_credit             → 还款 (恢复可用额度)

退款风控工具:
    assess_refund_risk       → 退款风控审核 (风险评估 + 处理策略)
    approve_refund           → 批准退款
    reject_refund            → 拒绝退款

启动:
    python -m services.mcp_risk_server          # stdio 传输 (默认)
    python -m services.mcp_risk_server --http   # SSE/HTTP 传输
"""

from __future__ import annotations

import sys

from services import risk_tools as rt

try:
    from mcp.server.fastmcp import FastMCP

    _MCP_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]
    _MCP_AVAILABLE = False

import structlog

logger = structlog.get_logger()

SERVER_NAME = "ecommerce-risk-control"


def _build_server() -> "FastMCP":
    """Construct the FastMCP server with all risk-control tools registered."""
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "mcp SDK 未安装, 请 `pip install mcp` 后运行本服务"
        )

    mcp = FastMCP(SERVER_NAME)

    # ---- 反欺诈工具 ----

    @mcp.tool()
    async def check_fraud(
        user_id: str,
        amount: float,
        payment_method: str = "alipay",
        device_id: str | None = None,
        ip_address: str | None = None,
        order_id: str | None = None,
    ) -> dict:
        """实时反欺诈检测 — 基于规则引擎的风险评分。

        检测IP/设备黑名单、高频交易、异地登录、新用户大额、历史欺诈等规则，
        返回风险等级(low/medium/high/critical)和建议动作(allow/review/block)。

        Args:
            user_id: 用户ID
            amount: 交易金额
            payment_method: 支付方式 (alipay/wechat/card/crypto)
            device_id: 设备指纹ID (可选)
            ip_address: 客户端IP地址 (可选)
            order_id: 关联订单号 (可选)

        Returns:
            {risk_level, risk_score, rules_hit, recommended_action, needs_human_review}
        """
        return await rt.check_fraud(
            user_id, amount, payment_method, device_id, ip_address, order_id
        )

    @mcp.tool()
    async def get_fraud_history(user_id: str, limit: int = 20) -> dict:
        """查询用户的历史欺诈检测记录。

        Args:
            user_id: 用户ID
            limit: 返回记录数上限 (默认20)

        Returns:
            {user_id, total_records, records}
        """
        return await rt.get_fraud_history(user_id, limit)

    @mcp.tool()
    async def add_to_blacklist(item_type: str, value: str, reason: str = "") -> dict:
        """将IP地址或设备ID加入风险黑名单。

        Args:
            item_type: 黑名单类型, "ip" 或 "device"
            value: IP地址或设备ID
            reason: 加入黑名单的原因 (可选)

        Returns:
            {status, type, value, reason}
        """
        return await rt.add_to_blacklist(item_type, value, reason)

    # ---- 信用授信工具 ----

    @mcp.tool()
    async def get_credit_profile(user_id: str) -> dict:
        """查询用户信用档案 (信用分/额度/状态/逾期记录)。

        Args:
            user_id: 用户ID

        Returns:
            {user_id, credit_score, credit_limit, available_limit, status, overdue_count}
        """
        return await rt.get_credit_profile(user_id)

    @mcp.tool()
    async def assess_credit(
        user_id: str,
        requested_amount: float,
        order_id: str | None = None,
    ) -> dict:
        """信用授信评估 — 判断是否可放款、放款额度、利率。

        根据用户信用分、可用额度、账户状态综合评估，返回是否批准及批准金额。
        批准后可用额度会自动扣减。

        Args:
            user_id: 用户ID
            requested_amount: 申请授信金额
            order_id: 关联订单号 (可选)

        Returns:
            {approved, approved_amount, credit_score, credit_limit,
             available_limit, interest_rate, tenure_days, status, reason}
        """
        return await rt.assess_credit(user_id, requested_amount, order_id)

    @mcp.tool()
    async def repay_credit(
        user_id: str,
        amount: float,
        order_id: str | None = None,
    ) -> dict:
        """信用还款 — 还款后恢复对应可用额度。

        Args:
            user_id: 用户ID
            amount: 还款金额
            order_id: 关联订单号 (可选)

        Returns:
            {status, user_id, repayment_amount, available_limit_before, available_limit_after}
        """
        return await rt.repay_credit(user_id, amount, order_id)

    # ---- 退款风控工具 ----

    @mcp.tool()
    async def assess_refund_risk(
        user_id: str,
        order_id: str,
        product_id: str,
        refund_amount: float,
        refund_reason: str = "",
    ) -> dict:
        """退款风控审核 — 评估退款风险等级并决定处理策略。

        检测高频退款、大额退款、收货即退、恶意退款历史、理由可疑等规则，
        返回风险等级和处理结果 (极速退款/待审核/人工审核/拒绝)。

        Args:
            user_id: 用户ID
            order_id: 订单号
            product_id: 商品ID
            refund_amount: 退款金额
            refund_reason: 用户填写的退款理由 (可选)

        Returns:
            {risk_level, risk_score, refund_status, flash_refund_eligible,
             needs_human_review, rejection_reason, rules_hit}
        """
        return await rt.assess_refund_risk(
            user_id, order_id, product_id, refund_amount, refund_reason
        )

    @mcp.tool()
    async def approve_refund(order_id: str, user_id: str, amount: float) -> dict:
        """[人工审核动作] 批准退款申请。

        Args:
            order_id: 订单号
            user_id: 用户ID
            amount: 退款金额

        Returns:
            {status, order_id, user_id, refund_amount, refund_id, processed_at}
        """
        return await rt.approve_refund(order_id, user_id, amount)

    @mcp.tool()
    async def reject_refund(order_id: str, user_id: str, reason: str) -> dict:
        """[人工审核动作] 拒绝退款申请。

        Args:
            order_id: 订单号
            user_id: 用户ID
            reason: 拒绝原因

        Returns:
            {status, order_id, user_id, rejection_reason, processed_at}
        """
        return await rt.reject_refund(order_id, user_id, reason)

    return mcp


def main() -> None:
    if not _MCP_AVAILABLE:
        logger.error("mcp.unavailable", hint="pip install mcp")
        sys.exit(1)

    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"

    mcp = _build_server()
    logger.info("mcp.server.start", name=SERVER_NAME, transport=transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
