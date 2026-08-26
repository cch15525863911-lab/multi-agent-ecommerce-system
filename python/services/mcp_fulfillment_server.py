"""
MCP 履约工具服务器 — 用 MCP 协议封装供应链业务工具。

通过 FastMCP (@mcp.tool()) 将 services/fulfillment_tools.py 中的六个业务
函数封装为标准 MCP 工具, 供外部 MCP 客户端调用:

    check_inventory        → 多仓实时库存查询
    reserve_inventory       → 分布式库存预占
    match_logistics_route   → 物流路线匹配 + 高价值加密保价
    create_order            → 订单自动创建
    release_inventory       → [Saga 补偿] 释放预占, 归还库存
    cancel_order            → [Saga 补偿] 取消订单 + 释放预占

启动:
    python -m services.mcp_fulfillment_server          # stdio 传输(默认)
    python -m services.mcp_fulfillment_server --http   # SSE/HTTP 传输

内部履约主链路由 services/saga.py 的 Saga 事务编排直接调用 fulfillment_tools.py，
不经过 MCP Server 和 LLM。本 MCP Server 作为标准工具暴露层，供外部系统
（其他 MCP 客户端、运维工具、第三方集成）以 stdio 或 HTTP 方式调用。
"""

from __future__ import annotations

import sys

from services import fulfillment_tools as ft

try:
    from mcp.server.fastmcp import FastMCP

    _MCP_AVAILABLE = True
except ImportError:  # mcp SDK 未安装时, 提供占位以避免导入失败
    FastMCP = None  # type: ignore[assignment,misc]
    _MCP_AVAILABLE = False

import structlog

logger = structlog.get_logger()

SERVER_NAME = "ecommerce-fulfillment"


def _build_server() -> "FastMCP":
    """Construct the FastMCP server with all fulfillment tools registered."""
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "mcp SDK 未安装, 请 `pip install mcp` 后运行本服务; "
            "Agent 端会自动降级为直接调用业务函数。"
        )

    mcp = FastMCP(SERVER_NAME)

    @mcp.tool()
    async def check_inventory(  # noqa: D401
        product_id: str, warehouse_id: str | None = None
    ) -> dict:
        """查询商品在多个仓库的实时可用库存(物理库存 - 已预占)。

        Args:
            product_id: 商品ID
            warehouse_id: 指定仓库ID, 为空则返回全部仓库库存

        Returns:
            {warehouses: [...], total_free}
        """
        return await ft.check_inventory(product_id, warehouse_id)

    @mcp.tool()
    async def reserve_inventory(  # noqa: D401
        product_id: str, quantity: int, warehouse_id: str
    ) -> dict:
        """分布式库存预占: 锁定指定仓库的库存, 返回预占单号与过期时间。

        采用 Redis SETNX 分布式锁 + 预占计数池防超卖; 预占 15 分钟内未下单
        自动释放。

        Args:
            product_id: 商品ID
            quantity: 预占数量
            warehouse_id: 仓库ID

        Returns:
            {status, reservation_id, expires_at} 或 {status:"insufficient"/"locked"}
        """
        return await ft.reserve_inventory(product_id, quantity, warehouse_id)

    @mcp.tool()
    async def match_logistics_route(  # noqa: D401
        product_id: str,
        warehouse_id: str,
        product_value: float,
        destination: str = "北京",
    ) -> dict:
        """为商品匹配物流路线; 高价值商品(>=3000)强制加密保价。

        高价值时选择支持保价的承运商并开启订单数据加密。

        Args:
            product_id: 商品ID
            warehouse_id: 发货仓库ID
            product_value: 商品单价(用于判断是否高价值)
            destination: 目的地城市

        Returns:
            {route_id, carrier, insured, insured_amount, encrypted, eta_hours}
        """
        return await ft.match_logistics_route(
            product_id, warehouse_id, product_value, destination
        )

    @mcp.tool()
    async def create_order(  # noqa: D401
        user_id: str,
        product_id: str,
        quantity: int,
        unit_price: float,
        reservation_id: str,
        route_id: str,
    ) -> dict:
        """基于已生效的预占单与物流路线自动创建订单。

        校验预占单有效后绑定预占号与物流路线, 生成订单号并落库。

        Args:
            user_id: 用户ID
            product_id: 商品ID
            quantity: 购买数量
            unit_price: 商品单价
            reservation_id: 预占单号
            route_id: 物流路线ID

        Returns:
            {order_id, status, total_amount}
        """
        return await ft.create_order(
            user_id, product_id, quantity, unit_price, reservation_id, route_id
        )

    @mcp.tool()
    async def release_inventory(reservation_id: str) -> dict:
        """[Saga 补偿] 释放指定预占单, 归还预占数量到可用库存池。

        当后续步骤(如 create_order)失败时调用, 回滚已生效的预占。

        Args:
            reservation_id: 预占单号

        Returns:
            {status: "released"} 或 {status: "not_found"/"already_consumed"}
        """
        return await ft.release_inventory(reservation_id)

    @mcp.tool()
    async def cancel_order(order_id: str) -> dict:
        """[Saga 补偿] 取消订单并释放关联的预占库存。

        将订单状态改为 cancelled, 同时释放绑定的预占。

        Args:
            order_id: 订单号

        Returns:
            {status: "cancelled", order_id, reservation_id} 或 {status: "not_found"}
        """
        return await ft.cancel_order(order_id)

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
