"""
性能基准测试 — 各 Agent 链路延迟对比。

运行方式: python -m tests.test_benchmark
或:       python tests/test_benchmark.py
或:       pytest tests/test_benchmark.py -s
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.fraud_service import FraudService
from services.credit_service import CreditService
from services.refund_service import RefundRiskService
from services.fulfillment_service import FulfillmentService
from services import risk_tools as rt
from services import fulfillment_tools as ft
from services.saga import run_fulfillment_saga
from models.schemas import FulfillmentRequest, Product


async def benchmark_fraud_deterministic():
    """风控检测（确定性规则引擎）"""
    agent = FraudService()
    start = time.perf_counter()
    await agent.run(
        user_id="U001",
        amount=2999,
        payment_method="alipay",
        device_id="D001",
        ip_address="192.168.1.1",
        order_id="ORD001",
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  风控检测（规则引擎）: {elapsed:.1f}ms")
    return elapsed


async def benchmark_credit_deterministic():
    """信用评估（确定性评分卡）"""
    agent = CreditService()
    start = time.perf_counter()
    await agent.run(user_id="U001", requested_amount=5000, order_id="ORD001")
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  信用评估（评分卡）: {elapsed:.1f}ms")
    return elapsed


async def benchmark_refund_deterministic():
    """退款风控（确定性规则引擎）"""
    agent = RefundRiskService()
    start = time.perf_counter()
    await agent.run(
        user_id="U001",
        order_id="ORD001",
        product_id="P001",
        refund_amount=2999,
        refund_reason="质量问题",
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  退款风控（规则引擎）: {elapsed:.1f}ms")
    return elapsed


async def benchmark_fulfillment_saga():
    """履约 Saga（4步事务）"""
    ft.reset_inmemory_state()
    request = FulfillmentRequest(
        user_id="U001",
        product=Product(
            product_id="P001",
            name="Dell XPS 15",
            category="笔记本",
            price=12999,
            brand="Dell",
            seller_id="S01",
            stock=500,
            tags=["旗舰"],
        ),
        quantity=1,
        destination="北京",
    )
    start = time.perf_counter()
    await run_fulfillment_saga(request)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  履约 Saga（4步事务）: {elapsed:.1f}ms")
    return elapsed


async def benchmark_metaagent_rule_fastpath():
    """MetaAgent 规则快通道"""
    from agents import MetaAgent
    from models.schemas import UserIntent

    meta = MetaAgent()
    agent_results = {
        "fraud_detection": type("R", (), {"risk_score": 10.0})(),
    }
    start = time.perf_counter()
    await meta.decide(UserIntent.FRAUD_CHECK, agent_results)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  MetaAgent 规则快通道: {elapsed:.1f}ms")
    return elapsed


async def run_all():
    print("=" * 50)
    print("性能基准测试")
    print("=" * 50)
    print()
    print("确定性链路（无 LLM 调用）:")
    await benchmark_fraud_deterministic()
    await benchmark_credit_deterministic()
    await benchmark_refund_deterministic()
    await benchmark_fulfillment_saga()
    await benchmark_metaagent_rule_fastpath()
    print()
    print("注: LLM 链路（营销文案/MetaAgent仲裁）延迟 1-3s，取决于 LLM 服务响应")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_all())
