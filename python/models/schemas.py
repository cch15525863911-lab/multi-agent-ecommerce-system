from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserSegment(str, Enum):
    NEW_USER = "new_user"
    ACTIVE = "active"
    HIGH_VALUE = "high_value"
    PRICE_SENSITIVE = "price_sensitive"
    CHURN_RISK = "churn_risk"


class UserProfile(BaseModel):
    user_id: str
    age: int | None = None
    gender: str | None = None
    city: str | None = None
    segments: list[UserSegment] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 10000.0)
    recent_views: list[str] = Field(default_factory=list)
    recent_purchases: list[str] = Field(default_factory=list)
    rfm_score: dict[str, float] = Field(default_factory=dict)
    real_time_tags: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    description: str = ""
    brand: str = ""
    seller_id: str = ""
    stock: int = 0
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    image_url: str = ""


class RecommendationRequest(BaseModel):
    user_id: str
    scene: str = "homepage"
    num_items: int = 10
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class UserProfileResult(AgentResult):
    agent_name: str = "user_profile"
    profile: UserProfile | None = None


class ProductRecResult(AgentResult):
    agent_name: str = "product_rec"
    products: list[Product] = Field(default_factory=list)
    recall_strategy: str = ""


class MarketingCopyResult(AgentResult):
    agent_name: str = "marketing_copy"
    copies: list[dict[str, str]] = Field(default_factory=list)
    prompt_template_used: str = ""


class InventoryResult(AgentResult):
    agent_name: str = "inventory"
    available_products: list[str] = Field(default_factory=list)
    low_stock_alerts: list[dict[str, Any]] = Field(default_factory=list)
    purchase_limits: dict[str, int] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    products: list[Product] = Field(default_factory=list)
    marketing_copies: list[dict[str, str]] = Field(default_factory=list)
    experiment_group: str = "control"
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# ---------- Supply chain / fulfillment ----------


class FulfillmentRequest(BaseModel):
    """A request to take a (high-value) product from recommendation to a placed order."""
    user_id: str
    product: Product
    quantity: int = 1
    destination: str = "北京"


class LogisticsRoute(BaseModel):
    route_id: str
    carrier: str
    warehouse_id: str
    destination: str
    insured: bool = False
    insured_amount: float = 0.0
    encrypted: bool = False
    eta_hours: int = 48


class Reservation(BaseModel):
    reservation_id: str
    product_id: str
    warehouse_id: str
    quantity: int
    expires_at: datetime
    status: str = "active"  # active | consumed | expired


class Order(BaseModel):
    order_id: str
    user_id: str
    product_id: str
    quantity: int
    reservation_id: str
    logistics_route_id: str
    status: str = "created"  # created | paid | shipped | cancelled
    total_amount: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)


class FulfillmentResult(AgentResult):
    agent_name: str = "supply_chain"
    order: Order | None = None
    reservation: Reservation | None = None
    logistics_route: LogisticsRoute | None = None


# ---------- Risk Control / Fraud Detection ----------


class FraudRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudCheckRequest(BaseModel):
    """实时反欺诈检测请求."""
    user_id: str
    order_id: str | None = None
    product_id: str | None = None
    amount: float = 0.0
    payment_method: str = "alipay"
    device_id: str | None = None
    ip_address: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class FraudRuleHit(BaseModel):
    rule_id: str
    rule_name: str
    risk_score: float
    description: str


class FraudCheckResult(AgentResult):
    agent_name: str = "fraud_detection"
    risk_level: FraudRiskLevel = FraudRiskLevel.LOW
    risk_score: float = 0.0
    rules_hit: list[FraudRuleHit] = Field(default_factory=list)
    recommended_action: str = "allow"  # allow | review | block
    needs_human_review: bool = False


# ---------- Credit /授信 ----------


class CreditStatus(str, Enum):
    NONE = "none"
    ACTIVE = "active"
    FROZEN = "frozen"
    OVERDUE = "overdue"


class CreditAssessmentRequest(BaseModel):
    """信用授信评估请求."""
    user_id: str
    requested_amount: float = 0.0
    order_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class CreditAssessmentResult(AgentResult):
    agent_name: str = "credit_assessment"
    credit_score: int = 0  # 300-900
    credit_limit: float = 0.0
    available_limit: float = 0.0
    credit_status: CreditStatus = CreditStatus.NONE
    approved: bool = False
    approved_amount: float = 0.0
    interest_rate: float = 0.0
    tenure_days: int = 30


# ---------- Refund / 售后退款风控 ----------


class RefundStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLASH_REFUND = "flash_refund"
    MANUAL_REVIEW = "manual_review"


class RefundRiskRequest(BaseModel):
    """售后退款风控审核请求."""
    user_id: str
    order_id: str
    product_id: str
    refund_amount: float
    refund_reason: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


class RefundRiskResult(AgentResult):
    agent_name: str = "refund_risk"
    risk_level: FraudRiskLevel = FraudRiskLevel.LOW
    risk_score: float = 0.0
    refund_status: RefundStatus = RefundStatus.PENDING
    rejection_reason: str = ""
    flash_refund_eligible: bool = False
    needs_human_review: bool = False


# ---------- Intent Routing / 意图路由 ----------


class UserIntent(str, Enum):
    RECOMMENDATION = "recommendation"
    FRAUD_CHECK = "fraud_check"
    CREDIT_ASSESSMENT = "credit_assessment"
    REFUND_REVIEW = "refund_review"
    FULFILLMENT = "fulfillment"
    UNKNOWN = "unknown"


class IntentRouteRequest(BaseModel):
    """统一的多场景请求 — 由意图路由器分发到对应 Agent 链路."""
    user_id: str
    intent: UserIntent | None = None  # 可由调用方显式指定, 否则由 LLM 识别
    query: str = ""  # 自然语言查询, 用于意图识别
    scene: str = "homepage"

    # 推荐相关
    num_items: int = 10

    # 交易/风控相关
    product: Product | None = None
    amount: float = 0.0
    order_id: str | None = None
    payment_method: str = ""
    device_id: str | None = None
    ip_address: str | None = None
    refund_reason: str = ""
    requested_amount: float = 0.0

    context: dict[str, Any] = Field(default_factory=dict)


class IntentRouteResult(AgentResult):
    agent_name: str = "intent_router"
    detected_intent: UserIntent = UserIntent.UNKNOWN
    confidence: float = 0.0
    routing_path: list[str] = Field(default_factory=list)


class MetaDecisionResult(AgentResult):
    agent_name: str = "meta_agent"
    final_decision: str = ""  # approve | reject | escalate | proceed
    decision_reason: str = ""
    aggregated_risks: dict[str, float] = Field(default_factory=dict)
    escalation_required: bool = False
    arbitration_source: str = "rule"  # rule_fast_approve | rule_fast_reject | rule | llm_arbitration | rule_fallback


class UnifiedResponse(BaseModel):
    """统一响应 — 根据意图返回对应领域的数据."""
    request_id: str
    user_id: str
    intent: UserIntent
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    meta_decision: MetaDecisionResult | None = None
    total_latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

    # 推荐场景数据
    products: list[Product] = Field(default_factory=list)
    marketing_copies: list[dict[str, str]] = Field(default_factory=list)
    experiment_group: str = "control"

    # 履约场景数据
    order: Order | None = None
    # Meta 拒绝/升级时生成的"待人工确认预订单" (不落真实订单、不占库存)
    pending_order: dict[str, Any] | None = None

    # 风控场景数据
    fraud_result: FraudCheckResult | None = None
    credit_result: CreditAssessmentResult | None = None
    refund_result: RefundRiskResult | None = None
