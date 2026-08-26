from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False

    # CORS — comma-separated list of allowed origins
    cors_origins: str = "http://localhost:3000,http://localhost:8080"

    # LLM — DeepSeek-R1 (reasoner) / DeepSeek-V3 (chat)
    llm_provider: str = "cloud"  # "cloud" (OpenAI-compatible API) or "vllm" (local vLLM)
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"  # DeepSeek-V3
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    # vLLM (local LLM inference, OpenAI-compatible API)
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: SecretStr = SecretStr("EMPTY")

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    feature_ttl_seconds: int = 86400

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "product_embeddings"

    # Embedding model for semantic vector search (sentence-transformers)
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # Database (PostgreSQL) — must be set via environment variable
    database_url: str = ""

    # Neo4j (Knowledge Graph)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    neo4j_database: str = "neo4j"

    # Profile source switch: "kg" (Neo4j, Cypher-based) or "redis" (legacy, LLM-based)
    profile_source: str = "kg"

    # A/B Testing
    ab_test_enabled: bool = True
    ab_test_default_bucket_count: int = 100

    # Agent timeouts (seconds) — Layer 2: independent timeout
    agent_timeout_user_profile: float = 5.0
    agent_timeout_product_rec: float = 8.0
    agent_timeout_marketing_copy: float = 10.0
    agent_timeout_inventory: float = 5.0
    agent_timeout_supply_chain: float = 12.0

    # Circuit breaker — Layer 4: sliding-window breaker
    circuit_failure_threshold: float = 0.5  # 50% error rate trips
    circuit_window_size: int = 10           # last 10 calls
    circuit_recovery_timeout: float = 30.0  # seconds before half-open probe

    # Supply chain / fulfillment
    high_value_threshold: float = 3000.0

    # Agent timeouts — 新增风控类 Agent 超时
    agent_timeout_fraud: float = 8.0
    agent_timeout_credit: float = 8.0
    agent_timeout_refund: float = 8.0

    # Intent Router — 意图路由
    intent_router_use_llm: bool = False  # 是否用LLM做意图识别 (关闭时用规则)

    # Model Routing — 多模型路由 (DeepSeek-V3 for general, DeepSeek-R1 for reasoning)
    model_routing_enabled: bool = True
    model_flash: str = "deepseek-chat"             # 简单任务: DeepSeek-V3 (快速低成本)
    model_general: str = "deepseek-chat"           # 通用任务: DeepSeek-V3
    model_reasoning: str = "deepseek-reasoner"     # 推理任务: DeepSeek-R1 (风控/决策)

    # Guardrails Gate — 安全网关
    guardrails_prompt_injection_enabled: bool = True
    guardrails_pii_enabled: bool = True
    # P0 安全修复: 默认开启鉴权, 履约等写操作必须认证; 关闭将显著放宽攻击面
    guardrails_jwt_enabled: bool = True
    guardrails_rate_limit_enabled: bool = True
    # JWT 密钥必须从环境变量注入 (ECOM_JWT_SECRET), 缺失且开启鉴权时启动直接报错;
    # 严禁在代码/配置中保留任何明文默认值。
    jwt_secret: str | None = None

    # Rate Limiting — 速率限制
    rate_limit_user_per_min: int = 60    # 每用户每分钟请求数
    rate_limit_ip_per_min: int = 300     # 每IP每分钟请求数

    # Tracing — 链路追踪
    tracing_enabled: bool = True
    tracing_max_traces: int = 1000

    # Workflow Engine — 工作流引擎 (Temporal 风格)
    workflow_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ECOM_",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.cors_origins:
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_api_key_str(self) -> str:
        return self.llm_api_key.get_secret_value()

    @property
    def vllm_api_key_str(self) -> str:
        return self.vllm_api_key.get_secret_value()

    @property
    def database_url_str(self) -> str:
        return self.database_url

    @property
    def neo4j_password_str(self) -> str:
        return self.neo4j_password.get_secret_value()


@lru_cache()
def get_settings() -> Settings:
    return Settings()
