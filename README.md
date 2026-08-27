# 🛒 多 Agent 电商系统（Multi-Agent E-Commerce System）v2.0

> 面向面试的企业级 Multi-Agent 工程化项目。基于 FastAPI + LangGraph，覆盖 **推荐 / 反欺诈 / 信用授信 / 售后退款 / 供应链履约** 5 大电商场景。
> 核心范式：**LLM 叠加在确定性管线之上并带降级** —— LLM 负责语义理解、创意生成与灰度仲裁；风控、授信、履约等强一致环节由确定性 Service 保障。

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python) ![LangGraph](https://img.shields.io/badge/LangGraph-v0.4-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-teal) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 目录

1. [这个项目是什么？](#-这个项目是什么)
2. [核心架构：LLM + 确定性混合](#-核心架构llm--确定性混合)
3. [五大业务场景](#-五大业务场景)
4. [快速上手](#-快速上手)
5. [API 接口文档](#-api-接口文档)
6. [项目文件结构](#-项目文件结构)
7. [可靠性设计](#-可靠性设计)
8. [测试](#-测试)
9. [面试要点](#-面试要点)
10. [已知问题与说明](#-已知问题与说明)

---

## 🤔 这个项目是什么？

### 一句话解释

> 让 **3 个 LLM Agent**（推荐重排、文案生成、跨域仲裁）与 **7 个确定性 Service**（画像、库存、反欺诈、授信、退款、履约等）协同工作，一个统一入口覆盖电商"**理解用户 → 推荐商品 → 生成文案 → 风控把关 → 履约下单**"全链路。

### 解决什么问题？

| 痛点 | 传统做法 | 本项目做法 |
|------|---------|-----------|
| 推荐结果与库存脱节 | 推荐了缺货商品 | **库存 Service** 实时校验，缺货自动剔除 |
| 营销文案千篇一律 | 所有人看同一段广告语 | **文案 Agent** 按用户分群生成个性化文案 |
| 各系统各自为战 | 推荐、风控、库存互不感知 | **动态引擎**统一编排，Meta-Agent 跨域仲裁 |
| 风控与业务脱节 | 下单后才发现欺诈/信用问题 | 履约前**风控预检**，Meta-Agent 批准后才允许真实下单 |

### 技术关键词（面试常考）

`Multi-Agent` · `LangGraph StateGraph (fan-out/fan-in)` · `意图路由` · `Meta-Agent 灰度仲裁` · `Saga 事务补偿` · `四层防护(重试/超时/降级/熔断)` · `安全网关(限流/JWT/PII脱敏/Prompt注入防护)` · `多模型路由(DeepSeek-V3/R1)` · `Redis Feature Store` · `Milvus 向量检索` · `Neo4j 知识图谱` · `GraphRAG` · `A/B Testing` · `Thompson Sampling` · `MCP 工具服务器` · `链路追踪` · `Temporal 风格工作流`

---

## 🏗 核心架构：LLM + 确定性混合

### 架构总览

```
                        用户请求（推荐 / 下单 / 退款 / 授信 / 风控）
                                      │
                                      ▼
                      ┌──────────────────────────────────────┐
                      │   安全网关 Guardrails Gate (中间件)     │
                      │   限流 → JWT鉴权 → PII脱敏 → 注入防护   │
                      └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │   Dynamic Engine (LangGraph 动态引擎)     │
                    │   意图路由 intent_router（规则优先,LLM兜底）│
                    └──────────────────┬──────────────────────┘
          ┌──────────────┬─────────────┼──────────────┬──────────────┐
          ▼              ▼             ▼              ▼              ▼
   ┌───────────┐  ┌───────────┐ ┌───────────┐ ┌───────────┐  ┌──────────────┐
   │推荐链路     │  │反欺诈链路  │ │授信链路    │ │退款链路    │  │履约链路       │
   │3阶段并行图  │  │FraudSvc   │ │CreditSvc  │ │RefundSvc  │  │风控预检→门控   │
   └─────┬─────┘  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘  └──────┬───────┘
         └──────────────┴─────────────┴─────────────┴───────────────┘
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │  Meta-Agent（LLM 灰度仲裁）                │
                    │  规则快通道 + LLM 仲裁 + 降级回规则         │
                    └─────────────────────────────────────────┘
                                      ▼
                               统一响应（含 meta_decision）
```

### 职责边界：哪些用 LLM，哪些不用？

| 层 | 模块 | 是否调用 LLM | 职责 |
|----|------|:---:|------|
| **LLM Agent** | `ProductRecAgent` | ✅ | 商品召回（协同过滤+向量+图谱+热度）+ LLM 语义重排 |
| | `MarketingCopyAgent` | ✅ | 按用户分群（新客/VIP/价格敏感/活跃/流失风险）生成文案 + 广告法合规过滤 |
| | `MetaAgent` | ✅（条件） | 跨域结果冲突时的灰度仲裁：approve / escalate / reject；规则快通道 <1ms |
| **确定性 Service** | `ProfileService` | ❌ | Neo4j 图谱特征 + RFM 规则分群（10-20x 快于 LLM，100% 可复现） |
| | `InventoryService` | ❌ | 实时库存校验、安全库存预警、限购策略 |
| | `FraudService` | ❌ | 反欺诈规则引擎（IP/设备/行为/黑名单） |
| | `CreditService` | ❌ | 信用评分卡授信、额度管理 |
| | `RefundRiskService` | ❌ | 退款风险规则、极速退款资格、恶意退款识别 |
| | `FulfillmentService` | ❌ | Saga 履约编排：库存预占 → 物流匹配 → 订单创建 + 逆序补偿 |
| | 支撑服务 | ❌ | A/B 测试、GraphRAG、安全网关、追踪、工作流、MCP Server ×2 |

### 两张 LangGraph 拓扑图

| 引擎 | 文件 | 拓扑 |
|------|------|------|
| 推荐三阶段图 | `orchestrator/graph.py` | START → 画像‖召回(并行) → 重排‖库存(并行) → 过滤 → 文案 → 聚合 → END |
| 动态引擎 | `orchestrator/dynamic_engine.py` | 意图路由 → 5 条领域分支 → Meta-Agent 决策 → END（履约走预检门控） |

> 💡 并行用 LangGraph 原生 fan-out / fan-in 边实现（不是手动 `asyncio.gather`），多分支结果通过 state reducer 自动合并，这是面试加分点。

---

## 🎯 五大业务场景

统一入口 `POST /api/v2/process`，通过 `intent` 字段路由：

| intent | 场景 | 链路 | 核心产出 |
|--------|------|------|---------|
| `recommendation` | 个性化推荐 | 画像‖召回 → 重排‖库存 → 文案 | 商品列表 + 个性化文案 + 实验分组 |
| `fraud_check` | 实时反欺诈 | 确定性规则引擎 | 风险等级/评分 + 建议动作 + 是否转人工 |
| `credit_assessment` | 信用授信 | 评分卡评估 | 是否批准 + 额度 + 利率 |
| `refund_review` | 售后退款风控 | 退款规则引擎 | 退款状态 + 极速退款资格 + 是否转人工 |
| `fulfillment` | 供应链履约 | **风控预检 → Meta-Agent 门控 → Saga 下单** | 订单 / 待人工确认预订单 |

> 履约场景的安全设计：先并行跑反欺诈 + 信用评估，Meta-Agent 决策 **approve 才允许真实下单 + 占库存**；reject/escalate 只生成"待人工确认预订单"，不落真实订单。

---

## 🚀 快速上手

### 前置条件

- Python 3.11+（推荐 3.12）
- LLM 后端二选一：
  - **云端 API（默认）**：DeepSeek / MiniMax 等 OpenAI 兼容接口，申请 API Key 即可
  - **本地 vLLM**：需要 NVIDIA GPU（≥16GB 显存），无需 API Key
- 存储依赖（均有降级，开发可先用 Mock）：Redis（特征/限流/协同过滤）、PostgreSQL（业务数据）、Milvus（商品向量）、Neo4j（知识图谱）

### 本地启动

```bash
cd python

# 1. 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置环境变量（重要！）
cp .env.example .env
#    JWT 鉴权默认开启（fail-closed）：.env.example 已提供 dev- 占位密钥，本地可直接启动；
#    调接口需带 Authorization: Bearer <token>。
#    生产部署必须将 ECOM_JWT_SECRET 换成 ≥32 字符的随机强密钥
#    （禁止 dev- 前缀，否则启动校验会拒绝）。

# 3. 启动服务
python main.py
# 看到 "Uvicorn running on http://0.0.0.0:8000" 即成功
```

### 灌入演示数据（可选）

```bash
cd python
python scripts/seed_kg.py          # 写入 Neo4j 用户-商品图谱
python -m database.init_db         # 初始化 PostgreSQL 建表 + 仓库/库存种子
```

### Docker Compose

```bash
# 全部依赖（Redis + PostgreSQL + Neo4j + Milvus + vLLM + API）
docker-compose up -d

# 仅核心依赖（不含 vLLM，使用云端 LLM API）—— 无需 GPU 的推荐方式
docker-compose up -d redis postgres neo4j milvus
```

> ⚠️ `api` 服务默认 `depends_on: vllm`：无 NVIDIA GPU 的环境请使用"仅核心依赖"方式启动，或临时从 `docker-compose.yml` 注释掉 vllm 依赖。

---

## 📡 API 接口文档

### 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/recommend` | 推荐（LangGraph 三阶段图，v1 兼容） |
| `POST` | `/api/v1/recommend/graph` | 推荐（显式图引擎，返回明细） |
| `POST` | `/api/v2/process` | ⭐ 统一动态引擎：意图路由 → 领域链路 → Meta-Agent 决策 |
| `POST` | `/api/v2/fraud/check` | 实时反欺诈检测（快捷接口） |
| `POST` | `/api/v2/credit/assess` | 信用授信评估（快捷接口） |
| `POST` | `/api/v2/refund/assess` | 售后退款风控审核（快捷接口） |
| `GET` | `/api/v1/experiments` | A/B 实验状态 |
| `POST` | `/api/v1/experiments/{id}/outcome` | 记录实验转化结果（更新 Thompson Sampling） |
| `GET` | `/api/v1/metrics` | 系统监控指标 |
| `GET` | `/api/v1/llm/status` | LLM Provider + 多模型路由状态 |
| `GET` | `/api/v2/traces` | 链路追踪列表 + 汇总统计 |
| `GET` | `/api/v2/traces/{trace_id}` | 单条追踪详情 |
| `GET` | `/api/v2/workflows` | 已注册工作流列表 |
| `POST` | `/api/v2/workflows/{type}/start` | 启动工作流 |
| `GET` | `/api/v2/workflows/{id}` | 查询工作流状态 |
| `GET` | `/health` | 健康检查（含各组件状态） |

### 请求示例：统一动态引擎

```bash
curl -X POST http://localhost:8000/api/v2/process \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "intent": "recommendation",
    "query": "推荐一下手机和耳机",
    "num_items": 5,
    "context": {"recent_views": ["手机", "耳机"], "avg_order_amount": 500}
  }'
```

响应含 `request_id`、`intent`、`agent_results`（各链路结果）、`meta_decision`（仲裁决策）、`total_latency_ms`。

### 请求示例：履约（含风控门控）

```bash
curl -X POST http://localhost:8000/api/v2/process \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "user_id": "user_001",
    "intent": "fulfillment",
    "product": {"product_id": "P001", "name": "iPhone 16 Pro", "price": 7999, "category": "手机"},
    "amount": 7999,
    "context": {"quantity": 1, "destination": "北京"}
  }'
```

> 履约需 `write` 权限 scope，否则返回 403。

---

## 📁 项目文件结构

```
multi-agent-ecommerce-system/
├── README.md                          # 本文件
├── plan.md                            # 开发计划（历史记录）
├── docker-compose.yml                 # Redis + PostgreSQL + Neo4j + Milvus + vLLM + API
├── .env.example                       # docker-compose 环境变量模板
│
├── docs/                              # （历史残留：HTML 文档库文件）
├── interview-prep/                    # （历史残留：面试 HTML 文档库文件）
│
└── python/                            # Python 实现（唯一代码库）
    ├── main.py                        # FastAPI 入口（v1 推荐 + v2 动态引擎 + 运维接口）
    ├── requirements.txt               # 依赖
    ├── .env.example                   # 应用环境变量模板（⚠️ 含 JWT 说明）
    ├── Dockerfile                     # 容器化部署
    │
    ├── agents/                        # ⭐ 3 个 LLM Agent（继承 BaseAgent 四层防护）
    │   ├── base_agent.py              # 基类：重试/独立超时/降级/熔断
    │   ├── product_rec_agent.py       # 商品推荐：4 路召回 + LLM 语义重排
    │   ├── marketing_copy_agent.py    # 营销文案：5 套分群模板 + 广告法过滤
    │   └── meta_agent.py              # 跨域仲裁：规则快通道 + LLM 灰度 + 降级
    │
    ├── orchestrator/                  # ⭐ LangGraph 编排层
    │   ├── graph.py                   # 推荐三阶段拓扑图（fan-out/fan-in）
    │   └── dynamic_engine.py          # 动态引擎：意图路由 + 5 领域分支 + 履约门控
    │
    ├── services/                      # 确定性 Service 与支撑组件
    │   ├── base_service.py            # Service 基类：熔断/超时/降级（无重试，防非幂等重复）
    │   ├── profile_service.py         # 用户画像（Neo4j + RFM 规则分群）
    │   ├── inventory_service.py       # 库存校验 / 预警 / 限购
    │   ├── fraud_service.py           # 反欺诈规则引擎
    │   ├── credit_service.py          # 信用授信评分卡
    │   ├── refund_service.py          # 退款风控规则引擎
    │   ├── fulfillment_service.py     # 供应链履约（Saga 事务编排）
    │   ├── saga.py                    # Saga 编排器 + 补偿机制（execute/compensate 独立熔断）
    │   ├── circuit_breaker.py         # 熔断器（CLOSED/OPEN/HALF_OPEN 三态）
    │   ├── ab_test.py                 # A/B 测试引擎（分桶 + Thompson Sampling）
    │   ├── cf_store.py                # 协同过滤（Redis / 内存降级）
    │   ├── milvus_store.py            # 商品向量检索（Milvus / 内存余弦降级）
    │   ├── kg_store.py                # Neo4j 知识图谱存储
    │   ├── graph_rag.py               # GraphRAG 上下文构建
    │   ├── feature_store.py           # Redis 实时特征（历史模块）
    │   ├── guardrails.py              # 安全网关：限流/鉴权/PII脱敏/注入防护
    │   ├── risk_tools.py              # 风控业务工具集（纯函数，供 Agent/MCP 调用）
    │   ├── mcp_fulfillment_server.py  # MCP Server：履约工具封装
    │   ├── mcp_risk_server.py         # MCP Server：风控工具封装
    │   ├── tracing.py                 # 链路追踪（类 LangSmith/Phoenix）
    │   ├── workflow_engine.py         # Temporal 风格工作流骨架
    │   ├── evaluation.py              # 评估框架（Precision@K/NDCG@K 等）
    │   ├── metrics.py                 # Prometheus 监控指标
    │   └── fulfillment_tools.py       # 履约业务工具（库存预占/订单/补偿落库）
    │
    ├── llm/                           # LLM 层
    │   ├── factory.py                 # 统一创建 LLM 客户端（cloud / vLLM 切换）
    │   └── router.py                  # 多模型路由（flash/general/reasoning 按任务选择）
    │
    ├── database/                      # PostgreSQL：engine / models / init_db
    ├── config/settings.py             # Pydantic 配置（ECOM_* 环境变量）
    ├── models/schemas.py              # Pydantic 数据模型（请求/响应/枚举）
    ├── scripts/seed_kg.py             # Neo4j 种子数据
    │
    ├── tests/                         # 19 个测试文件，143 个用例
    └── docs/                          # 项目真实文档
        ├── multi-agent.html           # 架构重构对比（Agent → Service 迁移）
        ├── 项目技术档案.md            # v2.0 完整技术档案（52K）
        ├── 简历-项目经历.md           # 简历可直接粘贴版
        └── 项目成果指标审查.md        # 简历量化指标逐条审查报告
```

---

## 🛡 可靠性设计

### 1. 分层防护

| 层 | 机制 | BaseAgent（LLM Agent） | BaseProtectedService（确定性 Service） |
|----|------|:---:|:---:|
| L1 | tenacity 指数退避重试 | ✅（LLM 调用可能瞬时抖动） | ❌（业务操作多非幂等，防重复扣减） |
| L2 | `asyncio.wait_for` 独立超时 | ✅ | ✅ |
| L3 | 降级返回 fallback | ✅ | ✅ |
| L4 | 滑动窗口熔断（错误率≥50% → OPEN，30s 后 HALF_OPEN 探测） | ✅ | ✅ |

### 2. Saga 事务补偿（履约链路）

库存校验 → 分布式预占（Redis SETNX）→ 物流匹配 → 订单创建，任一步失败逆序补偿（取消订单 + 释放预占）。**execute 与 compensate 各自持有独立熔断器**，互不影响。

### 3. 安全网关（Guardrails Gate）

- 限流：Redis 滑动窗口，按 user/IP 双维度（60/分钟/用户，300/分钟/IP）
- 鉴权：JWT Bearer Token + scope 校验（履约写操作要求 `write` scope，P0）
- PII 脱敏：手机号/身份证/银行卡/邮箱/地址 入站出站双向脱敏
- Prompt 注入防护：关键词黑名单 + 指令覆盖检测 + 间接注入检测

### 4. 可观测性

- 链路追踪 `/api/v2/traces`：记录每次调用的输入/输出/耗时/Token/错误
- 监控指标 `/api/v1/metrics`：Agent 调用统计 + 业务统计
- A/B 实验 `/api/v1/experiments`：推荐策略实验（rule_based vs LLM，各 50%）+ 文案风格实验（formal vs casual），支持 Thompson Sampling 动态调优

### 5. 多模型路由

`llm/router.py` 按任务类型选择模型：简单任务 → flash（deepseek-chat）、通用任务 → general（deepseek-chat）、推理任务（风控/决策）→ reasoning（deepseek-reasoner），主模型不可用时自动降级。

---

## 🧪 测试

```bash
cd python
pip install pytest pytest-asyncio
pytest
```

- **19 个测试文件，143 个用例，全部通过**（实测 143/143）
- 覆盖：Saga 事务补偿、熔断器、LangGraph 拓扑、A/B 引擎、KG 画像、GraphRAG、风控工具、安全网关、链路追踪、工作流、订单落库、性能基准

---

## 💼 面试要点

- **为什么 Multi-Agent 而不是单 Agent？** 上下文隔离（Token 少、推理准）、并行加速（延迟≈最慢 Agent）、独立演进（可独立 A/B）。
- **为什么 LLM 与确定性系统分离？** 面试中"LLM 设计过度"是减分项 —— 只有真正需要语义理解/创意/仲裁的模块才叫 Agent，确定性逻辑走可解释的 Service，并明确降级路径。
- **LangGraph 图编排 vs 手动 asyncio.gather？** 原生 fan-out/fan-in 边、state reducer 自动合并并行分支、checkpoint 持久化、可导出 DAG。
- **如何保证 Agent 调用稳定性？** 四层防护（重试/独立超时/降级/熔断），Saga 补偿也接入独立熔断。
- **履约安全怎么做？** 风控预检（反欺诈 + 授信并行）→ Meta-Agent 门控 → approve 才真实下单，reject/escalate 转人工预订单，避免"风控失效写操作"。

简历/面试材料见 `python/docs/`（技术档案、简历模板、指标审查）。

---

## ⚠️ 已知问题与说明

| 问题 | 状态 | 说明 |
|------|------|------|
| `test_recall_returns_products_without_profile` 不稳定 | ✅ 已修复 | `_recall` 排序改为基于 product_id 的确定性 hash 打散（替代 `random.random()`），结果可复现；测试断言修正为"fallback 候选来自 MOCK 池"的语义 |
| JWT 默认开启且无默认密钥 | ✅ 已统一 | 鉴权默认开启（fail-closed）；`.env.example` 提供 dev- 占位密钥可直接本地启动；启动时校验密钥 ≥32 字符、生产禁止 dev- 前缀（`ECOM_DEBUG=true` 除外） |
| LICENSE 文件缺失 | ✅ 已补全 | 已按 MIT 标准模板创建 `LICENSE` |
| `ECOM_PROFILE_SOURCE` 死配置 | 待清理 | settings 有字段但代码无引用；Redis 画像链路已随重构删除，docker-compose 中的该变量可移除 |
| vLLM 依赖 GPU | 环境相关 | 无 GPU 请用 `docker-compose up -d redis postgres neo4j milvus` 或注释 vllm |

---

## 📚 参考资料

| 项目 | 说明 |
|------|------|
| [NVIDIA Retail Agentic Commerce](https://github.com/NVIDIA-AI-Blueprints/Retail-Agentic-Commerce) | NVIDIA 企业级电商 Agent 蓝图 |
| [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) | LangGraph 状态图框架 |
| [Spring AI Alibaba](https://github.com/alibaba/spring-ai-alibaba) | Java 多 Agent 参考 |
| [DualAgent-Rec](https://github.com/GuilinDev/Dual-Agent-Recommendation) | 双 Agent 推荐系统 |
| [vLLM](https://github.com/vllm-project/vllm) | 本地 LLM 推理引擎 |

---

## 📄 License

[MIT License](LICENSE) — 随意使用、修改、商用，保留声明即可。
