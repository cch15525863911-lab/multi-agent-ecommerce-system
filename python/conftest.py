"""
pytest 全局配置 — 为测试环境提供必需的环境变量。

背景: JWT 鉴权默认开启 (fail-closed 安全默认), main.py 启动期会校验密钥强度
(缺失或 <32 字符直接 fail-fast)。test_neo4j_startup 等用例会调用 main.lifespan,
因此测试环境必须注入有效的 JWT 密钥, 否则初始化会被拒绝。

说明:
- 使用 setdefault: 若外部环境 (如 CI) 已设置 ECOM_JWT_SECRET, 则保留外部值。
- 测试密钥仅用于本地/CI 测试, 与生产密钥完全隔离。
"""

import os

os.environ.setdefault(
    "ECOM_JWT_SECRET",
    "test-only-jwt-secret-0123456789abcdefghij",
)
