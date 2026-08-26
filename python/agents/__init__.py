"""
LLM Agent 层 — 仅保留使用 LLM 的 Agent。

3 个 LLM Agent:
    - ProductRecAgent: LLM 语义重排
    - MarketingCopyAgent: LLM 文案生成
    - MetaAgent: LLM 灰度仲裁

传统业务逻辑 (风控/库存/履约/画像) 已迁移至 services/ 层。
"""

from .base_agent import BaseAgent
from .meta_agent import MetaAgent
from .product_rec_agent import ProductRecAgent
from .marketing_copy_agent import MarketingCopyAgent

__all__ = ["BaseAgent", "MetaAgent", "ProductRecAgent", "MarketingCopyAgent"]
