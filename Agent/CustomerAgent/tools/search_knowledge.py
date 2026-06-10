"""
Unified knowledge search tool.

Searches the current product's scene knowledge first, then falls back to
product_knowledge for static product details.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.knowledge_action_router import sanitize_formatted_knowledge
from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from core.di_container import container
from database.knowledge_service import KnowledgeService


SCENE_MAP = {
    "售前": "presale",
    "售中": "insale",
    "售后": "aftersale",
    "presale": "presale",
    "insale": "insale",
    "aftersale": "aftersale",
}


def _knowledge_service() -> KnowledgeService:
    try:
        return container.get(KnowledgeService)
    except Exception:
        return KnowledgeService()


def _normalize_scene(scene: Optional[str], query: str) -> str:
    if scene:
        mapped = SCENE_MAP.get(str(scene).strip())
        if mapped:
            return mapped
    return KnowledgeService.detect_customer_scene(query, default="presale")


class SearchKnowledgeParams(BaseModel):
    """Unified knowledge search params."""

    query: str = Field(..., description="客户原始问题")
    shop_id: int = Field(..., description="店铺ID")
    goods_id: Optional[int] = Field(None, description="当前商品ID")
    scene: Optional[str] = Field(None, description="售前/售中/售后")


@agent_tool(
    name="search_knowledge",
    description="查询当前商品知识。客户问商品参数、功能、图片里的按键/图标/部件用途、续航、充电、发货、物流、退换货、售后处理等问题时使用。",
    param_model=SearchKnowledgeParams,
)
def search_knowledge(params: SearchKnowledgeParams) -> str:
    if not params.shop_id:
        return "[错误：缺少店铺ID，无法查询知识]"
    if not params.query:
        return "[错误：缺少客户问题，无法查询知识]"
    if not params.goods_id:
        return "当前还没有锁定具体商品，请先确认客户咨询的是哪一款商品。"

    knowledge_service = _knowledge_service()
    scene_key = _normalize_scene(params.scene, params.query)

    scene_results = knowledge_service.search_scene_knowledge(
        scene=scene_key,
        shop_id=params.shop_id,
        goods_id=params.goods_id,
        query=params.query,
        limit=2,
    )
    if scene_results:
        return sanitize_formatted_knowledge(knowledge_service.format_scene_results(scene_results))

    product_result = knowledge_service.search_knowledge(
        shop_id=params.shop_id,
        query=params.query,
        goods_id=params.goods_id,
        search_scope="product",
    )
    formatted = knowledge_service.format_search_result(product_result)
    return sanitize_formatted_knowledge(formatted)
