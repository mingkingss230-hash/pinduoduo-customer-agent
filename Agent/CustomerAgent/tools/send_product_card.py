"""
Unified product-card tool.

If goods_id is known, send that product card directly. Otherwise fetch product
candidates and return them for the model to ask the customer to choose.
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from Channel.pinduoduo.utils.API.product_manager import ProductManager
from Channel.pinduoduo.utils.API.send_message import SendMessage
from utils.logger_loguru import get_logger

logger = get_logger("SendProductCardTool")


class SendProductCardParams(BaseModel):
    """Send product card params."""

    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(default=None, description="用户ID（账号ID）")
    recipient_uid: Optional[str] = Field(default=None, description="接收商品卡片的用户UID")
    goods_id: Optional[int] = Field(default=None, description="当前商品ID；已锁定商品时传入")
    query: Optional[str] = Field(default=None, description="客户原始问题，用于判断是否需要候选商品")


def _format_products_output(products: list, total: int) -> str:
    if not products:
        return "未找到可推荐商品。"

    lines = [f"可推荐商品列表（共{total}个，以下为前{len(products)}个）："]
    for index, product in enumerate(products, 1):
        goods_id = product.get("goods_id", "")
        goods_name = product.get("goods_name", "未命名商品")
        price = product.get("price", "")
        item = f"{index}. 商品名称：{goods_name}\n   商品ID：{goods_id}"
        if price:
            item += f"\n   价格：{price}元"
        lines.append(item)
    lines.append("如果客户没有明确选择哪一款，请先询问客户要哪一款，不要随便发送商品卡片。")
    return "\n\n".join(lines)


def _load_products(shop_id: Union[str, int], user_id: Union[str, int]) -> tuple[list, int, str]:
    product_manager = ProductManager(shop_id=shop_id, user_id=user_id)
    result = product_manager.get_product_list(page=1, size=10)
    if not result.get("success"):
        return [], 0, result.get("error_msg", "获取商品列表失败")
    return result.get("products", []), int(result.get("total", 0) or 0), ""


def _send_card(shop_id: Union[str, int], user_id: Union[str, int], recipient_uid: str, goods_id: int) -> str:
    if goods_id < 1000:
        logger.warning(f"商品ID可能错误: goods_id={goods_id} 太小，大概率是列表序号")
        return (
            f"发送失败：goods_id={goods_id} 无效。这个值像列表序号，不是真实商品ID；"
            "请根据候选商品里的商品ID重新选择。"
        )

    sender = SendMessage(str(shop_id), str(user_id))
    result = sender.send_mallGoodsCard(recipient_uid, goods_id, biz_type=2)
    if result and result.get("success"):
        logger.info(
            f"商品卡片发送成功: goods_id={goods_id}, recipient_uid={recipient_uid}, shop_id={shop_id}"
        )
        return "商品卡片发送成功"

    error_msg = result.get("error_msg", "发送失败") if result else "发送失败"
    logger.error(f"商品卡片发送失败: {error_msg}, goods_id={goods_id}, recipient_uid={recipient_uid}")
    return f"商品卡片发送失败: {error_msg}"


@agent_tool(
    name="send_product_card",
    description="发送当前商品卡片；如果未锁定商品，则查询候选商品列表供客户选择。",
    param_model=SendProductCardParams,
)
def send_product_card(params: SendProductCardParams) -> str:
    if not params.shop_id or not params.user_id:
        return "处理商品卡片失败：缺少 shop_id 或 user_id。"

    recipient_uid = str(params.recipient_uid or "").strip()
    goods_id = params.goods_id

    if goods_id:
        if not recipient_uid:
            return "发送商品卡片失败：缺少 recipient_uid。"
        return _send_card(params.shop_id, params.user_id, recipient_uid, int(goods_id))

    products, total, error_msg = _load_products(params.shop_id, params.user_id)
    if error_msg:
        return f"获取商品列表失败：{error_msg}"

    if len(products) == 1 and recipient_uid:
        only_goods_id = products[0].get("goods_id")
        if only_goods_id:
            return _send_card(params.shop_id, params.user_id, recipient_uid, int(only_goods_id))

    return _format_products_output(products, total)
