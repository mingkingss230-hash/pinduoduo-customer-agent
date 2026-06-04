from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base_request import BaseRequest

SIGNED_TRACE_KEYWORDS = (
    "包裹已签收",
    "包裹已签收！",
    "已签收",
    "快件已签收",
    "快件已签收，签收方式",
    "签收人是",
)


class OrderManager(BaseRequest):
    """Read-only PDD order queries for customer context."""

    USER_ALL_ORDER_URL = "https://mms.pinduoduo.com/latitude/order/userAllOrder"

    def __init__(self, shop_id: str = None, user_id: str = None, cookies=None):
        super().__init__(shop_id=shop_id, user_id=user_id)
        if cookies:
            self.update_cookies(cookies)

    def get_user_orders(self, uid: str, page_size: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent orders for a customer UID. This method never mutates order state."""
        uid_text = str(uid or "").strip()
        if not uid_text:
            return []

        anti_content = self.cookies.get("anti_content") or self.cookies.get("anti-content", "")
        headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "anti-content": anti_content,
            "Content-Type": "application/json",
            "origin": "https://mms.pinduoduo.com",
            "referer": "https://mms.pinduoduo.com/chat-merchant/index.html",
            "dnt": "1",
        }
        payload = {
            "pageNo": 1,
            "pageSize": int(page_size or 10),
            "showHistory": True,
            "uid": uid_text,
        }

        result = self.post(self.USER_ALL_ORDER_URL, json_data=payload, headers=headers, timeout=12)
        orders = ((result or {}).get("result") or {}).get("orders") or []
        if not isinstance(orders, list):
            self.logger.warning(f"订单列表响应格式异常: uid={uid_text}, result_type={type(orders).__name__}")
            return []
        return [order for order in orders if isinstance(order, dict)]


def build_order_context_text(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize PDD order response into compact text and scene hints."""
    normalized_orders = [_normalize_order_summary(order) for order in orders if isinstance(order, dict)]
    if not normalized_orders:
        return {
            "has_order": False,
            "scene_hint": "presale",
            "business_status": "无近期订单",
            "text": "【当前订单上下文】\n- 当前订单状态：无近期订单\n- 当前业务场景：售前/未查到订单",
        }

    scene_set = {
        item["scene_hint"]
        for item in normalized_orders
        if item.get("scene_hint")
    }
    if len(scene_set) > 1:
        lines = [
            "【当前订单上下文】",
            "- 当前客户存在多个不同状态订单，不能直接判断客户说的是哪一单",
            "- 当前业务场景：需要客户确认订单号",
        ]
        for item in normalized_orders[:5]:
            order_id = item.get("order_id") or "未知订单号"
            status = item.get("business_status") or "未知"
            goods_text = item.get("goods_text") or ""
            suffix = f"，商品：{goods_text}" if goods_text else ""
            lines.append(f"- 订单：{order_id}，状态：{status}{suffix}")
        return {
            "has_order": True,
            "needs_order_selection": True,
            "scene_hint": "mixed_orders",
            "business_status": "多个订单不同状态",
            "raw_count": len(normalized_orders),
            "orders": normalized_orders[:5],
            "text": "\n".join(lines),
        }

    primary = normalized_orders[0]
    lines = [
        "【当前订单上下文】",
        f"- 当前订单状态：{primary['business_status']}",
        f"- 当前业务场景：{_scene_label(primary['scene_hint'])}",
    ]
    if primary["order_id"]:
        lines.append(f"- 最近订单号：{primary['order_id']}")
    if primary["goods_text"]:
        lines.append(f"- 订单商品：{primary['goods_text']}")
    if primary["aftersale_status"]:
        lines.append(f"- 售后状态：{primary['aftersale_status']}")
    if primary["latest_trace"]:
        lines.append(f"- 最新物流状态：{primary['latest_trace']}")

    return {
        "has_order": True,
        "scene_hint": primary["scene_hint"],
        "business_status": primary["business_status"],
        "order_status": primary["order_status"],
        "shipping_status": primary["shipping_status"],
        "aftersale_status": primary["aftersale_status"],
        "latest_trace": primary["latest_trace"],
        "order_id": primary["order_id"],
        "goods_text": primary["goods_text"],
        "raw_count": len(normalized_orders),
        "text": "\n".join(lines),
    }


def _normalize_order_summary(order: Dict[str, Any]) -> Dict[str, str]:
    order_id = _first_present(order, "order_id", "order_sn", "orderSn", "orderSequenceNo", "id") or ""
    latest_trace = _extract_latest_trace(order)
    order_status = _normalize_order_status(order)
    shipping_status = _normalize_shipping_status(order)
    aftersale_status = _extract_aftersale_status(order)
    goods_text = _extract_goods_text(order)
    scene_hint = _infer_scene_hint(order_status, shipping_status, aftersale_status)
    business_status = _business_status_label(order_status, shipping_status, aftersale_status)
    return {
        "order_id": str(order_id),
        "order_status": order_status,
        "shipping_status": shipping_status,
        "aftersale_status": aftersale_status,
        "latest_trace": latest_trace,
        "goods_text": goods_text,
        "scene_hint": scene_hint,
        "business_status": business_status,
    }


def _first_present(data: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_order_status(order: Dict[str, Any]) -> str:
    raw = _first_present(order, "orderStatusStr", "order_status", "orderStatus", "status", "orderStatusDesc")
    text = str(raw or "").strip()
    if not text:
        return "unknown"
    lowered = text.casefold()
    rules = (
        (("待支付", "待付款", "pending_payment"), "pending_payment"),
        (("待发货", "已付款", "待成团", "processing", "paid"), "not_shipped"),
        (("已发货", "待收货", "shipped"), "shipped"),
        (("已签收", "已完成", "交易成功", "completed", "signed"), "signed"),
        (("已取消", "已关闭", "cancel"), "canceled"),
        (("退款中", "售后中", "refunding", "aftersale"), "aftersale"),
        (("已退款", "退款成功", "refunded"), "refunded"),
    )
    for tokens, normalized in rules:
        if any(token in lowered or token in text for token in tokens):
            return normalized
    return text


def _normalize_shipping_status(order: Dict[str, Any]) -> str:
    raw = _first_present(order, "shippingStatusStr", "shipping_status", "shippingStatus", "logisticsStatus")
    text = str(raw or "").strip()
    latest_trace = _extract_latest_trace(order)
    if _is_signed_delivery_trace(latest_trace):
        return "signed"

    combined = " ".join(
        part
        for part in (text, latest_trace, _extract_track_predict_text(order))
        if part
    )
    if not combined:
        return "unknown"
    rules = (
        (("未发货", "待发货", "not_shipped"), "not_shipped"),
        (("已发货", "运输中", "派送中", "待收货", "揽收", "中转", "shipped", "in_transit"), "in_transit"),
        (("signed", "delivered"), "signed"),
        (("退回", "拒收", "return"), "returning"),
    )
    lowered = combined.casefold()
    for tokens, normalized in rules:
        if any(token in lowered or token in combined for token in tokens):
            return normalized
    return text or "unknown"


def _extract_aftersale_status(order: Dict[str, Any]) -> str:
    direct = _first_present(
        order,
        "afterSalesStatusStr",
        "afterSaleStatusStr",
        "after_sales_status",
        "afterSalesStatus",
        "afterSaleStatus",
        "refundStatusStr",
        "refund_status",
    )
    if direct not in (None, ""):
        return str(direct).strip()

    for key in ("afterSalesInfo", "afterSaleInfo", "refundInfo", "afterSale"):
        info = order.get(key)
        if isinstance(info, dict):
            value = _first_present(info, "statusDesc", "statusStr", "status", "title", "desc")
            if value not in (None, ""):
                return str(value).strip()
    return ""


def _extract_latest_trace(order: Dict[str, Any]) -> str:
    final_trace = _extract_final_delivery_trace(order)
    if final_trace:
        return final_trace

    trace_items = order.get("traceInfoList") or order.get("trace_info_list") or order.get("trace_info") or []
    latest = _extract_latest_trace_from_items(trace_items)
    if latest:
        return latest
    return _extract_track_predict_text(order)


def _extract_final_delivery_trace(order: Dict[str, Any]) -> str:
    """Prefer final-leg logistics for consolidation orders after warehouse re-labeling."""
    for key in (
        "consolidationTraceInfoTwo",
        "splitDeliveryLogisticsOrderInfoVO",
    ):
        info = order.get(key)
        if isinstance(info, dict):
            latest = _extract_latest_trace_from_items(info.get("traceInfoList") or [])
            if latest:
                return latest

    warehouse_info = order.get("warehouseConsolidationTraceInfo")
    if isinstance(warehouse_info, dict):
        latest = _extract_latest_trace_from_items(warehouse_info.get("traceInfoList") or [])
        if latest:
            return latest
    return ""


def _extract_latest_trace_from_items(trace_items: Any) -> str:
    if isinstance(trace_items, list):
        for item in trace_items:
            if isinstance(item, dict):
                value = _first_present(item, "latest_node", "content", "info", "desc", "status_desc", "status", "title")
                if value not in (None, ""):
                    return str(value).strip()
            elif item:
                return str(item).strip()
    return ""


def _is_signed_delivery_trace(text: str) -> bool:
    value = str(text or "")
    return any(keyword in value for keyword in SIGNED_TRACE_KEYWORDS)


def _extract_track_predict_text(order: Dict[str, Any]) -> str:
    predict = order.get("trackPredictInfo")
    if not isinstance(predict, dict):
        return ""
    parts = [
        str(value).strip()
        for key in ("predictReplyText", "predictTimeText", "status")
        if (value := predict.get(key)) not in (None, "")
    ]
    return " ".join(parts)


def _extract_goods_text(order: Dict[str, Any]) -> str:
    values: List[str] = []
    for key in ("goods_name", "goodsName", "sku", "sku_name", "skuSpec", "goodsSpec"):
        value = order.get(key)
        if value not in (None, ""):
            values.append(str(value).strip())

    goods_list = order.get("orderGoodsList") or order.get("goodsList") or []
    if isinstance(goods_list, list):
        for item in goods_list:
            if not isinstance(item, dict):
                continue
            for key in ("goodsName", "goods_name", "skuSpec", "sku", "spec"):
                value = item.get(key)
                if value not in (None, ""):
                    values.append(str(value).strip())

    deduped: List[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return " / ".join(deduped[:3])


def _infer_scene_hint(order_status: str, shipping_status: str, aftersale_status: str) -> str:
    if shipping_status == "signed":
        return "aftersale"
    if order_status in {"not_shipped", "shipped", "aftersale", "refunded"} or shipping_status in {"not_shipped", "in_transit", "returning"}:
        return "insale"
    return "presale"


def _business_status_label(order_status: str, shipping_status: str, aftersale_status: str) -> str:
    if shipping_status == "signed":
        return "已签收"
    if order_status == "pending_payment":
        return "待支付"
    if order_status == "not_shipped" or shipping_status == "not_shipped":
        return "待发货"
    if order_status == "shipped" or shipping_status == "in_transit":
        return "已发货待收货"
    if order_status == "canceled":
        return "已取消"
    if order_status == "refunded":
        return "已退款"
    if order_status == "aftersale":
        return f"售后中（{aftersale_status or '状态未明'}）"
    return "未知"


def _scene_label(scene_hint: str) -> str:
    return {
        "presale": "售前/未查到订单",
        "insale": "售中",
        "aftersale": "售后倾向",
        "mixed_orders": "多订单待确认",
    }.get(scene_hint, scene_hint or "未知")
