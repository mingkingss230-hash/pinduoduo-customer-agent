"""
拼多多消息处理类
"""
from __future__ import annotations

import json
from enum import IntEnum
from typing import Any, Dict

from bridge.context import ContextType
from Message.message import ChatMessage


class PDDMsgType(IntEnum):
    """拼多多消息类型枚举"""

    TEXT = 0
    IMAGE = 1
    SERVICE_TODO = 8
    VIDEO = 14
    SYSTEM_HINT = 31
    GOODS_SOURCE = 41
    WITHDRAW = 1002
    EMOTION = 5
    GOODS_SPEC = 64
    TRANSFER = 24


class PDDSubType(IntEnum):
    """拼多多消息子类型枚举"""

    ORDER_INFO = 1
    GOODS_INQUIRY = 0


def _safe_get(data: Any, *keys: Any, default=None) -> Any:
    """安全获取嵌套字典/列表值。"""
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        elif isinstance(result, list) and isinstance(key, int):
            if 0 <= key < len(result):
                result = result[key]
            else:
                return default
        else:
            return default
        if result is None:
            return default
    return result


def _first_non_null(*values: Any) -> Any:
    """返回第一个非空值。"""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


class BaseMessageHandler:
    def __init__(self, msg):
        self.msg = msg
        self.data = msg.get("message", {})

    def get_basic_info(self):
        """获取基础信息"""
        return {
            "msg_id": self.data.get("msg_id"),
            "nickname": self.data.get("nickname"),
            "from_role": self.data.get("from", {}).get("role"),
            "from_uid": self.data.get("from", {}).get("uid"),
            "to_role": self.data.get("to", {}).get("role"),
            "to_uid": self.data.get("to", {}).get("uid"),
            "timestamp": self.data.get("time"),
        }


class MessageTypeHandler:
    """消息类型处理类"""

    @staticmethod
    def _parse_embedded_json(raw: Any) -> Dict[str, Any]:
        """解析嵌在字段里的 JSON 字符串。"""
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _get_content(msg_data: Dict[str, Any], context_type: ContextType, path: tuple) -> tuple:
        return context_type, _safe_get(msg_data, *path)

    @staticmethod
    def handle_text(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.TEXT, ("message", "content"))

    @staticmethod
    def handle_image(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.IMAGE, ("message", "content"))

    @staticmethod
    def handle_video(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.VIDEO, ("message", "content"))

    @staticmethod
    def handle_emotion(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.EMOTION, ("info", "description"))

    @staticmethod
    def handle_withdraw(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.WITHDRAW, ("info", "withdraw_hint"))

    @staticmethod
    def _extract_goods_fields(msg_data: Dict[str, Any]) -> Dict[str, Any]:
        """多路径提取商品信息，兼容不同卡片结构。"""
        callback_payload = MessageTypeHandler._parse_embedded_json(
            _safe_get(msg_data, "message", "biz_context", "mallOrderConfirmNewCardCallBackParam")
        )
        callback_payload_agree = MessageTypeHandler._parse_embedded_json(
            _safe_get(msg_data, "message", "biz_context", "orderConfirmAgreeNewCardCallBackParam")
        )
        embedded_payloads = [callback_payload, callback_payload_agree]

        return {
            "goods_id": _first_non_null(
                _safe_get(msg_data, "message", "info", "data", "goodsID"),
                _safe_get(msg_data, "message", "info", "data", "goodsId"),
                _safe_get(msg_data, "message", "info", "data", "goods_id"),
                _safe_get(msg_data, "message", "info", "goodsID"),
                _safe_get(msg_data, "message", "info", "goodsId"),
                _safe_get(msg_data, "message", "info", "goods_info", "goods_id"),
                _safe_get(msg_data, "message", "info", "data", "goods_info", "goods_id"),
                _safe_get(msg_data, "message", "info", "data", "button_click_action", "params", "goods_id"),
                _safe_get(msg_data, "message", "orders", 0, "orderGoodsList", "goodsId"),
                _safe_get(msg_data, "message", "orders", 0, "goodsId"),
                *(payload.get("goodsId") for payload in embedded_payloads),
            ),
            "goods_name": _first_non_null(
                _safe_get(msg_data, "message", "info", "data", "goodsName"),
                _safe_get(msg_data, "message", "info", "data", "goods_name"),
                _safe_get(msg_data, "message", "info", "goodsName"),
                _safe_get(msg_data, "message", "info", "goods_info", "goods_name"),
                _safe_get(msg_data, "message", "info", "data", "goods_info", "goods_name"),
                _safe_get(msg_data, "message", "orders", 0, "orderGoodsList", "goodsName"),
                *(payload.get("goodsName") for payload in embedded_payloads),
            ),
            "goods_price": _first_non_null(
                _safe_get(msg_data, "message", "info", "data", "goodsPrice"),
                _safe_get(msg_data, "message", "info", "data", "goods_price"),
                _safe_get(msg_data, "message", "info", "goodsPrice"),
                _safe_get(msg_data, "message", "info", "goods_info", "total_amount"),
                _safe_get(msg_data, "message", "info", "data", "goods_info", "total_amount"),
                _safe_get(msg_data, "message", "info", "data", "goods_info", "merchant_amount"),
                _safe_get(msg_data, "message", "info", "data", "coupon_promo_price"),
                _safe_get(msg_data, "message", "orders", 0, "orderGoodsList", "goodsPrice"),
                *(payload.get("merchantAmount") for payload in embedded_payloads),
                *(payload.get("totalAmount") for payload in embedded_payloads),
            ),
            "goods_thumb_url": _first_non_null(
                _safe_get(msg_data, "message", "info", "data", "goodsThumbUrl"),
                _safe_get(msg_data, "message", "info", "data", "thumb_url"),
                _safe_get(msg_data, "message", "info", "goodsThumbUrl"),
                _safe_get(msg_data, "message", "info", "goods_info", "goods_thumb_url"),
                _safe_get(msg_data, "message", "info", "data", "goods_info", "goods_thumb_url"),
                _safe_get(msg_data, "message", "orders", 0, "orderGoodsList", "thumbUrl"),
                *(payload.get("goodsThumbUrl") for payload in embedded_payloads),
            ),
            "link_url": _first_non_null(
                _safe_get(msg_data, "message", "info", "data", "linkUrl"),
                _safe_get(msg_data, "message", "info", "data", "link_url"),
                _safe_get(msg_data, "message", "info", "linkUrl"),
                _safe_get(msg_data, "message", "info", "goods_info", "mall_link_url"),
                _safe_get(msg_data, "message", "info", "data", "m_app_jump_url"),
                _safe_get(msg_data, "message", "info", "data", "button_click_action", "params", "jump_url"),
                _safe_get(msg_data, "message", "info", "data", "button_click_action", "params", "app_jump_url"),
                _safe_get(msg_data, "message", "info", "data", "click_action", "params", "jump_url"),
                _safe_get(msg_data, "message", "info", "data", "click_action", "params", "app_jump_url"),
                _safe_get(msg_data, "message", "info", "data", "m_button", "click_action", "params", "jump_url"),
                _safe_get(msg_data, "message", "info", "data", "click_action", "params", "jump_url"),
                _safe_get(msg_data, "message", "info", "data", "click_action", "params", "app_jump_url"),
                *(payload.get("linkUrl") for payload in embedded_payloads),
            ),
            "goods_spec": _first_non_null(
                _safe_get(msg_data, "message", "info", "data", "spec"),
                _safe_get(msg_data, "message", "info", "data", "goods_spec"),
                _safe_get(msg_data, "message", "info", "data", "goods_info", "extra"),
                _safe_get(msg_data, "message", "info", "data", "sku_data"),
                _safe_get(msg_data, "message", "info", "spec"),
                _safe_get(msg_data, "message", "orders", 0, "orderGoodsList", "spec"),
                *(payload.get("spec") for payload in embedded_payloads),
            ),
        }

    @staticmethod
    def handle_goods_inquiry(msg_data):
        """处理商品咨询消息"""
        goods_info = MessageTypeHandler._extract_goods_fields(msg_data)
        return ContextType.GOODS_INQUIRY, goods_info

    @staticmethod
    def handle_goods_spec(msg_data):
        """处理商品规格咨询消息"""
        goods_info = MessageTypeHandler._extract_goods_fields(msg_data)
        return ContextType.GOODS_SPEC, goods_info

    @staticmethod
    def handle_order_info(msg_data):
        """处理订单信息消息"""
        goods_info = MessageTypeHandler._extract_goods_fields(msg_data)
        order_info = {
            "order_id": _safe_get(msg_data, "message", "info", "orderSequenceNo"),
            "goods_id": goods_info.get("goods_id"),
            "goods_name": goods_info.get("goods_name"),
            "order_status": _safe_get(msg_data, "message", "info", "order_status"),
            "shipping_status": _safe_get(msg_data, "message", "info", "shipping_status"),
            "pay_status": _safe_get(msg_data, "message", "info", "pay_status"),
            "status": _safe_get(msg_data, "message", "info", "status"),
            "tracking_number": _safe_get(msg_data, "message", "info", "tracking_number"),
            "shipping_time": _safe_get(msg_data, "message", "info", "shipping_time"),
            "afterSalesStatus": _safe_get(msg_data, "message", "info", "afterSalesStatus"),
            "afterSalesType": _safe_get(msg_data, "message", "info", "afterSalesType"),
            "spec": goods_info.get("goods_spec"),
            "link_url": goods_info.get("link_url"),
        }
        return ContextType.ORDER_INFO, order_info

    @staticmethod
    def handle_mall_system_msg(msg_data):
        """处理商城消息"""
        system_msg = {
            "user_id": _safe_get(msg_data, "message", "data", "user_id"),
        }
        return ContextType.MALL_SYSTEM_MSG, system_msg

    @staticmethod
    def handle_auth(msg_data):
        """处理认证消息"""
        auth_info = {
            "uid": _safe_get(msg_data, "uid"),
            "result": _safe_get(msg_data, "auth", "result"),
            "status": _safe_get(msg_data, "status"),
        }
        return ContextType.AUTH, auth_info

    @staticmethod
    def handle_transfer(msg_data):
        """处理转接消息"""
        transfer_info = {
            "from_uid": _safe_get(msg_data, "message", "from", "uid"),
            "to_uid": _safe_get(msg_data, "message", "to", "uid"),
        }
        return ContextType.TRANSFER, transfer_info

    @staticmethod
    def handle_system_hint(msg_data):
        """处理系统提示/工单卡片消息"""
        content = _first_non_null(
            _safe_get(msg_data, "message", "content"),
            _safe_get(msg_data, "message", "info", "data", "title"),
            _safe_get(msg_data, "message", "info", "data", "summary"),
            _safe_get(msg_data, "message", "info", "title"),
            _safe_get(msg_data, "message", "info", "mall_content"),
        )
        return ContextType.SYSTEM_HINT, content or "[系统提示]"

    @staticmethod
    def is_system_action_card(msg_data: Dict[str, Any]) -> bool:
        """识别平台售后/工单卡，避免 type64 被误当商品规格卡给 Agent。"""
        template_name = str(_safe_get(msg_data, "message", "template_name") or "")
        info_key = str(_safe_get(msg_data, "message", "info", "key") or "")
        content = str(_safe_get(msg_data, "message", "content") or "")
        title = str(_safe_get(msg_data, "message", "info", "data", "title") or "")

        signals = (
            "aftersale",
            "mediate",
            "service_todo",
            "work_order",
            "warning",
        )
        haystack = f"{template_name} {info_key}".lower()
        if any(signal in haystack for signal in signals):
            return True
        return any(text in f"{content} {title}" for text in ("平台已帮您梳理售后问题", "工单", "消费者服务工单"))


class PDDChatMessage(ChatMessage):
    """拼多多消息实现类"""

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg
        self.base_handler = BaseMessageHandler(msg)

        basic_info = self.base_handler.get_basic_info()
        self.msg_id = basic_info.get("msg_id")
        self.nickname = basic_info.get("nickname")
        self.from_user = basic_info.get("from_role")
        self.from_uid = basic_info.get("from_uid")
        self.to_user = basic_info.get("to_role")
        self.to_uid = basic_info.get("to_uid")

        if self.from_user == "mall_cs":
            self.user_msg_type = ContextType.MALL_CS
            self.content = self.msg.get("message", {}).get("content")
            return

        self._process_message()

    def _process_message(self):
        """处理消息"""
        self.msg_type = self.msg.get("response")
        if self.msg_type == "push":
            user_msg_type = self.msg.get("message", {}).get("type")
            if user_msg_type == PDDMsgType.TEXT:
                sub_type = self.msg.get("message", {}).get("sub_type")
                if sub_type == PDDSubType.ORDER_INFO:
                    self.user_msg_type, self.content = MessageTypeHandler.handle_order_info(self.msg)
                elif sub_type == PDDSubType.GOODS_INQUIRY:
                    self.user_msg_type, self.content = MessageTypeHandler.handle_goods_inquiry(self.msg)
                else:
                    self.user_msg_type, self.content = MessageTypeHandler.handle_text(self.msg)
            elif user_msg_type == PDDMsgType.IMAGE:
                self.user_msg_type, self.content = MessageTypeHandler.handle_image(self.msg)
            elif user_msg_type == PDDMsgType.VIDEO:
                self.user_msg_type, self.content = MessageTypeHandler.handle_video(self.msg)
            elif user_msg_type == PDDMsgType.SYSTEM_HINT:
                self.user_msg_type, self.content = MessageTypeHandler.handle_system_hint(self.msg)
            elif user_msg_type == PDDMsgType.SERVICE_TODO:
                self.user_msg_type, self.content = MessageTypeHandler.handle_system_hint(self.msg)
            elif user_msg_type == PDDMsgType.WITHDRAW:
                self.user_msg_type, self.content = MessageTypeHandler.handle_withdraw(self.msg)
            elif user_msg_type == PDDMsgType.EMOTION:
                self.user_msg_type, self.content = MessageTypeHandler.handle_emotion(self.msg)
            elif user_msg_type == PDDMsgType.GOODS_SPEC:
                if MessageTypeHandler.is_system_action_card(self.msg):
                    self.user_msg_type, self.content = MessageTypeHandler.handle_system_hint(self.msg)
                else:
                    self.user_msg_type, self.content = MessageTypeHandler.handle_goods_spec(self.msg)
            elif user_msg_type == PDDMsgType.GOODS_SOURCE:
                self.user_msg_type, self.content = MessageTypeHandler.handle_goods_inquiry(self.msg)
            elif user_msg_type == PDDMsgType.TRANSFER:
                self.user_msg_type, self.content = MessageTypeHandler.handle_transfer(self.msg)
            else:
                self.user_msg_type = ContextType.SYSTEM_STATUS
                self.content = f"不支持的消息类型: {user_msg_type}"
        elif self.msg_type == "auth":
            self.user_msg_type, self.content = MessageTypeHandler.handle_auth(self.msg)
        elif self.msg_type == "mall_system_msg":
            self.user_msg_type, self.content = MessageTypeHandler.handle_mall_system_msg(self.msg)
        else:
            self.user_msg_type = ContextType.SYSTEM_STATUS
            self.content = f"不支持的消息类型: {self.msg_type}"
