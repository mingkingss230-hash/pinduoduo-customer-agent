#!/usr/bin/env python3
"""TurnContext 单元测试 — 覆盖 12 个场景。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Agent.CustomerAgent.custom.turn_context import (
    TurnContext,
    ProductCard,
    OrderCard,
    MediaInfo,
    TurnType,
    parse_turn_context,
    _price_fen_to_yuan,
)


def test_price_fen_to_yuan():
    """场景1: 价格分转元 — 1161 -> 11.61"""
    assert _price_fen_to_yuan("1161") == "11.61"
    assert _price_fen_to_yuan("2999") == "29.99"
    assert _price_fen_to_yuan("99") == "99"       # < 100 不转换
    assert _price_fen_to_yuan("12.5") == "12.5"   # 非纯数字不转换
    assert _price_fen_to_yuan("") == ""


def test_simple_text_only():
    """场景2: 纯文本消息"""
    tc = parse_turn_context("店家你好")
    assert tc.customer_text == "店家你好"
    assert tc.turn_type.has_text is True
    assert tc.turn_type.has_product_card is False
    assert tc.turn_type.has_order_card is False
    assert tc.turn_type.has_media is False
    assert tc.product_card.present is False
    assert tc.order_card.present is False


def test_product_card_with_fen_price():
    """场景3: 商品卡片 + 分价格"""
    raw = "内容：这款风扇怎么样\n商品卡片：商品：M11手持风扇，规格：白色款，价格：1161，商品ID：1234567"
    tc = parse_turn_context(raw)
    assert tc.customer_text == "这款风扇怎么样"
    assert tc.product_card.present is True
    assert tc.product_card.goods_id == "1234567"
    assert tc.product_card.goods_name == "M11手持风扇"
    assert tc.product_card.spec == "白色款"
    assert tc.product_card.price_raw == "1161"
    assert tc.product_card.price_yuan == "11.61"
    assert tc.turn_type.has_product_card is True


def test_order_card():
    """场景4: 订单卡片"""
    raw = "内容：我的快递到哪了\n订单卡片：订单号：260511-12345678，当前订单状态：已发货，订单主状态码：2，物流状态码：1，支付状态码：2，订单状态码：2，快递单号：SF1234567890"
    tc = parse_turn_context(raw)
    assert tc.customer_text == "我的快递到哪了"
    assert tc.order_card.present is True
    assert tc.order_card.order_sn == "260511-12345678"
    assert tc.order_card.order_status_text == "已发货"
    assert tc.order_card.main_status_code == "2"
    assert tc.order_card.logistics_status_code == "1"
    assert tc.order_card.payment_status_code == "2"
    assert tc.order_card.order_status_code == "2"
    assert tc.order_card.tracking_no == "SF1234567890"
    assert tc.turn_type.has_order_card is True


def test_media_image():
    """场景5: 图片消息"""
    raw = "内容：客户发送了图片：https://example.com/img1.jpg"
    tc = parse_turn_context(raw)
    assert tc.media.has_image is True
    assert tc.media.has_video is False
    assert "https://example.com/img1.jpg" in tc.media.image_urls
    assert tc.turn_type.has_media is True


def test_media_video():
    """场景6: 视频消息"""
    raw = "内容：[视频消息]"
    tc = parse_turn_context(raw)
    assert tc.media.has_video is True
    assert tc.media.has_image is False
    assert tc.turn_type.has_media is True


def test_customer_message_marker():
    """场景7: 客户消息：标记"""
    raw = "内容：客户消息：电池不耐用\n商品卡片：商品：M11，价格：999，商品ID：9999999"
    tc = parse_turn_context(raw)
    assert tc.customer_text == "电池不耐用"
    assert tc.product_card.present is True
    assert tc.product_card.goods_id == "9999999"
    assert "商品ID" not in tc.customer_text
    assert "9999999" not in tc.customer_text


def test_previous_customer_text():
    """场景8: 上一条客户问题"""
    raw = "内容：上一条客户问题：好的\n商品卡片：商品：M11，商品ID：8888888"
    tc = parse_turn_context(raw)
    assert tc.customer_text == "好的"
    assert tc.previous_customer_text == "好的"
    assert tc.product_card.present is True


def test_scene_hint():
    """场景9: 当前业务场景"""
    raw = "内容：这个能上飞机吗\n当前业务场景：presale"
    tc = parse_turn_context(raw)
    assert tc.raw_scene_hint == "presale"
    assert tc.customer_text == "这个能上飞机吗"


def test_empty_text_with_product_card():
    """场景10: 只有商品卡片、无客户文字"""
    raw = "商品卡片：商品：M11，价格：1999，商品ID：7777777"
    tc = parse_turn_context(raw)
    assert tc.customer_text == ""
    assert tc.turn_type.has_text is False
    assert tc.turn_type.has_product_card is True
    assert tc.product_card.present is True
    assert tc.product_card.goods_id == "7777777"


def test_empty_query():
    """场景11: 空查询"""
    tc = parse_turn_context("")
    assert tc.customer_text == ""
    assert tc.raw_query == ""
    assert tc.turn_type.has_text is False
    assert tc.turn_type.has_product_card is False
    assert tc.turn_type.has_order_card is False
    assert tc.turn_type.has_media is False


def test_metadata_stripping():
    """场景12: customer_text 不含元数据"""
    raw = "内容：你好\n商品卡片：商品：M11，商品ID：1234567"
    tc = parse_turn_context(raw)
    assert "1234567" not in tc.customer_text
    assert "商品ID" not in tc.customer_text
    assert "商品卡片" not in tc.customer_text
    assert tc.customer_text == "你好"


if __name__ == "__main__":
    tests = [
        test_price_fen_to_yuan,
        test_simple_text_only,
        test_product_card_with_fen_price,
        test_order_card,
        test_media_image,
        test_media_video,
        test_customer_message_marker,
        test_previous_customer_text,
        test_scene_hint,
        test_empty_text_with_product_card,
        test_empty_query,
        test_metadata_stripping,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\nResult: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
