"""
关键词检测处理器 - 检测转人工关键词并触发转人工流程
"""
import asyncio
from typing import Dict, Any
from bridge.context import Context, ContextType
from .base import BaseHandler
from database.db_manager import db_manager
from utils.logger_loguru import get_logger
from utils.night_mode import build_night_mode_key, get_night_mode_reply, is_night_mode
from utils.transfer_target import choose_transfer_candidate
from Channel.pinduoduo.utils.API.send_message import SendMessage

class KeywordDetectionHandler(BaseHandler):
    """关键词检测处理器 - 检测转人工关键词并触发转人工流程"""

    SAFE_TRANSFER_KEYWORDS = {
        "转人工", "人工客服", "真人客服", "找人工", "人工处理", "人工介入",
        "平台介入", "投诉", "举报",
        "退货地址", "退货码", "退款码", "取件码", "已寄回", "退回签收",
        "改地址", "地址错了", "拦截", "拒收", "补发", "开发票", "开票", "发票",
    }

    def __init__(self):
        super().__init__("KeywordDetectionHandler")
        self.logger = get_logger("KeywordDetectionHandler")
        self.keywords = self._load_keywords()

        # 记录加载的关键词数量
        self.logger.info(f"关键词检测处理器初始化完成，加载了 {len(self.keywords)} 个关键词")

    def _load_keywords(self):
        """从数据库加载关键词"""
        try:
            keywords_data = db_manager.get_all_keywords()
            raw_keywords = {item['keyword'].strip().lower() for item in keywords_data if item.get('keyword')}
            safe_keywords = {keyword.lower() for keyword in self.SAFE_TRANSFER_KEYWORDS}
            keywords = (raw_keywords & safe_keywords) | safe_keywords
            ignored = raw_keywords - safe_keywords
            if ignored:
                self.logger.warning(f"已忽略宽泛转人工关键词，避免误转: {sorted(ignored)}")
            self.logger.debug(f"生效转人工关键词: {keywords}")
            return keywords
        except Exception as e:
            self.logger.error(f"加载关键词失败: {e}")
            default_keywords = {keyword.lower() for keyword in self.SAFE_TRANSFER_KEYWORDS}
            self.logger.warning(f"使用默认关键词: {default_keywords}")
            return default_keywords

    def can_handle(self, context: Context) -> bool:
        """检查消息是否包含关键词"""
        self.logger.debug("关键词转人工入口已禁用，统一交由订单状态和Agent主链路处理")
        return False

        # 只处理文本类型的消息
        if context.type != ContextType.TEXT:
            return False

        # 检查消息内容是否存在且为字符串
        if not context.content or not isinstance(context.content, str):
            return False

        # 将消息内容转换为小写进行检测
        content_lower = context.content.lower()

        # 检查是否包含任何关键词
        for keyword in self.keywords:
            if keyword in content_lower:
                self.logger.debug(f"检测到关键词: '{keyword}' 在消息: '{context.content}'")
                return True

        return False

    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """转接到人工客服"""
        try:
            kwargs = context.kwargs
            shop_id = getattr(kwargs, 'shop_id', None) or (kwargs.get('shop_id') if isinstance(kwargs, dict) else None)
            user_id = getattr(kwargs, 'user_id', None) or (kwargs.get('user_id') if isinstance(kwargs, dict) else None)
            from_uid = getattr(kwargs, 'from_uid', None) or (kwargs.get('from_uid') if isinstance(kwargs, dict) else None)
            
            if not all([shop_id, user_id, from_uid]):
                return False

            if is_night_mode():
                reply = get_night_mode_reply(build_night_mode_key(shop_id, user_id, from_uid))
                sender = SendMessage(shop_id, user_id)
                send_result = await asyncio.to_thread(sender.send_text, from_uid, reply)
                if isinstance(send_result, dict) and send_result.get("success"):
                    self.logger.info(
                        f"夜间模式拦截关键词转人工: shop_id={shop_id}, user_id={user_id}, from_uid={from_uid}"
                    )
                    return True
                self.logger.warning(f"夜间模式提示发送失败: result={send_result}")
                return False
            
            # 获取可用的客服列表
            sender = SendMessage(shop_id, user_id)
            cs_list = await asyncio.to_thread(sender.getAssignCsList)
            
            if cs_list and isinstance(cs_list, dict):
                preferred = db_manager.get_transfer_target(
                    "pinduoduo",
                    str(shop_id),
                    str(user_id),
                )
                candidate = choose_transfer_candidate(
                    str(shop_id),
                    str(user_id),
                    cs_list,
                    preferred.get("target_user_id") if preferred else None,
                )

                if candidate:
                    cs_uid = candidate["raw_cs_uid"]
                    target_cs = cs_list[cs_uid]
                    cs_name = candidate["username"] or target_cs.get('username', '客服')

                    reply_sent = False
                    try:
                        send_result = await asyncio.to_thread(
                            sender.send_text,
                            from_uid,
                            "亲，我这边帮您转人工处理，请稍等。",
                        )
                        reply_sent = isinstance(send_result, dict) and send_result.get("success")
                    except Exception as send_error:
                        self.logger.warning(f"转人工前客户提示发送失败: {send_error}")
                    
                    # 转移会话
                    transfer_result = await asyncio.to_thread(sender.move_conversation, from_uid, cs_uid)
                    
                    if transfer_result and transfer_result.get('success'):
                        if not reply_sent:
                            try:
                                await asyncio.to_thread(sender.send_text, from_uid, "亲，已转人工为您处理，请稍等。")
                            except Exception as send_error:
                                self.logger.warning(f"转人工后客户提示发送失败: {send_error}")

                        self.logger.info(
                            f"会话已成功转接给 {cs_name} ({cs_uid}), "
                            f"configured_cs_uid={candidate['cs_uid']}"
                        )
                        return True
                    else:
                        self.logger.error("会话转接失败")
                        if reply_sent:
                            try:
                                await asyncio.to_thread(
                                    sender.send_text,
                                    from_uid,
                                    "亲，转人工暂时没成功，您先把问题发我，我这边继续帮您看。",
                                )
                            except Exception as send_error:
                                self.logger.warning(f"转人工失败提示发送失败: {send_error}")
                            return True
                else:
                    if preferred and preferred.get("target_user_id"):
                        self.logger.warning(
                            f"指定转人工客服不在可转接列表中: target_user_id={preferred.get('target_user_id')}"
                        )
                        await asyncio.to_thread(sender.send_text, from_uid, "抱歉，当前指定人工客服暂不可转接，请您稍后再试。")
                    else:
                        self.logger.warning("没有其他可用的客服进行转接")
                        await asyncio.to_thread(sender.send_text, from_uid, "抱歉，当前没有其他客服在线，请您稍后再试。")
            
            return False
            
        except Exception as e:
            self.logger.error(f"客服转接处理失败: {e}")
            return False
            
    def reload_keywords(self) -> None:
        """重新加载关键词（用于管理员更新关键词后刷新）"""
        old_count = len(self.keywords)
        self.keywords = self._load_keywords()
        new_count = len(self.keywords)
        self.logger.info(f"关键词重新加载完成: {old_count} -> {new_count}")

    def get_keyword_count(self) -> int:
        """获取当前关键词数量"""
        return len(self.keywords)

    def get_keywords(self) -> set:
        """获取当前关键词列表"""
        return self.keywords.copy()
