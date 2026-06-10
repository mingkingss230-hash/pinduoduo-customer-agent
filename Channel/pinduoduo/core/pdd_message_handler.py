# 娑堟伅澶勭悊妯″潡
import asyncio
import json
import time

from websockets import exceptions as ws_exceptions

from bridge.context import ChannelType, Context, ContextType
from Channel.pinduoduo.pdd_message import PDDChatMessage
from database import db_manager
from utils.logger_loguru import get_logger


class MessageHandlerMixin:
    """娑堟伅澶勭悊 Mixin"""

    QUEUE_DEBOUNCE_SECONDS = 1.0
    RECENT_TEXT_TTL_SECONDS = 90.0
    CARD_CONTEXT_TYPES = {
        ContextType.GOODS_INQUIRY,
        ContextType.GOODS_SPEC,
        ContextType.ORDER_INFO,
        ContextType.GOODS_CARD,
    }

    @staticmethod
    def _safe_json_dumps(data):
        """Safely serialize websocket data for trace logs."""
        try:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(data)

    @staticmethod
    def _extract_ws_meta(message_data):
        """Build a compact summary for websocket packets."""
        if not isinstance(message_data, dict):
            return {"response": None, "type": None, "sub_type": None}

        message = message_data.get("message", {})
        if not isinstance(message, dict):
            message = {}

        from_info = message.get("from", {})
        if not isinstance(from_info, dict):
            from_info = {}

        to_info = message.get("to", {})
        if not isinstance(to_info, dict):
            to_info = {}

        return {
            "response": message_data.get("response"),
            "type": message.get("type"),
            "sub_type": message.get("sub_type"),
            "msg_id": message.get("msg_id"),
            "from_uid": from_info.get("uid"),
            "from_role": from_info.get("role"),
            "to_uid": to_info.get("uid"),
            "to_role": to_info.get("role"),
            "nickname": message.get("nickname"),
            "time": message.get("time"),
        }

    def _log_websocket_raw(self, message_data, shop_id: str, user_id: str, username: str):
        """Log raw websocket payload for later field-path analysis."""
        meta = self._extract_ws_meta(message_data)
        self.logger.info(
            "PDD_WS_RAW shop_id=%s user_id=%s username=%s meta=%s payload=%s"
            % (
                shop_id,
                user_id,
                username,
                self._safe_json_dumps(meta),
                self._safe_json_dumps(message_data),
            )
        )

    def _log_websocket_parsed(self, pdd_message: PDDChatMessage, context: Context, queue_name: str):
        """Log parsed result to compare against the raw websocket payload."""
        kwargs = getattr(context, "kwargs", None)
        parsed_snapshot = {
            "queue_name": queue_name,
            "context_type": str(context.type) if context and context.type else "",
            "msg_id": getattr(kwargs, "msg_id", ""),
            "from_uid": getattr(kwargs, "from_uid", ""),
            "to_uid": getattr(kwargs, "to_uid", ""),
            "nickname": getattr(kwargs, "nickname", ""),
            "content": context.content if context else "",
            "pdd_user_msg_type": str(pdd_message.user_msg_type) if pdd_message.user_msg_type else "",
        }
        self.logger.info(f"PDD_WS_PARSED {self._safe_json_dumps(parsed_snapshot)}")

    async def _setup_message_consumer(self, queue_name: str):
        """璁剧疆娑堟伅娑堣垂鑰呭拰澶勭悊鍣ㄩ摼"""
        from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
        from Message import handler_chain, message_consumer_manager, queue_manager

        try:
            existing_consumer = message_consumer_manager.get_consumer(queue_name)
            if existing_consumer:
                self.logger.info(f"娑堣垂鑰?{queue_name} 宸插瓨鍦紝鍏堝仠姝㈠苟閲嶆柊鍒涘缓")
                try:
                    await message_consumer_manager.stop_consumer(queue_name)
                except Exception as e:
                    self.logger.warning(f"鍋滄鏃ф秷璐硅€呭け璐? {queue_name}, {e}")
                try:
                    queue_manager.recreate_queue(queue_name)
                except Exception as e:
                    self.logger.warning(f"閲嶆柊鍒涘缓闃熷垪澶辫触: {queue_name}, {e}")

            consumer = message_consumer_manager.create_consumer(queue_name, max_concurrent=10)

            try:
                from core.di_container import container

                bot = container.get(CustomerAgent)
            except Exception:
                bot = CustomerAgent()
            handlers = handler_chain(use_ai=True, businessHours=self.businessHours, bot=bot)
            for handler in handlers:
                consumer.add_handler(handler)

            await message_consumer_manager.start_consumer(queue_name)
            self.logger.debug(f"娑堟伅娑堣垂鑰呭凡鍚姩: {queue_name}")

        except Exception as e:
            self.logger.error(f"璁剧疆娑堟伅娑堣垂鑰呭け璐? {e}")
            raise

    @staticmethod
    def _conversation_key(queue_name: str, context: Context) -> tuple:
        kwargs = getattr(context, "kwargs", None)
        return (
            queue_name,
            str(getattr(kwargs, "shop_id", "") or ""),
            str(getattr(kwargs, "user_id", "") or ""),
            str(getattr(kwargs, "from_uid", "") or ""),
        )

    @staticmethod
    def _copy_context(context: Context, *, content: str, context_type: ContextType, msg_id: str) -> Context:
        kwargs = getattr(context, "kwargs", None)
        if hasattr(kwargs, "model_copy"):
            new_kwargs = kwargs.model_copy(update={"msg_id": msg_id, "user_msg_type": context_type})
        elif hasattr(kwargs, "copy"):
            new_kwargs = kwargs.copy(update={"msg_id": msg_id, "user_msg_type": context_type})
        else:
            new_kwargs = kwargs

        if hasattr(context, "model_copy"):
            return context.model_copy(update={"content": content, "type": context_type, "kwargs": new_kwargs})
        return context.copy(update={"content": content, "type": context_type, "kwargs": new_kwargs})

    @staticmethod
    def _parse_context_json(content: str) -> dict:
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _summarize_context_for_agent(self, context: Context) -> str:
        content = str(context.content or "").strip()
        if context.type == ContextType.TEXT:
            return f"客户消息：{content}" if content else ""
        if context.type in {ContextType.IMAGE, ContextType.VIDEO}:
            return f"客户发送了{'图片' if context.type == ContextType.IMAGE else '视频'}：{content}"
        if context.type in self.CARD_CONTEXT_TYPES:
            data = self._parse_context_json(content)
            if context.type == ContextType.ORDER_INFO:
                parts = [
                    f"订单号：{data.get('order_id') or ''}",
                    f"商品：{data.get('goods_name') or ''}",
                    f"商品ID：{data.get('goods_id') or ''}",
                    f"规格：{data.get('spec') or ''}",
                    f"订单主状态码：{data.get('order_status') if data.get('order_status') is not None else ''}",
                    f"物流状态码：{data.get('shipping_status') if data.get('shipping_status') is not None else ''}",
                    f"订单状态码：{data.get('status') if data.get('status') is not None else ''}",
                ]
                return "订单卡片：" + "，".join(part for part in parts if not part.endswith("："))

            parts = [
                f"商品：{data.get('goods_name') or ''}",
                f"商品ID：{data.get('goods_id') or ''}",
                f"价格：{data.get('goods_price') or ''}",
                f"规格：{data.get('goods_spec') or data.get('spec') or ''}",
                f"链接：{data.get('link_url') or ''}",
            ]
            return "商品卡片：" + "，".join(part for part in parts if not part.endswith("："))
        return content

    def _get_recent_text_for_context(self, key: tuple) -> str:
        recent = getattr(self, "_pdd_recent_customer_text", {}).get(key)
        if not recent:
            return ""
        text, created_at = recent
        if time.monotonic() - created_at > self.RECENT_TEXT_TTL_SECONDS:
            return ""
        return text

    def _merge_contexts_for_queue(self, key: tuple, contexts: list[Context]) -> Context:
        base = contexts[-1]
        msg_ids = []
        summaries = []
        has_text = False

        for item in contexts:
            kwargs = getattr(item, "kwargs", None)
            msg_id = str(getattr(kwargs, "msg_id", "") or "").strip()
            if msg_id:
                msg_ids.append(msg_id)
            if item.type == ContextType.TEXT:
                has_text = True
            summary = self._summarize_context_for_agent(item)
            if summary and summary not in summaries:
                summaries.append(summary)

        if not has_text and base.type in self.CARD_CONTEXT_TYPES:
            recent_text = self._get_recent_text_for_context(key)
            if recent_text:
                summaries.insert(0, f"上一条客户问题：{recent_text}")
                has_text = True

        if len(summaries) <= 1 and not has_text:
            return base

        merged_type = ContextType.TEXT if has_text else base.type
        merged_content = "\n".join(summaries).strip()
        merged_msg_id = "+".join(msg_ids[-4:]) if msg_ids else str(getattr(base.kwargs, "msg_id", "") or "")
        return self._copy_context(base, content=merged_content, context_type=merged_type, msg_id=merged_msg_id)

    async def _flush_debounced_context(self, queue_name: str, key: tuple):
        try:
            await asyncio.sleep(self.QUEUE_DEBOUNCE_SECONDS)
            buffers = getattr(self, "_pdd_reply_buffers", {})
            buffer = buffers.pop(key, None)
            if not buffer:
                return

            contexts = buffer.get("contexts") or []
            if not contexts:
                return

            merged_context = self._merge_contexts_for_queue(key, contexts)
            from Message import put_message

            msg_id = await put_message(queue_name, merged_context)
            self.logger.debug(
                f"合并入队: queue={queue_name}, merged_msg_id={msg_id}, "
                f"source_count={len(contexts)}, type={merged_context.type}"
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.logger.error(f"合并消息入队失败: queue={queue_name}, key={key}, error={exc}")

    async def _queue_message_with_debounce(self, queue_name: str, context: Context):
        if not hasattr(self, "_pdd_reply_buffers"):
            self._pdd_reply_buffers = {}
        if not hasattr(self, "_pdd_recent_customer_text"):
            self._pdd_recent_customer_text = {}

        key = self._conversation_key(queue_name, context)
        if context.type == ContextType.TEXT and str(context.content or "").strip():
            self._pdd_recent_customer_text[key] = (str(context.content).strip(), time.monotonic())

        buffer = self._pdd_reply_buffers.setdefault(key, {"contexts": [], "task": None})
        buffer["contexts"].append(context)

        old_task = buffer.get("task")
        if old_task and not old_task.done():
            old_task.cancel()

        buffer["task"] = asyncio.create_task(self._flush_debounced_context(queue_name, key))

    async def _process_websocket_message(self, message: str, shop_id: str, user_id: str, username: str, queue_name: str):
        """澶勭悊鍗曟潯WebSocket娑堟伅"""
        try:
            if not message or not message.strip():
                self.logger.debug(f"鏀跺埌绌烘秷鎭紝璺宠繃澶勭悊: {shop_id}-{username}")
                return

            message_data = json.loads(message)
            # self._log_websocket_raw(message_data, shop_id, user_id, username)  # 已关闭原始数据日志
            msg_type = message_data.get("message", {}).get("type", "unknown")
            from_uid_log = message_data.get("message", {}).get("from_uid", "unknown")
            self.logger.debug(f"鏀跺埌娑堟伅: type={msg_type}, from_uid={from_uid_log}, shop_id={shop_id}")

            try:
                pdd_message = PDDChatMessage(message_data)
            except Exception as pdd_error:
                self.logger.error(f"鍒涘缓PDD娑堟伅瀵硅薄澶辫触: {shop_id}-{username}, 閿欒: {pdd_error}")
                return

            try:
                context = self._convert_to_context(pdd_message, shop_id, user_id, username)
                if not context:
                    self.logger.debug(f"娑堟伅杞崲澶辫触锛岃烦杩囧鐞? {shop_id}-{username}")
                    return
                # self._log_websocket_parsed(pdd_message, context, queue_name)  # 已关闭解析数据日志
            except Exception as ctx_error:
                self.logger.error(f"杞崲Context澶辫触: {shop_id}-{username}, 閿欒: {ctx_error}")
                return

            if context:
                if self._should_process_immediately(context):
                    await self._handle_immediate_message(context, shop_id, user_id)
                    self.logger.debug(f"绔嬪嵆澶勭悊娑堟伅: {context.type}, ID: {pdd_message.msg_id}")
                elif self._should_queue_message(context):
                    await self._queue_message_with_debounce(queue_name, context)
                    self.logger.debug(f"消息等待合并入队: {queue_name}, 类型: {context.type}, ID: {pdd_message.msg_id}")
                else:
                    self.logger.debug(f"蹇界暐娑堟伅: {context.type}, ID: {pdd_message.msg_id}")
            else:
                self.logger.warning("娑堟伅杞崲澶辫触锛岃烦杩囧鐞?")

        except json.JSONDecodeError:
            self.logger.error(f"JSON瑙ｆ瀽澶辫触: {message}")
        except Exception as e:
            self.logger.error(f"澶勭悊WebSocket娑堟伅澶辫触: {e}")

    def _should_process_immediately(self, context: Context) -> bool:
        """鍒ゆ柇娑堟伅鏄惁闇€瑕佺珛鍗冲鐞?"""
        immediate_types = {
            ContextType.SYSTEM_STATUS,
            ContextType.AUTH,
            ContextType.WITHDRAW,
            ContextType.SYSTEM_HINT,
            ContextType.MALL_CS,
            ContextType.TRANSFER,
        }
        return context.type in immediate_types

    def _should_queue_message(self, context: Context) -> bool:
        """鍒ゆ柇娑堟伅鏄惁闇€瑕佹斁鍏ラ槦鍒楀鐞?"""
        queue_types = {
            ContextType.TEXT,
            ContextType.IMAGE,
            ContextType.VIDEO,
            ContextType.EMOTION,
            ContextType.GOODS_INQUIRY,
            ContextType.ORDER_INFO,
            ContextType.GOODS_CARD,
            ContextType.GOODS_SPEC,
        }
        return context.type in queue_types

    async def _handle_immediate_message(self, context: Context, shop_id: str, user_id: str):
        """绔嬪嵆澶勭悊娑堟伅"""
        username = context.kwargs.username
        recipient_uid = context.kwargs.from_uid
        try:
            from Channel.pinduoduo.utils.API.send_message import SendMessage

            send_message = SendMessage(shop_id, user_id)
            if context.type == ContextType.AUTH:
                auth_info = context.content
                if isinstance(auth_info, dict):
                    result = auth_info.get("result")
                    if result == "ok":
                        self.logger.info(f"{username}璁よ瘉鎴愬姛")
                    else:
                        self.logger.warning(f"{username}璁よ瘉澶辫触")

            elif context.type == ContextType.WITHDRAW:
                self.logger.info(f"鏀跺埌鎾ゅ洖娑堟伅: {context.content}")
                await asyncio.to_thread(send_message.send_text, recipient_uid, "[玫瑰]")

            elif context.type == ContextType.SYSTEM_STATUS:
                self.logger.debug(f"绯荤粺鐘舵€佹秷鎭? {context.content}")

            elif context.type == ContextType.SYSTEM_HINT:
                self.logger.info(f"绯荤粺鎻愮ず: {context.content}")
                await asyncio.to_thread(send_message.send_text, recipient_uid, "[玫瑰]")

            elif context.type == ContextType.MALL_CS:
                self.logger.debug(f"鏀跺埌瀹㈡湇娑堟伅: {context.content}")

            elif context.type == ContextType.SYSTEM_BIZ:
                self.logger.info(f"绯荤粺涓氬姟娑堟伅: {context.content}")

            elif context.type == ContextType.MALL_SYSTEM_MSG:
                self.logger.info(f"鍟嗗煄绯荤粺娑堟伅: {context.content}")

            elif context.type == ContextType.TRANSFER:
                self.logger.info(f"杞帴娑堟伅: {context.content}")
                await asyncio.to_thread(send_message.send_text, recipient_uid, "[玫瑰]")

        except Exception as e:
            self.logger.error(f"绔嬪嵆澶勭悊娑堟伅澶辫触: {e}")

    def _convert_to_context(self, pdd_message: PDDChatMessage, shop_id: str, user_id: str, username: str) -> Context:
        """灏嗘嫾澶氬娑堟伅杞崲涓篊ontext鏍煎紡"""
        shop_info = db_manager.get_shop(self.channel_name, shop_id)
        shop_name = shop_info.get("shop_name", "")

        content = pdd_message.content
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        else:
            content = str(content)

        context = Context.create_pinduoduo_context(
            content=content,
            msg_id=str(pdd_message.msg_id) if pdd_message.msg_id is not None else "",
            from_user=str(pdd_message.from_user) if pdd_message.from_user is not None else "",
            from_uid=str(pdd_message.from_uid) if pdd_message.from_uid is not None else "",
            to_user=str(pdd_message.to_user) if pdd_message.to_user is not None else "",
            to_uid=str(pdd_message.to_uid) if pdd_message.to_uid is not None else "",
            nickname=str(pdd_message.nickname) if pdd_message.nickname is not None else "",
            timestamp=pdd_message.timestamp,
            user_msg_type=pdd_message.user_msg_type,
            shop_id=str(shop_id),
            user_id=str(user_id),
            username=str(username),
            shop_name=str(shop_name),
            raw_data=pdd_message.raw_data,
            channel_type=ChannelType.PINDUODUO,
        )
        return context


__all__ = ["MessageHandlerMixin"]
