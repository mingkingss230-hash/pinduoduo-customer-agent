"""
自定义 CustomerAgent 实现

完全自主实现，不依赖 Agno 框架。

本模块已重构，职责分离为：
- agent_config.py: 配置管理
- llm_client.py: LLM 客户端封装
- message_builder.py: 消息和 Prompt 构建
- tool_executor.py: 工具执行器
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

from Agent.bot import Bot

# 导入工具模块，触发 @agent_tool 装饰器注册
from Agent.CustomerAgent.tools import (
    move_conversation,                 # noqa: F401  — 注册 transfer_conversation 工具
    search_knowledge,                 # noqa: F401  — 注册 search_knowledge 工具
    send_product_card,                # noqa: F401  — 注册 send_product_card 工具
)
from bridge.context import Context
from bridge.reply import Reply, ReplyType
from Agent.CustomerAgent.custom.session_manager import SessionManager
from Agent.CustomerAgent.custom.tool_decorator import get_tools_for_llm
from database.knowledge_service import KnowledgeService
from utils.logger_loguru import get_logger
from utils.runtime_path import get_resource_path
from Channel.pinduoduo.utils.API.order_manager import OrderManager, build_order_context_text

# 导入重构后的模块
from Agent.CustomerAgent.custom.agent_config import (
    AgentConfig,
    DEFAULT_DB_PATH,
    DEFAULT_TOKEN_WINDOW,
    DEFAULT_COMPRESS_RATIO,
    DEFAULT_RETAIN_COUNT,
    DEFAULT_MAX_LOOPS,
    DEFAULT_TEMPERATURE,
)
from Agent.CustomerAgent.custom.llm_client import LLMClient, LLMResponse
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from Agent.CustomerAgent.custom.tool_executor import ToolExecutor, ToolResult
from utils.night_mode import NIGHT_MODE_TRANSFER_RESULT_PREFIX, is_night_mode

logger = get_logger("CustomerAgent")
SCENE_PROMPT_FILES = {
    "presale": "runtime/scene_prompts_review/presale_prompt.txt",
    "insale": "runtime/scene_prompts_review/insale_prompt.txt",
    "aftersale": "runtime/scene_prompts_review/aftersale_prompt.txt",
}


class CustomerAgent(Bot):
    """
    自定义客服 Agent

    核心循环：
    1. 加载历史消息
    2. 检查上下文压缩
    3. 构建 messages 列表
    4. 调用 LLM → 解析 tool_calls
    5. 并行执行工具 → 回传结果
    6. 循环直到无工具调用
    7. 返回最终回复

    职责已分离到子模块：
    - AgentConfig: 配置管理
    - LLMClient: LLM API 调用
    - MessageBuilder: 消息和 Prompt 构建
    - ToolExecutor: 工具执行
    - SessionManager: 会话管理（已有独立模块）
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        token_window: int = DEFAULT_TOKEN_WINDOW,
        compress_ratio: float = DEFAULT_COMPRESS_RATIO,
        retain_count: int = DEFAULT_RETAIN_COUNT,
        max_loops: int = DEFAULT_MAX_LOOPS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__()
        self._is_initialized = False

        # 配置参数
        self._config = AgentConfig(
            db_path=db_path or DEFAULT_DB_PATH,
            token_window=token_window,
            compress_ratio=compress_ratio,
            retain_count=retain_count,
            max_loops=max_loops,
            temperature=temperature,
        )

        # 子组件（延迟初始化）
        self._llm_client: Optional[LLMClient] = None
        self._message_builder: Optional[MessageBuilder] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._session_manager: Optional[SessionManager] = None
        self._tools: List[Dict[str, Any]] = []
        self._scene_prompt_cache: Dict[str, str] = {}
        self._session_goods_id_cache: Dict[str, int] = {}  # session_id -> goods_id

        logger.info("CustomerAgent 实例创建成功")

    async def initialize_async(self) -> bool:
        """异步初始化 Agent"""
        if self._is_initialized:
            return True

        try:
            # 1. 从配置文件加载配置
            self._config = AgentConfig.load_from_config()

            # 2. 验证配置
            if not self._config.validate():
                return False

            # 3. 初始化 LLM 客户端
            self._llm_client = LLMClient(
                api_key=self._config.api_key,
                api_base=self._config.api_base,
                model_name=self._config.model_name,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                tool_call_max_tokens=self._config.tool_call_max_tokens,
                request_timeout_seconds=self._config.request_timeout_seconds,
                max_concurrent_requests=self._config.max_concurrent_requests,
                fallback_api_key=self._config.fallback_api_key,
                fallback_api_base=self._config.fallback_api_base,
                fallback_model_name=self._config.fallback_model_name,
                fallback_timeout_seconds=self._config.fallback_timeout_seconds,
                fallback_enabled=self._config.fallback_enabled,
            )
            await self._llm_client.initialize()

            # 4. 初始化会话管理器
            self._session_manager = SessionManager(
                db_path=self._config.db_path,
                token_window=self._config.token_window,
                compress_ratio=self._config.compress_ratio,
                retain_count=self._config.retain_count,
                model_name=self._config.model_name,
            )

            # 5. 初始化消息构建器
            self._message_builder = MessageBuilder()

            # 6. 初始化工具执行器
            self._tool_executor = ToolExecutor()

            # 7. 加载工具列表
            self._tools = get_tools_for_llm()
            self._llm_client.tools = self._tools
            tool_names = [t.get("function", {}).get("name", "unknown") for t in self._tools]
            logger.info(f"已加载 {len(self._tools)} 个工具: {tool_names}")

            self._is_initialized = True
            logger.info(f"CustomerAgent 初始化成功: model={self._config.model_name}")
            return True

        except Exception as e:
            logger.error(f"CustomerAgent 初始化失败: {e}")
            return False

    async def async_reply(self, query: str, context: Context = None) -> Reply:
        """异步回复接口"""
        # 延迟初始化
        if not self._is_initialized:
            if not await self.initialize_async():
                return Reply(ReplyType.TEXT, "AI客服初始化失败，请检查配置。")

        try:
            # 构建 session_id 和 dependencies
            if context and context.channel_type and hasattr(context.kwargs, "user_id"):
                dependencies = self._message_builder.build_dependencies(context)
                session_id = self._build_session_id(context, dependencies)
            else:
                # 降级：使用 query hash 作为 session_id
                session_id = f"fallback_{abs(hash(query)) % 100000}"
                dependencies = {}

            # goods_id 会话级缓存：首次获取后复用，避免后续纯文本消息丢失商品上下文
            current_gid = dependencies.get("goods_id")
            if current_gid:
                self._session_goods_id_cache[session_id] = current_gid
                # 防止缓存无限增长
                if len(self._session_goods_id_cache) > 5000:
                    oldest_keys = list(self._session_goods_id_cache.keys())[:2500]
                    for k in oldest_keys:
                        del self._session_goods_id_cache[k]
            elif session_id in self._session_goods_id_cache:
                dependencies["goods_id"] = self._session_goods_id_cache[session_id]

            await self._refresh_order_context(dependencies)

            # 加载历史并检查压缩
            history = self._session_manager.get_history(session_id)
            if self._session_manager.should_compress(session_id):
                logger.info(f"触发上下文压缩: session_id={session_id}")
                await self._compress_with_llm(session_id, history)

            # 构建 messages
            messages = self._message_builder.build_messages(query, history, dependencies)
            self._session_manager.add_message(
                session_id=session_id,
                role="user",
                content=query,
            )
            customer_scene = self._resolve_customer_scene(query, history, dependencies)
            dependencies["_customer_scene"] = customer_scene
            logger.info(
                "[场景判定] session={} scene={} shop_id={} goods_id={} context_type={} query={}".format(
                    session_id,
                    customer_scene,
                    dependencies.get("shop_id"),
                    dependencies.get("goods_id"),
                    dependencies.get("context_type"),
                    str(query or "")[:80],
                )
            )
            self._append_scene_prompt(messages, customer_scene)
            self._inject_pre_retrieved_knowledge(messages, query, dependencies, customer_scene)

            # 执行 Agent 循环
            final_content = await self._run_agent_loop(messages, dependencies)
            final_content = self._sanitize_daytime_night_mode_reply(final_content, messages)
            logger.info(
                "[最终回复] session={} scene={} reply_len={}".format(
                    session_id,
                    customer_scene,
                    len(final_content or ""),
                )
            )

            # 保存最终回复到历史
            self._session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=final_content,
            )

            return Reply(ReplyType.TEXT, final_content or "亲，客服正在为您处理，请稍等片刻哦～")

        except Exception as e:
            logger.error(f"CustomerAgent 回复失败: {e}")
            return Reply(ReplyType.TEXT, "亲，客服正在为您处理，请稍等片刻哦～")

    def _build_session_id(self, context: Context, dependencies: Dict[str, Any]) -> str:
        """按店铺账号和买家隔离会话历史，避免不同客户互相污染。"""
        channel = str(context.channel_type.value if context.channel_type else "unknown")
        shop_id = str(dependencies.get("shop_id") or getattr(context.kwargs, "shop_id", "") or "")
        user_id = str(dependencies.get("user_id") or getattr(context.kwargs, "user_id", "") or "")
        recipient_uid = str(
            dependencies.get("recipient_uid")
            or dependencies.get("from_uid")
            or getattr(context.kwargs, "from_uid", "")
            or ""
        )

        if recipient_uid:
            return f"{channel}:{shop_id}:{user_id}:{recipient_uid}"
        if shop_id or user_id:
            return f"{channel}:{shop_id}:{user_id}:unknown"
        return f"{channel}:fallback_{abs(hash(str(getattr(context, 'content', '')))) % 100000}"

    def _sanitize_daytime_night_mode_reply(
        self,
        content: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        """非夜间时清理被历史污染带出的夜间转人工话术。"""
        reply = str(content or "")
        if not reply or is_night_mode():
            return reply

        leak_markers = (
            "夜间时段",
            "夜间不转人工",
            "高级客服已下班",
            "当前高级客服不在线",
            "还没上班",
            "上班时间是早上8点",
            "上班时间为早上8点",
            "高级客服上班时间",
            "建议您晚点联系",
            "建议您晚点再联系",
        )
        if not any(marker in reply for marker in leak_markers):
            return reply

        tool_contents = [
            str(msg.get("content") or "")
            for msg in messages
            if isinstance(msg, dict) and msg.get("role") == "tool"
        ]
        if any(text.startswith(NIGHT_MODE_TRANSFER_RESULT_PREFIX) for text in tool_contents):
            return reply

        logger.warning(f"非夜间回复包含夜间话术，已清理: reply={reply[:120]}")
        if any(text.strip() == "会话转接成功" for text in tool_contents):
            return "亲，已经为您转接人工客服，会尽快为您处理，请稍等一下～"

        cleaned = re.sub(
            r"[^。！？!?]*?(夜间时段|夜间不转人工|高级客服已下班|当前高级客服不在线|还没上班|上班时间[是为]?早上8点|高级客服上班时间|建议您晚点(?:再)?联系)[^。！？!?]*[。！？!?]?",
            "",
            reply,
        ).strip(" ，,。.!！~～")
        return cleaned or "亲，客服正在为您处理，请稍等片刻哦～"

    async def _run_agent_loop(
        self,
        messages: List[Dict[str, Any]],
        dependencies: Dict[str, Any],
    ) -> str:
        """
        Agent 循环核心

        调用 LLM → 检查 tool_calls → 并行执行工具 → 回传结果 → 循环
        """
        loop_count = 0

        while loop_count < self._config.max_loops:
            # 1. 调用 LLM
            try:
                response = await self._llm_client.chat(messages, tool_choice="auto")
            except Exception as e:
                error_detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                logger.exception(f"LLM 调用失败: {error_detail}")
                if loop_count == 0:
                    return "亲，客服正在为您处理，请稍等片刻哦～"
                # 已有中间结果，返回已生成的内容
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        return msg["content"]
                return "亲，客服正在为您处理，请稍等片刻哦～"

            # 2. 解析响应
            if not response.has_tool_calls:
                # 无工具调用，返回内容
                content = response.content or ""
                messages.append({"role": "assistant", "content": content})
                return content

            # 3. 保存 assistant 消息（包含 tool_calls）
            assistant_msg = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            # 4. 检查循环上限
            if loop_count >= self._config.max_loops - 1:
                logger.warning(f"工具调用达到上限 {self._config.max_loops}，强制结束循环")
                messages.append({
                    "role": "user",
                    "content": "[已达到最大工具调用次数，请基于已有信息给出最终回复。]",
                })
                try:
                    final_response = await self._llm_client.chat(messages)
                    return final_response.content or assistant_msg["content"]
                except Exception:
                    return assistant_msg["content"]

            # 5. 并行执行所有工具调用
            tool_names = [tc.function.name for tc in response.tool_calls]
            logger.info(
                "[工具调用] tools={} scene={} shop_id={} goods_id={}".format(
                    tool_names,
                    dependencies.get("_customer_scene"),
                    dependencies.get("shop_id"),
                    dependencies.get("goods_id"),
                )
            )
            tool_results = await self._tool_executor.execute_parallel(
                response.tool_calls, dependencies
            )

            # 6. 将结果追加到消息列表
            for result in tool_results:
                messages.append(result.to_dict())

            loop_count += 1

        # 兜底
        return messages[-1].get("content", "")

    def _resolve_customer_scene(
        self,
        query: str,
        history: List[Dict[str, Any]],
        dependencies: Dict[str, Any],
    ) -> str:
        """基于问题和上下文判断当前会话场景。"""
        cached_scene = KnowledgeService.normalize_customer_scene(dependencies.get("_customer_scene")) or ""
        if cached_scene in {"presale", "insale", "aftersale"}:
            return cached_scene

        order_scene = str(dependencies.get("order_scene_hint") or "")
        if order_scene in {"aftersale", "insale"}:
            return order_scene

        combined = str(query or "")
        recent_user_text = " ".join(
            str(msg.get("content") or "")
            for msg in history[-6:]
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content")
        )
        if recent_user_text:
            combined = f"{combined} {recent_user_text}"

        scene = KnowledgeService.detect_customer_scene(combined, default="presale")
        return scene if scene in {"presale", "insale", "aftersale"} else "presale"

    async def _refresh_order_context(self, dependencies: Dict[str, Any]) -> None:
        """Fetch latest read-only PDD order context for scene detection."""
        shop_id = dependencies.get("shop_id")
        account_user_id = dependencies.get("user_id")
        customer_uid = dependencies.get("recipient_uid") or dependencies.get("from_uid")
        if not shop_id or not account_user_id or not customer_uid:
            return

        def load_context() -> Dict[str, Any]:
            manager = OrderManager(shop_id=str(shop_id), user_id=str(account_user_id))
            orders = manager.get_user_orders(str(customer_uid), page_size=5)
            return build_order_context_text(orders)

        try:
            context = await asyncio.to_thread(load_context)
        except Exception as exc:
            logger.warning(
                "[订单上下文] 获取失败 shop_id={} account_user_id={} customer_uid={} error={}".format(
                    shop_id,
                    account_user_id,
                    customer_uid,
                    exc,
                )
            )
            return

        dependencies["order_context_text"] = context.get("text") or ""
        dependencies["order_scene_hint"] = context.get("scene_hint") or ""
        dependencies["order_business_status"] = context.get("business_status") or ""
        dependencies["order_shipping_status"] = context.get("shipping_status") or ""
        dependencies["order_latest_trace"] = context.get("latest_trace") or ""
        dependencies["order_id"] = context.get("order_id") or ""
        logger.info(
            "[订单上下文] scene_hint={} business_status={} shipping_status={} order_id={} latest_trace={}".format(
                dependencies.get("order_scene_hint"),
                dependencies.get("order_business_status"),
                dependencies.get("order_shipping_status"),
                dependencies.get("order_id"),
                str(dependencies.get("order_latest_trace") or "")[:100],
            )
        )

    def _load_scene_prompt(self, customer_scene: str) -> str:
        scene_key = KnowledgeService.normalize_customer_scene(customer_scene) or "presale"
        cached = self._scene_prompt_cache.get(scene_key)
        if cached:
            return cached
        relative_path = SCENE_PROMPT_FILES.get(scene_key)
        if not relative_path:
            return ""
        try:
            prompt = get_resource_path(relative_path).read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning(f"场景 Prompt 读取失败: scene={scene_key}, error={exc}")
            return ""
        self._scene_prompt_cache[scene_key] = prompt
        return prompt

    def _append_scene_prompt(self, messages: List[Dict[str, Any]], customer_scene: str) -> None:
        prompt = self._load_scene_prompt(customer_scene)
        if not prompt:
            return

        scene_label = KnowledgeService.customer_scene_label(customer_scene)
        scene_note = (
            "【当前会话场景规则】\n"
            f"当前场景规则按“{scene_label}”加载；场景名仅供内部判断，不要对客户输出。\n\n"
            f"{prompt}"
        )

        if messages and messages[0].get("role") == "system":
            existing = str(messages[0].get("content") or "").strip()
            if scene_note not in existing:
                messages[0]["content"] = f"{existing}\n\n{scene_note}" if existing else scene_note
        else:
            messages.insert(0, {"role": "system", "content": scene_note})

    def _inject_pre_retrieved_knowledge(
        self,
        messages: List[Dict[str, Any]],
        query: str,
        dependencies: Dict[str, Any],
        customer_scene: str,
    ) -> None:
        """预检索知识并注入 system prompt，提高第一轮回复稳定性。"""
        try:
            # 从 dependencies 获取 shop_id 和 goods_id
            shop_id = dependencies.get("shop_id")
            goods_id = dependencies.get("goods_id")

            # 参数校验
            if not shop_id or not goods_id or not customer_scene:
                return
            if customer_scene not in {"presale", "insale", "aftersale"}:
                return

            retrieval_query = self._knowledge_retrieval_query(query)

            # 调用检索
            ks = KnowledgeService()
            results = ks.search_scene_knowledge(
                scene=customer_scene,
                shop_id=shop_id,
                goods_id=goods_id,
                query=retrieval_query,
                limit=3,
            )

            if not results:
                logger.info(
                    "[预检索] scene={} shop_id={} goods_id={} hit=0 query={}".format(
                        customer_scene,
                        shop_id,
                        goods_id,
                        str(retrieval_query or "")[:80],
                    )
                )
                return

            top1 = results[0] if results else {}
            logger.info(
                "[预检索] scene={} shop_id={} goods_id={} hit={} top1_section={} top1_intent={} match={} score={} query={}".format(
                    customer_scene,
                    shop_id,
                    goods_id,
                    len(results),
                    top1.get("section_title") or "",
                    top1.get("sub_intent") or "",
                    top1.get("match_type") or "",
                    top1.get("score") or "",
                    str(retrieval_query or "")[:80],
                )
            )

            # 构建注入内容，控制长度
            knowledge_lines = []
            total_length = 0
            max_single = 300
            max_total = 1200

            for i, r in enumerate(results, 1):
                answer = (r.get("answer") or "")[:max_single]
                section = r.get("section_title") or ""
                sub_intent = r.get("sub_intent") or ""

                entry = f"{i}. 分类：{section}\n   意图：{sub_intent}\n   答案：{answer}"
                if total_length + len(entry) > max_total:
                    break
                knowledge_lines.append(entry)
                total_length += len(entry)

            if not knowledge_lines:
                return

            # 组装注入文本
            inject_text = (
                "【本轮预检索知识】\n"
                "以下知识由系统根据当前店铺、商品、场景和客户问题自动检索，仅供本轮回复使用。\n"
                "如果知识能回答客户问题，优先按知识直接回复。\n"
                "不要告诉客户“知识库、RAG、预检索、系统检索”等内部信息。\n\n"
                + "\n\n".join(knowledge_lines)
            )

            # 注入到 messages[0]（system prompt）
            if messages and messages[0].get("role") == "system":
                existing = str(messages[0].get("content") or "").strip()
                messages[0]["content"] = f"{existing}\n\n{inject_text}" if existing else inject_text
            else:
                messages.insert(0, {"role": "system", "content": inject_text})

            logger.debug(f"预检索知识注入完成: scene={customer_scene}, 条数={len(knowledge_lines)}")

        except Exception as e:
            logger.warning(f"预检索知识注入失败（不影响正常流程）: {e}")

    @staticmethod
    def _knowledge_retrieval_query(query: str) -> str:
        """预检索只使用客户真实文本，避免商品卡片价格/标题污染匹配。"""
        text = str(query or "").strip()
        marker = "客户消息："
        if marker not in text:
            return text

        customer_part = text.split(marker, 1)[1]
        stop_markers = ("\n商品卡片：", "\n商品：", "\n订单信息：", "\n物流信息：")
        for stop in stop_markers:
            if stop in customer_part:
                customer_part = customer_part.split(stop, 1)[0]
        return customer_part.strip() or text

    async def _compress_with_llm(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
    ) -> None:
        """使用 LLM 生成摘要并压缩历史"""

        def summary_llm(messages: List[Dict[str, Any]]) -> str:
            """同步调用 LLM 生成摘要"""
            summary_prompt = (
                "请简洁地总结以下对话的要点，保留关键信息和用户意图。\n\n"
                f"对话内容（共 {len(messages)} 条消息）：\n"
                + "\n".join(
                    f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')[:200]}"
                    for msg in messages
                    if msg.get("content")
                )
            )

            # 在线程中使用 asyncio.run 创建新的事件循环执行协程
            try:
                response = asyncio.run(
                    self._llm_client.chat(
                        messages=[
                            {"role": "system", "content": "你是一个对话摘要助手。请简洁地总结对话要点。"},
                            {"role": "user", "content": summary_prompt},
                        ],
                        tool_choice="none",
                    )
                )
                return response.content or "[摘要生成失败]"
            except RuntimeError:
                # 如果当前已有事件循环（某些环境中），降级为同步返回
                return "[对话历史摘要]"
            except Exception:
                return "[摘要生成失败]"

        self._session_manager.compress_history(session_id, summary_llm)
