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
from config import get_config
from bridge.context import Context
from bridge.reply import Reply, ReplyType
from Agent.CustomerAgent.custom.session_manager import SessionManager
from Agent.CustomerAgent.custom.tool_decorator import execute_tool, get_tools_for_llm
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
from Agent.CustomerAgent.custom.knowledge_action_router import sanitize_final_reply
from Agent.CustomerAgent.custom.turn_context import TurnContext, parse_turn_context
from utils.night_mode import NIGHT_MODE_TRANSFER_RESULT_PREFIX, is_night_mode

logger = get_logger("CustomerAgent")
SCENE_PROMPT_FILES = {
    "presale": "runtime/scene_prompts_review/m11_售前场景prompt_待审.txt",
    "insale": "runtime/scene_prompts_review/m11_售中场景prompt_待审.txt",
    "aftersale": "runtime/scene_prompts_review/m11_售后场景prompt_待审.txt",
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

            # TurnContext: log-only 模式
            turn_context = dependencies.get("turn_context")
            if turn_context is None and get_config("enable_turn_context", False):
                turn_context = parse_turn_context(str(query or ""))
                dependencies["turn_context"] = turn_context
            if turn_context is not None and get_config("enable_turn_context_log_only", True):
                self._log_turn_context(session_id, turn_context)

            await self._refresh_order_context(dependencies)

            # 加载历史并检查压缩；场景判定需要历史和订单上下文。
            history = self._session_manager.get_history(session_id)
            if self._session_manager.should_compress(session_id):
                logger.info(f"触发上下文压缩: session_id={session_id}")
                await self._compress_with_llm(session_id, history)

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

            # 售后场景收到图片/视频时，直接转人工，避免模型看图/看视频臆断。
            if customer_scene == "aftersale" and self._has_media_input(dependencies):
                self._session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=query,
                )
                final_content = await self._transfer_to_human(dependencies, session_id, reason="aftersale_media")
                self._session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_content,
                )
                return Reply(ReplyType.TEXT, final_content)

            # 售后场景反馈找不到充电口/插电口时，直接转人工，避免模型编造接口位置。
            if customer_scene == "aftersale" and self._is_charge_port_aftersale_issue(query):
                self._session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=query,
                )
                final_content = await self._transfer_to_human(
                    dependencies,
                    session_id,
                    reason="charge_port_issue",
                )
                self._session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_content,
                )
                return Reply(ReplyType.TEXT, final_content)

            # 视频/图片无文字时直接追问，不走 LLM
            if self._is_media_only_query(query, dependencies):
                return Reply(ReplyType.TEXT, "麻烦您说下具体想确认哪里")

            # 构建 messages
            messages = self._message_builder.build_messages(query, history, dependencies)
            self._session_manager.add_message(
                session_id=session_id,
                role="user",
                content=query,
            )
            self._append_scene_prompt(messages, customer_scene)
            self._append_night_mode_constraint(messages)
            self._append_order_hard_constraints(messages, customer_scene, dependencies, session_id, query)
            self._append_image_grounding_constraint(messages, dependencies, session_id)
            self._inject_pre_retrieved_knowledge(messages, query, dependencies, customer_scene)
            self._append_missing_goods_knowledge_constraint(messages, query, dependencies, session_id)

            # 执行 Agent 循环
            final_content = await self._run_agent_loop(messages, dependencies)
            final_content = self._sanitize_daytime_night_mode_reply(final_content, messages)
            final_content = sanitize_final_reply(final_content)

            # 回复去重：与最近一条 assistant 回复比较
            final_content = await self._dedup_reply(
                final_content, messages, history, session_id,
            )
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
            "高级客服下班了",
            "高级客服目前下班",
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
        logger.warning(f"非夜间回复包含夜间话术，已清理: reply={reply[:120]}")
        if any(text.strip() == "会话转接成功" for text in tool_contents):
            return "亲，已经为您转接人工客服，会尽快为您处理，请稍等一下～"

        cleaned = re.sub(
            r"[^。！？!?]*?(夜间时段|夜间不转人工|高级客服已下班|高级客服(?:目前)?下班了|高级客服目前下班|当前高级客服不在线|还没上班|上班时间[是为]?早上8点|高级客服上班时间|建议您晚点(?:再)?联系)[^。！？!?]*[。！？!?]?",
            "",
            reply,
        ).strip(" ，,。.!！~～")
        return cleaned or "亲，客服正在为您处理，请稍等片刻哦～"

    @staticmethod
    def _log_turn_context(session_id: str, tc: TurnContext) -> None:
        """log-only 模式：记录 TurnContext 结构化数据，不参与回复流程。"""
        logger.info(
            "[TurnContext] session={session} customer_text={ct} "
            "has_product_card={hpc} has_order_card={hoc} has_media={hm} "
            "goods_id={gid} order_sn={osn} scene={scene} warnings={warn}".format(
                session=session_id,
                ct=str(tc.customer_text or "")[:80],
                hpc=tc.turn_type.has_product_card,
                hoc=tc.turn_type.has_order_card,
                hm=tc.turn_type.has_media,
                gid=tc.product_card.goods_id or "",
                osn=tc.order_card.order_sn or "",
                scene=tc.raw_scene_hint or "",
                warn=tc.parse_warnings,
            )
        )

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

            # 6. 夜间转人工结果直接返回，不交给 LLM 二次生成
            for result in tool_results:
                if result.content and result.content.startswith(NIGHT_MODE_TRANSFER_RESULT_PREFIX):
                    # 提取前缀后的客户可见回复
                    night_reply = result.content
                    sep_idx = night_reply.find("：")
                    if sep_idx != -1:
                        night_reply = night_reply[sep_idx + 1:]
                    night_reply = night_reply.strip()
                    if night_reply:
                        logger.info(f"[夜间转人工直返] reply={night_reply[:80]}")
                        return night_reply

            # 7. 将结果追加到消息列表
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
        if order_scene == "mixed_orders":
            return "aftersale"

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

    _SIGNED_TRACE_KEYWORDS = (
        "包裹已签收",
        "快件已签收",
        "签收人是",
        "已签收",
        "已收货",
        "已取件",
    )

    @classmethod
    def _is_order_signed(cls, dependencies: Dict[str, Any]) -> bool:
        """判断订单是否已签收。"""
        if dependencies.get("order_shipping_status") == "signed":
            return True
        biz = str(dependencies.get("order_business_status") or "")
        if "已签收" in biz or "已收货" in biz:
            return True
        trace = str(dependencies.get("order_latest_trace") or "")
        if any(kw in trace for kw in cls._SIGNED_TRACE_KEYWORDS):
            return True
        return False

    _ORDER_HARD_CONSTRAINT = (
        '【订单硬约束】\n'
        '系统根据订单状态和最新物流判断：当前客户已收到商品。\n'
        '禁止使用"收到货后、等收到后、到货后、拿到货后、先试用、先试试"等暗示客户尚未收到商品的话术。\n'
        '如果客户反馈商品问题，例如噪音、异响、风小、不转、充不进电、坏了，应按售后问题处理；'
        '无法直接解决时使用转人工工具。\n'
    )

    _USAGE_FEEDBACK_KEYWORDS = (
        "好响", "声音大", "噪音大", "吵", "滋滋", "异响",
        "没反应", "坏了", "不转", "充不进", "充不上",
        "正在用", "收到", "收到货", "已收到",
        "风小", "没风", "风力小",
    )

    def _append_order_hard_constraints(
        self,
        messages: List[Dict[str, Any]],
        customer_scene: str,
        dependencies: Dict[str, Any],
        session_id: str,
        query: str = "",
    ) -> None:
        """已签收或客户有使用反馈时注入订单硬约束（所有场景）。"""
        scene = KnowledgeService.normalize_customer_scene(customer_scene)
        is_signed = self._is_order_signed(dependencies)
        has_usage_feedback = any(kw in str(query or "") for kw in self._USAGE_FEEDBACK_KEYWORDS)

        if not is_signed and not has_usage_feedback:
            return
        # presale 场景且无签收信息时跳过（避免纯售前咨询误注入）
        if scene == "presale" and not is_signed:
            return

        order_id = dependencies.get("order_id") or ""
        shipping = dependencies.get("order_shipping_status") or ""
        logger.info(
            "[订单硬约束] aftersale signed injected: session={} order_id={} shipping_status={}".format(
                session_id, order_id, shipping,
            )
        )

        if messages and messages[0].get("role") == "system":
            existing = str(messages[0].get("content") or "").strip()
            if self._ORDER_HARD_CONSTRAINT not in existing:
                messages[0]["content"] = f"{existing}\n\n{self._ORDER_HARD_CONSTRAINT}"
        else:
            messages.insert(0, {"role": "system", "content": self._ORDER_HARD_CONSTRAINT})

    _MISSING_GOODS_PARAMETER_KEYWORDS = (
        "续航",
        "用多久",
        "能用多久",
        "充一次电",
        "充满",
        "几个小时",
        "电池多大",
        "电池容量",
        "多少毫安",
        "几毫安",
        "毫安",
        "mah",
        "几档",
        "多少档",
        "档位",
        "最高档",
        "最大档",
        "风大",
        "风力",
        "风速",
        "凉快",
        "制冷",
        "半导体",
        "挂绳",
        "挂脖",
        "底座",
        "支架",
        "充电头",
        "充电线",
        "充电器",
        "配件",
        "按键",
        "按钮",
        "开关",
        "图标",
        "标识",
        "中间",
        "摇头",
        "怎么用",
        "颜色",
        "库存",
        "快递",
        "发货地",
        "哪里发货",
    )
    _MISSING_GOODS_KNOWLEDGE_CONSTRAINT = (
        "【商品知识状态】\n"
        "当前会话没有识别到 goods_id，无法加载该商品的专属知识。\n"
        "如果客户询问续航、电池容量、档位、按键/图标/功能、颜色、配件、发货地、快递等依赖具体商品的信息，"
        "不要根据商品名、昵称、图片符号或经验猜测具体参数，不要编造小时数、毫安数、档位数、功能或赠品承诺。\n"
        "应回复：亲，不同款式/规格参数会不一样，麻烦您发一下具体商品链接或点一下商品卡片，我按对应款式帮您确认哦。\n"
        "不要向客户提到“知识库、goods_id、系统无法加载、商品知识状态”等内部信息。\n"
    )

    @classmethod
    def _is_missing_goods_id(cls, dependencies: Dict[str, Any]) -> bool:
        goods_id = dependencies.get("goods_id")
        if goods_id is None:
            return True
        text = str(goods_id).strip().lower()
        return text in {"", "none", "null", "0"}

    @classmethod
    def _is_product_parameter_query(cls, query: str) -> bool:
        text = str(query or "").lower()
        return any(keyword in text for keyword in cls._MISSING_GOODS_PARAMETER_KEYWORDS)

    def _append_missing_goods_knowledge_constraint(
        self,
        messages: List[Dict[str, Any]],
        query: str,
        dependencies: Dict[str, Any],
        session_id: str,
    ) -> None:
        """无 goods_id 的商品参数问题注入约束，避免模型编造具体商品参数。"""
        if not self._is_missing_goods_id(dependencies):
            return
        if not self._is_product_parameter_query(query):
            return

        logger.info(
            "[无商品知识约束] injected: session={} shop_id={} query={}".format(
                session_id,
                dependencies.get("shop_id"),
                str(query or "")[:80],
            )
        )
        messages.append({"role": "system", "content": self._MISSING_GOODS_KNOWLEDGE_CONSTRAINT})

    _NIGHT_MODE_FAULT_CONSTRAINT = (
        '【夜间模式约束】\n'
        '当前为夜间值守时段（23:00-08:00），无法转接人工客服。\n'
        '禁止回复"已为您转接""稍后会有专员""已转人工"等虚假转接话术。\n'
        '如果客户反馈商品故障（坏了、没反应、不转、充不进电、噪音大），'
        '应回复：已记录您的问题，夜间无法转接人工，建议您早上8点后联系，会有专人为您处理。\n'
        '如果客户反复反馈同一问题，不要重复相同话术，简短确认已记录即可。\n'
    )

    _IMAGE_GROUNDING_CONSTRAINT = (
        "【图片理解硬约束】\n"
        "图片只能作为客户现场/截图的辅助信息，不是商品功能依据。\n"
        "禁止根据图片里的 X、+、-、雪花、风扇、灯、圆点、金属片、按键形状等视觉符号，"
        "推断摇头、制冷、档位、灯光、充电等商品功能。\n"
        "客户问图中按钮、图标、标识、部件用途时，必须以预检索知识或 search_knowledge 的明确答案为准；"
        "没有明确答案时回复：亲，仅凭图片看不出这个位置的具体功能，麻烦您说下是哪个按键/位置，或点一下商品卡片，我按对应款式帮您确认哦。\n"
        "不要说“X 是摇头功能”等没有知识依据的结论。\n"
        "不要向客户提到图片理解硬约束、预检索、search_knowledge、知识库等内部信息。\n"
    )

    @staticmethod
    def _has_media_input(dependencies: Dict[str, Any]) -> bool:
        """判断本轮是否包含图片或视频。"""
        context_type = str(dependencies.get("context_type") or "")
        media_type = str(dependencies.get("media_type") or "")
        media_url = str(dependencies.get("media_url") or "").lower()
        if context_type in {"image", "video"} or media_type in {"image", "video"}:
            return True
        return bool(media_url and ("chat-img" in media_url or "video" in media_url))

    @staticmethod
    def _has_image_input(dependencies: Dict[str, Any]) -> bool:
        context_type = str(dependencies.get("context_type") or "")
        media_type = str(dependencies.get("media_type") or "")
        media_url = str(dependencies.get("media_url") or "")
        return context_type == "image" or media_type == "image" or bool(media_url and "chat-img" in media_url)

    _CHARGE_PORT_AFTERSALE_ISSUE_KEYWORDS = (
        "没有充电口",
        "没充电口",
        "怎么没有充电口",
        "找不到充电口",
        "充电口在哪",
        "充电口在哪里",
        "充电口在那",
        "充电口在哪儿",
        "没有插电口",
        "没插电口",
        "找不到插电口",
        "插电口在哪",
        "插电口在哪里",
        "插电口在那",
        "插电口在哪儿",
        "在哪里插电",
        "在哪插电",
        "哪里插电",
    )

    @classmethod
    def _is_charge_port_aftersale_issue(cls, query: str) -> bool:
        text = str(query or "").replace(" ", "")
        return any(keyword in text for keyword in cls._CHARGE_PORT_AFTERSALE_ISSUE_KEYWORDS)

    @staticmethod
    def _is_media_only_query(query: str, dependencies: Dict[str, Any]) -> bool:
        """判断是否只有图片/视频、没有文字问题。"""
        context_type = str(dependencies.get("context_type") or "")
        media_type = str(dependencies.get("media_type") or "")
        is_media = context_type in {"image", "video"} or media_type in {"image", "video"}
        if not is_media:
            return False
        # 检查 query 是否包含客户实际文字（排除 URL、[视频消息]、[图片消息] 等）
        text = str(query or "").strip()
        # 去掉 URL
        text_no_url = re.sub(r"https?://[^\s]+", "", text).strip()
        # 去掉标记
        text_no_url = re.sub(r"\[(视频|图片|video|image)消息\]", "", text_no_url, flags=re.IGNORECASE).strip()
        text_no_url = re.sub(r"客户消息[：:]\s*", "", text_no_url).strip()
        text_no_url = re.sub(r"客户发送了(图片|视频)[：:]*\s*", "", text_no_url).strip()
        return len(text_no_url) < 2

    def _append_image_grounding_constraint(
        self,
        messages: List[Dict[str, Any]],
        dependencies: Dict[str, Any],
        session_id: str,
    ) -> None:
        """有图片输入时约束模型不要把可见符号臆断为商品功能。"""
        if not self._has_image_input(dependencies):
            return

        logger.info(
            "[图片硬约束] injected: session={} shop_id={} goods_id={} media_type={}".format(
                session_id,
                dependencies.get("shop_id"),
                dependencies.get("goods_id"),
                dependencies.get("media_type"),
            )
        )
        if messages and messages[0].get("role") == "system":
            existing = str(messages[0].get("content") or "").strip()
            if self._IMAGE_GROUNDING_CONSTRAINT not in existing:
                messages[0]["content"] = (
                    f"{existing}\n\n{self._IMAGE_GROUNDING_CONSTRAINT}"
                    if existing
                    else self._IMAGE_GROUNDING_CONSTRAINT
                )
        else:
            messages.insert(0, {"role": "system", "content": self._IMAGE_GROUNDING_CONSTRAINT})

    def _append_night_mode_constraint(self, messages: List[Dict[str, Any]]) -> None:
        """夜间模式时注入约束，防止 LLM 生成虚假转接话术。"""
        if not is_night_mode():
            return
        if messages and messages[0].get("role") == "system":
            existing = str(messages[0].get("content") or "").strip()
            if self._NIGHT_MODE_FAULT_CONSTRAINT not in existing:
                messages[0]["content"] = f"{existing}\n\n{self._NIGHT_MODE_FAULT_CONSTRAINT}"
        else:
            messages.insert(0, {"role": "system", "content": self._NIGHT_MODE_FAULT_CONSTRAINT})

    async def _transfer_to_human(
        self,
        dependencies: Dict[str, Any],
        session_id: str,
        reason: str,
    ) -> str:
        """高风险问题直转人工，不经过 LLM。"""
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            execute_tool,
            "transfer_conversation",
            "{}",
            dependencies,
        )
        content = str(result or "").strip()
        logger.info(f"[售后直转人工] session={session_id} reason={reason} result={content[:120]}")

        if content.startswith(NIGHT_MODE_TRANSFER_RESULT_PREFIX):
            sep_idx = content.find("：")
            if sep_idx != -1:
                content = content[sep_idx + 1:]
            content = content.strip()
            if content:
                return sanitize_final_reply(content)

        if "会话转接成功" in content:
            return "亲，已转人工为您处理，请稍等。"

        if "当前无可用" in content or "不可转接" in content:
            return "亲，当前人工客服暂时不可转接，您先把问题发我，我这边继续帮您记录。"

        if "转接失败" in content or "缺少必要的会话信息" in content or "工具执行错误" in content:
            return "亲，转人工暂时没成功，您先把问题发我，我这边继续帮您看。"

        return "亲，已为您转接人工处理，请稍等。"

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
        stop_markers = (
            "\n商品卡片：",
            "\n商品：",
            "\n订单信息：",
            "\n物流信息：",
            "\n客户发送了图片",
            "\n客户发送了视频",
        )
        for stop in stop_markers:
            if stop in customer_part:
                customer_part = customer_part.split(stop, 1)[0]
        return customer_part.strip() or text

    async def _dedup_reply(
        self,
        content: str,
        messages: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        session_id: str,
    ) -> str:
        """如果最终回复与上一条 assistant 回复完全相同，重新生成。"""
        if not content or len(content) < 4:
            return content

        # 找最近一条 assistant 回复
        last_assistant = ""
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = str(msg["content"]).strip()
                break

        if not last_assistant:
            return content

        # 规范化比较：去除空白、标点
        def _norm(text: str) -> str:
            return re.sub(r"[\s。！？!?，,~～.]+", "", text)

        if _norm(content) != _norm(last_assistant):
            return content

        # 相同：注入去重提示，重新生成
        logger.warning(f"[回复去重] 检测到重复回复，重新生成: session={session_id}, reply={content[:60]}")
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                "[系统提示] 你的上一条回复与之前重复了。"
                "请根据客户最新的消息内容，给出有针对性的递进回复，不要重复之前的表述。"
            ),
        })
        try:
            response = await self._llm_client.chat(messages, tool_choice="none")
            new_content = response.content or ""
            new_content = self._sanitize_daytime_night_mode_reply(new_content, messages)
            new_content = sanitize_final_reply(new_content)
            if new_content and _norm(new_content) != _norm(content):
                return new_content
        except Exception as e:
            logger.warning(f"[回复去重] 重新生成失败: {e}")
        return content

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
