"""
LLM 客户端模块

封装与 LLM API 的交互，提供类型安全的请求和响应处理。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("openai package is required: pip install openai>=1.109.1")

from utils.logger_loguru import get_logger
from utils.volcengine_models import ChatCompletionsRequest

logger = get_logger("LLMClient")

_LLM_SEMAPHORE_LOCK = asyncio.Lock()
_LLM_SEMAPHORE: Optional[asyncio.Semaphore] = None
_LLM_SEMAPHORE_LIMIT = 0
_LLM_SEMAPHORE_LOOP: Optional[asyncio.AbstractEventLoop] = None


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    content: Optional[str]
    tool_calls: Optional[List[Any]]
    raw_response: Any
    reasoning_content: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        """是否有工具调用"""
        return self.tool_calls is not None and len(self.tool_calls) > 0


class LLMClient:
    """LLM 客户端封装"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        tool_call_max_tokens: int = 256,
        request_timeout_seconds: float = 20.0,
        max_concurrent_requests: int = 2,
        fallback_api_key: str = "",
        fallback_api_base: str = "",
        fallback_model_name: str = "",
        fallback_timeout_seconds: float = 20.0,
        fallback_enabled: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥
            api_base: API 基础地址
            model_name: 模型名称
            temperature: 温度参数
            tools: 可用工具列表
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max(1, int(max_tokens))
        self.tool_call_max_tokens = max(self.max_tokens, int(tool_call_max_tokens))
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds or 20.0))
        self.max_concurrent_requests = max(1, int(max_concurrent_requests or 2))
        self.fallback_api_key = fallback_api_key
        self.fallback_api_base = fallback_api_base
        self.fallback_model_name = fallback_model_name
        self.fallback_timeout_seconds = max(1.0, float(fallback_timeout_seconds or self.request_timeout_seconds))
        self.fallback_enabled = bool(fallback_enabled and fallback_api_key and fallback_model_name)
        self.tools = tools or []

        self._client: Optional[AsyncOpenAI] = None
        self._fallback_client: Optional[AsyncOpenAI] = None

    @staticmethod
    async def _get_global_semaphore(limit: int) -> asyncio.Semaphore:
        global _LLM_SEMAPHORE, _LLM_SEMAPHORE_LIMIT, _LLM_SEMAPHORE_LOOP
        normalized_limit = max(1, int(limit or 1))
        current_loop = asyncio.get_running_loop()
        async with _LLM_SEMAPHORE_LOCK:
            # 检查信号量是否绑定到不同的事件循环
            if _LLM_SEMAPHORE is not None and _LLM_SEMAPHORE_LOOP is not current_loop:
                logger.warning("检测到事件循环变化，重新创建信号量")
                _LLM_SEMAPHORE = None
            if _LLM_SEMAPHORE is None or _LLM_SEMAPHORE_LIMIT != normalized_limit:
                _LLM_SEMAPHORE = asyncio.Semaphore(normalized_limit)
                _LLM_SEMAPHORE_LIMIT = normalized_limit
                _LLM_SEMAPHORE_LOOP = current_loop
                logger.info(f"LLM API 并发限制已设置为 {normalized_limit}")
            return _LLM_SEMAPHORE

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """确保 system 消息只出现在最前面。"""
        if not messages:
            return []

        system_contents: List[str] = []
        other_messages: List[Dict[str, Any]] = []

        for message in messages:
            if message.get("role") == "system":
                content = str(message.get("content") or "").strip()
                if content:
                    system_contents.append(content)
            else:
                other_messages.append(message)

        normalized_messages: List[Dict[str, Any]] = []
        if system_contents:
            normalized_messages.append({
                "role": "system",
                "content": "\n\n".join(system_contents),
            })
        normalized_messages.extend(other_messages)
        return normalized_messages

    @staticmethod
    def _normalize_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """确保每个 tool 都显式带上 type=function。"""
        normalized_tools: List[Dict[str, Any]] = []

        for tool in tools or []:
            function = dict(tool.get("function", {}))
            parameters = dict(function.get("parameters", {}))
            function["parameters"] = parameters

            normalized_tools.append({
                "type": "function",
                "function": function,
            })

        return normalized_tools

    @staticmethod
    def _should_disable_chat_template_thinking_for(api_base: str, model_name: str) -> bool:
        """MiMo/local chat templates need thinking disabled, otherwise content can be empty."""
        api_base = str(api_base or "").lower()
        model_name = str(model_name or "").lower()
        return (
            "127.0.0.1" in api_base
            or "localhost" in api_base
            or "xiaomimimo.com" in api_base
            or model_name.startswith("mimo-")
            or model_name.startswith("glm-")
        )

    def _should_disable_chat_template_thinking(self) -> bool:
        return self._should_disable_chat_template_thinking_for(self.api_base, self.model_name)

    async def initialize(self) -> None:
        """初始化 OpenAI 客户端"""
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base or None,
            timeout=self.request_timeout_seconds,
        )
        if self.fallback_enabled:
            self._fallback_client = AsyncOpenAI(
                api_key=self.fallback_api_key,
                base_url=self.fallback_api_base or None,
                timeout=self.fallback_timeout_seconds,
            )
            logger.info(
                f"LLM兜底模型已启用: primary={self.model_name}, fallback={self.fallback_model_name}, "
                f"timeout={self.request_timeout_seconds}s/{self.fallback_timeout_seconds}s"
            )
        logger.debug(f"LLM 客户端初始化成功: model={self.model_name}")

    async def _create_completion(
        self,
        client: AsyncOpenAI,
        payload: Dict[str, Any],
        model_name: str,
        timeout_seconds: float,
    ) -> Any:
        request_payload = dict(payload)
        request_payload["model"] = model_name
        return await asyncio.wait_for(
            client.chat.completions.create(**request_payload),
            timeout=timeout_seconds,
        )

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tool_choice: str = "auto",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        发送聊天请求到 LLM

        Args:
            messages: 消息列表
            tool_choice: 工具选择策略

        Returns:
            LLMResponse 封装的响应
        """
        if not self._client:
            raise RuntimeError("LLM 客户端未初始化，请先调用 initialize()")

        normalized_messages = self._normalize_messages(messages)
        selected_tools = self.tools if tools is None else tools
        normalized_tools = self._normalize_tools(selected_tools)

        # 1. 构建请求参数字典
        effective_max_tokens = self.max_tokens
        if normalized_tools and tool_choice != "none":
            effective_max_tokens = self.tool_call_max_tokens

        request_dict: Dict[str, Any] = {
            "model": self.model_name,
            "messages": normalized_messages,
            "temperature": self.temperature,
            "max_tokens": effective_max_tokens,
        }

        if normalized_tools and tool_choice != "none":
            request_dict["tools"] = normalized_tools
            request_dict["tool_choice"] = tool_choice

        # 2. 使用 Pydantic 模型验证请求参数
        try:
            validated_request = ChatCompletionsRequest(**request_dict)
            logger.debug("请求参数验证通过")
        except Exception as e:
            logger.error(f"请求参数验证失败: {e}")
            raise

        # 3. 调试日志：输出发送给 LLM 的消息（限制内容长度，避免泄露敏感信息）
        logger.debug(f"发送给 LLM 的消息数: {len(normalized_messages)}")
        for i, msg in enumerate(normalized_messages):
            role = msg.get("role", "unknown")
            # 只记录消息角色和长度，不记录内容（避免泄露用户隐私）
            content = str(msg.get("content", ""))
            logger.debug(f"消息 {i} [{role}]: 长度={len(content)}")

        # 4. 调用 API
        payload = validated_request.model_dump(exclude_none=True)
        payload["messages"] = normalized_messages
        logger.info(f"LLM请求 max_tokens={payload.get('max_tokens')}, effective={effective_max_tokens}, model={self.model_name}")

        if normalized_tools and tool_choice != "none":
            payload["tools"] = normalized_tools
            payload["tool_choice"] = tool_choice
        else:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

        if not payload.get("logprobs"):
            payload.pop("logprobs", None)
            payload.pop("top_logprobs", None)

        # 禁用 thinking 模式，避免 reasoning 吃光 tokens
        if self._should_disable_chat_template_thinking():
            payload["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            }

        semaphore = await self._get_global_semaphore(self.max_concurrent_requests)
        async with semaphore:
            try:
                response = await self._create_completion(
                    self._client,
                    payload,
                    self.model_name,
                    self.request_timeout_seconds,
                )
                used_model = self.model_name
                message = response.choices[0].message
                # 提取 reasoning_content（思考模型会返回）
                reasoning_content = getattr(message, 'reasoning_content', None)
                if reasoning_content:
                    logger.debug(f"LLM reasoning ({used_model}): {str(reasoning_content)[:300]}...")
                # 用 content 作为实际回复
                actual_content = message.content
                if not message.tool_calls and not str(actual_content or "").strip():
                    # 如果 content 为空但有 reasoning，说明 reasoning 吃掉了所有 tokens
                    if reasoning_content:
                        logger.warning(f"LLM content 为空但 reasoning 存在 ({used_model}): reasoning长度={len(str(reasoning_content))}")
                        raise RuntimeError(f"LLM返回空内容(推理占用): model={used_model}")
                    raise RuntimeError(f"LLM返回空内容: model={used_model}")
            except Exception as primary_exc:
                # 记录主模型失败时的详细信息（token、reasoning等）
                primary_resp = locals().get('response')
                primary_reasoning = locals().get('reasoning_content')
                primary_actual = locals().get('actual_content')
                primary_msg = locals().get('message')
                if primary_resp and hasattr(primary_resp, 'usage') and primary_resp.usage:
                    logger.warning(
                        f"主模型失败Token详情: model={self.model_name}, "
                        f"total={primary_resp.usage.total_tokens}, "
                        f"prompt={primary_resp.usage.prompt_tokens}, "
                        f"completion={primary_resp.usage.completion_tokens}, "
                        f"reasoning_chars={len(str(primary_reasoning)) if primary_reasoning else 0}, "
                        f"content_chars={len(str(primary_actual)) if primary_actual else 0}, "
                        f"has_tool_calls={bool(primary_msg and primary_msg.tool_calls)}"
                    )
                elif primary_resp:
                    logger.warning(f"主模型失败但无usage数据: model={self.model_name}")
                else:
                    logger.warning(f"主模型调用异常(无response): model={self.model_name}, error={primary_exc}")

                if not self.fallback_enabled or not self._fallback_client:
                    raise
                primary_error = (
                    f"{type(primary_exc).__name__}: {primary_exc}"
                    if str(primary_exc)
                    else type(primary_exc).__name__
                )
                logger.opt(exception=primary_exc).warning(
                    f"主模型调用失败，切换兜底模型: primary={self.model_name}, "
                    f"fallback={self.fallback_model_name}, error={primary_error}"
                )
                fallback_payload = dict(payload)
                # 兜底模型也禁用 thinking
                if self._should_disable_chat_template_thinking_for(
                    self.fallback_api_base,
                    self.fallback_model_name,
                ):
                    fallback_payload["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        }
                    }
                response = await self._create_completion(
                    self._fallback_client,
                    fallback_payload,
                    self.fallback_model_name,
                    self.fallback_timeout_seconds,
                )
                used_model = self.fallback_model_name
                message = response.choices[0].message
                # 提取 reasoning_content
                reasoning_content = getattr(message, 'reasoning_content', None)
                if reasoning_content:
                    logger.debug(f"LLM reasoning ({used_model}): {str(reasoning_content)[:300]}...")
                actual_content = message.content

        # 空内容检查
        if not message.tool_calls and not str(actual_content or "").strip():
            if reasoning_content:
                logger.warning(f"LLM content 为空但 reasoning 存在 ({used_model}): reasoning长度={len(str(reasoning_content))}")
                raise RuntimeError(f"LLM返回空内容(推理占用): model={used_model}")
            raise RuntimeError(f"LLM返回空内容: model={used_model}")

        # 5. 记录 token 使用情况
        if response.usage:
            logger.info(f"Token使用: model={used_model}, total={response.usage.total_tokens}, "
                        f"prompt={response.usage.prompt_tokens}, "
                        f"completion={response.usage.completion_tokens}, "
                        f"reasoning_chars={len(str(reasoning_content)) if reasoning_content else 0}, "
                        f"content_chars={len(str(actual_content)) if actual_content else 0}")

        # 6. 调试日志：输出 LLM 的响应
        if message.tool_calls:
            tool_names = [tc.function.name for tc in message.tool_calls]
            logger.info(f"LLM 决定调用工具: {tool_names}, model={used_model}")
        else:
            logger.debug(f"LLM 直接回复: model={used_model}, content={str(actual_content)[:200]}...")

        return LLMResponse(
            content=actual_content,
            tool_calls=message.tool_calls,
            raw_response=response,
            reasoning_content=reasoning_content,
        )
