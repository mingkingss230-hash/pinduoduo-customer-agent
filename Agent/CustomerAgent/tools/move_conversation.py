"""
会话转接工具

将当前会话转接给人工客服。
"""
from typing import Optional, Union

from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from Channel.pinduoduo.utils.API.send_message import SendMessage
from database.db_manager import db_manager
from utils.logger_loguru import get_logger
from utils.night_mode import (
    NIGHT_MODE_TRANSFER_RESULT_PREFIX,
    build_night_mode_key,
    get_night_mode_reply,
    is_night_mode,
)
from utils.transfer_target import choose_transfer_candidate

logger = get_logger("TransferConversationTool")


class TransferConversationParams(BaseModel):
    """会话转接参数"""

    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID；如果需要传入，纯数字也按字符串传入")
    user_id: Optional[Union[str, int]] = Field(default=None, description="用户ID（账号ID）；如果需要传入，纯数字也按字符串传入")
    recipient_uid: Optional[str] = Field(
        default=None,
        description=(
            "接收转接的客户UID，必须是字符串。即使内容全是数字，也必须加引号，"
            "例如传 \"4813704555\"，不要传 4813704555。"
        ),
    )


@agent_tool(
    name="transfer_conversation",
    description=(
        "将当前会话转接给人工客服。仅在需要人工执行动作或升级处理时使用，"
        "例如改地址、拦截、拒收、改派、补发、开发票、退货地址/取件码/寄回核验、"
        "平台介入、投诉举报，或售后处理动作明确返回需要转人工。"
        "客户明确要求人工、人工客服、售后专员、售后处理、联系人工、转人工时，应直接调用。"
        "客户出现已签收异常、坏了不能用、噪音大、风力小、充不进电、少件漏发、退款/退货/赔付等明确售后诉求时，也应优先转人工。"
        "调用本工具时，所有 ID 参数都必须按字符串传入；纯数字 ID 也必须加引号，"
        "尤其 recipient_uid 必须传字符串，例如 \"4813704555\"，不能传数字 4813704555。"
    ),
    param_model=TransferConversationParams,
)
def transfer_conversation(params: TransferConversationParams) -> str:
    """将当前会话转接给人工客服。"""
    try:
        if is_night_mode():
            key = build_night_mode_key(params.shop_id, params.user_id, params.recipient_uid)
            reply = get_night_mode_reply(key)
            logger.info(
                "夜间模式拦截转人工: "
                f"shop_id={params.shop_id}, user_id={params.user_id}, recipient_uid={params.recipient_uid}"
            )
            return f"{NIGHT_MODE_TRANSFER_RESULT_PREFIX}：{reply}"

        recipient_uid = str(params.recipient_uid or "").strip()
        if recipient_uid.lower() in {"", "null", "none"}:
            recipient_uid = ""

        if not all([params.shop_id, params.user_id, recipient_uid]):
            logger.warning(
                "会话转接失败: 缺少必要的会话信息 "
                f"(shop_id={params.shop_id}, user_id={params.user_id}, recipient_uid={params.recipient_uid})"
            )
            return (
                "转接失败：缺少必要的会话信息 "
                f"(shop_id={params.shop_id}, user_id={params.user_id}, recipient_uid={params.recipient_uid})"
            )

        sender = SendMessage(str(params.shop_id), str(params.user_id))
        cs_list = sender.getAssignCsList()
        if cs_list and isinstance(cs_list, dict):
            preferred = db_manager.get_transfer_target(
                "pinduoduo",
                str(params.shop_id),
                str(params.user_id),
            )
            candidate = choose_transfer_candidate(
                str(params.shop_id),
                str(params.user_id),
                cs_list,
                preferred.get("target_user_id") if preferred else None,
            )

            if candidate:
                cs_uid = candidate["raw_cs_uid"]
                transfer_result = sender.move_conversation(recipient_uid, cs_uid)

                if transfer_result and transfer_result.get("success"):
                    logger.info(
                        f"会话转接成功: recipient_uid={recipient_uid}, "
                        f"to_cs_uid={cs_uid}, configured_cs_uid={candidate['cs_uid']}, "
                        f"target_username={candidate['username']}"
                    )
                    return "会话转接成功"

                logger.warning(f"会话转接失败: transfer_result={transfer_result}")
                return "会话转接失败"

            if preferred and preferred.get("target_user_id"):
                logger.warning(
                    "会话转接失败: 指定客服不在可转接列表中 "
                    f"(shop_id={params.shop_id}, target_user_id={preferred.get('target_user_id')})"
                )
                return "指定人工客服当前不可转接"

            logger.warning(f"会话转接失败: 当前无可用的人工客服 (shop_id={params.shop_id})")
            return "当前无可用的人工客服"

        logger.warning("会话转接失败：无法获取客服列表")
        return "会话转接失败：无法获取客服列表"

    except Exception as exc:
        logger.error(f"转接过程中发生错误: {exc}")
        return f"转接过程中发生错误: {exc}"
