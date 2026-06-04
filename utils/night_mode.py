"""
Night-mode handoff helpers.

During night mode we keep replying, but do not move conversations to human CS.
"""
from __future__ import annotations

from datetime import datetime, time
from threading import Lock
from typing import Dict


NIGHT_MODE_START_HOUR = 23
NIGHT_MODE_END_HOUR = 8

NIGHT_MODE_FIRST_REPLY = (
    "亲，当前问题需要高级客服为您处理，高级客服上班时间为早上8点~晚上11点，"
    "建议您晚点联系这边由高级客服为您处理哦！"
)
NIGHT_MODE_SECOND_REPLY = (
    "亲，专业的高级客服下班了，还没上班，上班时间联系这边，会为您妥善处理的，您耐心等待下。"
)
NIGHT_MODE_REPLIES = (
    NIGHT_MODE_FIRST_REPLY,
    NIGHT_MODE_SECOND_REPLY,
    "亲，您的问题这边已经收到啦，目前高级客服不在线，早上8点后会有专人继续帮您处理，请您先放心。",
    "亲，您先别着急，夜间无法转接高级客服，早上8点后客服上班会继续为您核实处理的。",
    "亲，您反馈的情况我已经了解，目前夜间只能先为您记录，高级客服上班后会优先处理。",
    "亲，现在是夜间值守时段，高级客服暂时不在线，您可以先把情况补充完整，早上会继续处理。",
    "亲，已经帮您记录诉求了，当前时段无法转人工，早上8点后高级客服会接着为您处理。",
    "亲，您连续发的消息我这边都收到了，请您先耐心等一下，高级客服上班后会为您处理。",
)
NIGHT_MODE_TRANSFER_RESULT_PREFIX = "夜间不转人工"

_state_lock = Lock()
_reply_stages: Dict[str, int] = {}


def is_night_mode(now: datetime | None = None) -> bool:
    current = now or datetime.now()
    start, end = get_night_mode_time_range()
    current_time = current.time().replace(second=0, microsecond=0)
    if start == end:
        return False
    if start < end:
        return start <= current_time < end
    return current_time >= start or current_time < end


def get_night_mode_time_range() -> tuple[time, time]:
    start_text = f"{NIGHT_MODE_START_HOUR:02d}:00"
    end_text = f"{NIGHT_MODE_END_HOUR:02d}:00"
    try:
        from config import config

        start_text = str(config.get("night_mode.start", start_text) or start_text)
        end_text = str(config.get("night_mode.end", end_text) or end_text)
    except Exception:
        pass

    try:
        start = datetime.strptime(start_text, "%H:%M").time()
        end = datetime.strptime(end_text, "%H:%M").time()
        return start, end
    except Exception:
        return time(NIGHT_MODE_START_HOUR, 0), time(NIGHT_MODE_END_HOUR, 0)


def build_night_mode_key(shop_id: object = None, user_id: object = None, recipient_uid: object = None) -> str:
    return f"{shop_id or ''}:{user_id or ''}:{recipient_uid or ''}"


def get_night_mode_reply(key: str | None = None) -> str:
    if not key:
        return NIGHT_MODE_FIRST_REPLY

    with _state_lock:
        stage = _reply_stages.get(key, 0)
        _reply_stages[key] = stage + 1

    return NIGHT_MODE_REPLIES[stage % len(NIGHT_MODE_REPLIES)]


def reset_night_mode_reply_state() -> None:
    with _state_lock:
        _reply_stages.clear()
