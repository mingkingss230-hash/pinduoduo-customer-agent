"""Customer service knowledge action routing and sanitization."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


INTERNAL_ACTION_TERMS = (
    "转人工",
    "人工客服",
)

DIRECT_TRANSFER_REPLIES = {
    "转人工",
    "转人工处理",
    "联系人工",
    "联系人工客服",
}

SAFE_TRANSFER_REPLY = "亲，已转人工为您处理，请稍等。"
FORBIDDEN_REPLY_REPLACEMENTS = {
    "运费险": "退货包运费服务",
}


@dataclass
class KnowledgeAction:
    action_type: str
    reason: str = ""
    customer_reply: str = ""


@dataclass
class RoutedKnowledge:
    sanitized_text: str
    action: Optional[KnowledgeAction] = None


def _normalize_action_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip("。？！?!，,")


def _extract_first_standard_answer(formatted_knowledge: str) -> str:
    text = str(formatted_knowledge or "")
    match = re.search(r"标准答案[:：]\s*(.+?)(?:\n\s*\n|\Z)", text, flags=re.S)
    if match:
        return match.group(1).strip()

    for line in text.splitlines():
        item = line.strip()
        if not item or item.startswith("【") or re.match(r"^\d+\.", item):
            continue
        return item
    return ""


def _strip_internal_parentheses(text: str) -> str:
    result = str(text or "")
    for open_p, close_p in (("（", "）"), ("(", ")")):
        pattern = rf"\{open_p}[^\{open_p}\{close_p}]*(?:转人工|人工客服)[^\{open_p}\{close_p}]*\{close_p}"
        result = re.sub(pattern, "", result)
    return result


def sanitize_customer_service_text(text: str) -> str:
    """Remove internal action hints from knowledge text."""
    result = _strip_internal_parentheses(text)

    compact = _normalize_action_text(result)
    if compact in {_normalize_action_text(item) for item in DIRECT_TRANSFER_REPLIES}:
        return SAFE_TRANSFER_REPLY

    for term in INTERNAL_ACTION_TERMS:
        result = result.replace(term, "")

    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"[。？！?!，,]\s*$", "", result).strip()
    return result


def sanitize_formatted_knowledge(formatted_knowledge: str) -> str:
    """Clean the rendered knowledge text before it reaches the model."""
    text = str(formatted_knowledge or "")
    if not text.strip():
        return text

    def replace_answer(match: re.Match[str]) -> str:
        answer = match.group(1)
        suffix = match.group(2)
        safe_answer = sanitize_customer_service_text(answer)
        return f"{safe_answer}{suffix}"

    return re.sub(
        r"标准答案[:：]\s*(.+?)(\n\s*\n|\Z)",
        replace_answer,
        text,
        flags=re.S,
    ).strip()


def route_customer_service_knowledge(formatted_knowledge: str, query: str = "") -> RoutedKnowledge:
    """Route internal transfer actions and return cleaned knowledge text."""
    return RoutedKnowledge(sanitized_text=sanitize_formatted_knowledge(formatted_knowledge))


def sanitize_final_reply(reply: str) -> str:
    """Final safety pass before sending text to the customer."""
    raw = str(reply or "")
    if _normalize_action_text(raw) in {_normalize_action_text(item) for item in DIRECT_TRANSFER_REPLIES}:
        return SAFE_TRANSFER_REPLY

    text = _strip_internal_parentheses(raw)
    for term in INTERNAL_ACTION_TERMS:
        text = text.replace(term, "")
    for forbidden, replacement in FORBIDDEN_REPLY_REPLACEMENTS.items():
        text = text.replace(forbidden, replacement)
    text = re.sub(r"\s+", " ", text).strip()
    return text
