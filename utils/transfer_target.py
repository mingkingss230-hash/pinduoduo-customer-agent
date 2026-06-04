"""Helpers for resolving configured manual-transfer targets."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


CS_UID_FIELDS = ("cs_uid", "csUid", "csid", "csId", "cs_id")
USER_ID_FIELDS = (
    "user_id",
    "userId",
    "uid",
    "id",
    "staffId",
    "staff_id",
    "operatorId",
    "csUserId",
    "cs_user_id",
    "pddUserId",
)
USERNAME_FIELDS = ("username", "name", "nickname", "nickName", "displayName", "realName")


def normalize_cs_uid(shop_id: str, value: Any) -> Optional[str]:
    """Return the stable cs uid format used by local configuration."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.startswith("cs_"):
        return text
    return f"cs_{shop_id}_{text}"


def _first_text(info: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def resolve_stable_cs_uid(shop_id: str, raw_cs_uid: Any, info: Any = None) -> Optional[str]:
    """Build a stable cs uid from a backend cs-list item."""
    info = info if isinstance(info, dict) else {}

    explicit_cs_uid = _first_text(info, CS_UID_FIELDS)
    if explicit_cs_uid:
        return normalize_cs_uid(shop_id, explicit_cs_uid)

    raw_text = str(raw_cs_uid).strip() if raw_cs_uid is not None else ""
    if raw_text.startswith("cs_"):
        return raw_text

    user_id = _first_text(info, USER_ID_FIELDS)
    if user_id:
        return normalize_cs_uid(shop_id, user_id)

    return normalize_cs_uid(shop_id, raw_text)


def resolve_cs_username(raw_cs_uid: Any, info: Any = None) -> str:
    """Return the best display name for a backend cs-list item."""
    info = info if isinstance(info, dict) else {}
    return _first_text(info, USERNAME_FIELDS) or str(raw_cs_uid)


def build_transfer_candidates(shop_id: str, source_user_id: str, cs_list: Dict[str, Any]) -> list[dict]:
    """Normalize backend online CS list into UI/runtime transfer candidates."""
    source_raw_uid = str(source_user_id)
    source_stable_uid = normalize_cs_uid(shop_id, source_user_id)
    candidates = []

    for raw_cs_uid, info in cs_list.items():
        raw_text = str(raw_cs_uid).strip()
        stable_cs_uid = resolve_stable_cs_uid(shop_id, raw_text, info)
        if not stable_cs_uid:
            continue
        if raw_text in {source_raw_uid, source_stable_uid} or stable_cs_uid == source_stable_uid:
            continue

        candidates.append(
            {
                "raw_cs_uid": raw_text,
                "cs_uid": stable_cs_uid,
                "username": resolve_cs_username(raw_text, info),
                "info": info if isinstance(info, dict) else {},
            }
        )

    return candidates


def choose_transfer_candidate(
    shop_id: str,
    source_user_id: str,
    cs_list: Dict[str, Any],
    preferred_value: Any = None,
) -> Optional[dict]:
    """Choose the configured transfer candidate, or auto-pick when no preference exists."""
    candidates = build_transfer_candidates(shop_id, source_user_id, cs_list)
    if not candidates:
        return None

    preferred_raw = str(preferred_value).strip() if preferred_value else ""
    preferred_stable = normalize_cs_uid(shop_id, preferred_raw) if preferred_raw else None

    if preferred_raw:
        for candidate in candidates:
            if candidate["raw_cs_uid"] == preferred_raw or candidate["cs_uid"] == preferred_stable:
                return candidate
        return None

    return candidates[0]
