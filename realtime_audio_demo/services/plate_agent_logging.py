from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Any

from realtime_audio_demo.config import (
    PLATE_AGENT_LOG_DETAIL_MAX_CHARS,
    PLATE_AGENT_TRACE_DIR,
    PLATE_AGENT_TRACE_ENABLED,
)


logger = logging.getLogger("uvicorn.error")
CURRENT_SESSION_ID: ContextVar[str] = ContextVar("plate_agent_session_id", default="")
CURRENT_TURN_BEFORE_STATE: ContextVar[dict[str, Any] | None] = ContextVar("plate_agent_turn_before_state", default=None)
_TRACE_LOCK = threading.Lock()
_TRACE_FILENAMES_BY_SESSION: dict[str, str] = {}


def state_change_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = [
        "car_plate",
        "confirmed",
        "final_car_plate",
        "vehicle_type",
        "ack_sent",
        "plate_chars",
        "need_confirm_chars",
        "confirmed_chars",
        "confusions",
        "assistant_reply",
    ]
    changes: dict[str, dict[str, Any]] = {}
    for key in keys:
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value != after_value:
            changes[key] = {
                "before": before_value,
                "after": after_value,
            }
    return changes


def log_node_output(node: str, output: dict[str, Any]) -> None:
    before_state = output.get("before_state") or CURRENT_TURN_BEFORE_STATE.get()
    after_state = output.get("after_state") or output.get("state")
    payload: dict[str, Any] = {
        "session_id": CURRENT_SESSION_ID.get() or None,
        "method": node,
        "event": "node_output",
        "output": output,
    }
    if isinstance(before_state, dict):
        payload["before_state"] = before_state
    if isinstance(after_state, dict):
        payload["after_state"] = after_state
    if isinstance(before_state, dict) and isinstance(after_state, dict):
        payload["state_diff"] = state_change_summary(before_state, after_state)
    write_session_trace(payload)
    logger.info("plate_agent %s", format_node_log_summary(payload))


def log_agent_line(message: str, **fields: Any) -> None:
    payload: dict[str, Any] = {
        "session_id": CURRENT_SESSION_ID.get() or None,
        "event": "trace_line",
        "说明": message,
    }
    if fields:
        payload["详情"] = fields
    write_session_trace(payload)
    logger.info("plate_agent_trace %s", format_trace_line_summary(payload))


def write_session_trace(payload: dict[str, Any]) -> None:
    if not PLATE_AGENT_TRACE_ENABLED:
        return
    session_id = str(payload.get("session_id") or "no-session").strip() or "no-session"
    trace_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    try:
        line = json.dumps(trace_payload, ensure_ascii=False, default=str)
        with _TRACE_LOCK:
            trace_path = PLATE_AGENT_TRACE_DIR / trace_filename_for_session(session_id)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:
        logger.warning("plate_agent trace write failed session_id=%s error=%s", session_id, exc)


def safe_session_filename(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)[:120] or "no-session"


def trace_filename_for_session(session_id: str) -> str:
    """同一个 session 固定写入同一个按时间命名的 JSONL 文件。"""
    safe_session_id = safe_session_filename(session_id)
    filename = _TRACE_FILENAMES_BY_SESSION.get(safe_session_id)
    if filename:
        return filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}_{safe_session_id}.jsonl"
    _TRACE_FILENAMES_BY_SESSION[safe_session_id] = filename
    return filename


def format_node_log_summary(payload: dict[str, Any]) -> str:
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    before_state = payload.get("before_state") if isinstance(payload.get("before_state"), dict) else {}
    after_state = payload.get("after_state") if isinstance(payload.get("after_state"), dict) else {}
    state = after_state or (output.get("state") if isinstance(output.get("state"), dict) else {})
    fields = {
        "session_id": payload.get("session_id"),
        "method": payload.get("method"),
        "stage": output.get("stage"),
        "action": output.get("action"),
        "car_plate": state.get("car_plate") or output.get("car_plate"),
        "before_plate": before_state.get("car_plate"),
        "after_plate": after_state.get("car_plate"),
        "status": output.get("task_status") or output.get("status"),
        "latency_ms": output.get("latency_ms"),
        "pending_count": len(state.get("need_confirm_chars") or []),
        "confirmed_count": len(state.get("confirmed_chars") or []),
    }
    return compact_kv(fields)


def format_trace_line_summary(payload: dict[str, Any]) -> str:
    details = payload.get("详情") if isinstance(payload.get("详情"), dict) else {}
    fields = {
        "session_id": payload.get("session_id"),
        "message": payload.get("说明"),
        "car_plate": first_present(details, "当前车牌", "候选车牌", "执行后车牌"),
        "stage": details.get("阶段") or details.get("stage"),
        "action": details.get("action"),
        "detail": short_text(select_detail_text(details)),
    }
    return compact_kv(fields)


def compact_kv(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None or value == "" or value == []:
            continue
        parts.append(f"{key}={short_text(value)}")
    return " ".join(parts)


def first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if not is_empty_value(value):
            return value
    return None


def is_empty_value(value: Any) -> bool:
    return value is None or value == "" or value == []


def select_detail_text(details: dict[str, Any]) -> Any:
    for key in ("说明", "原因", "错误信息", "回复", "模型判断", "是否修改", "是否还有未处理修改", "编辑是否有效"):
        if key in details:
            return details.get(key)
    return ""


def short_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    max_chars = max(80, PLATE_AGENT_LOG_DETAIL_MAX_CHARS)
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text
