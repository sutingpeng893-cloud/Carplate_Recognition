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
CURRENT_USER_AUDIO_PATH: ContextVar[str] = ContextVar("plate_agent_user_audio_path", default="")
CURRENT_AI_RAW_DIALOG: ContextVar[list[dict[str, Any]] | None] = ContextVar("plate_agent_ai_raw_dialog", default=None)
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
    collect_ai_raw_dialog(node, output)
    payload: dict[str, Any] = {
        "session_id": CURRENT_SESSION_ID.get() or None,
        "user_audio_path": CURRENT_USER_AUDIO_PATH.get() or None,
        "ai_raw_dialog": list(CURRENT_AI_RAW_DIALOG.get() or []),
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
    if payload.get("method") != "turn_result":
        return
    session_id = str(payload.get("session_id") or "no-session").strip() or "no-session"
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with _TRACE_LOCK:
            trace_path = PLATE_AGENT_TRACE_DIR / trace_filename_for_session(session_id)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            history = read_session_history(trace_path, session_id=session_id, timestamp=timestamp)
            turn = build_turn_history_record(payload, timestamp=timestamp)
            turn["turn_index"] = len(history["turns"]) + 1
            history["updated_at"] = timestamp
            history["turns"].append(turn)
            trace_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        logger.warning("plate_agent history write failed session_id=%s error=%s", session_id, exc)


def read_session_history(trace_path: Any, *, session_id: str, timestamp: str) -> dict[str, Any]:
    if trace_path.exists():
        try:
            data = json.loads(trace_path.read_text(encoding="utf-8"))
            turns = data.get("turns") if isinstance(data, dict) else None
            if isinstance(turns, list):
                return {
                    "session_id": data.get("session_id") or (None if session_id == "no-session" else session_id),
                    "created_at": data.get("created_at") or timestamp,
                    "updated_at": data.get("updated_at") or timestamp,
                    "turns": turns,
                }
        except Exception:
            pass
    return {
        "session_id": None if session_id == "no-session" else session_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "turns": [],
    }


def build_turn_history_record(payload: dict[str, Any], *, timestamp: str) -> dict[str, Any]:
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    after_state = payload.get("after_state") if isinstance(payload.get("after_state"), dict) else {}
    state = after_state or (output.get("state") if isinstance(output.get("state"), dict) else {})
    assistant_json = parse_json_text(output.get("text"))
    record = {
        "timestamp": timestamp,
        "user_audio_path": payload.get("user_audio_path"),
        "ai_inference_metadata": clean_empty_values(
            {
                "stage": output.get("stage"),
                "task_status": assistant_json.get("task_status") or output.get("task_status"),
                "car_plate": state.get("car_plate") or assistant_json.get("car_plate") or output.get("car_plate"),
                "final_car_plate": (
                    state.get("final_car_plate")
                    or assistant_json.get("final_plate_number")
                    or assistant_json.get("final_car_plate")
                ),
                "vehicle_type": state.get("vehicle_type"),
                "latency_ms": output.get("latency_ms"),
                "need_confirmation_count": len(state.get("need_confirm_chars") or []),
                "confirmed_count": len(state.get("confirmed_chars") or []),
            }
        ),
        "ai_raw_dialog": payload.get("ai_raw_dialog"),
        "history": clean_empty_values(
            {
                "assistant_reply": output.get("speech_text")
                or assistant_json.get("assistant_reply")
                or state.get("assistant_reply"),
                "assistant_json": assistant_json or None,
                "history_text": output.get("text"),
            }
        ),
    }
    return clean_empty_values(record)


def clean_empty_values(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None and value != "" and value != {} and value != []}


def collect_ai_raw_dialog(node: str, output: dict[str, Any]) -> None:
    dialog = CURRENT_AI_RAW_DIALOG.get()
    if dialog is None:
        return
    record = build_ai_raw_dialog_record(node, output, index=len(dialog) + 1)
    if record:
        dialog.append(record)


def build_ai_raw_dialog_record(node: str, output: dict[str, Any], *, index: int) -> dict[str, Any]:
    raw = output.get("raw")
    if node == "detect_plate_presence":
        return assistant_raw_record(node, raw)
    if node == "extract_car_plate.step1_extract_with_pronunciation":
        return assistant_raw_record(node, raw)
    if node == "extract_car_plate.normalize" and output.get("province_retry_raw"):
        return assistant_raw_record(node, output.get("province_retry_raw"))
    if node == "detect_confirmation":
        return assistant_raw_record(node, raw)
    if node == "update_car_plate.react_action":
        return assistant_raw_record(
            node,
            raw,
            tool_calls=[
                function_tool_call(
                    index,
                    "plate_edit_command",
                    clean_empty_values(
                        {
                            "step": output.get("step"),
                            "input_plate": output.get("input_plate"),
                            "command": output.get("command"),
                            "commands": output.get("commands"),
                        }
                    ),
                )
            ],
        )
    if node == "update_car_plate.edit_result":
        return tool_result_record(
            node,
            "plate_edit_apply",
            clean_empty_values(
                {
                    "step": output.get("step"),
                    "input_plate": output.get("input_plate"),
                    "command": output.get("command"),
                    "commands": output.get("commands"),
                    "edit_result": output.get("edit_result"),
                }
            ),
        )
    if node == "confirmation_state.detect_actions":
        return assistant_raw_record(
            node,
            raw,
            tool_calls=[
                function_tool_call(
                    index,
                    "confirmation_state_actions",
                    clean_empty_values(
                        {
                            "model_actions": output.get("model_actions"),
                            "actions": output.get("actions"),
                        }
                    ),
                )
            ],
        )
    if node == "confirmation_state.apply_actions":
        return tool_result_record(
            node,
            "confirmation_state_apply",
            clean_empty_values(
                {
                    "source": output.get("source"),
                    "actions": output.get("actions"),
                }
            ),
        )
    if node == "update_car_plate.review":
        return assistant_raw_record(
            node,
            raw,
            tool_calls=[function_tool_call(index, "plate_update_review", output.get("review"))],
        )
    return {}


def assistant_raw_record(node: str, content: Any, *, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if content is None or content == "":
        return {}
    return clean_empty_values(
        {
            "node": node,
            "messages": [
                clean_empty_values(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or None,
                    }
                )
            ],
        }
    )


def tool_result_record(node: str, name: str, content: Any) -> dict[str, Any]:
    return clean_empty_values(
        {
            "node": node,
            "messages": [
                clean_empty_values(
                    {
                        "role": "tool",
                        "name": name,
                        "content": content,
                    }
                )
            ],
        }
    )


def function_tool_call(index: int, name: str, arguments: Any) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments or {},
        },
    }


def parse_json_text(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def safe_session_filename(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)[:120] or "no-session"


def trace_filename_for_session(session_id: str) -> str:
    """同一个 session 固定写入同一个按时间命名的 JSON history 文件。"""
    safe_session_id = safe_session_filename(session_id)
    filename = _TRACE_FILENAMES_BY_SESSION.get(safe_session_id)
    if filename:
        return filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}_{safe_session_id}.json"
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
