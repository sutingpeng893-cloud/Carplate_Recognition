from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any


logger = logging.getLogger("uvicorn.error")
CURRENT_SESSION_ID: ContextVar[str] = ContextVar("plate_agent_session_id", default="")
CURRENT_TURN_BEFORE_STATE: ContextVar[dict[str, Any] | None] = ContextVar("plate_agent_turn_before_state", default=None)


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
    logger.info("plate_agent event=%s", json.dumps(payload, ensure_ascii=False, default=str))


def log_agent_line(message: str, **fields: Any) -> None:
    payload: dict[str, Any] = {
        "session_id": CURRENT_SESSION_ID.get() or None,
        "说明": message,
    }
    if fields:
        payload["详情"] = fields
    logger.info("plate_agent_trace %s", json.dumps(payload, ensure_ascii=False, default=str))
