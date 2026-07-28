from __future__ import annotations

import json
from typing import Any

from realtime_audio_demo.services.plate_agent_rules import clean_plate_text, describe_plate_char, with_relative_confusion_reasons
from realtime_audio_demo.services.plate_agent_types import PlateAgentState


def build_output_json(
    *,
    task_status: str,
    car_plate: str,
    assistant_reply: str,
    final_car_plate: str = "",
) -> str:
    data: dict[str, Any] = {
        "task_status": task_status,
        "car_plate": clean_plate_text(car_plate),
        "assistant_reply": assistant_reply,
    }
    if final_car_plate:
        data["final_plate_number"] = clean_plate_text(final_car_plate)
    return json.dumps(data, ensure_ascii=False, indent=2)


def fallback_reply(state: PlateAgentState) -> str:
    plate = state.car_plate or "当前车牌"
    parts = [f"我这边暂时识别到的车牌号是{plate}。"]
    descriptions = pending_confirmation_descriptions(state)
    if descriptions:
        parts.append("请您再确认一下：" + "；".join(descriptions) + "。")
    else:
        parts.append("请您确认一下是否正确。")
    if state.vehicle_type == "new_energy":
        parts.append("另外这是新能源号牌吗？")
    return "".join(parts)


def reply_with_pending_confirmation(base_reply: str, state: PlateAgentState) -> str:
    reply = str(base_reply or "").strip()
    descriptions = pending_confirmation_descriptions(state)
    if descriptions:
        suffix = "当前仍需您确认：" + "；".join(descriptions) + "。"
    else:
        plate = state.car_plate or "当前保留的车牌"
        suffix = f"请您确认{plate}是否正确。"
    if not reply:
        return suffix
    if not reply.endswith(("。", "！", "？", ".", "!", "?")):
        reply += "。"
    return reply + suffix


def pending_confirmation_descriptions(state: PlateAgentState) -> list[str]:
    descriptions: list[str] = []
    if state.need_confirm_chars:
        for item in state.need_confirm_chars:
            reason = normalize_confirmation_reason(item.reason)
            if not reason:
                reason = f"第{item.position}位当前识别为{describe_plate_char(item.value)}，请您确认是否正确"
            descriptions.append(reason)
        return descriptions

    for item in with_relative_confusion_reasons(state.car_plate, state.confusions):
        reason = normalize_confirmation_reason(item.reason)
        if not reason:
            reason = f"第{item.position}位当前识别为{describe_plate_char(item.value)}，请您确认是否正确"
        descriptions.append(reason)
    return descriptions


def normalize_confirmation_reason(value: str) -> str:
    reason = str(value or "").strip().rstrip("。")
    if not reason:
        return ""
    reason = reason.replace("请用户确认是否正确", "请您确认是否正确")
    reason = reason.replace("请用户确认", "请您确认")
    return reason
