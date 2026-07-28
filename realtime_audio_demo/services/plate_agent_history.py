from __future__ import annotations

from typing import Any

from realtime_audio_demo.services.plate_agent_rules import clean_plate_text, describe_plate_char
from realtime_audio_demo.services.plate_agent_types import PlateAgentResult, PlateAgentState


MAX_SUMMARY_TEXT_LENGTH = 900
MAX_ASSISTANT_REPLY_LENGTH = 180


def build_plate_turn_summary(
    *,
    turn_index: int,
    before_state: PlateAgentState,
    result: PlateAgentResult,
) -> str:
    before_plate = clean_plate_text(before_state.car_plate)
    after_state = result.state
    after_plate = clean_plate_text(after_state.car_plate)

    parts = [f"第{turn_index}轮：用户输入了一段语音。"]
    parts.append(describe_turn_change(before_state=before_state, after_state=after_state))
    if before_plate != after_plate:
        parts.append(f"车牌从{before_plate or '空'}更新为{after_plate or '空'}。")
    elif after_plate:
        parts.append(f"当前暂存车牌为{after_plate}。")

    confirmed_text = describe_char_list(after_state.confirmed_chars)
    if confirmed_text:
        parts.append(f"已确认字符：{confirmed_text}。")

    pending_text = describe_char_list(after_state.need_confirm_chars)
    if pending_text:
        parts.append(f"仍需二次确认：{pending_text}。")

    if after_state.final_car_plate:
        parts.append(f"最终车牌为{after_state.final_car_plate}。")

    assistant_reply = trim_text(result.speech_text or after_state.assistant_reply, MAX_ASSISTANT_REPLY_LENGTH)
    if assistant_reply:
        parts.append(f"AI回复：{assistant_reply}")

    return trim_text("".join(parts), MAX_SUMMARY_TEXT_LENGTH)


def describe_turn_change(*, before_state: PlateAgentState, after_state: PlateAgentState) -> str:
    before_plate = clean_plate_text(before_state.car_plate)
    after_plate = clean_plate_text(after_state.car_plate)
    if not before_state.has_car_plate and after_state.has_car_plate:
        return f"系统首轮识别到车牌{after_plate}。"
    if not before_state.has_car_plate and not after_state.has_car_plate:
        return "系统本轮还没有形成可用车牌。"
    if after_state.final_car_plate:
        return "用户确认了当前车牌。"
    if before_plate != after_plate:
        return "用户进行了车牌纠错。"
    if confirmed_positions(before_state) != confirmed_positions(after_state):
        return "用户确认了部分待确认字符。"
    return "系统保留当前车牌，继续等待用户确认或纠错。"


def confirmed_positions(state: PlateAgentState) -> list[int]:
    return sorted(item.position for item in state.confirmed_chars if item.confirmed and item.position > 0)


def describe_char_list(items: list[Any]) -> str:
    descriptions: list[str] = []
    for item in items:
        position = getattr(item, "position", 0)
        value = getattr(item, "value", "")
        if position <= 0 or not value:
            continue
        descriptions.append(f"第{position}位={describe_plate_char(value)}")
    return "，".join(descriptions)


def trim_text(value: Any, max_length: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"
