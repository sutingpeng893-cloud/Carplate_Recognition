import json
import re
from dataclasses import dataclass
from typing import Any

from realtime_audio_demo.services.plate_agent_messages import (
    NORMALIZED_INVALID_PLATE_REPLY,
    NORMALIZED_SHORT_PLATE_REPLY,
)


@dataclass
class AssistantOutput:
    raw_text: str
    display_text: str
    history_text: str
    speech_text: str
    is_json: bool


def normalize_assistant_output(text: str | None) -> AssistantOutput:
    raw_text = (text or "").strip()
    if not raw_text:
        return AssistantOutput(
            raw_text="",
            display_text="",
            history_text="",
            speech_text="",
            is_json=False,
        )

    candidate = extract_json_candidate(raw_text)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return AssistantOutput(
            raw_text=raw_text,
            display_text=raw_text,
            history_text=raw_text,
            speech_text=raw_text,
            is_json=False,
        )

    if not isinstance(value, dict):
        display_text = format_json_value(value)
        return AssistantOutput(
            raw_text=raw_text,
            display_text=display_text,
            history_text=display_text,
            speech_text="",
            is_json=True,
        )

    display_text = format_json_value(value)

    return AssistantOutput(
        raw_text=raw_text,
        display_text=display_text,
        history_text=display_text,
        speech_text=display_text,
        is_json=True,
    )


def extract_json_candidate(text: str, *, prefer_object: bool = False) -> str:
    stripped = text.strip()
    return extract_last_balanced_json_candidate(stripped, prefer_object=prefer_object) or stripped


def extract_last_balanced_json_candidate(text: str, *, prefer_object: bool = False) -> str:
    candidates: list[tuple[str, str]] = []
    stack: list[tuple[str, int]] = []
    closing_to_opening = {"}": "{", "]": "["}
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"' and stack:
            in_string = True
            continue

        if char in {"{", "["}:
            stack.append((char, index))
            continue

        expected_opening = closing_to_opening.get(char)
        if not expected_opening:
            continue

        match_index = -1
        for stack_index in range(len(stack) - 1, -1, -1):
            if stack[stack_index][0] == expected_opening:
                match_index = stack_index
                break
        if match_index < 0:
            stack.clear()
            continue

        opening_char, start = stack[match_index]
        del stack[match_index:]
        candidate = text[start : index + 1].strip()
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        candidates.append((candidate, opening_char))

    if prefer_object:
        for candidate, opening_char in reversed(candidates):
            if opening_char == "{":
                return candidate
    if candidates:
        return candidates[-1][0]
    return ""


def parse_assistant_json(text: str | None) -> dict[str, Any] | None:
    candidate = extract_json_candidate((text or "").strip())
    if not candidate:
        return None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate_current_normalized_plate(text: str | None) -> str | None:
    data = parse_assistant_json(text)
    if not data or "normalized" not in data:
        return text

    normalized = re.sub(r"\s+", "", str(data.get("normalized") or ""))
    data["normalized"] = normalized
    plate_length = len(normalized)
    if plate_length in {7, 8}:
        return format_json_value(data)

    if plate_length < 7:
        data["task_status"] = "need_more_info"
        data["assistant_reply"] = NORMALIZED_SHORT_PLATE_REPLY
    else:
        data["task_status"] = "invalid"
        data["assistant_reply"] = NORMALIZED_INVALID_PLATE_REPLY

    return format_json_value(data)


def format_json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
