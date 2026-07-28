from __future__ import annotations

import json
import re
import time
from typing import Any

from realtime_audio_demo.services.output_filter import extract_json_candidate
from realtime_audio_demo.services.plate_agent_edit import normalize_spoken_plate_chars, parse_positive_int
from realtime_audio_demo.services.plate_agent_rules import (
    PROVINCE_ABBREVIATIONS,
    SPECIAL_PLATE_TAIL_CHARS,
    clean_plate_text,
    normalize_plate_format,
)


PLATE_JSON_KEY_ALIASES = {
    "carplate",
    "plate",
    "platenumber",
    "licenseplate",
    "licenseplatenumber",
    "finalplate",
    "finalcarplate",
    "finalplatenumber",
    "finalcarnumber",
    "车牌",
    "车牌号",
    "最终车牌",
    "最终车牌号",
}

UNKNOWN_PLATE_VALUES = {"", "?", "？", "UNKNOWN", "NONE", "NULL", "INVALID", "无", "未知", "不确定"}


def parse_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(extract_json_candidate(raw, prefer_object=True))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def extract_plate_from_json_object(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key, value in data.items():
        normalized_key = normalize_json_key(key)
        if normalized_key in PLATE_JSON_KEY_ALIASES:
            plate = sanitize_extracted_plate_text(value)
            if plate:
                return plate
    return ""


def sanitize_extracted_plate_text(value: Any) -> str:
    if isinstance(value, dict):
        return extract_plate_from_json_object(value)
    if not isinstance(value, str):
        return ""

    raw = value.strip()
    if not raw:
        return ""

    for variant in jsonish_text_variants(raw):
        if looks_like_json_fragment(variant):
            nested_plate = extract_plate_from_json_object(parse_json_object(variant))
            if nested_plate:
                return nested_plate
            return ""

    converted = normalize_spoken_plate_chars(raw)
    compact = clean_plate_text(converted).upper()
    if compact in UNKNOWN_PLATE_VALUES:
        return ""

    chars: list[str] = []
    for char in compact:
        if char in PROVINCE_ABBREVIATIONS or char in SPECIAL_PLATE_TAIL_CHARS:
            chars.append(char)
        elif char.isascii() and char.isalnum():
            chars.append(char.upper())

    plate = "".join(chars)
    if not plate or plate in UNKNOWN_PLATE_VALUES:
        return ""
    if len(plate) < 2:
        return ""
    if normalize_json_key(plate) in PLATE_JSON_KEY_ALIASES or "CARPLATE" in plate:
        return ""
    return plate


def jsonish_text_variants(value: str) -> list[str]:
    variants = [value]
    if "\\" in value:
        variants.append(value.replace('\\"', '"').replace("\\", ""))
    return unique_text_values(variants)


def looks_like_json_fragment(value: str) -> bool:
    text = value.strip()
    return ("{" in text and "}" in text) or ("[" in text and "]" in text)


def normalize_json_key(value: Any) -> str:
    return re.sub(r"[\s_\\/:：\"'`{}【】\[\]()-]+", "", str(value or "")).lower()


def unique_text_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_final_plate_from_text(text: Any) -> str:
    raw = str(text or "")
    matches = re.findall(r"^最终车牌[:：]\s*(.+)$", raw, flags=re.MULTILINE)
    if not matches:
        return ""
    return matches[-1].strip()


def parse_bool_text(text: str, *, default: bool) -> bool:
    value = str(text or "").strip().lower()
    if "true" in value:
        return True
    if "false" in value:
        return False
    return default


def parse_json_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"true", "yes", "1", "是", "对", "需要"}:
        return True
    if raw in {"false", "no", "0", "否", "不", "不需要"}:
        return False
    return default


def parse_position_list(value: Any, max_position: int) -> list[int]:
    raw_items = value if isinstance(value, list) else [value]
    positions: list[int] = []
    for item in raw_items:
        position = parse_positive_int(item)
        if 1 <= position <= max_position:
            positions.append(position)
    return unique_positions(positions)


def unique_positions(values: list[int]) -> list[int]:
    seen: set[int] = set()
    positions: list[int] = []
    for value in values:
        try:
            position = int(value)
        except (TypeError, ValueError):
            continue
        if position <= 0 or position in seen:
            continue
        seen.add(position)
        positions.append(position)
    return positions


def parse_yes_no(text: str, *, default: bool) -> bool:
    value = str(text or "").strip().lower()
    if re.search(r"\byes\b", value) or "true" in value:
        return True
    if re.search(r"\bno\b", value) or "false" in value:
        return False
    return default


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
