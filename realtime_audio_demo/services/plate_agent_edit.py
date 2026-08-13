from __future__ import annotations

import json
import re
from typing import Any

from realtime_audio_demo.services.output_filter import extract_json_candidate
from realtime_audio_demo.services.plate_agent_messages import (
    EDIT_UNCLEAR_REPLY,
    build_char_not_found_reply,
    build_duplicate_char_reply,
    build_keep_current_plate_reply,
)
from realtime_audio_demo.services.plate_agent_types import PlateEditCommand, PlateEditResult


MODEL_EDIT_ACTIONS = {"replace_position", "replace_char", "insert_position", "delete_position", "none"}
INTERNAL_EDIT_ACTIONS = MODEL_EDIT_ACTIONS | {"unknown"}
SPOKEN_PLATE_CHAR_REPLACEMENTS = {
    "零": "0",
    "〇": "0",
    "洞": "0",
    "一": "1",
    "幺": "1",
    "么": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "是": "4",
    "五": "5",
    "六": "6",
    "陆": "6",
    "七": "7",
    "拐": "7",
    "八": "8",
    "九": "9",
    "吸": "C",
    "勾": "J",
    "沟儿": "J",
    "圈": "Q",
}
def parse_plate_edit_command(text: Any) -> PlateEditCommand:
    """解析文本为单条编辑指令，取列表第一项；解析失败返回 unknown 指令。"""
    commands = parse_plate_edit_commands(text)
    return commands[0] if commands else unknown_edit_command()


def parse_plate_edit_commands(text: Any) -> list[PlateEditCommand]:
    """解析文本为编辑指令列表；支持 JSON 数组、actions/commands 字段和 tool_calls 格式。"""
    value = parse_json_value(text)
    command_items = extract_command_items(value)
    commands = [parse_plate_edit_command_data(item) for item in command_items if isinstance(item, dict)]
    return commands or [unknown_edit_command()]


def extract_command_items(value: Any) -> list[dict[str, Any]]:
    """从 JSON 结构中提取编辑指令 dict 列表，兼容 actions/commands/tool_calls 等多种模型输出格式。"""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []

    for key in ("actions", "commands", "edits"):
        items = value.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]

    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        commands: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function_data = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function_data, dict):
                continue
            arguments = parse_arguments_object(function_data.get("arguments"))
            if isinstance(arguments, dict):
                commands.append({**arguments, "action": value.get("action") or function_data.get("name") or arguments.get("action")})
            else:
                commands.append({"action": value.get("action") or function_data.get("name")})
        if commands:
            return commands

    return [value]


def parse_plate_edit_command_data(data: dict[str, Any]) -> PlateEditCommand:
    """将单个 dict 解析为 PlateEditCommand，兼容 tool_calls/function_call/arguments 嵌套格式。"""
    tool_calls = data.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first_tool = tool_calls[0]
        function_data = first_tool.get("function") if isinstance(first_tool, dict) else None
        if isinstance(function_data, dict):
            arguments = parse_arguments_object(function_data.get("arguments"))
            if isinstance(arguments, dict):
                data = {**arguments, "action": data.get("action") or function_data.get("name") or arguments.get("action")}
            else:
                data = {**data, "action": data.get("action") or function_data.get("name")}

    function_call = data.get("function_call")
    if isinstance(function_call, dict):
        arguments = parse_arguments_object(function_call.get("arguments"))
        if isinstance(arguments, dict):
            data = {**arguments, "action": data.get("action") or function_call.get("name") or arguments.get("action")}
        else:
            data = {**data, "action": data.get("action") or function_call.get("name")}

    arguments = data.get("arguments")
    if isinstance(arguments, dict):
        data = {**arguments, "action": data.get("action") or data.get("name") or arguments.get("action")}

    return PlateEditCommand(
        action=normalize_edit_action(data.get("action") or data.get("name") or data.get("function")),
        position=parse_positive_int(data.get("position") or data.get("index") or data.get("target_position") or data.get("anchor_position")),
        value=normalize_edit_value(
            data.get("value")
            or data.get("new_value")
            or data.get("new_char")
            or data.get("char")
            or data.get("insert_value")
            or data.get("replacement")
        ),
        old_value=normalize_edit_value(
            data.get("old_value")
            or data.get("old_char")
            or data.get("target_value")
            or data.get("target_char")
            or data.get("source_value")
            or data.get("source_char")
            or data.get("from")
        ),
        relation=normalize_relation(data.get("relation") or data.get("where")),
        occurrence=normalize_occurrence(data.get("occurrence") or data.get("which")),
        raw=data,
    )


def unknown_edit_command() -> PlateEditCommand:
    """创建 action="unknown" 的默认占位编辑指令。"""
    return PlateEditCommand(action="unknown", raw={})


def apply_plate_edit_command(current_plate: str, command: PlateEditCommand) -> PlateEditResult:
    """对当前车牌执行单条编辑指令（replace_position/replace_char/insert_position/delete_position），纯规则无模型。"""
    plate = normalize_plate_text(current_plate)
    action = command.action
    if action == "none":
        return PlateEditResult(
            car_plate=plate,
            changed=False,
            command=command,
            error=build_keep_current_plate_reply(plate),
        )
    if action == "unknown":
        return PlateEditResult(car_plate=plate, changed=False, command=command, error=EDIT_UNCLEAR_REPLY)

    chars = list(plate)
    if action == "replace_position":
        if not valid_existing_position(command.position, chars) or not command.value:
            return edit_error(plate, command, EDIT_UNCLEAR_REPLY)
        chars[command.position - 1] = command.value
        return edit_success(chars, command, changed_positions=[command.position])

    if action == "replace_char":
        return apply_replace_char(plate, chars, command)

    if action == "insert_position":
        if not command.value or command.position <= 0:
            return edit_error(plate, command, EDIT_UNCLEAR_REPLY)
        insert_index = command.position if command.relation == "after" else command.position - 1
        if insert_index < 0 or insert_index > len(chars):
            return edit_error(plate, command, EDIT_UNCLEAR_REPLY)
        chars.insert(insert_index, command.value)
        return edit_success(chars, command, changed_positions=[insert_index + 1])

    if action == "delete_position":
        if not valid_existing_position(command.position, chars):
            return edit_error(plate, command, EDIT_UNCLEAR_REPLY)
        del chars[command.position - 1]
        return edit_success(chars, command, changed_positions=[])

    return edit_error(plate, command, EDIT_UNCLEAR_REPLY)


def apply_plate_edit_commands(current_plate: str, commands: list[PlateEditCommand]) -> PlateEditResult:
    """顺序执行多条编辑指令，任一步失败则立即终止并返回当前结果，纯规则无模型。"""
    plate = normalize_plate_text(current_plate)
    normalized_commands = commands or [unknown_edit_command()]
    if len(normalized_commands) == 1:
        return apply_plate_edit_command(plate, normalized_commands[0])

    changed = False
    changed_positions: list[int] = []
    command_steps: list[dict[str, Any]] = []
    primary_command: PlateEditCommand | None = None

    for index, command in enumerate(normalized_commands, start=1):
        if command.action == "none":
            command_steps.append(
                {
                    "batch_index": index,
                    "input_plate": plate,
                    "command": command.to_dict(),
                    "output_plate": plate,
                    "changed": False,
                    "error": "",
                }
            )
            continue

        result = apply_plate_edit_command(plate, command)
        if primary_command is None:
            primary_command = command
        command_steps.append(
            {
                "batch_index": index,
                "input_plate": plate,
                "command": command.to_dict(),
                "output_plate": result.car_plate,
                "changed": result.changed,
                "changed_positions": result.changed_positions,
                "error": result.error,
            }
        )
        if not result.changed:
            return PlateEditResult(
                car_plate=plate,
                changed=changed,
                command=command,
                changed_positions=unique_ints(changed_positions),
                steps=command_steps,
                error=result.error or EDIT_UNCLEAR_REPLY,
                raw=result.raw,
            )
        plate = result.car_plate
        changed = True
        changed_positions.extend(result.changed_positions)

    if not changed:
        command = normalized_commands[0]
        return PlateEditResult(
            car_plate=plate,
            changed=False,
            command=command,
            changed_positions=[],
            steps=command_steps,
            error=build_keep_current_plate_reply(plate),
        )
    return PlateEditResult(
        car_plate=plate,
        changed=True,
        command=primary_command or normalized_commands[0],
        changed_positions=unique_ints(changed_positions),
        steps=command_steps,
    )


def apply_replace_char(plate: str, chars: list[str], command: PlateEditCommand) -> PlateEditResult:
    """按旧字符值查找并替换为新字符，支持 first/last/all 三种出现位置模式。"""
    if not command.old_value or not command.value:
        return edit_error(plate, command, EDIT_UNCLEAR_REPLY)
    indexes = [index for index, char in enumerate(chars) if char == command.old_value]
    if not indexes:
        return edit_error(plate, command, build_char_not_found_reply(command.old_value))
    if command.occurrence == "all":
        changed_positions = [index + 1 for index in indexes]
        for index in indexes:
            chars[index] = command.value
        return edit_success(chars, command, changed_positions=changed_positions)
    if len(indexes) > 1 and command.occurrence not in {"first", "last"}:
        return edit_error(plate, command, build_duplicate_char_reply(command.old_value))
    index = indexes[0] if command.occurrence != "last" else indexes[-1]
    chars[index] = command.value
    return edit_success(chars, command, changed_positions=[index + 1])


def edit_success(chars: list[str], command: PlateEditCommand, *, changed_positions: list[int]) -> PlateEditResult:
    """构造编辑成功的 PlateEditResult 结果对象。"""
    return PlateEditResult(
        car_plate=normalize_plate_text("".join(chars)),
        changed=True,
        command=command,
        changed_positions=changed_positions,
    )


def edit_error(plate: str, command: PlateEditCommand, error: str) -> PlateEditResult:
    """构造编辑失败的 PlateEditResult 结果对象。"""
    return PlateEditResult(
        car_plate=normalize_plate_text(plate),
        changed=False,
        command=command,
        error=error,
    )


def parse_json_object(text: Any) -> dict[str, Any]:
    """从文本解析 JSON，返回 dict；失败返回空 dict。"""
    value = parse_json_value(text)
    return value if isinstance(value, dict) else {}


def parse_json_value(text: Any) -> Any:
    """从文本中提取并解析 JSON 值（借助 extract_json_candidate 提取候选片段）。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(extract_json_candidate(raw))
    except json.JSONDecodeError:
        return None
    return value


def parse_arguments_object(value: Any) -> dict[str, Any] | None:
    """将字符串或 dict 解析为参数 dict；非法输入返回 None。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(extract_json_candidate(value, prefer_object=True))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def normalize_edit_action(value: Any) -> str:
    """将 action 字符串标准化为已知动作名；不在合法集合中则返回 "unknown"。"""
    raw = str(value or "").strip().lower()
    return raw if raw in INTERNAL_EDIT_ACTIONS else "unknown"


def normalize_edit_value(value: Any) -> str:
    """将口语字符转换后标准化为单个大写车牌字符；多字符或空值返回空串。"""
    text = normalize_spoken_plate_chars(value)
    if not text:
        return ""
    text = normalize_plate_format(text)
    return text[0] if len(text) == 1 else ""


def normalize_spoken_plate_chars(value: Any) -> str:
    """将口语发音替换为对应车牌字符（如"幺"→"1"、"拐"→"7"、"圈"→"Q"）。"""
    text = clean_plate_text(value)
    for source, target in SPOKEN_PLATE_CHAR_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def normalize_relation(value: Any) -> str:
    """标准化插入关系字符串为 "before" 或 "after"，默认 "before"。"""
    raw = str(value or "").strip().lower()
    return raw if raw in {"before", "after"} else "before"


def normalize_occurrence(value: Any) -> str:
    """标准化出现位置字符串为 "first"/"last"/"all"；不合法则返回空串。"""
    raw = str(value or "").strip().lower()
    return raw if raw in {"first", "last", "all"} else ""


def parse_positive_int(value: Any) -> int:
    """解析正整数，支持阿拉伯数字和中文数字；非正整数返回 0。"""
    raw = str(value or "").strip()
    try:
        number = int(raw)
    except (TypeError, ValueError):
        number = parse_chinese_position(raw)
    return number if number > 0 else 0


def parse_chinese_position(value: str) -> int:
    """将中文位置文本（如"第三位"、"十二"）转换为对应整数。"""
    raw = re.sub(r"\s+", "", str(value or ""))
    raw = raw.replace("第", "").replace("位", "").replace("个", "")
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if raw in digits:
        return digits[raw]
    if raw == "十":
        return 10
    if raw.startswith("十") and raw[1:] in digits:
        return 10 + digits[raw[1:]]
    if "十" in raw:
        left, right = raw.split("十", 1)
        if left in digits:
            return digits[left] * 10 + (digits.get(right, 0) if right else 0)
    return 0


def valid_existing_position(position: int, chars: list[str]) -> bool:
    """检查 1-based 位置是否在字符列表的有效范围 [1, len(chars)] 内。"""
    return 1 <= position <= len(chars)


def unique_ints(values: list[int]) -> list[int]:
    """去重保序，过滤掉非正整数，返回有序不重复的正整数列表。"""
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def clean_plate_text(value: Any) -> str:
    """去除空白字符（edit 模块内独立定义，逻辑同 plate_agent_rules）。"""
    return re.sub(r"\s+", "", str(value or "")).strip()


def normalize_plate_format(value: Any) -> str:
    """去空白并转大写（edit 模块内独立定义）。"""
    return clean_plate_text(value).upper()


def normalize_plate_text(value: Any) -> str:
    """标准化车牌，含 G→冀（edit 模块内独立定义）。"""
    plate = normalize_plate_format(value)
    if plate.startswith("G"):
        return "冀" + plate[1:]
    return plate

