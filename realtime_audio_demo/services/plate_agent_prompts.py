from __future__ import annotations

import json
import re
from typing import Any

from realtime_audio_demo.services.plate_agent_rules import describe_plate_char
from realtime_audio_demo.services.plate_agent_types import PlateAgentState


def build_plate_presence_prompt() -> str:
    return (
        "任务：判断用户语音中是否包含车牌号相关内容。"
        "只回答 true 或 false，不要输出其它内容。"
        "如果用户说了省份、字母、数字、车牌片段或完整车牌，回答 true。"
    )


def build_plate_edit_command_prompt(
    state: PlateAgentState,
    *,
    current_plate: str | None = None,
    edit_steps: list[dict[str, Any]] | None = None,
) -> str:
    plate = clean_prompt_plate(current_plate if current_plate is not None else state.car_plate)
    state_context = state.to_context()
    context = {
        "current_car_plate": plate,
        "plate_length": len(plate),
        "confirmed": state.is_confirmed,
        "plate_chars": state_context.get("plate_chars") or plate_chars_context(plate),
        "vehicle_type": state.vehicle_type,
        "need_confirm_chars": state_context.get("need_confirm_chars", []),
        "confirmed_chars": state_context.get("confirmed_chars", []),
        "previous_assistant_reply": state.assistant_reply,
        "executed_edit_steps": edit_steps or [],
        "recent_turn_summaries": state_context.get("turn_summaries", []),
    }
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    context_text = build_plate_edit_context_text(context)
    return f"""
任务：听用户最新音频，判断用户是在确认当前车牌，还是在纠正当前车牌，并输出可执行的车牌编辑命令。

请先做简洁、可验证的分析，再给出最终命令。不要输出冗长推理，不要编造用户没有说过的信息。

可用编辑动作：
- replace_position：按位置替换一位。例：第 3 位改成 R。JSON：{{"action":"replace_position","position":3,"value":"R"}}
- replace_char：按已有字符替换。例：把临改成 0。JSON：{{"action":"replace_char","old_value":"临","value":"0"}}
- insert_position：在某一位前面或后面加字符。例：第 3 位后面加 5。JSON：{{"action":"insert_position","position":3,"value":"5","relation":"after"}}
- delete_position：删除某一位。例：删掉最后一位。JSON：{{"action":"delete_position","position":{len(plate)}}}
- none：用户只是确认，或者听不出明确修改。JSON：{{"action":"none"}}

如果用户一次说了多个明确修改，最后输出 actions 列表，例如：
{{"actions":[{{"action":"replace_position","position":3,"value":"R"}},{{"action":"delete_position","position":{len(plate)}}}]}}

常见读法：洞/零/〇=0，幺/么=1，二/两=2，是=4，陆=6，拐=7，吸=C，勾/沟儿=J，圈=Q。
位置从 1 开始数。

输出要求：
【问题理解】
用一句话说明你听到的用户意图：确认、替换、插入、删除，或不明确。

【关键分析】
简洁说明你如何根据当前车牌、历史摘要、待确认字符和已确认字符定位到要处理的位置或字符。
如果用户说的是读音，说明转换后的标准字符。
如果用户没有给出明确修改，说明为什么不能执行修改。

【最终答案】
先用一句话说明最终选择的动作。
然后单独输出一个完整 JSON 对象；后端只读取最后这个 JSON。

【置信度】
输出高、中或低，并简要说明原因。

当前车牌状态：
{context_text}

结构化状态：
{context_json}
""".strip()


def build_confirmation_state_action_prompt(context: dict[str, Any]) -> str:
    context_text = build_confirmation_state_context_text(context)
    return f"""
任务：根据用户最新音频和当前车牌状态，更新二次确认列表和已确认字符。

你只处理确认状态，不修改车牌号本身。可以先简短判断用户这句话确认了什么、还有什么需要继续确认；最后单独输出一个 JSON，后端只读取最后这个 JSON。

可用动作：
- clear_need_confirmation：重新计算当前车牌的二次确认列表。
- add_need_confirmation：某一位还需要继续问用户确认。例：{{"action":"add_need_confirmation","position":3}}
- remove_need_confirmation：某一位不需要再二次确认。
- add_confirmed：用户已经明确确认某一位是对的，或者本轮成功修改到了这一位。
- remove_confirmed：用户否定了之前确认过的某一位。
- none：确认状态没有明确变化。

多个动作输出 actions 列表，例如：
{{"actions":[{{"action":"clear_need_confirmation"}},{{"action":"add_confirmed","position":1}},{{"action":"add_need_confirmation","position":3}}]}}

当前确认状态：
{context_text}

结构化确认状态：
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


def build_confirmation_state_context_text(context: dict[str, Any]) -> str:
    lines = [
        f"- 当前暂存车牌：{context.get('car_plate') or '空'}",
        f"- 当前车牌长度：{context.get('plate_length') or 0} 位",
        f"- 车辆类型：{vehicle_type_text(context.get('vehicle_type'))}",
    ]
    need_confirm = describe_state_chars(context.get("need_confirm_chars"))
    lines.append(f"- 当前还需要二次确认：{need_confirm or '无'}")
    confirmed = describe_state_chars(context.get("confirmed_chars"))
    lines.append(f"- 已经确认的字符：{confirmed or '无'}")
    rule_confusions = describe_state_chars(context.get("rule_confusions"))
    lines.append(f"- 按规则当前可能需要确认的位置：{rule_confusions or '无'}")
    review_positions = context.get("confirmed_positions_from_review") or []
    if review_positions:
        lines.append(f"- 上一步复核认为用户本轮已经确认的位置：{review_positions}")
    assistant_reply = str(context.get("assistant_reply") or "").strip()
    if assistant_reply:
        lines.append(f"- 上一轮 AI 回复：{assistant_reply}")
    summaries = [str(item).strip() for item in context.get("recent_turn_summaries") or [] if str(item).strip()]
    if summaries:
        lines.append("- 最近历史摘要：")
        lines.extend(f"  {index}. {summary}" for index, summary in enumerate(summaries[-6:], start=1))
    return "\n".join(lines)


def build_plate_edit_context_text(context: dict[str, Any]) -> str:
    plate = str(context.get("current_car_plate") or "").strip()
    lines = [
        f"- 当前暂存车牌：{plate or '空'}",
        f"- 当前车牌长度：{context.get('plate_length') or 0} 位",
        f"- 当前是否已经整车牌确认：{'是' if context.get('confirmed') else '否'}",
        f"- 车辆类型：{vehicle_type_text(context.get('vehicle_type'))}",
    ]

    plate_chars = describe_plate_chars(context.get("plate_chars"))
    if plate_chars:
        lines.append(f"- 当前车牌逐位内容：{plate_chars}")

    need_confirm = describe_state_chars(context.get("need_confirm_chars"))
    lines.append(f"- 当前还需要用户二次确认：{need_confirm or '无'}")

    confirmed = describe_state_chars(context.get("confirmed_chars"))
    lines.append(f"- 用户已经明确确认过的字符：{confirmed or '无'}")

    previous_reply = str(context.get("previous_assistant_reply") or "").strip()
    if previous_reply:
        lines.append(f"- 上一轮 AI 回复给用户的话：{previous_reply}")

    summaries = [str(item).strip() for item in context.get("recent_turn_summaries") or [] if str(item).strip()]
    if summaries:
        lines.append("- 最近历史摘要：")
        lines.extend(f"  {index}. {summary}" for index, summary in enumerate(summaries[-6:], start=1))

    edit_steps = context.get("executed_edit_steps")
    if isinstance(edit_steps, list) and edit_steps:
        lines.append(f"- 本轮已经执行过 {len(edit_steps)} 次编辑尝试，后续判断不要重复已经完成的动作。")

    return "\n".join(lines)


def describe_plate_chars(value: Any) -> str:
    items = value if isinstance(value, list) else []
    descriptions: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        position = item.get("position")
        char = str(item.get("value") or item.get("char") or "").strip()
        if not position or not char:
            continue
        descriptions.append(f"第{position}位是{describe_plate_char(char)}")
    return "，".join(descriptions)


def describe_state_chars(value: Any) -> str:
    items = value if isinstance(value, list) else []
    descriptions: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        position = item.get("position")
        char = str(item.get("value") or item.get("char") or "").strip()
        if not position or not char:
            continue
        reason = str(item.get("reason") or "").strip().rstrip("。")
        text = f"第{position}位={describe_plate_char(char)}"
        if reason:
            text += f"（{reason}）"
        descriptions.append(text)
    return "，".join(descriptions)


def vehicle_type_text(value: Any) -> str:
    raw = str(value or "").strip()
    if raw == "fuel":
        return "普通燃油车号牌"
    if raw == "new_energy":
        return "新能源车号牌"
    return raw or "未知"


def build_confirmation_detection_prompt(previous_ai_reply: str) -> str:
    return build_confirmation_detection_prompt_with_history(previous_ai_reply, [])


def build_confirmation_detection_prompt_with_history(previous_ai_reply: str, turn_summaries: list[str]) -> str:
    history_text = "\n".join(f"- {item}" for item in turn_summaries[-6:] if str(item).strip())
    history_block = f"\n\n## 历史摘要\n{history_text}" if history_text else ""
    return (
        "任务：判断用户是否在确认上一轮 AI 所说的车牌信息。\n\n"
        "## 判断逻辑\n"
        "分析用户语音是否明确确认了 AI 刚才说的车牌号。"
        "用户可能说的确认话术包括：对、是的、没错、正确、就是这个、确认、嗯对、就是这样、可以了。\n"
        "用户可能说的否认话术包括：不对、修改、不是、某一位错了、听错了、不是这个、我重新说。\n\n"
        "## 输出要求\n"
        "只回答 yes 或 no，不要输出其它内容。\n"
        "yes = 用户确认了上一轮 AI 说出的车牌号\n"
        "no = 用户否认、纠正或要求修改\n\n"
        "## 上一轮 AI 对用户说的话\n"
        f"{previous_ai_reply}"
        f"{history_block}"
    )


def build_plate_update_review_prompt(context: dict[str, Any]) -> str:
    return f"""
任务：对刚才的车牌编辑步骤做一次 ReAct 复核，但不要直接改车牌。

你需要根据同一段用户语音和当前上下文判断：
1. 用户这轮到底是在确认哪些位、修改哪些位，还是还有未处理的修改意图。
2. 已执行的 command / commands 是否已经覆盖用户意图。
3. after_plate 是否是一个合理的中间或最终结果。
4. 用户本轮明确确认过的车牌位置有哪些，需要加入 confirmed_positions。

当前上下文：
{json.dumps(context, ensure_ascii=False)}

判断规则：
1. 如果用户明确说某个待确认字符是对的，例如“天津的津对”“那个 R 没错”，把对应 position 放进 confirmed_positions。
2. 如果用户明确把某一位改成新字符，且本轮 command / commands 正确执行了这个修改，也把该 position 放进 confirmed_positions。
3. 如果修改后的字符已经不是易混淆字符，后端会自动从 need_confirm_chars 删除；你只需要把用户明确确认的位置输出出来。
4. 如果用户还表达了另一个未处理的修改动作，needs_more_edit=true。
5. 如果当前编辑动作和用户语音明显不一致，valid_result=false。
6. 如果本轮一个或多个编辑动作已经覆盖用户意图，needs_more_edit=false。
7. 不要输出推理过程，不要输出自然语言，只输出 JSON。

输出 JSON 字段：
{{
  "confirmed_positions": [1, 4],
  "needs_more_edit": false,
  "valid_result": true,
  "reason": "简短说明"
}}
""".strip()


def build_province_retry_prompt(formatted_plate: str) -> str:
    return f"""
任务：当前暂时的车牌识别结果第一位不是省份简称，请根据用户音频重新识别车牌号。
当前暂时识别结果：{formatted_plate}

请输出带省份简称的完整车牌号码。
车牌第一位必须是省份简称，例如：京、津、冀、晋、蒙、辽、吉、黑、沪、苏、浙、皖、闽、赣、鲁、豫、鄂、湘、粤、桂、琼、渝、川、贵、云、藏、陕、甘、青、宁、新。
只输出 JSON 对象，字段为 car_plate。
""".strip()


def clean_prompt_plate(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().upper()


def plate_chars_context(plate: str) -> list[dict[str, Any]]:
    return [
        {
            "position": index,
            "value": value,
            "confirmed": False,
            "needs_confirmation": False,
            "candidates": [],
            "reason": "",
        }
        for index, value in enumerate(plate, start=1)
    ]
