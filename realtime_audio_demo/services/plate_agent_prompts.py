from __future__ import annotations

import json
import re
from typing import Any

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
    }
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    return f"""
任务：听用户最新音频，判断用户是在确认当前车牌，还是在纠正当前车牌。

请先自然地想清楚用户这句话的意思，可以简短说明你的判断；最后单独输出一个 JSON，后端只读取最后这个 JSON。

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

当前车牌状态：
{context_json}
""".strip()


def build_confirmation_state_action_prompt(context: dict[str, Any]) -> str:
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
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


def build_confirmation_detection_prompt(previous_ai_reply: str) -> str:
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


def build_reply_generation_prompt() -> str:
    return (
        "任务：根据当前暂时识别的车牌、易混淆字符列表、车辆类型，生成口语化客服回复。"
        "回复要求：1. 说出当前识别到的车牌；"
        "2. need_confirm_chars 是当前还没有确认、必须继续向用户确认的权威列表，不能忽略、合并或省略；"
        "3. 如果 need_confirm_chars 不为空，必须逐项按 reason 的描述向用户确认当前识别结果；已经在 confirmed_chars 里的字符不要再确认；"
        "4. 必须把具体位置说给用户，可以说“第1位”“第2位”“第几位”；"
        "5. 如果同一类易混淆字符出现多次，要逐项说清楚对应第几位；"
        "6. 不要编造候选值，不要说“是 A 还是 B”，只说当前识别为 reason 里的内容并请用户确认是否正确；"
        "7. 不要向用户解释易混淆规则，只确认当前识别到的具体字符；"
        "8. 如果有多个易混淆字符，要全部说出来，不能只确认其中一个；"
        "9. 如果是 8 位新能源号牌，要询问用户是不是新能源电车；"
        "10. 简短自然，不要解释系统逻辑。"
        "只输出 JSON 对象，字段为 car_plate 和 assistant_reply。"
    )


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
