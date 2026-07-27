from __future__ import annotations

import json
import re
from typing import Any

from realtime_audio_demo.services.plate_agent_types import PlateAgentState


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
任务：根据用户最新音频，把用户的修改意图转换成一个车牌编辑函数调用。

判断步骤：
1. 用户说了什么：简短复述用户这轮语音里的确认或修改意图。
2. 应该怎么改：说明要操作当前车牌的哪个位置或哪个字符；如果不需要修改，说明是确认。
3. 如何输入命令：选择唯一 action，并确定 position、value、old_value、relation、occurrence 中需要的字段。

重要原则：
1. 只判断用户这句话应该调用哪一个编辑函数，并输出这个函数需要的参数。
2. 只能选择一个 action；如果已经有 executed_edit_steps，本次只输出尚未处理的下一个单一动作。
3. action 必须只使用英文枚举：replace_position、replace_char、insert_position、delete_position、none。
4. relation 必须只使用 before 或 after；occurrence 必须只使用 first、last、all 或空字符串。
5. position 使用 1 开始计数，必须对应 current_car_plate / plate_chars 里的当前位置。
6. value 和 old_value 只能是单个车牌字符：省份简称、英文字母、数字、警、临、学、领、挂。
7. 需要处理报号发音：洞=0，幺=1，是=4，陆=6，拐=7，吸=C，勾/沟儿=J，圈=Q。
8. 用户只是确认某个待确认字符正确，例如“天津的津是对的”“那个 R 没错”，输出 none，不要输出替换动作。
9. 如果用户修改意图已经被 executed_edit_steps 覆盖完，输出 none。
10. 必须先按“用户说了什么 / 应该怎么改 / 如何输入命令”输出简短判断，再输出一个完整 JSON 对象。
11. 后端会从输出尾部提取最后一个完整 JSON 对象作为唯一解析依据；JSON 对象必须只包含一个 action 及其参数。

可调用编辑函数：

def replace_position(position: int, value: str)
说明：用户明确说要把某个位置改成某个字符时使用。
适用表达：第 3 位改成 R、倒数第 1 位改成临、前面那个改成 E。
必须条件：能确定 position，并且能确定新的单字符 value。
输出：{{"action":"replace_position","position":3,"value":"R"}}

def replace_char(old_value: str, value: str, occurrence: str = "")
说明：用户明确说把当前车牌里的某个具体字符替换成另一个字符时使用。
适用表达：把临改成 0、不是 R 是 2、那个 E 改成 1。
必须条件：能确定 old_value 和 value；如果 old_value 在当前车牌里出现多次，必须能确定 occurrence。
occurrence 可选值：first 表示前面的，last 表示后面的，all 表示全部；只出现一次可以留空。
输出：{{"action":"replace_char","old_value":"临","value":"0"}}

def insert_position(position: int, value: str, relation: str)
说明：用户明确说要在某个位置前面或后面加一个字符时使用。
适用表达：第 3 位后面加 5、R 前面加 2、最后一位前面插个 A。
必须条件：能确定锚点 position、插入字符 value、插入关系 relation。
relation 只能是 before 或 after；如果用户说“插到第 3 位”，按 before 处理。
输出：{{"action":"insert_position","position":3,"value":"5","relation":"after"}}

def delete_position(position: int)
说明：用户明确说删除当前车牌某个位置的字符时使用。
适用表达：删掉最后一位、不要第 4 位、把前面那个多余的字符去掉。
必须条件：能确定要删除的 position。
输出：{{"action":"delete_position","position":{len(plate)}}}

def none()
说明：用户是在确认当前车牌或确认某个待确认字符，没有要求修改车牌时使用。
适用表达：对、没错、天津的津没错、那个 R 是对的。
输出：{{"action":"none"}}

输出格式：
用户说了什么：用户是在纠正第 3 位。
应该怎么改：把当前车牌第 3 位替换为字母 R。
如何输入命令：使用 replace_position，position=3，value=R。
{{"action":"replace_position","position":3,"value":"R"}}

读取下面的当前车牌状态后，结合用户音频完成判断。当前车牌状态是本轮唯一动态上下文：
{context_json}
""".strip()


def build_plate_update_review_prompt(context: dict[str, Any]) -> str:
    return f"""
任务：对刚才的车牌编辑步骤做一次 ReAct 复核，但不要直接改车牌。

你需要根据同一段用户语音和当前上下文判断：
1. 用户这轮到底是在确认哪些位、修改哪些位，还是还有未处理的修改意图。
2. 已执行编辑步骤是否已经覆盖用户意图。
3. after_plate 是否是一个合理的中间或最终结果。
4. 用户本轮明确确认过的车牌位置有哪些，需要加入 confirmed_positions。

当前上下文：
{json.dumps(context, ensure_ascii=False)}

判断规则：
1. 如果用户明确说某个待确认字符是对的，例如“天津的津对”“那个 R 没错”，把对应 position 放进 confirmed_positions。
2. 如果用户明确把某一位改成新字符，且本轮编辑正确执行了这个修改，也把该 position 放进 confirmed_positions。
3. 如果修改后的字符已经不是易混淆字符，后端会自动从 need_confirm_chars 删除；你只需要把用户明确确认的位置输出出来。
4. 如果用户还表达了另一个未处理的修改动作，needs_more_edit=true。
5. 如果当前编辑动作和用户语音明显不一致，valid_result=false。
6. 如果编辑已经覆盖用户意图，needs_more_edit=false。
7. 不要输出推理过程，不要输出自然语言，只输出 JSON。

输出 JSON 字段：
{{
  "confirmed_positions": [1, 4],
  "needs_more_edit": false,
  "valid_result": true,
  "reason": "简短说明"
}}
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
