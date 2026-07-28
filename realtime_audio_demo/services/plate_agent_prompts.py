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
任务：根据用户最新音频，把用户的纠正意图转换成一个车牌编辑函数调用。

当前任务不是重新识别一个新车牌，而是对当前已经暂存的车牌做最小必要编辑。
你需要把用户语音里的纠正、插入、删除、确认意图，转换成后端可以执行的编辑命令。
如果用户一句话里包含多个明确修改，可以输出 actions 列表，后端会按列表顺序执行。

判断步骤：
1. 用户说了什么：简短复述用户这轮语音里的确认或纠正意图。
2. 当前车牌怎么定位：根据 current_car_plate / plate_chars 找到用户指向的是哪个位置或哪个已有字符。
3. 字符怎么标准化：把用户说的报号发音转换成车牌字符，例如洞=0、幺=1、拐=7、圈=Q、勾=J。
4. 应该怎么改：说明是替换某一位、替换某个字符、插入、删除，还是只是在确认。
5. 修改结果预演：在心里把编辑动作应用到 current_car_plate 上，确认动作能得到用户想要的变化。
6. 如何输入命令：选择单个 action 或 actions 列表，并确定 position、value、old_value、relation、occurrence 中需要的字段。

编辑前检查闭环：
1. 先逐位读取 current_car_plate，不要跳过当前已有字符。
2. 再从用户语音里找纠正关键词：不是、改成、换成、前面、后面、第几位、最后一位、加、删、不要、对、没错。
3. 如果用户说的是“这个、那个、前面的、后面的”，必须结合 need_confirm_chars、previous_assistant_reply 和 current_car_plate 判断指向。
4. 如果用户说的是读音，必须先转换成标准车牌字符，再决定 value / old_value。
5. 如果能确定修改动作，先预演一次修改后的车牌；如果预演结果明显不符合用户语义，就不要输出这个动作。
6. 如果一句话里包含多个修改，且每个修改都明确可执行，可以按用户语义顺序输出 actions 列表。
7. 如果一句话里包含多个修改但只有一部分明确，只输出明确的动作；不明确的动作不要猜。
8. 如果没有办法形成确定、可执行的编辑命令，输出 none，不要猜。

重要原则：
1. current_car_plate 是唯一可编辑对象；必须先把用户语音和当前车牌对齐，不要凭空生成新的完整车牌。
2. 优先选择最小编辑；一个修改用单个 action，多个明确修改用 actions 列表；如果 executed_edit_steps 已经覆盖本轮意图，本次输出 none。
3. action 只能使用 replace_position、replace_char、insert_position、delete_position、none；relation 只能是 before/after；occurrence 只能是 first/last/all 或空字符串。
4. position 使用 1 开始计数，必须对应 current_car_plate / plate_chars 里的当前位置；最后一位、倒数第一位等说法要换算成正向位置。
5. 根据用户表达选择函数：第几位改成用 replace_position；不是 X 是 Y、X 改成 Y 用 replace_char；加或插入用 insert_position；删掉或不要用 delete_position。
6. 用户只是在确认当前车牌或待确认字符正确时输出 none；如果无法形成确定、可执行的编辑命令，也输出 none，不要猜。
7. replace_char 的 old_value 如果在当前车牌里出现多次，必须根据用户说的前面、后面、全部设置 occurrence；无法确定时输出 none。
8. value 和 old_value 只能是单个车牌字符：省份简称、英文字母、数字、警、临、学、领、挂。
9. 必须处理报号发音：零/洞=0，幺/么=1，二/两=2，是=4，陆=6，拐=7，吸=C，勾/沟儿=J，圈=Q。
10. actions 列表最多输出 3 个动作；列表中不要混入 none；每个动作都必须能独立执行。
11. actions 列表会按顺序执行；后续动作的 position 必须按前一个动作执行后的车牌重新计算。

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
必须先按“用户说了什么 / 当前车牌怎么定位 / 字符怎么标准化 / 应该怎么改 / 修改结果预演 / 如何输入命令”输出简短判断。
最后必须输出一个完整 JSON 对象；后端会从输出尾部提取最后一个完整 JSON 对象作为唯一解析依据。
单个修改可以输出单个 action 对象；多个明确修改必须输出 actions 列表对象。

用户说了什么：用户是在纠正第 3 位。
当前车牌怎么定位：当前车牌第 3 位是 2。
字符怎么标准化：用户说的 R 是字母 R。
应该怎么改：把当前车牌第 3 位替换为字母 R。
修改结果预演：执行后只有第 3 位从 2 变成 R，其它字符不变。
如何输入命令：使用 replace_position，position=3，value=R。
{{"action":"replace_position","position":3,"value":"R"}}

多个修改输出示例：
用户说了什么：用户要求把第 3 位改成 R，并在最后一位后面加临。
当前车牌怎么定位：当前车牌第 3 位是 2，最后一位位置是 {len(plate)}。
字符怎么标准化：R 是字母 R，临是特殊尾字临。
应该怎么改：先替换第 3 位，再在当前最后一位后面插入临。
修改结果预演：两个动作按顺序执行，前一个动作改变第 3 位，后一个动作在末尾追加临。
如何输入命令：使用 actions 列表，按 replace_position、insert_position 的顺序执行。
{{"actions":[{{"action":"replace_position","position":3,"value":"R"}},{{"action":"insert_position","position":{len(plate)},"value":"临","relation":"after"}}]}}

读取下面的当前车牌状态后，结合用户音频完成判断。当前车牌状态是本轮唯一动态上下文：
{context_json}
""".strip()


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
