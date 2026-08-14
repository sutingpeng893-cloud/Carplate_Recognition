from __future__ import annotations

from typing import Any

from realtime_audio_demo.config import PLATE_REPLY_INCLUDE_CONFIRMATION
from realtime_audio_demo.services.plate_agent_rules import clean_plate_text, describe_plate_char, with_relative_confusion_reasons
from realtime_audio_demo.services.plate_agent_types import PlateAgentState, PlateEditCommand


# 产品话术集中维护文件。
#
# 给产品或运营改文案时看这里：
# 1. 可以直接修改等号右侧的中文字符串。
# 2. 不要修改变量名、函数名、字典 key，例如 SESSION_OPENING_TEXT、ACK_MESSAGES_BY_SCENE、initial、update。
# 3. 带花括号的内容是后端占位符，必须保留：
#    {plate} = 当前车牌号，{char} = 某个字符的口语说明，{pending} = 需要用户确认的位置列表。
# 4. ACK_MESSAGES_BY_SCENE 每组 1 句话，音频收到后立即发送一次过渡话术。

# 新会话开始时返回给用户的开场白。
# 触发位置：/api/chatbox/audio/session/start 接口。
# 用户看到/听到：页面刚开始时 AI 主动提示用户开始说车牌。
SESSION_OPENING_TEXT = "您好，请告诉我您的车牌号。"

# 音频识别还没有出最终结果时，按时间点返回给前端展示的衔接语。
# initial：首轮还没有暂存车牌。
# update：多轮已有暂存车牌，用户可能是在确认或纠错。
ACK_MESSAGES_BY_SCENE = {
    "initial": (
        "您稍等，正在为您查询车牌。",
    ),
    "update": (
        "您稍等，正在为您更新车牌。",
    ),
}

# 首轮没有听到任何车牌相关内容时返回。
# 典型场景：用户没说车牌、音频为空、环境噪声导致模型判断没有车牌内容。
NO_PLATE_REPLY = "我没有听到车牌号内容，请告诉我车牌号。"

# 首轮提取到了内容，但后端校验发现不是合法车牌时返回。
# 典型场景：模型提取结果位数、首位省份、字符规则不满足车牌规则。
INVALID_PLATE_REPLY = "您好，您当前的车牌号并不是有效号码，请重新输入。"

# 多轮纠错时，模型没有形成可执行编辑动作时返回。
# 典型场景：用户只说“不对”、没有说明改哪一位，或者语音太模糊。
EDIT_UNCLEAR_REPLY = "我没有听清您要修改车牌的哪一处，当前仍保留原来的车牌{plate}。请您说明要替换、插入或删除哪一位。"

# 多轮纠错执行后，新车牌不符合格式时返回。
# {plate} 必须保留，后端会替换成当前保留的旧车牌。
# 典型场景：删除/插入/替换后车牌位数不合法，系统不会覆盖旧车牌。
EDIT_INVALID_REPLY = "按这次修改后车牌格式不符合规则，当前仍保留车牌{plate}。请您重新说明要改哪一处。"

# 多轮纠错包含多个动作时，部分动作已经处理，但 ReAct 复核认为还没完全覆盖用户意图时返回。
# 典型场景：用户一句话说了多个修改，系统只确认处理到了其中一部分。
EDIT_MULTI_STEP_PARTIAL_REPLY = "这次修改包含多步内容，我先处理到目前能确定的位置，请您继续确认或说明剩余要改的部分。"

# 用户本轮没有实际修改车牌时返回。
# {plate} 必须保留，后端会替换成当前暂存车牌。
# 典型场景：模型 action=none，或者用户只是表达确认但还没整车牌确认完成。
EDIT_KEEP_CURRENT_PLATE_TEMPLATE = "当前仍保留原来的车牌{plate}，请您确认是否正确。"

# 用户要求“把某个字符改成另一个字符”，但当前车牌里没有这个字符时返回。
# {char} 必须保留，后端会替换成“数字 2 / 字母 R / 天津的津”这类口语说明。
# 典型场景：用户说“把 R 改成 2”，但当前车牌里没有 R。
EDIT_CHAR_NOT_FOUND_TEMPLATE = "当前车牌里没有{char}，所以没有修改。当前仍保留原来的车牌。"

# 用户按字符修改，但当前车牌里同一个字符出现多次，无法确定改哪一个时返回。
# {char} 必须保留，后端会替换成对应字符说明。
# 典型场景：车牌里有两个 1，用户只说“把 1 改成 E”，没有说前一个还是后一个。
EDIT_DUPLICATE_CHAR_TEMPLATE = "当前车牌里有多个{char}，请您说明要改前面的还是后面的。"

# 旧 JSON 输出过滤逻辑里，normalized 字段长度小于 7 位时返回。
# 这是兼容旧链路的兜底话术，plate agent 主流程一般不走这里。
NORMALIZED_SHORT_PLATE_REPLY = "当前您提供的车牌位数不符合要求。请您重新输入车牌。"

# 旧 JSON 输出过滤逻辑里，normalized 字段长度超过规则或格式不合法时返回。
# 这是兼容旧链路的兜底话术，plate agent 主流程一般不走这里。
NORMALIZED_INVALID_PLATE_REPLY = "请重新输入车牌，车牌格式不符合要求。"

# 首轮识别成功后的基础回复。
# {plate} 必须保留，后端会替换成首轮识别出的车牌。
# 后面会自动追加二次确认内容，例如“需要您确认：第1位是天津的津。”
INITIAL_SUCCESS_TEMPLATE = "我识别到的车牌号是{plate}。"

# 多轮纠错成功后的基础回复。
# {plate} 必须保留，后端会替换成修改后的完整车牌。
# 后面会自动追加二次确认内容，继续确认易混淆字符。
UPDATE_SUCCESS_TEMPLATE = "已按您的说明更新为{plate}。"

# 用户只确认了部分待确认字符，但还没有确认完整车牌时返回。
# 后面会自动追加剩余待确认内容；如果没有剩余待确认项，会提示确认整车牌。
PARTIAL_CONFIRMATION_TEMPLATE = "好的，已记录您的确认。"

# 用户明确确认完整车牌后返回。
# {plate} 必须保留，后端会替换成最终车牌；这句话通常是任务结束话术。
CONFIRMED_REPLY_TEMPLATE = "好的，已确认您的车牌号是{plate}。"

# 有待确认字符时追加到基础回复后面。
# {pending} 必须保留，后端会替换成待确认列表，例如“第1位是天津的津，第4位是数字 2”。
PENDING_CONFIRMATION_TEMPLATE = "需要您确认：{pending}。"

# 没有待确认字符，但还没有整车牌确认时追加到基础回复后面。
# {plate} 必须保留，后端会替换成当前暂存车牌。
NO_PENDING_CONFIRMATION_TEMPLATE = "请您确认{plate}是否正确。"

# 当前车牌被判断为 8 位新能源号牌时追加。
# 典型场景：后端 vehicle_type=new_energy，需要用户确认是不是新能源车牌。
NEW_ENERGY_CONFIRMATION_TEXT = "另外这是新能源号牌吗？"

# pending action 话术：大模型识别出修改意图后追问用户是否修改。
# {old_plate}=当前车牌，{action_desc}=修改描述，{new_plate}=预执行后车牌。
PENDING_ACTION_CONFIRM_TEMPLATE = "识别到您要{action_desc}，是否先执行这一步？"

# pending action 话术：用户确认执行后回复。
# {plate}=已写入的新车牌。
PENDING_ACTION_APPLIED_TEMPLATE = "已修改为{plate}，请确认是否正确。"

# pending action 话术：用户拒绝执行后回复。
# {plate}=保留的原车牌。
PENDING_ACTION_DISCARDED_TEMPLATE = "好的，当前保留原车牌{plate}，请继续确认或告知修改。"


def describe_edit_command(cmd: PlateEditCommand) -> str:
    if cmd.action == "replace_position" and cmd.position and cmd.value:
        return f"将第{cmd.position}位替换为{cmd.value}"
    if cmd.action == "replace_char" and cmd.old_value and cmd.value:
        return f"将{cmd.old_value}替换为{cmd.value}"
    if cmd.action == "insert_position" and cmd.position and cmd.value:
        relation = "之后" if cmd.relation == "after" else "之前"
        return f"在第{cmd.position}位{relation}插入{cmd.value}"
    if cmd.action == "delete_position" and cmd.position:
        return f"删除第{cmd.position}位"
    if cmd.action == "replace_substring" and cmd.old_value and cmd.new_value:
        return f"将{cmd.old_value}改为{cmd.new_value}"
    return ""


def describe_edit_commands(commands: list[PlateEditCommand]) -> str:
    parts = [describe_edit_command(cmd) for cmd in commands if cmd.action not in ("none", "unknown")]
    return "，".join(p for p in parts if p)


def build_pending_action_confirm_reply(old_plate: str, commands: list[PlateEditCommand], new_plate: str) -> str:  # noqa: ARG001
    action_desc = describe_edit_commands(commands)
    return PENDING_ACTION_CONFIRM_TEMPLATE.format(action_desc=action_desc)


def build_pending_action_applied_reply(state: PlateAgentState) -> str:
    base = PENDING_ACTION_APPLIED_TEMPLATE.format(plate=display_plate(state))
    return with_pending_confirmation(base, state)


def build_pending_action_discarded_reply(plate: str) -> str:
    return PENDING_ACTION_DISCARDED_TEMPLATE.format(plate=clean_plate_text(plate))


def build_initial_success_reply(state: PlateAgentState) -> str:
    return with_pending_confirmation(INITIAL_SUCCESS_TEMPLATE.format(plate=display_plate(state)), state)


def build_update_success_reply(state: PlateAgentState) -> str:
    return with_pending_confirmation(UPDATE_SUCCESS_TEMPLATE.format(plate=display_plate(state)), state)


def build_partial_confirmation_reply(state: PlateAgentState) -> str:
    return with_pending_confirmation(PARTIAL_CONFIRMATION_TEMPLATE, state)


def build_confirmed_reply(plate: str) -> str:
    return CONFIRMED_REPLY_TEMPLATE.format(plate=clean_plate_text(plate))


def build_edit_invalid_reply(state: PlateAgentState) -> str:
    return with_pending_confirmation(EDIT_INVALID_REPLY.format(plate=display_plate(state)), state)


def build_edit_unclear_reply(state: PlateAgentState, base_reply: str | None = None) -> str:
    return with_pending_confirmation(base_reply or EDIT_UNCLEAR_REPLY, state)


def build_keep_current_plate_reply(plate: str) -> str:
    return EDIT_KEEP_CURRENT_PLATE_TEMPLATE.format(plate=clean_plate_text(plate))


def build_char_not_found_reply(value: str) -> str:
    return EDIT_CHAR_NOT_FOUND_TEMPLATE.format(char=describe_plate_char(value))


def build_duplicate_char_reply(value: str) -> str:
    return EDIT_DUPLICATE_CHAR_TEMPLATE.format(char=describe_plate_char(value))


def build_fixed_reply(state: PlateAgentState, *, changed: bool, scene: str = "") -> str:
    """根据当前业务场景选择固定回复模板。"""
    if scene == "initial_success":
        return build_initial_success_reply(state)
    if scene == "partial_confirmation":
        return build_partial_confirmation_reply(state)
    if scene == "update_success" or changed:
        return build_update_success_reply(state)
    return build_edit_unclear_reply(state)


def with_pending_confirmation(base_reply: str, state: PlateAgentState) -> str:
    """在基础回复后追加二次确认内容。"""
    reply = ensure_sentence(base_reply)
    if not PLATE_REPLY_INCLUDE_CONFIRMATION:
        return reply
    pending = pending_confirmation_text(state)
    if pending:
        reply += PENDING_CONFIRMATION_TEMPLATE.format(pending=pending)
    else:
        reply += NO_PENDING_CONFIRMATION_TEMPLATE.format(plate=display_plate(state))
    if state.vehicle_type == "new_energy":
        reply += NEW_ENERGY_CONFIRMATION_TEXT
    return reply


def pending_confirmation_text(state: PlateAgentState) -> str:
    descriptions = pending_confirmation_descriptions(state)
    return "，".join(descriptions)


def pending_confirmation_descriptions(state: PlateAgentState) -> list[str]:
    """把待确认字符转成用户可听懂的描述，例如“第1位是天津的津”。"""
    items = state.need_confirm_chars or with_relative_confusion_reasons(state.car_plate, state.confusions)
    descriptions: list[str] = []
    seen_positions: set[int] = set()
    for item in items:
        position = int(getattr(item, "position", 0) or 0)
        value = str(getattr(item, "value", "") or "").strip()
        if position <= 0 or not value or position in seen_positions:
            continue
        seen_positions.add(position)
        descriptions.append(f"第{position}位是{describe_plate_char(value)}")
    return descriptions


def display_plate(state: PlateAgentState | Any) -> str:
    """清理车牌里的空格和换行，保证话术中展示的是紧凑车牌号。"""
    if isinstance(state, PlateAgentState):
        return clean_plate_text(state.car_plate) or "当前车牌"
    return clean_plate_text(state) or "当前车牌"


def ensure_sentence(value: str) -> str:
    """确保基础文案以句号、问号或感叹号结束，方便后面拼接确认句。"""
    text = str(value or "").strip()
    if not text:
        return ""
    if not text.endswith(("。", "！", "？", ".", "!", "?")):
        text += "。"
    return text
