from __future__ import annotations

import json
from typing import Any

from realtime_audio_demo.services.plate_agent_rules import describe_plate_char
from realtime_audio_demo.services.plate_agent_tooling import compact_tool_results
from realtime_audio_demo.services.plate_agent_types import PlateAgentState


def build_plate_agent_system_prompt() -> str:
    """Agent 静态说明：定义角色、目标、工具协议，不放动态状态。"""

    return """
你是一个车牌语音识别 Agent，目标是通过多轮语音把用户车牌识别清楚并确认。

工作方式：
1. 每次先读取 user 消息里的 <agent_status>，那里是后端用代码维护的真实状态。
2. 听最新音频后，自主决定下一步要不要调用工具、调用哪个工具、传什么参数。
3. 车牌状态只能通过工具改变，不能只靠文字结论改变。
4. 工具执行结果会回填到下一次 <agent_status>，你再决定继续调用工具还是 finish。
5. 首轮没有可用车牌时，不要编车牌；已有暂存车牌时，不要因为一次修改失败就清空旧车牌。
6. validate、detect_confusions、refresh_confirmation 都是工具能力，不是固定流程。你要自己判断什么时候需要调用。

可用工具：
- get_current_state()：读取当前车牌状态。
- validate_plate_rules(car_plate?)：校验某个车牌或当前车牌是否符合规则，只返回结果，不修改状态。
- detect_confusions_by_rules(car_plate?)：扫描某个车牌或当前车牌里的易混淆字符，只返回结果，不修改状态。
- set_plate(car_plate, confirmed_positions?, confirmed?, preserve_confirmed?)：设置完整车牌。只在你从音频里听到了完整、合法车牌时使用。
- replace_position(position, value)：把第 position 位替换成 value。
- replace_char(old_value, value, occurrence?)：把当前车牌中的某个字符替换成 value；occurrence 可为 first、last、all。
- insert_position(position, value, relation?)：在第 position 位 before/after 插入 value。
- delete_position(position)：删除第 position 位，删除后后面的字符会自动前移。
- refresh_confirmation_by_rules(confirmed_positions?)：按易混淆规则写入需要二次确认的字符。只有你调用它，状态里的待确认列表才会刷新。
- add_need_confirmation(position)：把某一位加入待二次确认。
- remove_need_confirmation(position)：把某一位从待二次确认中移除。
- add_confirmed(position)：记录用户已经明确确认某一位。
- remove_confirmed(position)：撤销某一位已确认状态。
- confirm_all()：用户明确确认整车牌正确时调用。

输出协议：
- 只输出一个 JSON 对象。
- 需要工具时输出 {"thought":"简短判断","tool_calls":[{"name":"工具名","arguments":{...}}]}。
- 工具执行后还需要继续处理，可以继续输出 tool_calls。
- 不需要再调用工具时输出 {"thought":"简短判断","finish":{"task_status":"need_more_info|need_confirmation|confirmed|invalid|unclear","reply_scene":"initial_success|update_success|partial_confirmation|confirmed|need_more_info|edit_unclear|invalid"}}。
- 如果输出 tool_calls，本次不要同时 finish，等工具结果回填后再 finish。

车牌基本规则：
- 第 1 位是中文省份简称。
- 第 2 位是英文字母。
- 普通燃油车 7 位，新能源车 8 位。
- 后续字符通常是数字或大写英文字母，特殊尾字可为警、临、学、领、挂。
- 常见语音转换：洞/零/〇=0，幺/么=1，二/两=2，是=4，陆=6，拐=7，吸=C，勾/沟儿=J，圈=Q。
- 易混淆且需要二次确认的字符：2、R、1、E、甘、赣、津、京、桂、贵、冀、吉。

注意：
- 你可以先设置或修改车牌，再调用 validate_plate_rules / detect_confusions_by_rules 检查，也可以在有把握时直接 finish。
- 如果希望用户确认易混淆字符，需要调用 refresh_confirmation_by_rules 或 add_need_confirmation 更新状态。
- 如果用户明确确认整车牌，调用 confirm_all。
""".strip()


def build_plate_agent_turn_instruction(
    *,
    state: PlateAgentState,
    session_id: str,
    iteration: int,
    max_iterations: int,
    tool_results: list[dict[str, Any]],
) -> str:
    """动态状态栏：作为本次 user 消息文本和音频一起送给模型。"""

    return (
        f"{build_plate_agent_status_bar(state=state, session_id=session_id, iteration=iteration, max_iterations=max_iterations, tool_results=tool_results)}\n\n"
        "请基于本条用户语音、状态栏和前面的 tool_results，决定下一步 tool_calls 或 finish。"
    )


def build_plate_agent_status_bar(
    *,
    state: PlateAgentState,
    session_id: str,
    iteration: int,
    max_iterations: int,
    tool_results: list[dict[str, Any]],
) -> str:
    """把代码维护的真实状态压缩为模型每轮都能看到的状态栏。"""

    context = state.to_context()
    current_plate = str(context.get("car_plate") or "").strip()
    stage = "多轮确认/纠错" if state.has_car_plate else "首轮识别"
    status = {
        "stage": stage,
        "tool_round": f"{iteration}/{max_iterations}",
        "has_car_plate": state.has_car_plate,
        "current_car_plate": current_plate or "空",
        "plate_length": len(current_plate),
        "vehicle_type": vehicle_type_text(context.get("vehicle_type")),
        "confirmed": state.is_confirmed,
        "final_car_plate": context.get("final_car_plate") or "",
    }
    lines = ["<agent_status>", "Current State:"]
    lines.extend(f"- {key}: {value}" for key, value in status.items())

    plate_chars = describe_plate_chars(context.get("plate_chars"))
    lines.append(f"- plate_chars: {plate_chars or '无'}")
    need_confirm = describe_state_chars(context.get("need_confirm_chars"))
    lines.append(f"- need_confirm_chars: {need_confirm or '无'}")
    confirmed = describe_state_chars(context.get("confirmed_chars"))
    lines.append(f"- confirmed_chars: {confirmed or '无'}")

    summaries = [str(item).strip() for item in context.get("turn_summaries") or [] if str(item).strip()]
    if summaries:
        lines.append("Recent History:")
        lines.extend(f"- {summary}" for summary in summaries[-6:])
    else:
        lines.append("Recent History: 无")

    compacted_results = compact_tool_results(tool_results)
    if compacted_results:
        lines.append("Previous Tool Results:")
        lines.append(json.dumps(compacted_results, ensure_ascii=False, indent=2))
    else:
        lines.append("Previous Tool Results: 无")

    lines.append("</agent_status>")
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
        descriptions.append(f"第{position}位={describe_plate_char(char)}")
    return "，".join(descriptions)


def vehicle_type_text(value: Any) -> str:
    raw = str(value or "").strip()
    if raw == "fuel":
        return "普通燃油车号牌"
    if raw == "new_energy":
        return "新能源车号牌"
    return raw or "未知"
