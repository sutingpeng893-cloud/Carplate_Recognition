from __future__ import annotations

import time
from typing import Any

from realtime_audio_demo.config import QWEN_MODEL
from realtime_audio_demo.services.interfaces import ChatModel
from realtime_audio_demo.services.plate_agent_ack import ack_schedule_for_state
from realtime_audio_demo.services.plate_agent_confirmation import (
    apply_confirmation_actions,
    confirmation_actions_from_confusions,
)
from realtime_audio_demo.services.plate_agent_edit import apply_plate_edit_commands, parse_plate_edit_commands
from realtime_audio_demo.services.plate_agent_logging import (
    CURRENT_AI_RAW_DIALOG,
    CURRENT_LLM_CALL_TIMINGS,
    CURRENT_SESSION_ID,
    CURRENT_TURN_BEFORE_STATE,
    CURRENT_USER_AUDIO_PATH,
    log_agent_line,
    log_node_output,
    logger,
)
from realtime_audio_demo.services.plate_agent_messages import (
    EDIT_UNCLEAR_REPLY,
    NO_PLATE_REPLY,
    build_confirmed_reply,
    build_edit_invalid_reply,
    build_pending_action_applied_reply,
    build_pending_action_confirm_reply,
    build_pending_action_discarded_reply,
)
from realtime_audio_demo.services.plate_agent_nodes import PlateAgentNodesMixin
from realtime_audio_demo.services.plate_agent_parsing import elapsed_ms
from realtime_audio_demo.services.plate_agent_response import build_output_json
from realtime_audio_demo.services.plate_agent_rules import (
    detect_initial_confusions_by_rule,
    is_valid_plate_number,
    plate_length,
    vehicle_type_by_length,
)
from realtime_audio_demo.services.plate_agent_state import clone_state, refresh_plate_state
from realtime_audio_demo.services.plate_agent_types import PlateAgentResult, PlateAgentState, PlateConfirmationAction


class PlateAgentService(PlateAgentNodesMixin):
    """车牌语音 Agent 主服务。

    这个文件只保留一轮音频进入后的主流程编排；具体节点能力放在
    plate_agent_nodes.py / plate_agent_confirmation.py / plate_agent_edit.py 等文件里。
    """

    def __init__(self, model_client: ChatModel) -> None:
        """初始化服务，注入 ChatModel 客户端实例。"""
        self.model_client = model_client

    async def handle_audio_turn(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
        session_id: str = "",
        user_audio_path: str = "",
        on_ack: Any = None,
        turn_summaries: list[str] | None = None,
    ) -> PlateAgentResult:
        """处理一轮用户音频。

        流程分两类：
        1. 首轮：还没有暂存车牌，先判断有没有车牌内容，再提取车牌并生成待确认列表。
        2. 多轮：已有暂存车牌，先判断用户是否确认；不是确认时进入纠错 action 流程。
        """
        started = time.perf_counter()
        debug: dict[str, Any] = {}

        # 每轮都克隆一份状态在 working 上处理，避免中途失败时污染调用方传入的旧状态。
        working = clone_state(state)
        if turn_summaries is not None:
            working.turn_summaries = list(turn_summaries)[-6:]
        before_state = working.to_context()

        # 记录 session id 和本轮处理前状态，后续所有节点日志都会自动带上这些上下文。
        CURRENT_SESSION_ID.set(str(session_id or "").strip())
        CURRENT_TURN_BEFORE_STATE.set(before_state)
        CURRENT_USER_AUDIO_PATH.set(str(user_audio_path or "").strip())
        CURRENT_AI_RAW_DIALOG.set([])
        CURRENT_LLM_CALL_TIMINGS.set([])
        log_node_output(
            "handle_audio_turn.start",
            {
                "action": "start_audio_turn",
                "model": model or QWEN_MODEL,
                "wav_bytes": len(wav_bytes),
                "before_state": before_state,
                "state": working.to_context(),
            },
        )

        # 兼容旧调用方传入的 on_ack：这里只发第一个衔接语；新流式 ack 逻辑在接口层按时间表发送。
        if on_ack is not None:
            try:
                _, ack_text = ack_schedule_for_state(working)[0]
                await on_ack(ack_text)
                log_node_output(
                    "handle_audio_turn.compat_on_ack",
                    {
                        "action": "emit_compat_ack",
                        "ack_text": ack_text,
                        "state": working.to_context(),
                    },
                )
            except Exception as exc:
                logger.warning("plate_agent compat on_ack failed: %s", exc)

        # 首轮：当前没有任何暂存车牌，先确认音频里是否包含车牌信息。
        if not working.has_car_plate:
            log_agent_line(
                "首轮识别中",
                当前状态=working.to_context(),
                说明="当前还没有暂存车牌，先判断音频里有没有车牌内容。",
            )
            has_plate = await self.detect_plate_presence(model=model, wav_bytes=wav_bytes)
            debug["has_plate"] = has_plate

            # 首轮没有听到车牌内容：不清空状态，只提示用户继续说车牌。
            if not has_plate:
                log_agent_line("首轮未听到车牌", 模型判断=has_plate, 回复=NO_PLATE_REPLY)
                output = build_output_json(
                    task_status="need_more_info",
                    car_plate="",
                    assistant_reply=NO_PLATE_REPLY,
                )
                working.assistant_reply = NO_PLATE_REPLY
                working.ack_sent = False
                latency_ms = elapsed_ms(started)
                log_node_output(
                    "turn_result",
                    {
                        "stage": "no_plate",
                        "text": output,
                        "speech_text": NO_PLATE_REPLY,
                        "state": working.to_context(),
                        "latency_ms": latency_ms,
                    },
                )
                return PlateAgentResult(
                    text=output,
                    history_text=output,
                    speech_text=NO_PLATE_REPLY,
                    state=working,
                    latency_ms=latency_ms,
                    debug=debug,
                    llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []),
                )

            working.ack_sent = False

            # 首轮听到车牌内容：使用首轮提取 prompt 从音频中提取候选车牌。
            car_plate = await self.extract_car_plate(model=model, wav_bytes=wav_bytes)
            log_agent_line("首轮识别到候选车牌", 候选车牌=car_plate)
            vehicle_type = vehicle_type_by_length(car_plate)

            # 首轮候选车牌位数不合法：返回 invalid，后续音频会继续拼接后重新走首轮识别。
            if vehicle_type == "unknown":
                log_agent_line("首轮车牌格式不合法", 候选车牌=car_plate, 车辆类型=vehicle_type)
                return self.build_invalid_plate_result(
                    started=started,
                    working=working,
                    car_plate=car_plate,
                    debug=debug,
                    stage="invalid_initial_plate",
                )

            # 首轮候选车牌合法：先写入暂存车牌和逐位状态，暂时还没有最终确认车牌。
            refresh_plate_state(
                working,
                car_plate,
                confusions=[],
                confirmed=False,
                preserve_confirmed=False,
            )
            working.final_car_plate = ""
            working.ack_sent = False
            log_node_output(
                "resolve_vehicle_type_by_length",
                {
                    "car_plate": working.car_plate,
                    "plate_length": plate_length(working.car_plate),
                    "vehicle_type": working.vehicle_type,
                },
            )

            # 根据固定易混淆规则生成首轮待确认 action，例如津/京、2/R、1/E。
            confusions = detect_initial_confusions_by_rule(working.car_plate)
            log_node_output(
                "detect_confusions",
                {
                    "source": "rule",
                    "car_plate": working.car_plate,
                    "confusions": [item.to_dict() for item in confusions],
                },
            )
            apply_confirmation_actions(
                working,
                confirmation_actions_from_confusions(confusions),
                source="initial_rule_confusions",
            )

            # 根据当前暂存车牌和待确认列表，生成给用户听到的自然语言回复。
            assistant_reply = await self.generate_reply(
                model=model,
                state=working,
                changed=True,
                scene="initial_success",
            )
            working.assistant_reply = assistant_reply
            output = build_output_json(
                task_status="need_confirmation",
                car_plate=working.car_plate,
                assistant_reply=assistant_reply,
            )
            latency_ms = elapsed_ms(started)
            result_debug = {
                **debug,
                "car_plate": working.car_plate,
                "vehicle_type": working.vehicle_type,
                "confusions": [item.to_dict() for item in working.confusions],
            }
            log_node_output(
                "turn_result",
                {
                    "stage": "initial_plate",
                    "text": output,
                    "speech_text": assistant_reply,
                    "state": working.to_context(),
                    "latency_ms": latency_ms,
                },
            )
            return PlateAgentResult(
                text=output,
                history_text=output,
                speech_text=assistant_reply,
                state=working,
                latency_ms=latency_ms,
                debug=result_debug,
                llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []),
            )

        # 多轮：已有暂存车牌。入口分流：是否有待确认的 pending 修改动作。
        if working.pending_plate:
            # ──────────────────────────────────────────────────────────────
            # 分支 B：等待确认待执行修改轮（1 次 LLM）
            # ──────────────────────────────────────────────────────────────
            log_agent_line("分支B：等待确认pending修改", pending车牌=working.pending_plate, 当前车牌=working.car_plate)
            pending_response = await self.classify_pending_response(model=model, wav_bytes=wav_bytes, state=working)
            debug["pending_response"] = pending_response.to_dict()
            working.ack_sent = False

            if pending_response.intent == "execute":
                # B1：用户确认执行 → 将 pending_plate 写入暂存车牌
                log_agent_line("B1：用户确认执行pending修改", pending车牌=working.pending_plate)
                working.car_plate = working.pending_plate
                working.pending_plate = ""
                working.pending_commands = []
                # 规则扫描新车牌的易混淆位（零LLM），写入状态，回复中自动追加
                new_confusions = detect_initial_confusions_by_rule(working.car_plate)
                refresh_plate_state(working, working.car_plate, confusions=new_confusions, confirmed=False, preserve_confirmed=True)
                apply_confirmation_actions(working, confirmation_actions_from_confusions(new_confusions), source="pending_executed_rule_confusions")
                working.final_car_plate = ""
                assistant_reply = build_pending_action_applied_reply(working)
                working.assistant_reply = assistant_reply
                output = build_output_json(
                    task_status="need_confirmation",
                    car_plate=working.car_plate,
                    assistant_reply=assistant_reply,
                )
                latency_ms = elapsed_ms(started)
                log_node_output("turn_result", {"stage": "pending_executed", "text": output, "speech_text": assistant_reply, "state": working.to_context(), "latency_ms": latency_ms})
                return PlateAgentResult(text=output, history_text=output, speech_text=assistant_reply, state=working, latency_ms=latency_ms, debug=debug, llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []))

            if pending_response.intent == "reject":
                # B2：用户拒绝执行，无新修改意见 → 清空 pending，保留原车牌
                log_agent_line("B2：用户拒绝pending修改，保留原车牌", 当前车牌=working.car_plate)
                working.pending_plate = ""
                working.pending_commands = []
                assistant_reply = build_pending_action_discarded_reply(working.car_plate)
                working.assistant_reply = assistant_reply
                output = build_output_json(
                    task_status="need_confirmation",
                    car_plate=working.car_plate,
                    assistant_reply=assistant_reply,
                )
                latency_ms = elapsed_ms(started)
                log_node_output("turn_result", {"stage": "pending_rejected", "text": output, "speech_text": assistant_reply, "state": working.to_context(), "latency_ms": latency_ms})
                return PlateAgentResult(text=output, history_text=output, speech_text=assistant_reply, state=working, latency_ms=latency_ms, debug=debug, llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []))

            # B3：用户拒绝执行并给出新修改意见 → 清空旧 pending，预执行新 action
            # B4：用户同意执行并同时给出新修改意见 → 先执行 pending，再设置新 pending
            is_execute_with_new = pending_response.intent == "execute_with_new_edit"
            if is_execute_with_new:
                log_agent_line("B4：用户同意执行并追加新修改意见，先执行pending", pending车牌=working.pending_plate, 新actions=[cmd.to_dict() for cmd in pending_response.commands])
                working.car_plate = working.pending_plate
                new_confusions = detect_initial_confusions_by_rule(working.car_plate)
                refresh_plate_state(working, working.car_plate, confusions=new_confusions, confirmed=False, preserve_confirmed=True)
                apply_confirmation_actions(working, confirmation_actions_from_confusions(new_confusions), source="pending_executed_rule_confusions")
                working.final_car_plate = ""
            else:
                log_agent_line("B3：用户给出新修改意见，预执行", 新actions=[cmd.to_dict() for cmd in pending_response.commands])
            working.pending_plate = ""
            working.pending_commands = []
            if pending_response.commands:
                new_edit_result = apply_plate_edit_commands(working.car_plate, pending_response.commands)
                if new_edit_result.changed and is_valid_plate_number(new_edit_result.car_plate):
                    new_plate = new_edit_result.car_plate
                    working.pending_plate = new_plate
                    working.pending_commands = [cmd.to_dict() for cmd in pending_response.commands]
                    assistant_reply = build_pending_action_confirm_reply(working.car_plate, pending_response.commands, new_plate)
                elif new_edit_result.changed:
                    assistant_reply = build_edit_invalid_reply(working)
                else:
                    assistant_reply = new_edit_result.error or EDIT_UNCLEAR_REPLY
            else:
                assistant_reply = build_pending_action_applied_reply(working) if is_execute_with_new else EDIT_UNCLEAR_REPLY
            working.assistant_reply = assistant_reply
            output = build_output_json(task_status="need_confirmation", car_plate=working.car_plate, assistant_reply=assistant_reply)
            latency_ms = elapsed_ms(started)
            log_node_output("turn_result", {"stage": "pending_new_edit", "text": output, "speech_text": assistant_reply, "state": working.to_context(), "latency_ms": latency_ms})
            return PlateAgentResult(text=output, history_text=output, speech_text=assistant_reply, state=working, latency_ms=latency_ms, debug=debug, llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []))

        # ──────────────────────────────────────────────────────────────
        # 分支 A：普通修改轮（pending_plate 为空，最多 2 次 LLM）
        # ──────────────────────────────────────────────────────────────
        log_agent_line(
            "分支A：普通修改轮",
            当前车牌=working.car_plate,
            说明="无 pending 动作，先判断用户是否在确认整车牌。",
        )
        confirmation = await self.detect_confirmation(model=model, wav_bytes=wav_bytes, state=working)
        debug["confirmation"] = confirmation

        # A1：用户明确确认当前车牌
        if confirmation:
            log_agent_line("A1：用户确认当前车牌", 当前车牌=working.car_plate)
            apply_confirmation_actions(
                working,
                [PlateConfirmationAction(action="confirm_all")],
                source="full_plate_confirmation",
            )
            working.final_car_plate = working.car_plate
            working.ack_sent = False
            assistant_reply = build_confirmed_reply(working.final_car_plate)
            working.assistant_reply = assistant_reply
            output = build_output_json(
                task_status="confirmed",
                car_plate=working.car_plate,
                assistant_reply=assistant_reply,
                final_car_plate=working.final_car_plate,
            )
            latency_ms = elapsed_ms(started)
            log_node_output("turn_result", {"stage": "confirmed", "text": output, "speech_text": assistant_reply, "state": working.to_context(), "latency_ms": latency_ms})
            return PlateAgentResult(text=output, history_text=output, speech_text=assistant_reply, state=working, latency_ms=latency_ms, debug=debug, llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []))

        working.ack_sent = False

        # A2：用户给出修改意见 → 提取 action，预执行，不写入状态
        log_agent_line("A2：提取编辑意图并预执行", 当前车牌=working.car_plate)
        edit_result = await self.update_car_plate(model=model, wav_bytes=wav_bytes, state=working)
        debug["edit_result"] = edit_result.to_dict()

        # A2-1：未识别到有效修改动作
        if not edit_result.changed:
            assistant_reply = edit_result.error or EDIT_UNCLEAR_REPLY
            working.confirmed = False
            working.final_car_plate = ""
            working.assistant_reply = assistant_reply
            output = build_output_json(task_status="need_confirmation", car_plate=working.car_plate, assistant_reply=assistant_reply)
            latency_ms = elapsed_ms(started)
            log_node_output("turn_result", {"stage": "edit_unclear", "text": output, "speech_text": assistant_reply, "state": working.to_context(), "latency_ms": latency_ms, "edit_result": edit_result.to_dict()})
            return PlateAgentResult(text=output, history_text=output, speech_text=assistant_reply, state=working, latency_ms=latency_ms, debug=debug, llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []))

        # A2-2：识别到有效修改，检验格式
        new_plate = edit_result.car_plate
        if not is_valid_plate_number(new_plate):
            # A2-2-1：格式不合规
            return self.build_invalid_update_result(
                started=started,
                working=working,
                attempted_plate=new_plate,
                debug=debug,
                stage="invalid_updated_plate",
            )

        # A2-2-2：格式合规 → 保存为 pending，追问用户是否执行
        commands = parse_plate_edit_commands(edit_result.raw)
        working.pending_plate = new_plate
        working.pending_commands = [cmd.to_dict() for cmd in commands]
        working.ack_sent = False
        assistant_reply = build_pending_action_confirm_reply(working.car_plate, commands, new_plate)
        working.assistant_reply = assistant_reply
        output = build_output_json(task_status="need_confirmation", car_plate=working.car_plate, assistant_reply=assistant_reply)
        latency_ms = elapsed_ms(started)
        log_node_output("turn_result", {"stage": "pending_saved", "text": output, "speech_text": assistant_reply, "state": working.to_context(), "latency_ms": latency_ms})
        return PlateAgentResult(
            text=output,
            history_text=output,
            speech_text=assistant_reply,
            state=working,
            latency_ms=latency_ms,
            debug={**debug, "pending_plate": new_plate},
            llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []),
        )


    async def audio_call(self, *, model: str, wav_bytes: bytes, prompt: str, max_tokens: int, node: str = "") -> str:
        """统一的音频模型调用入口。

        上层节点只负责传入不同 prompt；这里统一设置模型、音频、token 和本轮音频任务指令。
        node 参数用于日志标记，方便追踪每次调用的耗时。
        """
        call_start = time.perf_counter()
        completion = await self.model_client.complete_audio(
            model=model or QWEN_MODEL,
            wav_bytes=wav_bytes,
            prompt=prompt,
            history=[],
            max_tokens=max_tokens,
            turn_instruction="请根据这段用户语音完成当前任务。",
        )
        call_ms = int((time.perf_counter() - call_start) * 1000)
        timings = CURRENT_LLM_CALL_TIMINGS.get()
        if timings is not None:
            timings.append({"node": node or "unknown", "duration_ms": call_ms, "max_tokens": max_tokens, "output_chars": len(completion.text or "")})
        log_agent_line(
            "LLM调用耗时",
            节点=node or "unknown",
            耗时ms=call_ms,
            max_tokens=max_tokens,
            输出字数=len(completion.text or ""),
        )
        log_node_output(
            f"llm_call_timing.{node}" if node else "llm_call_timing",
            {
                "node": node or "unknown",
                "duration_ms": call_ms,
                "max_tokens": max_tokens,
                "output_chars": len(completion.text or ""),
            },
        )
        if completion.raw_response and completion.raw_response.get("status_code"):
            raise RuntimeError(str(completion.raw_response.get("message") or "upstream audio request failed"))
        return completion.text or ""


plate_agent_service: PlateAgentService | None = None


def get_plate_agent_service(model_client: ChatModel) -> PlateAgentService:
    """返回单例 PlateAgentService，避免每个请求重复创建服务对象。"""
    global plate_agent_service
    if plate_agent_service is None or plate_agent_service.model_client is not model_client:
        plate_agent_service = PlateAgentService(model_client)
    return plate_agent_service
