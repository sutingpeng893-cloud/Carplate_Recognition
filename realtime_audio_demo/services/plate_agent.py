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
from realtime_audio_demo.services.plate_agent_logging import (
    CURRENT_SESSION_ID,
    CURRENT_TURN_BEFORE_STATE,
    log_agent_line,
    log_node_output,
    logger,
)
from realtime_audio_demo.services.plate_agent_messages import (
    EDIT_UNCLEAR_REPLY,
    NO_PLATE_REPLY,
    build_confirmed_reply,
)
from realtime_audio_demo.services.plate_agent_nodes import PlateAgentNodesMixin
from realtime_audio_demo.services.plate_agent_parsing import elapsed_ms, unique_positions
from realtime_audio_demo.services.plate_agent_response import build_output_json, reply_with_pending_confirmation
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
        self.model_client = model_client

    async def handle_audio_turn(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
        session_id: str = "",
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
            )

        # 多轮：已经有暂存车牌，先判断用户这轮是在确认整车牌，还是在纠错。
        log_agent_line(
            "多轮确认或纠错中",
            当前车牌=working.car_plate,
            二次确认列表=[item.to_dict() for item in working.need_confirm_chars],
            已确认字符=[item.to_dict() for item in working.confirmed_chars],
            说明="当前已有暂存车牌，先判断用户是在确认还是纠错。",
        )
        confirmation = await self.detect_confirmation(model=model, wav_bytes=wav_bytes, state=working)
        debug["confirmation"] = confirmation

        # 用户明确确认当前车牌：清空待确认列表，把所有字符标记为已确认，并输出最终车牌。
        if confirmation:
            log_agent_line("用户确认当前车牌", 当前车牌=working.car_plate)
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
            log_node_output(
                "turn_result",
                {
                    "stage": "confirmed",
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
                debug=debug,
            )

        working.ack_sent = False

        # 用户不是确认：进入纠错流程，模型输出编辑 action，后端按 action 修改当前暂存车牌。
        log_agent_line("多轮纠错中", 当前车牌=working.car_plate, 说明="用户不是确认，进入编辑动作推理和执行。")
        edit_result = await self.update_car_plate(model=model, wav_bytes=wav_bytes, state=working)
        debug["edit_result"] = edit_result.to_dict()
        review_confirmed_positions = edit_result.review.confirmed_positions if edit_result.review else []

        # 没有实际改动车牌：可能是用户只确认了某些易混淆位，也可能是纠错意图不清晰。
        if not edit_result.changed:
            if (edit_result.command and edit_result.command.action == "none") or review_confirmed_positions:
                # 用户只确认了部分字符：刷新二次确认列表和已确认字符，然后继续让用户确认剩余内容。
                confusions = await self.refresh_confusions_after_audio(
                    model=model,
                    wav_bytes=wav_bytes,
                    working=working,
                    confirmed_positions=review_confirmed_positions,
                )
                assistant_reply = (
                    await self.generate_reply(
                        model=model,
                        state=working,
                        changed=False,
                        scene="partial_confirmation",
                    )
                    if (edit_result.command and edit_result.command.action == "none") or review_confirmed_positions
                    else reply_with_pending_confirmation(edit_result.error or EDIT_UNCLEAR_REPLY, working)
                )
                working.assistant_reply = assistant_reply
                output = build_output_json(
                    task_status="need_confirmation",
                    car_plate=working.car_plate,
                    assistant_reply=assistant_reply,
                )
                latency_ms = elapsed_ms(started)
                log_node_output(
                    "turn_result",
                    {
                        "stage": "partial_confirmation",
                        "text": output,
                        "speech_text": assistant_reply,
                        "state": working.to_context(),
                        "latency_ms": latency_ms,
                        "confusions": [item.to_dict() for item in confusions],
                        "edit_result": edit_result.to_dict(),
                    },
                )
                return PlateAgentResult(
                    text=output,
                    history_text=output,
                    speech_text=assistant_reply,
                    state=working,
                    latency_ms=latency_ms,
                    debug={
                        **debug,
                        "car_plate": working.car_plate,
                        "vehicle_type": working.vehicle_type,
                        "confusions": [item.to_dict() for item in working.confusions],
                    },
                )

            assistant_reply = reply_with_pending_confirmation(edit_result.error or EDIT_UNCLEAR_REPLY, working)
            working.confirmed = False
            working.final_car_plate = ""
            working.assistant_reply = assistant_reply
            output = build_output_json(
                task_status="need_confirmation",
                car_plate=working.car_plate,
                assistant_reply=assistant_reply,
            )
            latency_ms = elapsed_ms(started)
            log_node_output(
                "turn_result",
                {
                    "stage": "edit_unclear",
                    "text": output,
                    "speech_text": assistant_reply,
                    "state": working.to_context(),
                    "latency_ms": latency_ms,
                    "edit_result": edit_result.to_dict(),
                },
            )
            return PlateAgentResult(
                text=output,
                history_text=output,
                speech_text=assistant_reply,
                state=working,
                latency_ms=latency_ms,
                debug=debug,
            )

        # 纠错 action 修改出了新车牌：先校验格式，不合法则保留旧车牌并继续让用户说明。
        new_car_plate = edit_result.car_plate
        if new_car_plate:
            if not is_valid_plate_number(new_car_plate):
                return self.build_invalid_update_result(
                    started=started,
                    working=working,
                    attempted_plate=new_car_plate,
                    debug=debug,
                    stage="invalid_updated_plate",
                )

            # 新车牌格式合法：写入暂存车牌，后面再统一更新待确认列表和已确认字符。
            refresh_plate_state(
                working,
                new_car_plate,
                confusions=[],
                confirmed=False,
                preserve_confirmed=True,
            )
            working.final_car_plate = ""
            log_node_output(
                "resolve_vehicle_type_by_length",
                {
                    "car_plate": working.car_plate,
                    "plate_length": plate_length(working.car_plate),
                    "vehicle_type": working.vehicle_type,
                },
            )

        # 修改成功后，按规则扫描易混淆位，再让模型输出确认状态 action，更新两个确认列表。
        confusions = await self.refresh_confusions_after_audio(
            model=model,
            wav_bytes=wav_bytes,
            working=working,
            confirmed_positions=unique_positions([*edit_result.changed_positions, *review_confirmed_positions]),
        )
        assistant_reply = await self.generate_reply(
            model=model,
            state=working,
            changed=True,
            scene="update_success",
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
                "stage": "updated_plate",
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
        )

    async def audio_call(self, *, model: str, wav_bytes: bytes, prompt: str, max_tokens: int) -> str:
        """统一的音频模型调用入口。

        上层节点只负责传入不同 prompt；这里统一设置模型、音频、token 和本轮音频任务指令。
        """
        completion = await self.model_client.complete_audio(
            model=model or QWEN_MODEL,
            wav_bytes=wav_bytes,
            prompt=prompt,
            history=[],
            max_tokens=max_tokens,
            turn_instruction="请根据这段用户语音完成当前任务。",
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
