from __future__ import annotations

from typing import Any

from realtime_audio_demo.services.plate_agent_constants import CAR_PLATE_EXTRACTION_PROMPT
from realtime_audio_demo.services.plate_agent_confirmation import (
    apply_confirmation_actions,
    complete_confirmation_actions,
    parse_confirmation_actions,
)
from realtime_audio_demo.services.plate_agent_edit import apply_plate_edit_commands, parse_plate_edit_command_data, parse_plate_edit_commands
from realtime_audio_demo.services.plate_agent_logging import log_agent_line, log_node_output, CURRENT_LLM_CALL_TIMINGS
from realtime_audio_demo.services.plate_agent_parsing import (
    elapsed_ms,
    extract_final_plate_from_text,
    extract_plate_from_json_object,
    parse_bool_text,
    parse_json_bool,
    parse_json_object,
    parse_position_list,
    parse_yes_no,
    sanitize_extracted_plate_text,
    unique_positions,
)
from realtime_audio_demo.services.plate_agent_prompts import (
    build_classify_intent_prompt,
    build_classify_pending_commands_prompt,
    build_confirmation_detection_prompt_with_history,
    build_confirmation_state_action_prompt,
    build_plate_edit_command_prompt,
    build_plate_presence_prompt,
    build_plate_update_review_prompt,
    build_province_retry_prompt,
)
from realtime_audio_demo.services.plate_agent_response import (
    build_output_json,
    reply_with_pending_confirmation,
)
from realtime_audio_demo.services.plate_agent_messages import (
    EDIT_UNCLEAR_REPLY,
    INVALID_PLATE_REPLY,
    build_edit_invalid_reply,
    build_fixed_reply,
    describe_edit_commands,
)
from realtime_audio_demo.services.plate_agent_rules import (
    clean_plate_text,
    detect_initial_confusions_by_rule,
    first_char_is_ascii_letter_or_digit,
    normalize_plate_format,
    normalize_plate_text,
    plate_length,
    replace_leading_g_with_ji,
    vehicle_type_by_length,
)
from realtime_audio_demo.services.plate_agent_state import (
    extract_batch_commands,
)
from realtime_audio_demo.services.plate_agent_types import (
    PlateAgentResult,
    PlateAgentState,
    PlateConfirmationAction,
    PlateConfusion,
    PlateEditResult,
    PlateUpdateReview,
    PendingResponseResult,
)


class PlateAgentNodesMixin:
    def build_invalid_plate_result(
        self,
        *,
        started: float,
        working: PlateAgentState,
        car_plate: str,
        debug: dict[str, Any],
        stage: str,
    ) -> PlateAgentResult:
        """构造车牌格式不合法时的结果对象，清空 state 中所有车牌字段并返回 invalid 状态，无模型调用。"""
        output = build_output_json(
            task_status="invalid",
            car_plate=car_plate,
            assistant_reply=INVALID_PLATE_REPLY,
        )
        working.car_plate = ""
        working.plate_chars = []
        working.confirmed = False
        working.need_confirm_chars = []
        working.confirmed_chars = []
        working.vehicle_type = "unknown"
        working.confusions = []
        working.final_car_plate = ""
        working.assistant_reply = INVALID_PLATE_REPLY
        working.ack_sent = False
        latency_ms = elapsed_ms(started)
        result_debug = {
            **debug,
            "invalid_car_plate": clean_plate_text(car_plate),
            "plate_length": plate_length(car_plate),
        }
        log_node_output(
            "validate_plate_length",
            {
                "car_plate": clean_plate_text(car_plate),
                "plate_length": plate_length(car_plate),
                "valid": False,
                "assistant_reply": INVALID_PLATE_REPLY,
            },
        )
        log_node_output(
            "turn_result",
            {
                "stage": stage,
                "text": output,
                "speech_text": INVALID_PLATE_REPLY,
                "state": working.to_context(),
                "latency_ms": latency_ms,
            },
        )
        return PlateAgentResult(
            text=output,
            history_text=output,
            speech_text=INVALID_PLATE_REPLY,
            state=working,
            latency_ms=latency_ms,
            debug=result_debug,
            llm_calls=list(CURRENT_LLM_CALL_TIMINGS.get() or []),
        )

    def build_invalid_update_result(
        self,
        *,
        started: float,
        working: PlateAgentState,
        attempted_plate: str,
        debug: dict[str, Any],
        stage: str,
    ) -> PlateAgentResult:
        """构造更新后车牌不合法的结果对象，保留原有车牌并返回 need_confirmation 状态，无模型调用。"""
        assistant_reply = build_edit_invalid_reply(working)
        working.confirmed = False
        working.final_car_plate = ""
        working.assistant_reply = assistant_reply
        working.ack_sent = False
        output = build_output_json(
            task_status="need_confirmation",
            car_plate=working.car_plate,
            assistant_reply=assistant_reply,
        )
        latency_ms = elapsed_ms(started)
        result_debug = {
            **debug,
            "attempted_car_plate": clean_plate_text(attempted_plate),
            "kept_car_plate": working.car_plate,
        }
        log_node_output(
            "validate_updated_plate",
            {
                "attempted_car_plate": clean_plate_text(attempted_plate),
                "kept_car_plate": working.car_plate,
                "valid": False,
                "assistant_reply": assistant_reply,
            },
        )
        log_node_output(
            "turn_result",
            {
                "stage": stage,
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

    async def refresh_confusions_after_audio(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        working: PlateAgentState,
        confirmed_positions: list[int] | None = None,
    ) -> list[PlateConfusion]:
        """刷新易混淆位列表：先规则扫描，再调用 detect_confirmation_state_actions（调用大模型）输出确认状态 action，最后原地更新 state。"""
        previous_confirmed_positions = {
            item.position for item in working.confirmed_chars if item.confirmed and item.position > 0
        }
        merged_confirmed_positions = unique_positions([*previous_confirmed_positions, *(confirmed_positions or [])])
        rule_confusions = detect_initial_confusions_by_rule(working.car_plate)
        log_agent_line(
            "更新二次确认列表：开始复核",
            当前车牌=working.car_plate,
            规则扫描结果=[item.to_dict() for item in rule_confusions],
            之前已确认位置=sorted(previous_confirmed_positions),
            本轮新增确认位置=confirmed_positions or [],
        )
        log_node_output(
            "detect_confusions.rule_scan",
            {
                "source": "rule_before_model",
                "car_plate": working.car_plate,
                "confusions": [item.to_dict() for item in rule_confusions],
            },
        )
        actions = await self.detect_confirmation_state_actions(
            model=model,
            wav_bytes=wav_bytes,
            state=working,
            rule_confusions=rule_confusions,
            confirmed_positions=merged_confirmed_positions,
        )
        confusions = apply_confirmation_actions(
            working,
            actions,
            source="refresh_confusions_after_audio",
        )
        log_agent_line(
            "更新二次确认列表：复核后结果",
            当前车牌=working.car_plate,
            action执行后仍需二次确认=[item.to_dict() for item in confusions],
            action执行后已确认字符=[item.to_dict() for item in working.confirmed_chars],
        )
        return confusions

    async def detect_plate_presence(self, *, model: str, wav_bytes: bytes) -> bool:
        """调用大模型判断音频中是否含有车牌信息，返回 bool（max_tokens=8）。"""
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=build_plate_presence_prompt(),
            max_tokens=8,
            node="detect_plate_presence",
        )
        has_plate = parse_bool_text(result, default=False)
        log_node_output("detect_plate_presence", {"raw": result, "has_plate": has_plate})
        return has_plate

    async def extract_car_plate(self, *, model: str, wav_bytes: bytes) -> str:
        """两阶段提取车牌：第一步调用大模型提取原始文本，第二步调用 normalize_plate_result 做省份归一化（可能再次调用大模型）。"""
        extraction_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=CAR_PLATE_EXTRACTION_PROMPT,
            max_tokens=1024,
            node="extract_car_plate",
        )
        log_agent_line("首轮模型推理是什么", 模型输出=extraction_result)
        summarized_raw = extract_final_plate_from_text(extraction_result)
        summarized_plate = sanitize_extracted_plate_text(summarized_raw)
        parsed_plate = extract_plate_from_json_object(parse_json_object(extraction_result))
        extraction_plate = summarized_plate or parsed_plate
        log_agent_line(
            "首轮车牌提取结果",
            摘要中的最终车牌=summarized_raw,
            清洗后车牌=extraction_plate,
            JSON车牌=parsed_plate,
        )
        log_node_output(
            "extract_car_plate.step1_extract_with_pronunciation",
            {
                "raw": extraction_result,
                "summary_raw": summarized_raw,
                "summary_car_plate": summarized_plate,
                "json_car_plate": parsed_plate,
                "car_plate": extraction_plate,
            },
        )
        final_plate = await self.normalize_plate_result(
            model=model,
            wav_bytes=wav_bytes,
            car_plate=extraction_plate,
            node="extract_car_plate.normalize",
        )
        log_node_output(
            "extract_car_plate",
            {
                "step1_car_plate": extraction_plate,
                "car_plate": final_plate,
            },
        )
        return final_plate

    async def detect_confirmation_state_actions(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
        rule_confusions: list[PlateConfusion],
        confirmed_positions: list[int],
    ) -> list[PlateConfirmationAction]:
        """调用大模型输出确认状态 action 列表（max_tokens=512），再与规则混淆结果合并补全，返回最终 action 列表。"""
        plate = clean_plate_text(state.car_plate)
        context = {
            "car_plate": plate,
            "plate_length": len(plate),
            "vehicle_type": state.vehicle_type,
            "plate_chars": [item.to_dict() for item in state.plate_chars],
            "need_confirm_chars": [item.to_dict() for item in state.need_confirm_chars],
            "confirmed_chars": [item.to_dict() for item in state.confirmed_chars],
            "rule_confusions": [item.to_dict() for item in rule_confusions],
            "confirmed_positions_from_review": confirmed_positions,
            "assistant_reply": state.assistant_reply,
            "recent_turn_summaries": list(state.turn_summaries),
        }
        raw = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=build_confirmation_state_action_prompt(context),
            max_tokens=512,
            node="detect_confirmation_state_actions",
        )
        model_actions = parse_confirmation_actions(raw)
        actions = complete_confirmation_actions(
            plate=plate,
            rule_confusions=rule_confusions,
            model_actions=model_actions,
            confirmed_positions=confirmed_positions,
        )
        log_agent_line(
            "确认状态更新：模型推理是什么",
            当前车牌=plate,
            模型输出=raw,
        )
        log_agent_line(
            "确认状态更新：模型 action",
            模型actions=[item.to_dict() for item in model_actions],
            最终执行actions=[item.to_dict() for item in actions],
        )
        log_node_output(
            "confirmation_state.detect_actions",
            {
                "raw": raw,
                "context": context,
                "model_actions": [item.to_dict() for item in model_actions],
                "actions": [item.to_dict() for item in actions],
            },
        )
        return actions

    async def detect_confirmation(self, *, model: str, wav_bytes: bytes, state: PlateAgentState) -> bool:
        """调用大模型判断用户是否在整体确认当前车牌，输出 yes/no（max_tokens=8），返回 bool。"""
        previous_ai_reply = (state.assistant_reply or "").strip()
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=build_confirmation_detection_prompt_with_history(previous_ai_reply, state.turn_summaries),
            max_tokens=8,
            node="detect_confirmation",
        )
        confirmed = parse_yes_no(result, default=False)
        log_context = {
            "raw": result,
            "confirmed": confirmed,
            "assistant_reply": previous_ai_reply,
            "state": state.to_context(),
        }
        log_node_output("detect_confirmation", log_context)
        return confirmed

    async def update_car_plate(self, *, model: str, wav_bytes: bytes, state: PlateAgentState) -> PlateEditResult:
        current_plate = normalize_plate_text(state.car_plate)
        if not current_plate:
            return PlateEditResult(car_plate="", changed=False, error=EDIT_UNCLEAR_REPLY)

        command_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=build_plate_edit_command_prompt(
                state,
                current_plate=current_plate,
                edit_steps=[],
            ),
            max_tokens=1024,
            node="update_car_plate.edit_command",
        )
        commands = parse_plate_edit_commands(command_result)
        command = commands[0]
        log_agent_line(
            "多轮纠错：模型推理是什么",
            当前车牌=current_plate,
            模型输出=command_result,
        )
        log_agent_line(
            "多轮纠错：action 是什么",
            actions=[item.to_dict() for item in commands],
            说明="后端会按 actions 顺序执行；单个修改时 actions 只有一项。",
        )
        log_node_output(
            "update_car_plate.action",
            {
                "raw": command_result,
                "previous_state": state.to_context(),
                "input_plate": current_plate,
                "command": command.to_dict(),
                "commands": [item.to_dict() for item in commands],
            },
        )
        edit_result = apply_plate_edit_commands(current_plate, commands)
        edit_result.raw = command_result
        log_agent_line(
            "多轮纠错：action 执行结果",
            执行前车牌=current_plate,
            执行后车牌=edit_result.car_plate,
            是否修改=edit_result.changed,
            修改位置=edit_result.changed_positions,
            错误信息=edit_result.error,
        )
        log_node_output(
            "update_car_plate.edit_result",
            {
                "input_plate": current_plate,
                "command": command.to_dict(),
                "commands": [item.to_dict() for item in commands],
                "edit_result": edit_result.to_dict(),
            },
        )
        return edit_result

    async def classify_pending_response(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
    ) -> PendingResponseResult:
        """两阶段意图分类：
        1. 轻量级调用（max_tokens=16）判断 execute / reject / unclear。
        2. 仅当 unclear 时触发重量级调用（max_tokens=128）提取编辑命令。
        execute 和 reject 只需1次 LLM 调用；new_edit 需要 2 次。
        """
        action_desc = describe_edit_commands(
            [parse_plate_edit_command_data(cmd) for cmd in state.pending_commands if isinstance(cmd, dict)]
        )
        context = {
            "car_plate": state.car_plate,
            "pending_plate": state.pending_plate,
            "action_desc": action_desc,
            "previous_assistant_reply": state.assistant_reply,
        }

        # ── 第一阶段：轻量级意图分类（max_tokens=16）──
        intent_raw = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=build_classify_intent_prompt(context),
            max_tokens=16,
            node="classify_intent",
        )
        intent_text = intent_raw.strip().lower()
        log_agent_line("等待确认轮：轻量级意图分类", 模型输出=intent_raw, 解析结果=intent_text)
        log_node_output(
            "classify_intent",
            {"raw": intent_raw, "intent_text": intent_text, "context": context},
        )

        if intent_text in ("execute", "reject"):
            # 快速路径：无需第二次调用
            result = PendingResponseResult(intent=intent_text, commands=[], raw=intent_raw)
            log_node_output(
                "classify_pending_response",
                {"fast_path": True, "raw": intent_raw, "context": context, "result": result.to_dict()},
            )
            return result

        # ── 第二阶段：重量级调用提取编辑命令（max_tokens=128）──
        commands_raw = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=build_classify_pending_commands_prompt(context),
            max_tokens=128,
            node="classify_pending_commands",
        )
        log_agent_line("等待确认轮：提取编辑命令", 模型输出=commands_raw)
        data = parse_json_object(commands_raw)
        intent = str(data.get("intent") or "").strip()
        if intent not in ("execute", "reject", "new_edit", "execute_with_new_edit"):
            intent = "reject"
        commands_list = data.get("commands")
        commands = parse_plate_edit_commands(commands_list) if isinstance(commands_list, list) and commands_list else []
        result = PendingResponseResult(intent=intent, commands=commands, raw=commands_raw)
        log_node_output(
            "classify_pending_response",
            {"fast_path": False, "raw": commands_raw, "context": context, "result": result.to_dict()},
        )
        return result

    async def review_plate_update(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
        before_plate: str,
        edit_result: PlateEditResult,
        steps: list[dict[str, Any]],
    ) -> PlateUpdateReview:
        """调用大模型对单步编辑结果复核（max_tokens=256），输出 confirmed_positions、needs_more_edit、valid_result。"""
        after_plate = normalize_plate_text(edit_result.car_plate or before_plate)
        context = {
            "previous_state": state.to_context(),
            "before_plate": before_plate,
            "after_plate": after_plate,
            "command": edit_result.command.to_dict() if edit_result.command else None,
            "commands": extract_batch_commands(edit_result),
            "changed": edit_result.changed,
            "changed_positions": edit_result.changed_positions,
            "existing_steps": steps,
        }
        raw = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=build_plate_update_review_prompt(context),
            max_tokens=256,
            node="review_plate_update",
        )
        log_agent_line(
            "多轮纠错：review 模型推理是什么",
            修改前车牌=before_plate,
            修改后车牌=after_plate,
            模型输出=raw,
        )
        data = parse_json_object(raw)
        review = PlateUpdateReview(
            confirmed_positions=parse_position_list(data.get("confirmed_positions"), plate_length(after_plate)),
            needs_more_edit=parse_json_bool(data.get("needs_more_edit"), default=False),
            valid_result=parse_json_bool(data.get("valid_result"), default=True),
            reason=str(data.get("reason") or "").strip(),
            raw=raw,
        )
        log_agent_line(
            "多轮纠错：review 结果",
            已确认位置=review.confirmed_positions,
            是否还有未处理修改=review.needs_more_edit,
            编辑是否有效=review.valid_result,
            原因=review.reason,
        )
        log_node_output(
            "update_car_plate.review",
            {
                "raw": raw,
                "context": context,
                "review": review.to_dict(),
            },
        )
        return review

    async def normalize_plate_result(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        car_plate: str,
        node: str,
    ) -> str:
        """若提取车牌首字符为 ASCII 字母/数字（省份识别失败），则条件性调用大模型重试省份识别（max_tokens=128）；最后执行 G→冀 兜底替换。"""
        formatted_plate = normalize_plate_format(sanitize_extracted_plate_text(car_plate))
        corrected_plate = formatted_plate
        retry_raw = ""
        retry_plate = ""
        if first_char_is_ascii_letter_or_digit(formatted_plate):
            retry_raw = await self.audio_call(
                model=model,
                wav_bytes=wav_bytes,
                prompt=build_province_retry_prompt(formatted_plate),
                max_tokens=128,
                node="normalize_plate_result",
            )
            retry_plate = normalize_plate_format(extract_plate_from_json_object(parse_json_object(retry_raw)))
            if retry_plate:
                corrected_plate = retry_plate
        final_plate = replace_leading_g_with_ji(corrected_plate)
        log_node_output(
            node,
            {
                "input_car_plate": car_plate,
                "formatted_car_plate": formatted_plate,
                "province_retry_used": bool(retry_raw),
                "province_retry_raw": retry_raw,
                "province_retry_car_plate": retry_plate,
                "car_plate": final_plate,
            },
        )
        return final_plate

    async def generate_reply(self, *, model: str, state: PlateAgentState, changed: bool, scene: str = "") -> str:
        """根据当前 state 和 scene 标记生成固定模板回复文本，不调用大模型。"""
        reply = build_fixed_reply(state, changed=changed, scene=scene)
        log_node_output(
            "generate_reply",
            {
                "source": "fixed_message_template",
                "scene": scene,
                "changed": changed,
                "assistant_reply": reply,
                "state": state.to_context(),
            },
        )
        return reply
