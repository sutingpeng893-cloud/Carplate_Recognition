from __future__ import annotations

import json
import logging
import re
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from realtime_audio_demo.config import QWEN_MODEL
from realtime_audio_demo.services.interfaces import ChatModel
from realtime_audio_demo.services.output_filter import extract_json_candidate
from realtime_audio_demo.services.plate_agent_ack import ack_schedule_for_state
from realtime_audio_demo.services.plate_agent_edit import (
    apply_plate_edit_command,
    normalize_spoken_plate_chars,
    parse_plate_edit_command,
    parse_positive_int,
)
from realtime_audio_demo.services.plate_agent_prompts import (
    build_plate_edit_command_prompt,
    build_plate_update_review_prompt,
)
from realtime_audio_demo.services.plate_agent_types import (
    PlateAgentResult,
    PlateAgentState,
    PlateCharState,
    PlateConfusion,
    PlateEditResult,
    PlateUpdateReview,
)


logger = logging.getLogger("uvicorn.error")
CURRENT_SESSION_ID: ContextVar[str] = ContextVar("plate_agent_session_id", default="")
CURRENT_TURN_BEFORE_STATE: ContextVar[dict[str, Any] | None] = ContextVar("plate_agent_turn_before_state", default=None)

NO_PLATE_REPLY = "我没有听到车牌号内容，请告诉我车牌号。"
INVALID_PLATE_REPLY = "您好，您当前的车牌号并不是有效号码，请重新输入。"
EDIT_UNCLEAR_REPLY = "我没有听清您要修改车牌的哪一处，当前仍保留原来的车牌。请您说明要替换、插入或删除哪一位。"
EDIT_INVALID_REPLY = "我按这次修改后得到的车牌格式不符合规则，当前仍保留原来的车牌。请您重新说明要改哪一处。"

PROVINCE_ABBREVIATIONS = {
    "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏", "浙", "皖", "闽",
    "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵", "云", "藏",
    "陕", "甘", "青", "宁", "新",
}
SPECIAL_PLATE_TAIL_CHARS = {"警", "临", "学", "领", "挂"}

CAR_PLATE_EXTRACTION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "car_plate_extraction_prompt.md"
CAR_PLATE_EXTRACTION_PROMPT = CAR_PLATE_EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
CONFUSION_PROVINCE_CHARS = {"甘", "赣", "津", "京", "桂", "贵", "冀", "吉"}
CONFUSION_ALNUM_CHARS = {"2", "R", "1", "E"}


def log_node_output(node: str, output: dict[str, Any]) -> None:
    before_state = output.get("before_state") or CURRENT_TURN_BEFORE_STATE.get()
    after_state = output.get("after_state") or output.get("state")
    payload: dict[str, Any] = {
        "session_id": CURRENT_SESSION_ID.get() or None,
        "method": node,
        "event": "node_output",
        "output": output,
    }
    if isinstance(before_state, dict):
        payload["before_state"] = before_state
    if isinstance(after_state, dict):
        payload["after_state"] = after_state
    if isinstance(before_state, dict) and isinstance(after_state, dict):
        payload["state_diff"] = state_change_summary(before_state, after_state)
    logger.info("plate_agent event=%s", json.dumps(payload, ensure_ascii=False, default=str))


class PlateAgentService:
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
    ) -> PlateAgentResult:
        started = time.perf_counter()
        debug: dict[str, Any] = {}
        working = clone_state(state)
        before_state = working.to_context()
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

        if not working.has_car_plate:
            has_plate = await self.detect_plate_presence(model=model, wav_bytes=wav_bytes)
            debug["has_plate"] = has_plate
            if not has_plate:
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

            car_plate = await self.extract_car_plate(model=model, wav_bytes=wav_bytes)
            vehicle_type = vehicle_type_by_length(car_plate)
            if vehicle_type == "unknown":
                return self.build_invalid_plate_result(
                    started=started,
                    working=working,
                    car_plate=car_plate,
                    debug=debug,
                    stage="invalid_initial_plate",
                )
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
            confusions = detect_initial_confusions_by_rule(working.car_plate)
            log_node_output(
                "detect_confusions",
                {
                    "source": "rule",
                    "car_plate": working.car_plate,
                    "confusions": [item.to_dict() for item in confusions],
                },
            )
            working.confusions = confusions
            refresh_plate_state(
                working,
                working.car_plate,
                confusions=confusions,
                confirmed=False,
                preserve_confirmed=False,
            )
            assistant_reply = await self.generate_reply(model=model, state=working, changed=True)
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

        confirmation = await self.detect_confirmation(model=model, wav_bytes=wav_bytes, state=working)
        debug["confirmation"] = confirmation
        if confirmation:
            working.final_car_plate = working.car_plate
            refresh_plate_state(
                working,
                working.car_plate,
                confusions=[],
                confirmed=True,
                preserve_confirmed=False,
            )
            working.ack_sent = False
            assistant_reply = f"好的，已确认您的车牌号是{working.final_car_plate}。"
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

        edit_result = await self.update_car_plate(model=model, wav_bytes=wav_bytes, state=working)
        debug["edit_result"] = edit_result.to_dict()
        review_confirmed_positions = edit_result.review.confirmed_positions if edit_result.review else []
        if not edit_result.changed:
            if (edit_result.command and edit_result.command.action == "none") or review_confirmed_positions:
                confusions = await self.refresh_confusions_after_audio(
                    model=model,
                    wav_bytes=wav_bytes,
                    working=working,
                    confirmed_positions=review_confirmed_positions,
                )
                assistant_reply = (
                    await self.generate_reply(model=model, state=working, changed=False)
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
            refresh_plate_state(
                working,
                new_car_plate,
                confusions=[],
                confirmed=False,
                confirmed_positions=unique_positions([*edit_result.changed_positions, *review_confirmed_positions]),
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
        confusions = await self.refresh_confusions_after_audio(
            model=model,
            wav_bytes=wav_bytes,
            working=working,
            confirmed_positions=unique_positions([*edit_result.changed_positions, *review_confirmed_positions]),
        )
        assistant_reply = await self.generate_reply(model=model, state=working, changed=True)
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

    def build_invalid_plate_result(
        self,
        *,
        started: float,
        working: PlateAgentState,
        car_plate: str,
        debug: dict[str, Any],
        stage: str,
    ) -> PlateAgentResult:
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
        assistant_reply = reply_with_pending_confirmation(
            f"{EDIT_INVALID_REPLY}当前保留的车牌是{working.car_plate}。",
            working,
        )
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
        )

    async def refresh_confusions_after_audio(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        working: PlateAgentState,
        confirmed_positions: list[int] | None = None,
    ) -> list[PlateConfusion]:
        previous_confirmed_positions = {
            item.position for item in working.confirmed_chars if item.confirmed and item.position > 0
        }
        rule_confusions = detect_initial_confusions_by_rule(working.car_plate)
        log_node_output(
            "detect_confusions.rule_scan",
            {
                "source": "rule_before_model",
                "car_plate": working.car_plate,
                "confusions": [item.to_dict() for item in rule_confusions],
            },
        )
        if rule_confusions:
            confusions = await self.detect_confusions(
                model=model,
                wav_bytes=wav_bytes,
                car_plate=working.car_plate,
                state=working,
                rule_confusions=rule_confusions,
            )
        else:
            confusions = []

        confirmed_position_set = set(confirmed_positions or [])
        confirmed_position_set.update(previous_confirmed_positions)
        confirmed_position_set.update(resolved_confusion_positions(rule_confusions, confusions))
        if confirmed_position_set:
            confusions = [item for item in confusions if item.position not in confirmed_position_set]
        refresh_plate_state(
            working,
            working.car_plate,
            confusions=confusions,
            confirmed=False,
            confirmed_positions=sorted(confirmed_position_set),
            preserve_confirmed=True,
        )
        return confusions

    async def detect_plate_presence(self, *, model: str, wav_bytes: bytes) -> bool:
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=(
                "任务：判断用户语音中是否包含车牌号相关内容。"
                "只回答 true 或 false，不要输出其它内容。"
                "如果用户说了省份、字母、数字、车牌片段或完整车牌，回答 true。"
            ),
            max_tokens=8,
        )
        has_plate = parse_bool_text(result, default=False)
        log_node_output("detect_plate_presence", {"raw": result, "has_plate": has_plate})
        return has_plate

    async def extract_car_plate(self, *, model: str, wav_bytes: bytes) -> str:
        extraction_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=CAR_PLATE_EXTRACTION_PROMPT,
            max_tokens=512,
        )
        summarized_raw = extract_final_plate_from_text(extraction_result)
        summarized_plate = sanitize_extracted_plate_text(summarized_raw)
        parsed_plate = extract_plate_from_json_object(parse_json_object(extraction_result))
        extraction_plate = summarized_plate or parsed_plate
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

    async def detect_confusions(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        car_plate: str,
        state: PlateAgentState | None = None,
        rule_confusions: list[PlateConfusion] | None = None,
    ) -> list[PlateConfusion]:
        plate = clean_plate_text(car_plate)
        current_context = {
            "car_plate": plate,
            "plate_chars": [
                {"position": idx, "char": char}
                for idx, char in enumerate(plate, start=1)
            ],
            "plate_length": len(plate),
            "vehicle_type": (state.vehicle_type if state else vehicle_type_by_length(car_plate)),
            "need_confirm_chars": [item.to_dict() for item in (state.need_confirm_chars if state else [])],
            "confirmed_chars": [item.to_dict() for item in (state.confirmed_chars if state else [])],
            "assistant_reply": state.assistant_reply if state else "",
        }
        rule_confusions_context = json.dumps(
            [item.to_dict() for item in (rule_confusions or [])],
            ensure_ascii=False,
        )
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=(
                "任务：结合用户最新音频、当前识别车牌、当前候选混淆列表，判断哪些混淆点还需要二次确认。"
                f"当前识别车牌：{car_plate}。"
                f"当前识别状态：{json.dumps(current_context, ensure_ascii=False)}。"
                f"当前候选混淆列表：{rule_confusions_context}。"
                "只根据当前车牌号输出本轮 confusions，不要参考上一轮的 confusions、assistant_reply、历史纠错结果或之前的车牌。"
                "当前识别车牌中的字符位置已经给出，必须以这些位置上的字符为准，不要自行新增不存在的字符或位置。"
                "当前候选混淆列表来自固定规则扫描，表示当前新车牌里命中的潜在混淆字符。"
                "当前已经确认字符列表里的位置，本轮不要再输出为 confusions，除非用户最新音频明确说这个位置之前确认错了。"
                "不要参考上一轮旧的 confusions；不要把上一轮旧混淆点带入本轮结果。"
                "处理顺序必须如下："
                "第一步，只从当前候选混淆列表里判断哪些还需要用户二次确认。"
                "如果用户最新音频已经明确确认了某个候选混淆点，就不要输出这个混淆点。"
                "如果当前车牌号里没有这个候选混淆点，就不要凭历史结果或旧状态补出来。"
                "如果用户最新音频没有明确确认某个候选混淆点，或者音频里仍然听起来不确定，就输出这个混淆点。"
                "第二步，不要额外新增当前候选混淆列表之外的位置。"
                "只要输出混淆点，每一个字符位置都必须单独输出一条 confusions 记录。"
                "不能只返回第一个命中的位置，不能合并多个位置。"
                "当前候选混淆列表的规则来源如下："
                "1. 如果某个字符是 2 或 R，必须输出这个字符的 position，candidates 输出空数组，reason 直接说明第几位当前识别为什么，请用户确认是否正确；"
                "2. 如果某个字符是 1 或 E，必须输出这个字符的 position，candidates 输出空数组，reason 直接说明第几位当前识别为什么，请用户确认是否正确；"
                "3. 如果省份简称是 甘 或 赣，必须输出 position=1，candidates 输出空数组，reason 直接说明第1位当前识别为什么，请用户确认是否正确；"
                "4. 如果省份简称是 津 或 京，必须输出 position=1，candidates 输出空数组，reason 直接说明第1位当前识别为什么，请用户确认是否正确；"
                "5. 如果省份简称是 桂 或 贵，必须输出 position=1，candidates 输出空数组，reason 直接说明第1位当前识别为什么，请用户确认是否正确；"
                "6. 如果省份简称是 冀 或 吉，必须输出 position=1，candidates 输出空数组，reason 直接说明第1位当前识别为什么，请用户确认是否正确；"
                "7. 当前候选混淆列表之外的位置一律不要输出；0 和 临 不属于本轮需要二次确认的易混淆字符。"
                "只输出 JSON 对象，字段为 confusions。confusions 是数组，每一项包含 position、value、candidates、reason。"
                "position 是给后端使用的数字位置；reason 是给用户确认的自然语言，必须使用“第几位”说明位置。"
                "如果没有需要二次确认的易混淆字符，confusions 输出空数组。"
            ),
            max_tokens=256,
        )
        data = parse_json_object(result)
        raw_items = data.get("confusions")
        if not isinstance(raw_items, list):
            log_node_output("detect_confusions", {"raw": result, "car_plate": car_plate, "confusions": []})
            return []
        items = [PlateConfusion.from_value(item) for item in raw_items]
        parsed_items = [item for item in items if item is not None]
        allowed_positions = {item.position for item in (rule_confusions or [])}
        if allowed_positions:
            parsed_items = [item for item in parsed_items if item.position in allowed_positions]
        confirmed_positions = {item.position for item in (state.confirmed_chars if state else []) if item.confirmed}
        if confirmed_positions:
            parsed_items = [item for item in parsed_items if item.position not in confirmed_positions]
        confusions = with_relative_confusion_reasons(car_plate, parsed_items)
        log_node_output(
            "detect_confusions",
            {
                "raw": result,
                "car_plate": car_plate,
                "current_context": current_context,
                "rule_confusions": [item.to_dict() for item in (rule_confusions or [])],
                "confusions": [item.to_dict() for item in confusions],
            },
        )
        return confusions

    async def detect_confirmation(self, *, model: str, wav_bytes: bytes, state: PlateAgentState) -> bool:
        previous_ai_reply = (state.assistant_reply or "").strip()
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=(
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
            ),
            max_tokens=8,
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

        tentative_state = clone_state(state)
        tentative_plate = current_plate
        steps: list[dict[str, Any]] = []
        changed_positions: list[int] = []
        final_result: PlateEditResult | None = None

        for step_index in range(1, 4):
            command_result = await self.audio_call(
                model=model,
                wav_bytes=wav_bytes,
                prompt=build_plate_edit_command_prompt(
                    tentative_state,
                    current_plate=tentative_plate,
                    edit_steps=steps,
                ),
                max_tokens=256,
            )
            command = parse_plate_edit_command(command_result)
            log_node_output(
                "update_car_plate.react_action",
                {
                    "step": step_index,
                    "raw": command_result,
                    "previous_state": state.to_context(),
                    "tentative_state": tentative_state.to_context(),
                    "input_plate": tentative_plate,
                    "command": command.to_dict(),
                },
            )
            edit_result = apply_plate_edit_command(tentative_plate, command)
            edit_result.raw = command_result
            log_node_output(
                "update_car_plate.edit_result",
                {
                    "step": step_index,
                    "input_plate": tentative_plate,
                    "command": command.to_dict(),
                    "edit_result": edit_result.to_dict(),
                },
            )
            review = await self.review_plate_update(
                model=model,
                wav_bytes=wav_bytes,
                state=state,
                before_plate=tentative_plate,
                edit_result=edit_result,
                steps=steps,
            )
            edit_result.review = review
            step = {
                "step": step_index,
                "input_plate": tentative_plate,
                "raw": command_result,
                "command": command.to_dict(),
                "edit_result": edit_result.to_dict(),
                "review": review.to_dict(),
            }
            steps.append(step)
            log_node_output(
                "update_car_plate.react_step_done",
                {
                    "step": step_index,
                    "previous_state": state.to_context(),
                    "tentative_state": tentative_state.to_context(),
                    "command": command.to_dict(),
                    "edit_result": edit_result.to_dict(),
                    "review": review.to_dict(),
                },
            )

            if not review.valid_result:
                return PlateEditResult(
                    car_plate=current_plate,
                    changed=False,
                    command=command,
                    changed_positions=unique_positions([*changed_positions, *review.confirmed_positions]),
                    review=review,
                    steps=steps,
                    error=EDIT_UNCLEAR_REPLY,
                    raw=command_result,
                )

            changed_positions = unique_positions(
                [*changed_positions, *edit_result.changed_positions, *review.confirmed_positions]
            )

            if edit_result.changed:
                tentative_plate = edit_result.car_plate
                if not is_valid_plate_number(tentative_plate):
                    edit_result.changed_positions = changed_positions
                    edit_result.steps = steps
                    return edit_result
                refresh_plate_state(
                    tentative_state,
                    tentative_plate,
                    confusions=[],
                    confirmed=False,
                    confirmed_positions=changed_positions,
                    preserve_confirmed=True,
                )

            edit_result.car_plate = tentative_plate
            edit_result.changed_positions = changed_positions
            edit_result.steps = steps
            final_result = edit_result

            if not review.needs_more_edit:
                return edit_result

            if command.action in {"none", "unknown"} or not edit_result.changed:
                edit_result.error = edit_result.error or EDIT_UNCLEAR_REPLY
                return edit_result

        if final_result is not None:
            final_result.error = final_result.error or "这次修改包含多步内容，我先处理到目前能确定的位置，请您继续确认或说明剩余要改的部分。"
            return final_result
        return PlateEditResult(car_plate=current_plate, changed=False, error=EDIT_UNCLEAR_REPLY)

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
        after_plate = normalize_plate_text(edit_result.car_plate or before_plate)
        context = {
            "previous_state": state.to_context(),
            "before_plate": before_plate,
            "after_plate": after_plate,
            "command": edit_result.command.to_dict() if edit_result.command else None,
            "changed": edit_result.changed,
            "changed_positions": edit_result.changed_positions,
            "existing_steps": steps,
        }
        raw = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=build_plate_update_review_prompt(context),
            max_tokens=256,
        )
        data = parse_json_object(raw)
        review = PlateUpdateReview(
            confirmed_positions=parse_position_list(data.get("confirmed_positions"), plate_length(after_plate)),
            needs_more_edit=parse_json_bool(data.get("needs_more_edit"), default=False),
            valid_result=parse_json_bool(data.get("valid_result"), default=True),
            reason=str(data.get("reason") or "").strip(),
            raw=raw,
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
        formatted_plate = normalize_plate_format(sanitize_extracted_plate_text(car_plate))
        corrected_plate = formatted_plate
        retry_raw = ""
        retry_plate = ""
        if first_char_is_ascii_letter_or_digit(formatted_plate):
            retry_raw = await self.audio_call(
                model=model,
                wav_bytes=wav_bytes,
                prompt=f"""
任务：当前暂时的车牌识别结果第一位不是省份简称，请根据用户音频重新识别车牌号。
当前暂时识别结果：{formatted_plate}

请输出带省份简称的完整车牌号码。
车牌第一位必须是省份简称，例如：京、津、冀、晋、蒙、辽、吉、黑、沪、苏、浙、皖、闽、赣、鲁、豫、鄂、湘、粤、桂、琼、渝、川、贵、云、藏、陕、甘、青、宁、新。
只输出 JSON 对象，字段为 car_plate。
""",
                max_tokens=128,
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

    async def generate_reply(self, *, model: str, state: PlateAgentState, changed: bool) -> str:
        result, status_code = await self.model_client.complete_text(
            model=model or QWEN_MODEL,
            text=json.dumps(
                {
                    "car_plate": state.car_plate,
                    "plate_chars": [item.to_dict() for item in state.plate_chars],
                    "confirmed": state.is_confirmed,
                    "vehicle_type": state.vehicle_type,
                    "need_confirm_chars": [item.to_dict() for item in state.need_confirm_chars],
                    "confirmed_chars": [item.to_dict() for item in state.confirmed_chars],
                    "confusions": [item.to_dict() for item in with_relative_confusion_reasons(state.car_plate, state.confusions)],
                    "changed": changed,
                },
                ensure_ascii=False,
            ),
            prompt=(
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
            ),
            history=[],
            max_tokens=256,
            output_audio=False,
        )
        if status_code >= 400:
            reply = fallback_reply(state)
            log_node_output(
                "generate_reply",
                {
                    "status_code": status_code,
                    "raw": result.get("text"),
                    "assistant_reply": reply,
                    "fallback_used": True,
                    "state": state.to_context(),
                },
            )
            return reply
        parsed = parse_json_object(result.get("text"))
        parsed_reply = str(parsed.get("assistant_reply") or "").strip()
        fallback_used = not parsed_reply or contains_absolute_position_text(parsed_reply)
        reply = parsed_reply if not fallback_used else fallback_reply(state)
        log_node_output(
            "generate_reply",
            {
                "status_code": status_code,
                "raw": result.get("text"),
                "assistant_reply": reply,
                "fallback_used": fallback_used,
                "state": state.to_context(),
            },
        )
        return reply

    async def audio_call(self, *, model: str, wav_bytes: bytes, prompt: str, max_tokens: int) -> str:
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


def clone_state(state: PlateAgentState) -> PlateAgentState:
    confusions: list[PlateConfusion] = []
    for item in state.confusions:
        cloned = PlateConfusion.from_value(item.to_dict())
        if cloned is not None:
            confusions.append(cloned)
    plate_chars = clone_plate_char_states(state.plate_chars)
    need_confirm_chars = clone_plate_char_states(state.need_confirm_chars)
    confirmed_chars = clone_plate_char_states(state.confirmed_chars)
    cloned_state = PlateAgentState(
        car_plate=state.car_plate,
        plate_chars=plate_chars,
        confirmed=state.confirmed,
        need_confirm_chars=need_confirm_chars,
        confirmed_chars=confirmed_chars,
        vehicle_type=state.vehicle_type,
        confusions=confusions,
        final_car_plate=state.final_car_plate,
        assistant_reply=state.assistant_reply,
        ack_sent=state.ack_sent,
    )
    if cloned_state.car_plate and not cloned_state.plate_chars:
        refresh_plate_state(
            cloned_state,
            cloned_state.car_plate,
            confusions=confusions,
            confirmed=cloned_state.is_confirmed,
            preserve_confirmed=False,
        )
    return cloned_state


def clone_plate_char_states(items: list[PlateCharState]) -> list[PlateCharState]:
    cloned_items: list[PlateCharState] = []
    for item in items:
        cloned = PlateCharState.from_value(item.to_dict())
        if cloned is not None:
            cloned_items.append(cloned)
    return cloned_items


def build_output_json(
    *,
    task_status: str,
    car_plate: str,
    assistant_reply: str,
    final_car_plate: str = "",
) -> str:
    data: dict[str, Any] = {
        "task_status": task_status,
        "car_plate": clean_plate_text(car_plate),
        "assistant_reply": assistant_reply,
    }
    if final_car_plate:
        data["final_plate_number"] = clean_plate_text(final_car_plate)
    return json.dumps(data, ensure_ascii=False, indent=2)


def state_change_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = [
        "car_plate",
        "confirmed",
        "final_car_plate",
        "vehicle_type",
        "ack_sent",
        "plate_chars",
        "need_confirm_chars",
        "confirmed_chars",
        "confusions",
        "assistant_reply",
    ]
    changes: dict[str, dict[str, Any]] = {}
    for key in keys:
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value != after_value:
            changes[key] = {
                "before": before_value,
                "after": after_value,
            }
    return changes


def refresh_plate_state(
    state: PlateAgentState,
    car_plate: str,
    *,
    confusions: list[PlateConfusion] | None = None,
    confirmed: bool = False,
    confirmed_positions: list[int] | None = None,
    preserve_confirmed: bool = True,
) -> None:
    before_state = state.to_context()
    plate = normalize_plate_text(car_plate)
    normalized_confusions = with_relative_confusion_reasons(plate, confusions or [])
    confirmed_position_set = {position for position in (confirmed_positions or []) if position > 0}
    previous_confirmed = collect_confirmed_char_keys(state) if preserve_confirmed else set()
    confusion_by_position = {item.position: item for item in normalized_confusions if item.position > 0}

    plate_chars: list[PlateCharState] = []
    for position, value in enumerate(plate, start=1):
        confusion = confusion_by_position.get(position)
        needs_confirmation = confusion is not None
        is_confirmed = bool(confirmed) or (
            not needs_confirmation
            and ((position, value) in previous_confirmed or position in confirmed_position_set)
        )
        plate_chars.append(
            PlateCharState(
                position=position,
                value=value,
                confirmed=is_confirmed,
                needs_confirmation=needs_confirmation,
                candidates=(confusion.candidates if confusion else []),
                reason=(confusion.reason if confusion else ""),
            )
        )

    state.car_plate = plate
    state.plate_chars = plate_chars
    state.vehicle_type = vehicle_type_by_length(plate)
    state.confusions = normalized_confusions
    state.confirmed = bool(confirmed)
    if state.confirmed:
        state.final_car_plate = plate
    else:
        state.final_car_plate = ""
    state.need_confirm_chars = [item for item in plate_chars if item.needs_confirmation]
    state.confirmed_chars = [item for item in plate_chars if item.confirmed]
    log_node_output(
        "refresh_plate_state",
        {
            "action": "refresh_plate_state",
            "input": {
                "car_plate": car_plate,
                "normalized_car_plate": plate,
                "confirmed": confirmed,
                "confirmed_positions": confirmed_positions or [],
                "preserve_confirmed": preserve_confirmed,
                "confusions": [item.to_dict() for item in (confusions or [])],
            },
            "before_state": before_state,
            "state": state.to_context(),
        },
    )


def collect_confirmed_char_keys(state: PlateAgentState) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for item in [*state.confirmed_chars, *state.plate_chars]:
        if item.confirmed and item.position > 0 and item.value:
            keys.add((item.position, item.value))
    return keys


def resolved_confusion_positions(
    rule_confusions: list[PlateConfusion],
    remaining_confusions: list[PlateConfusion],
) -> set[int]:
    remaining_positions = {item.position for item in remaining_confusions}
    return {item.position for item in rule_confusions if item.position > 0 and item.position not in remaining_positions}


def parse_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(extract_json_candidate(raw, prefer_object=True))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


PLATE_JSON_KEY_ALIASES = {
    "carplate",
    "plate",
    "platenumber",
    "licenseplate",
    "licenseplatenumber",
    "finalplate",
    "finalcarplate",
    "finalplatenumber",
    "finalcarnumber",
    "车牌",
    "车牌号",
    "最终车牌",
    "最终车牌号",
}

UNKNOWN_PLATE_VALUES = {"", "?", "？", "UNKNOWN", "NONE", "NULL", "INVALID", "无", "未知", "不确定"}


def extract_plate_from_json_object(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key, value in data.items():
        normalized_key = normalize_json_key(key)
        if normalized_key in PLATE_JSON_KEY_ALIASES:
            plate = sanitize_extracted_plate_text(value)
            if plate:
                return plate
    return ""


def sanitize_extracted_plate_text(value: Any) -> str:
    if isinstance(value, dict):
        return extract_plate_from_json_object(value)
    if not isinstance(value, str):
        return ""

    raw = value.strip()
    if not raw:
        return ""

    for variant in jsonish_text_variants(raw):
        if looks_like_json_fragment(variant):
            nested_plate = extract_plate_from_json_object(parse_json_object(variant))
            if nested_plate:
                return nested_plate
            return ""

    converted = normalize_spoken_plate_chars(raw)
    compact = clean_plate_text(converted).upper()
    if compact in UNKNOWN_PLATE_VALUES:
        return ""

    chars: list[str] = []
    for char in compact:
        if char in PROVINCE_ABBREVIATIONS or char in SPECIAL_PLATE_TAIL_CHARS:
            chars.append(char)
        elif char.isascii() and char.isalnum():
            chars.append(char.upper())

    plate = "".join(chars)
    if not plate or plate in UNKNOWN_PLATE_VALUES:
        return ""
    if len(plate) < 2:
        return ""
    if normalize_json_key(plate) in PLATE_JSON_KEY_ALIASES or "CARPLATE" in plate:
        return ""
    return plate


def jsonish_text_variants(value: str) -> list[str]:
    variants = [value]
    if "\\" in value:
        variants.append(value.replace('\\"', '"').replace("\\", ""))
    return unique_text_values(variants)


def looks_like_json_fragment(value: str) -> bool:
    text = value.strip()
    return ("{" in text and "}" in text) or ("[" in text and "]" in text)


def normalize_json_key(value: Any) -> str:
    return re.sub(r"[\s_\\/:：\"'`{}【】\[\]()-]+", "", str(value or "")).lower()


def unique_text_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_final_plate_from_text(text: Any) -> str:
    raw = str(text or "")
    matches = re.findall(r"^最终车牌[:：]\s*(.+)$", raw, flags=re.MULTILINE)
    if not matches:
        return ""
    return matches[-1].strip()


def parse_bool_text(text: str, *, default: bool) -> bool:
    value = str(text or "").strip().lower()
    if "true" in value:
        return True
    if "false" in value:
        return False
    return default


def parse_json_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"true", "yes", "1", "是", "对", "需要"}:
        return True
    if raw in {"false", "no", "0", "否", "不", "不需要"}:
        return False
    return default


def parse_position_list(value: Any, max_position: int) -> list[int]:
    raw_items = value if isinstance(value, list) else [value]
    positions: list[int] = []
    for item in raw_items:
        position = parse_positive_int(item)
        if 1 <= position <= max_position:
            positions.append(position)
    return unique_positions(positions)


def unique_positions(values: list[int]) -> list[int]:
    seen: set[int] = set()
    positions: list[int] = []
    for value in values:
        try:
            position = int(value)
        except (TypeError, ValueError):
            continue
        if position <= 0 or position in seen:
            continue
        seen.add(position)
        positions.append(position)
    return positions


def parse_yes_no(text: str, *, default: bool) -> bool:
    value = str(text or "").strip().lower()
    if re.search(r"\byes\b", value) or "true" in value:
        return True
    if re.search(r"\bno\b", value) or "false" in value:
        return False
    return default


def clean_plate_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def normalize_plate_format(value: Any) -> str:
    return clean_plate_text(value).upper()


def first_char_is_ascii_letter_or_digit(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    return first.isascii() and first.isalnum()


def replace_leading_g_with_ji(value: str) -> str:
    plate = normalize_plate_format(value)
    if plate.startswith("G"):
        return "冀" + plate[1:]
    return plate


def normalize_plate_text(value: Any) -> str:
    return replace_leading_g_with_ji(value)


def plate_length(car_plate: str) -> int:
    return len(clean_plate_text(car_plate))


def vehicle_type_by_length(car_plate: str) -> str:
    length = plate_length(car_plate)
    if length == 7:
        return "fuel"
    if length == 8:
        return "new_energy"
    return "unknown"


def is_valid_plate_number(car_plate: str) -> bool:
    plate = normalize_plate_text(car_plate)
    if vehicle_type_by_length(plate) == "unknown":
        return False
    if not plate or plate[0] not in PROVINCE_ABBREVIATIONS:
        return False
    if len(plate) < 2 or not re.fullmatch(r"[A-Z]", plate[1]):
        return False
    for index, char in enumerate(plate[2:], start=3):
        is_tail = index == len(plate)
        if char in SPECIAL_PLATE_TAIL_CHARS:
            if not is_tail:
                return False
            continue
        if not re.fullmatch(r"[A-Z0-9]", char):
            return False
    return True


def detect_initial_confusions_by_rule(car_plate: str) -> list[PlateConfusion]:
    plate = clean_plate_text(car_plate)
    confusions: list[PlateConfusion] = []
    if plate and plate[0] in CONFUSION_PROVINCE_CHARS:
        confusions.append(build_confusion(position=1, value=plate[0]))
    for index, value in enumerate(plate, start=1):
        if value in CONFUSION_ALNUM_CHARS:
            confusions.append(build_confusion(position=index, value=value))
    return with_relative_confusion_reasons(plate, confusions)


def build_confusion(*, position: int, value: str) -> PlateConfusion:
    return PlateConfusion(
        position=position,
        value=value,
        reason=f"第{position}位当前识别为{describe_plate_char(value)}，请用户确认。",
    )


def with_relative_confusion_reasons(car_plate: str, confusions: list[PlateConfusion]) -> list[PlateConfusion]:
    plate = clean_plate_text(car_plate)
    normalized: list[PlateConfusion] = []
    for item in confusions:
        value = item.value
        index = item.position - 1
        if 0 <= index < len(plate):
            value = plate[index]
        candidates = list(item.candidates)
        if value and candidates and value not in candidates:
            candidates.insert(0, value)
        normalized.append(
            PlateConfusion(
                position=item.position,
                value=value,
                candidates=candidates,
                reason=relative_confusion_reason(plate, PlateConfusion(item.position, value, item.reason, candidates)),
            )
        )
    return normalized


def relative_confusion_reason(car_plate: str, item: PlateConfusion) -> str:
    plate = clean_plate_text(car_plate)
    index = item.position - 1
    value = item.value
    if 0 <= index < len(plate):
        value = value or plate[index]
    value_label = describe_plate_char(value)
    if 0 <= index < len(plate):
        return f"第{item.position}位当前识别为{value_label}，请用户确认是否正确。"
    return f"当前识别为{value_label}的位置，请用户确认是否正确。"


def describe_plate_char(value: str) -> str:
    labels = {
        "赣": "江西的赣",
        "甘": "甘肃的甘",
        "津": "天津的津",
        "京": "北京的京",
        "桂": "广西的桂",
        "贵": "贵州的贵",
        "冀": "河北的冀",
        "吉": "吉林的吉",
        "临": "临时车牌的临",
        "警": "警车的警",
        "学": "学车的学",
        "领": "领馆的领",
        "挂": "挂车的挂",
    }
    if value in labels:
        return labels[value]
    if value.isdigit():
        return f"数字 {value}"
    if re.fullmatch(r"[A-Za-z]", value):
        return f"字母 {value.upper()}"
    return value


def fallback_reply(state: PlateAgentState) -> str:
    plate = state.car_plate or "当前车牌"
    parts = [f"我这边暂时识别到的车牌号是{plate}。"]
    descriptions = pending_confirmation_descriptions(state)
    if descriptions:
        parts.append("请您再确认一下：" + "；".join(descriptions) + "。")
    else:
        parts.append("请您确认一下是否正确。")
    if state.vehicle_type == "new_energy":
        parts.append("另外这是新能源号牌吗？")
    return "".join(parts)


def reply_with_pending_confirmation(base_reply: str, state: PlateAgentState) -> str:
    reply = str(base_reply or "").strip()
    descriptions = pending_confirmation_descriptions(state)
    if descriptions:
        suffix = "当前仍需您确认：" + "；".join(descriptions) + "。"
    else:
        plate = state.car_plate or "当前保留的车牌"
        suffix = f"请您确认{plate}是否正确。"
    if not reply:
        return suffix
    if not reply.endswith(("。", "！", "？", ".", "!", "?")):
        reply += "。"
    return reply + suffix


def pending_confirmation_descriptions(state: PlateAgentState) -> list[str]:
    descriptions: list[str] = []
    if state.need_confirm_chars:
        for item in state.need_confirm_chars:
            reason = normalize_confirmation_reason(item.reason)
            if not reason:
                reason = f"第{item.position}位当前识别为{describe_plate_char(item.value)}，请您确认是否正确"
            descriptions.append(reason)
        return descriptions

    for item in with_relative_confusion_reasons(state.car_plate, state.confusions):
        reason = normalize_confirmation_reason(item.reason)
        if not reason:
            reason = f"第{item.position}位当前识别为{describe_plate_char(item.value)}，请您确认是否正确"
        descriptions.append(reason)
    return descriptions


def normalize_confirmation_reason(value: str) -> str:
    reason = str(value or "").strip().rstrip("。")
    if not reason:
        return ""
    reason = reason.replace("请用户确认是否正确", "请您确认是否正确")
    reason = reason.replace("请用户确认", "请您确认")
    return reason


def contains_absolute_position_text(value: str) -> bool:
    return bool(re.search(r"第\s*[0-9一二三四五六七八九十]+\s*位", str(value or "")))


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


plate_agent_service: PlateAgentService | None = None


def get_plate_agent_service(model_client: ChatModel) -> PlateAgentService:
    global plate_agent_service
    if plate_agent_service is None or plate_agent_service.model_client is not model_client:
        plate_agent_service = PlateAgentService(model_client)
    return plate_agent_service
