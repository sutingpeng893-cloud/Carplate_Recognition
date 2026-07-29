from __future__ import annotations

import time
from typing import Any

from realtime_audio_demo.config import QWEN_MODEL
from realtime_audio_demo.services.interfaces import ChatModel
from realtime_audio_demo.services.plate_agent_ack import ack_schedule_for_state
from realtime_audio_demo.services.plate_agent_logging import (
    CURRENT_SESSION_ID,
    CURRENT_TURN_BEFORE_STATE,
    log_agent_line,
    log_node_output,
    log_session_event,
    logger,
)
from realtime_audio_demo.services.plate_agent_messages import (
    EDIT_UNCLEAR_REPLY,
    INVALID_PLATE_REPLY,
    NO_PLATE_REPLY,
    build_confirmed_reply,
    build_edit_invalid_reply,
    build_fixed_reply,
)
from realtime_audio_demo.services.plate_agent_parsing import elapsed_ms
from realtime_audio_demo.services.plate_agent_prompts import (
    build_plate_agent_system_prompt,
    build_plate_agent_turn_instruction,
)
from realtime_audio_demo.services.plate_agent_response import build_output_json, reply_with_pending_confirmation
from realtime_audio_demo.services.plate_agent_state import clone_state
from realtime_audio_demo.services.plate_agent_tooling import (
    PlateAgentPlan,
    PlateToolExecutor,
    build_tool_result_history_message,
    parse_agent_plan,
)
from realtime_audio_demo.services.plate_agent_types import (
    PlateAgentResult,
    PlateAgentState,
)


MAX_AGENT_TOOL_ROUNDS = 4


class PlateAgentService:
    """车牌语音 Agent 主服务。

    主流程只负责搭建 Agent 循环：
    1. 注入后端维护的状态栏。
    2. 让模型输出 tool_calls 或 finish。
    3. 后端执行工具并把结果回填到下一轮状态栏。
    4. 根据最终状态生成接口需要的 JSON 和播报话术。
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
        started = time.perf_counter()
        working = clone_state(state)
        if turn_summaries is not None:
            working.turn_summaries = list(turn_summaries)[-6:]

        before_state = working.to_context()
        CURRENT_SESSION_ID.set(str(session_id or "").strip())
        CURRENT_TURN_BEFORE_STATE.set(before_state)
        log_node_output(
            "handle_audio_turn.start",
            {
                "action": "start_agent_audio_turn",
                "model": model or QWEN_MODEL,
                "wav_bytes": len(wav_bytes),
                "before_state": before_state,
                "state": working.to_context(),
            },
        )
        await self.emit_compat_ack_if_needed(on_ack=on_ack, state=working)

        log_agent_line(
            "Agent 回合开始",
            阶段="多轮确认或纠错" if working.has_car_plate else "首轮识别",
            当前状态=working.to_context(),
            说明="后续由模型根据状态栏自主选择工具调用。",
        )

        executor = PlateToolExecutor(working)
        tool_results: list[dict[str, Any]] = []
        agent_history: list[dict[str, Any]] = []
        plans: list[dict[str, Any]] = []
        last_plan = PlateAgentPlan(raw="")

        for iteration in range(1, MAX_AGENT_TOOL_ROUNDS + 1):
            raw_plan = await self.plan_next_action(
                model=model,
                wav_bytes=wav_bytes,
                state=working,
                session_id=session_id,
                iteration=iteration,
                tool_results=tool_results,
                agent_history=agent_history,
            )
            last_plan = parse_agent_plan(raw_plan)
            plans.append(last_plan.to_dict())
            agent_history.append({"role": "assistant", "content": raw_plan})
            log_session_event(
                "llm_response",
                iteration=iteration,
                raw_output=raw_plan,
                parsed_plan=last_plan.to_dict(),
                agent_history=compact_agent_history(agent_history),
                state=working.to_context(),
            )
            log_agent_line(
                "Agent 规划：模型推理是什么",
                第几轮=iteration,
                模型输出=raw_plan,
            )
            log_agent_line(
                "Agent 规划：tool_calls 是什么",
                第几轮=iteration,
                thought=last_plan.thought,
                tool_calls=[item.to_dict() for item in last_plan.tool_calls],
                finish=last_plan.finish,
            )
            log_node_output(
                "agent.plan",
                {
                    "iteration": iteration,
                    "raw": raw_plan,
                    "plan": last_plan.to_dict(),
                    "state": working.to_context(),
                    "tool_results": tool_results,
                    "agent_history": compact_agent_history(agent_history),
                },
            )

            if last_plan.tool_calls:
                current_tool_results = executor.execute_all(last_plan.tool_calls)
                tool_results.extend(current_tool_results)
                agent_history.append(
                    {
                        "role": "user",
                        "content": build_tool_result_history_message(current_tool_results),
                    }
                )
                continue

            if last_plan.finish:
                break

            log_agent_line(
                "Agent 规划为空",
                第几轮=iteration,
                说明="模型没有输出可执行工具，也没有输出 finish，结束循环并走兜底回复。",
            )
            break

        return self.build_final_result(
            started=started,
            before_state=before_state,
            working=working,
            last_plan=last_plan,
            plans=plans,
            tool_results=tool_results,
            agent_history=agent_history,
        )

    async def emit_compat_ack_if_needed(self, *, on_ack: Any, state: PlateAgentState) -> None:
        """兼容旧调用方：如果还传 on_ack，只发送第一条衔接语。"""

        if on_ack is None:
            return
        try:
            _, ack_text = ack_schedule_for_state(state)[0]
            await on_ack(ack_text)
            log_node_output(
                "handle_audio_turn.compat_on_ack",
                {
                    "action": "emit_compat_ack",
                    "ack_text": ack_text,
                    "state": state.to_context(),
                },
            )
        except Exception as exc:
            logger.warning("plate_agent compat on_ack failed: %s", exc)

    async def plan_next_action(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
        session_id: str,
        iteration: int,
        tool_results: list[dict[str, Any]],
        agent_history: list[dict[str, Any]],
    ) -> str:
        """调用音频模型，让模型根据状态栏输出 tool_calls 或 finish。"""

        system_prompt = build_plate_agent_system_prompt()
        turn_instruction = build_plate_agent_turn_instruction(
            state=state,
            session_id=session_id,
            iteration=iteration,
            max_iterations=MAX_AGENT_TOOL_ROUNDS,
            tool_results=tool_results,
        )
        log_session_event(
            "llm_request",
            iteration=iteration,
            model=model or QWEN_MODEL,
            input_type="audio" if iteration == 1 else "text",
            audio_bytes=len(wav_bytes) if iteration == 1 else 0,
            turn_instruction=turn_instruction,
            agent_history=compact_agent_history(agent_history),
            state=state.to_context(),
            previous_tool_results=tool_results,
        )
        if iteration == 1:
            completion = await self.model_client.complete_audio(
                model=model or QWEN_MODEL,
                wav_bytes=wav_bytes,
                prompt=system_prompt,
                history=agent_history,
                max_tokens=1024,
                turn_instruction=turn_instruction,
            )
            if completion.raw_response and completion.raw_response.get("status_code"):
                raise RuntimeError(str(completion.raw_response.get("message") or "upstream audio request failed"))
            return completion.text or ""

        response, status_code = await self.model_client.complete_text(
            model=model or QWEN_MODEL,
            text=turn_instruction,
            prompt=system_prompt,
            history=agent_history,
            max_tokens=1024,
            output_audio=False,
        )
        if status_code >= 400:
            raise RuntimeError(str(response.get("message") or "upstream text request failed"))
        return str(response.get("text") or "")

    def build_final_result(
        self,
        *,
        started: float,
        before_state: dict[str, Any],
        working: PlateAgentState,
        last_plan: PlateAgentPlan,
        plans: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        agent_history: list[dict[str, Any]],
    ) -> PlateAgentResult:
        """把 Agent 最终状态转成前端接口仍然兼容的输出格式。"""

        finish_status = normalize_finish_status(last_plan.finish.get("task_status"))
        reply_scene = str(last_plan.finish.get("reply_scene") or "").strip()
        failed_result = last_failed_tool_result(tool_results)
        state_changed = agent_changed_state(before_state, working, tool_results)

        if finish_status == "confirmed" and working.final_car_plate:
            assistant_reply = build_confirmed_reply(working.final_car_plate)
            task_status = "confirmed"
        elif not working.has_car_plate:
            task_status, assistant_reply = self.reply_without_plate(
                finish_status=finish_status,
                failed_result=failed_result,
            )
        elif finish_status == "invalid" and failed_result is not None:
            task_status = "need_confirmation"
            assistant_reply = build_edit_invalid_reply(working)
        elif (not last_plan.finish or finish_status == "unclear" or failed_result is not None) and not state_changed:
            task_status = "need_confirmation"
            assistant_reply = self.reply_for_failed_or_unclear_edit(working, failed_result)
        else:
            task_status = "need_confirmation"
            assistant_reply = self.reply_for_current_state(
                working,
                before_state=before_state,
                reply_scene=reply_scene,
            )

        working.assistant_reply = assistant_reply
        working.ack_sent = False
        output = build_output_json(
            task_status=task_status,
            car_plate=working.car_plate,
            assistant_reply=assistant_reply,
            final_car_plate=working.final_car_plate if task_status == "confirmed" else "",
        )
        latency_ms = elapsed_ms(started)
        debug = {
            "agent_plans": plans,
            "tool_results": tool_results,
            "agent_history": compact_agent_history(agent_history),
            "finish": last_plan.finish,
            "car_plate": working.car_plate,
            "vehicle_type": working.vehicle_type,
        }
        log_node_output(
            "turn_result",
            {
                "stage": task_status,
                "text": output,
                "speech_text": assistant_reply,
                "state": working.to_context(),
                "latency_ms": latency_ms,
                "agent_plans": plans,
                "tool_results": tool_results,
            },
        )
        log_session_event(
            "final_response",
            task_status=task_status,
            response_text=output,
            speech_text=assistant_reply,
            latency_ms=latency_ms,
            state=working.to_context(),
            agent_plans=plans,
            tool_results=tool_results,
            agent_history=compact_agent_history(agent_history),
        )
        return PlateAgentResult(
            text=output,
            history_text=output,
            speech_text=assistant_reply,
            state=working,
            latency_ms=latency_ms,
            debug=debug,
        )

    def reply_without_plate(
        self,
        *,
        finish_status: str,
        failed_result: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if finish_status == "invalid" or failed_tool_name(failed_result) == "set_plate":
            return "invalid", INVALID_PLATE_REPLY
        return "need_more_info", NO_PLATE_REPLY

    def reply_for_failed_or_unclear_edit(
        self,
        working: PlateAgentState,
        failed_result: dict[str, Any] | None,
    ) -> str:
        message = str((failed_result or {}).get("message") or "").strip()
        if message and "格式不合法" in message:
            return build_edit_invalid_reply(working)
        return reply_with_pending_confirmation(message or EDIT_UNCLEAR_REPLY, working)

    def reply_for_current_state(
        self,
        working: PlateAgentState,
        *,
        before_state: dict[str, Any],
        reply_scene: str,
    ) -> str:
        if reply_scene in {"initial_success", "update_success", "partial_confirmation"}:
            return build_fixed_reply(working, changed=reply_scene == "update_success", scene=reply_scene)
        before_plate = str(before_state.get("car_plate") or "").strip()
        if not before_plate and working.car_plate:
            return build_fixed_reply(working, changed=True, scene="initial_success")
        if before_plate and before_plate != working.car_plate:
            return build_fixed_reply(working, changed=True, scene="update_success")
        return build_fixed_reply(working, changed=False, scene="partial_confirmation")


def normalize_finish_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "need_more_info": "need_more_info",
        "more_info": "need_more_info",
        "need_confirmation": "need_confirmation",
        "confirmation": "need_confirmation",
        "confirmed": "confirmed",
        "success": "confirmed",
        "invalid": "invalid",
        "unclear": "unclear",
        "unknown": "unclear",
    }
    return aliases.get(raw, "")


def last_failed_tool_result(tool_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(tool_results):
        if item.get("success") is False:
            return item
    return None


def agent_changed_state(
    before_state: dict[str, Any],
    working: PlateAgentState,
    tool_results: list[dict[str, Any]],
) -> bool:
    current_state = working.to_context()
    for key in ("car_plate", "confirmed", "final_car_plate", "need_confirm_chars", "confirmed_chars"):
        if before_state.get(key) != current_state.get(key):
            return True
    return False


def failed_tool_name(result: dict[str, Any] | None) -> str:
    return str((result or {}).get("name") or "").strip()


def compact_agent_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    for item in history[-10:]:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if not role or not content:
            continue
        compacted.append({"role": role, "content": content[:3000]})
    return compacted


plate_agent_service: PlateAgentService | None = None


def get_plate_agent_service(model_client: ChatModel) -> PlateAgentService:
    """返回单例 PlateAgentService，避免每个请求重复创建服务对象。"""

    global plate_agent_service
    if plate_agent_service is None or plate_agent_service.model_client is not model_client:
        plate_agent_service = PlateAgentService(model_client)
    return plate_agent_service
