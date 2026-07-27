from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from realtime_audio_demo.services.plate_agent_types import PlateAgentState


ACK_SCHEDULE_SECONDS = (0.0, 1.0, 3.0, 5.0)

ACK_MESSAGES_BY_SCENE = {
    "initial": (
        "语音已收到，正在判断是否包含车牌信息。",
        "正在识别车牌号码内容。",
        "还在结合车牌规则和发音做确认。",
        "识别还在处理，请稍等。",
    ),
    "update": (
        "语音已收到，正在判断您是在确认还是修改。",
        "正在结合当前车牌处理您的这次回复。",
        "还在复核修改结果和需要确认的位置。",
        "处理还在继续，请稍等。",
    ),
}


def ack_scene_for_state(state: PlateAgentState) -> str:
    return "update" if state.has_car_plate else "initial"


def ack_schedule_for_state(state: PlateAgentState) -> list[tuple[float, str]]:
    messages = ACK_MESSAGES_BY_SCENE[ack_scene_for_state(state)]
    return list(zip(ACK_SCHEDULE_SECONDS, messages))


async def emit_scheduled_acks(
    *,
    state: PlateAgentState,
    on_ack: Callable[[str], Awaitable[None]],
    is_result_ready: Callable[[], bool],
) -> None:
    """Send the scheduled bridge text only when the agent result is still not ready."""
    started = time.perf_counter()
    for delay_seconds, message in ack_schedule_for_state(state):
        sleep_seconds = delay_seconds - (time.perf_counter() - started)
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)
        if is_result_ready():
            return
        await on_ack(message)
