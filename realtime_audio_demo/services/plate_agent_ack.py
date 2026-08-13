from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from realtime_audio_demo.services.plate_agent_messages import ACK_MESSAGES_BY_SCENE
from realtime_audio_demo.services.plate_agent_types import PlateAgentState


ACK_SCHEDULE_SECONDS = (0.0,)


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
