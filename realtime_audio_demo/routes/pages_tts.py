from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from realtime_audio_demo.config import (
    EASYTURN_ACK_TEXT,
    EASYTURN_API_URL,
    EASYTURN_ENABLED,
    MAX_HISTORY_TURNS,
    PAIC_TTS_API_URL,
    PAIC_TTS_AUDIO_PATH,
    PAIC_TTS_EMOTION,
    PAIC_TTS_HEAD_SILENCE,
    PAIC_TTS_MODEL_FIELD,
    PAIC_TTS_PITCH,
    PAIC_TTS_SAMPLE_RATE,
    PAIC_TTS_SOUND_LIBRARY_ID,
    PAIC_TTS_SOUND_LIBRARY_ID_FIELD,
    PAIC_TTS_SPEED,
    PAIC_TTS_TEXT_FIELD,
    PAIC_TTS_TAIL_SILENCE,
    PAIC_TTS_VOICE,
    PAIC_TTS_VOICE_FIELD,
    PAIC_TTS_VOLUME,
    PREFILL_MODE,
    QWEN_API_BASE,
    QWEN_MODALITIES,
    QWEN_MODEL,
    QWEN_SPEAKER,
    SESSION_TTL,
    SILERO_VAD_ENABLED,
    SILERO_VAD_MAX_SPEECH_MS,
    SILERO_VAD_MIN_SILENCE_MS,
    SILERO_VAD_MIN_SPEECH_MS,
    SILERO_VAD_PRELOAD,
    SILERO_VAD_THRESHOLD,
    STATIC_DIR,
    STREAM_FINAL_OUTPUT,
    SYSTEM_PROMPT_PATH,
    TARGET_SAMPLE_RATE,
    TTS_API_BASE,
    TTS_BACKEND,
    TTS_MODEL,
    TTS_RATE,
    TTS_SAMPLE_RATE,
    TTS_STREAM_CHUNK_MS,
    TTS_STREAM_RESPONSE_FORMAT,
    TTS_VOICE,
    resolved_provider,
)
from realtime_audio_demo.services.speech import (
    diagnose_paic_connectivity,
    paic_runtime_snapshot,
    speech_backend_status,
    speech_streaming_supported,
)
from realtime_audio_demo.services.silero_vad import silero_vad_status


router = APIRouter()
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
}


@router.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/chatbox")


@router.get("/chatbox")
async def chatbox() -> FileResponse:
    return FileResponse(STATIC_DIR / "chatbox.html", headers=NO_CACHE_HEADERS)


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "qwen_api_base": QWEN_API_BASE,
            "model": QWEN_MODEL,
            "prefill_mode": PREFILL_MODE,
            "provider": resolved_provider(),
            "modalities": QWEN_MODALITIES,
            "speaker": QWEN_SPEAKER or None,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "max_history_turns": MAX_HISTORY_TURNS,
            "stream_final_output": STREAM_FINAL_OUTPUT,
            "silero_vad": {
                "enabled": SILERO_VAD_ENABLED,
                "preload": SILERO_VAD_PRELOAD,
                "threshold": SILERO_VAD_THRESHOLD,
                "min_speech_ms": SILERO_VAD_MIN_SPEECH_MS,
                "min_silence_ms": SILERO_VAD_MIN_SILENCE_MS,
                "max_speech_ms": SILERO_VAD_MAX_SPEECH_MS,
                "status": silero_vad_status(),
                "startup": getattr(request.app.state, "silero_vad", None),
            },
            "easy_turn": {
                "enabled": EASYTURN_ENABLED,
                "api_configured": bool(EASYTURN_API_URL),
                "ack_text": EASYTURN_ACK_TEXT,
            },
            "tts": {
                "backend": TTS_BACKEND,
                "api_base": TTS_API_BASE or None,
                "model": TTS_MODEL,
                "voice": TTS_VOICE or None,
                "rate": TTS_RATE,
                "sample_rate": TTS_SAMPLE_RATE,
                "stream_chunk_ms": TTS_STREAM_CHUNK_MS,
                "stream_response_format": TTS_STREAM_RESPONSE_FORMAT,
                "streaming_supported": speech_streaming_supported(),
                "paic": {
                    "api_url": PAIC_TTS_API_URL or None,
                    "voice": PAIC_TTS_VOICE or None,
                    "text_field": PAIC_TTS_TEXT_FIELD,
                    "voice_field": PAIC_TTS_VOICE_FIELD or None,
                    "model_field": PAIC_TTS_MODEL_FIELD or None,
                    "audio_path": PAIC_TTS_AUDIO_PATH,
                    "sound_library_id_field": PAIC_TTS_SOUND_LIBRARY_ID_FIELD,
                    "sound_library_id": PAIC_TTS_SOUND_LIBRARY_ID or None,
                    "sample_rate": PAIC_TTS_SAMPLE_RATE,
                    "pitch": PAIC_TTS_PITCH,
                    "speed": PAIC_TTS_SPEED,
                    "volume": PAIC_TTS_VOLUME,
                    "emotion": PAIC_TTS_EMOTION,
                    "head_silence": PAIC_TTS_HEAD_SILENCE,
                    "tail_silence": PAIC_TTS_TAIL_SILENCE,
                    "runtime": paic_runtime_snapshot(),
                },
                "resolved": speech_backend_status(),
            },
            "system_prompt_path": str(SYSTEM_PROMPT_PATH),
            "session_ttl": SESSION_TTL,
        }
    )


@router.get("/health/tts/paic")
async def health_tts_paic() -> JSONResponse:
    return JSONResponse(await diagnose_paic_connectivity())
