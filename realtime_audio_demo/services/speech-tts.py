import asyncio
import base64
import json
import logging
import os
import re
import socket
import ssl
import time
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx

from realtime_audio_demo.config import (
    FINAL_MAX_TOKENS,
    PAIC_TTS_API_URL,
    PAIC_TTS_AUDIO_PATH,
    PAIC_TTS_CONNECT_TIMEOUT,
    PAIC_TTS_DIAG_TIMEOUT,
    PAIC_TTS_EMOTION,
    PAIC_TTS_EXTRA_BODY_JSON,
    PAIC_TTS_HEAD_SILENCE,
    PAIC_TTS_HEADERS_JSON,
    PAIC_TTS_MODEL_FIELD,
    PAIC_TTS_PITCH,
    PAIC_TTS_SAMPLE_RATE,
    PAIC_TTS_SOUND_LIBRARY_ID,
    PAIC_TTS_SOUND_LIBRARY_ID_FIELD,
    PAIC_TTS_SPEED,
    PAIC_TTS_STREAM_TIMEOUT,
    PAIC_TTS_TAIL_SILENCE,
    PAIC_TTS_TEXT_FIELD,
    PAIC_TTS_TIMEOUT,
    PAIC_TTS_TRUST_ENV,
    PAIC_TTS_VOICE,
    PAIC_TTS_VOICE_FIELD,
    PAIC_TTS_VOLUME,
    TTS_API_BASE,
    TTS_BACKEND,
    TTS_MODEL,
    TTS_RATE,
    TTS_REF_AUDIO,
    TTS_REF_TEXT,
    TTS_RESPONSE_FORMAT,
    TTS_SAMPLE_RATE,
    TTS_STREAM_CHUNK_MS,
    TTS_STREAM_FORMAT,
    TTS_STREAM_RESPONSE_FORMAT,
    TTS_TASK_TYPE,
    TTS_VOICE,
)
from realtime_audio_demo.services.interfaces import ChatModel, SpeechSynthesizer
from realtime_audio_demo.services.local_tts import LocalTtsError, WindowsSapiTtsBackend
from realtime_audio_demo.services.model_gateway import model_gateway
from realtime_audio_demo.utils.audio import wav_bytes_to_pcm16le_bytes

logger = logging.getLogger("uvicorn.error")
_speech_backend_status: dict[str, object] = {}
_PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


def build_openai_tts_payload(*, model: str, text: str, response_format: str, stream: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "input": text,
        "response_format": response_format,
    }
    if TTS_VOICE:
        payload["voice"] = TTS_VOICE
    if TTS_TASK_TYPE:
        payload["task_type"] = TTS_TASK_TYPE
    if TTS_REF_AUDIO:
        payload["ref_audio"] = TTS_REF_AUDIO
        payload.setdefault("task_type", "Base")
    if TTS_REF_TEXT:
        payload["ref_text"] = TTS_REF_TEXT
    if stream:
        payload["stream"] = True
        payload["stream_format"] = TTS_STREAM_FORMAT
    return payload


def resolve_qwen_talker_model(request_model: str | None = None) -> str:
    configured = (TTS_MODEL or "").strip()
    if configured:
        return configured
    fallback = (request_model or "").strip()
    if fallback:
        return fallback
    return "qwen3-omni"


def _parse_json_mapping(raw: str, *, label: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("%s is not valid JSON: %s", label, exc)
        return {}
    if isinstance(value, dict):
        return value
    logger.warning("%s must be a JSON object", label)
    return {}


_PATH_TOKEN_RE = re.compile(r"([^[.\]]+)|\[(\d+)\]")


def _split_path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for chunk in [item for item in path.split(".") if item]:
        matches = list(_PATH_TOKEN_RE.finditer(chunk))
        if not matches:
            tokens.append(chunk)
            continue
        consumed = "".join(match.group(0) for match in matches)
        if consumed != chunk:
            tokens.append(chunk)
            continue
        for match in matches:
            key = match.group(1)
            index = match.group(2)
            if key is not None:
                tokens.append(key)
            elif index is not None:
                tokens.append(int(index))
    return tokens


def _dig_value(data: Any, path: str) -> Any:
    current = data
    for part in _split_path_tokens(path):
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return None
            current = current[part]
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _iter_audio_candidates(data: Any):
    if isinstance(data, dict):
        for key in ("audio_base64", "audio", "audioData", "data", "result"):
            if key in data:
                value = data[key]
                if isinstance(value, (str, bytes)):
                    yield value
                elif isinstance(value, (dict, list)):
                    yield from _iter_audio_candidates(value)
        for value in data.values():
            yield from _iter_audio_candidates(value)
        return
    if isinstance(data, list):
        for item in data:
            yield from _iter_audio_candidates(item)
        return
    if isinstance(data, (str, bytes)):
        yield data


def _as_audio_data_url(audio_bytes: bytes, fmt: str) -> str:
    return f"data:audio/{fmt};base64,{base64.b64encode(audio_bytes).decode('ascii')}"


def _decode_audio_text_value(value: str) -> bytes | None:
    text = value.strip()
    if not text:
        return None
    if text.startswith("data:audio/") and "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        return None


def _extract_audio_data_url_from_json(data: Any, *, response_format: str) -> str | None:
    candidates: list[Any] = []
    if PAIC_TTS_AUDIO_PATH:
        candidate = _dig_value(data, PAIC_TTS_AUDIO_PATH)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(list(_iter_audio_candidates(data)))
    for candidate in candidates:
        if isinstance(candidate, str):
            if candidate.startswith("data:audio/"):
                return candidate
            if candidate.startswith("http://") or candidate.startswith("https://"):
                return candidate
            audio_bytes = _decode_audio_text_value(candidate)
            if audio_bytes:
                return _as_audio_data_url(audio_bytes, response_format)
        if isinstance(candidate, bytes):
            return _as_audio_data_url(candidate, response_format)
    return None


def _extract_audio_bytes_from_json(data: Any) -> bytes | None:
    candidates: list[Any] = []
    if PAIC_TTS_AUDIO_PATH:
        candidate = _dig_value(data, PAIC_TTS_AUDIO_PATH)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(list(_iter_audio_candidates(data)))
    for candidate in candidates:
        if isinstance(candidate, bytes):
            return candidate
        if isinstance(candidate, str):
            audio_bytes = _decode_audio_text_value(candidate)
            if audio_bytes:
                return audio_bytes
    return None


def _preview_json_shape(data: Any, *, depth: int = 2, max_items: int = 6) -> Any:
    if depth <= 0:
        return type(data).__name__
    if isinstance(data, dict):
        preview: dict[str, Any] = {}
        for index, (key, value) in enumerate(data.items()):
            if index >= max_items:
                preview["..."] = f"+{len(data) - max_items} more keys"
                break
            preview[str(key)] = _preview_json_shape(value, depth=depth - 1, max_items=max_items)
        return preview
    if isinstance(data, list):
        preview = [_preview_json_shape(item, depth=depth - 1, max_items=max_items) for item in data[:max_items]]
        if len(data) > max_items:
            preview.append(f"... +{len(data) - max_items} more items")
        return preview
    if isinstance(data, str):
        text = data.strip()
        if text.startswith("data:audio/"):
            return f"str(len={len(data)}, data_url)"
        if text.startswith("http://") or text.startswith("https://"):
            return f"str(len={len(data)}, url)"
        return f"str(len={len(data)})"
    if isinstance(data, bytes):
        return f"bytes(len={len(data)})"
    return type(data).__name__


def _redact_url(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    hostname = parts.hostname or ""
    if not hostname:
        return text
    port = f":{parts.port}" if parts.port else ""
    userinfo = ""
    if parts.username:
        userinfo = parts.username
        if parts.password is not None:
            userinfo += ":***"
        userinfo += "@"
    return urlunsplit((parts.scheme, f"{userinfo}{hostname}{port}", parts.path, "", ""))


def _proxy_env_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            snapshot[key] = _redact_url(value)
    return snapshot


def _paic_target() -> tuple[str, str, int]:
    parts = urlsplit(PAIC_TTS_API_URL)
    scheme = (parts.scheme or "https").lower()
    port = parts.port or (443 if scheme == "https" else 80)
    return scheme, parts.hostname or "", port


def _paic_http_timeout(total_timeout: float) -> httpx.Timeout:
    connect_timeout = PAIC_TTS_CONNECT_TIMEOUT
    if total_timeout > 0:
        connect_timeout = min(connect_timeout, total_timeout)
    return httpx.Timeout(total_timeout, connect=connect_timeout)


def paic_runtime_snapshot() -> dict[str, Any]:
    scheme, host, port = _paic_target()
    header_names = sorted(_parse_json_mapping(PAIC_TTS_HEADERS_JSON, label="PAIC_TTS_HEADERS_JSON").keys())
    return {
        "api_url": _redact_url(PAIC_TTS_API_URL),
        "scheme": scheme,
        "host": host or None,
        "port": port,
        "connect_timeout": PAIC_TTS_CONNECT_TIMEOUT,
        "request_timeout": PAIC_TTS_TIMEOUT,
        "stream_timeout": PAIC_TTS_STREAM_TIMEOUT,
        "diag_timeout": PAIC_TTS_DIAG_TIMEOUT,
        "trust_env": PAIC_TTS_TRUST_ENV,
        "proxy_env": _proxy_env_snapshot(),
        "headers_configured": bool(header_names),
        "header_names": header_names,
    }


def _probe_error(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


async def _run_http_probe(*, url: str, trust_env: bool, timeout_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=_paic_http_timeout(timeout_seconds),
            trust_env=trust_env,
            follow_redirects=False,
        ) as client:
            response = await client.post(url, json={}, headers={"Content-Type": "application/json"})
        return {
            "ok": True,
            "trust_env": trust_env,
            "reachable": True,
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        result = _probe_error(exc)
        result.update(
            {
                "trust_env": trust_env,
                "reachable": False,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }
        )
        return result


def _summarize_paic_probe(report: dict[str, Any]) -> str | None:
    dns = report.get("dns") if isinstance(report.get("dns"), dict) else {}
    tcp = report.get("tcp") if isinstance(report.get("tcp"), dict) else {}
    tls = report.get("tls") if isinstance(report.get("tls"), dict) else {}
    http_probe = report.get("http_probe") if isinstance(report.get("http_probe"), dict) else {}
    http_probe_no_env = (
        report.get("http_probe_without_env") if isinstance(report.get("http_probe_without_env"), dict) else {}
    )
    if dns and not dns.get("ok"):
        return "DNS failed on the server. Check private DNS, resolver reachability, or /etc/resolv.conf."
    if tcp and not tcp.get("ok"):
        return "DNS resolves, but direct TCP to the PAIC host:port failed. Check server egress firewall, route table, or security policy for outbound 443."
    if report.get("scheme") == "https" and tls and not tls.get("ok"):
        return "TCP connects, but the TLS handshake failed. Check HTTPS inspection, internal CA trust, or certificate policy."
    if http_probe.get("reachable"):
        status_code = http_probe.get("status_code")
        return f"HTTP connectivity is working because the PAIC endpoint returned status {status_code}. Next check cookies, auth, and request body."
    if http_probe_no_env.get("reachable") and not http_probe.get("reachable"):
        return "Direct HTTP without environment proxies works while the current runtime does not. Likely HTTPS_PROXY/ALL_PROXY/NO_PROXY mismatch; try PAIC_TTS_TRUST_ENV=0 or add the host to NO_PROXY."
    if http_probe.get("error_type") == "ConnectTimeout":
        return "The request timed out during connect. That usually means the server cannot establish a path to the target 443 port or is trying the wrong proxy."
    return None


async def diagnose_paic_connectivity() -> dict[str, Any]:
    report = paic_runtime_snapshot()
    scheme = str(report.get("scheme") or "https")
    host = str(report.get("host") or "")
    port = int(report.get("port") or (443 if scheme == "https" else 80))
    timeout_seconds = max(1.0, PAIC_TTS_DIAG_TIMEOUT)
    if not host:
        report.update({"ok": False, "summary": "PAIC_TTS_API_URL is empty or invalid."})
        return report

    try:
        addrinfo = await asyncio.wait_for(
            asyncio.to_thread(socket.getaddrinfo, host, port, socket.AF_UNSPEC, socket.SOCK_STREAM),
            timeout=timeout_seconds,
        )
        addresses: list[str] = []
        for entry in addrinfo:
            address = entry[4][0]
            if address not in addresses:
                addresses.append(address)
        report["dns"] = {
            "ok": True,
            "addresses": addresses[:8],
            "address_count": len(addresses),
        }
    except Exception as exc:
        report["dns"] = _probe_error(exc if isinstance(exc, Exception) else Exception(str(exc)))
        report["ok"] = False
        report["summary"] = _summarize_paic_probe(report)
        return report

    tcp_started = time.perf_counter()
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host=host, port=port), timeout=timeout_seconds)
        report["tcp"] = {
            "ok": True,
            "elapsed_ms": int((time.perf_counter() - tcp_started) * 1000),
            "peer": writer.get_extra_info("peername"),
        }
        writer.close()
        await writer.wait_closed()
    except Exception as exc:
        report["tcp"] = _probe_error(exc)
        report["tcp"]["elapsed_ms"] = int((time.perf_counter() - tcp_started) * 1000)
        report["ok"] = False
        report["summary"] = _summarize_paic_probe(report)
        return report

    if scheme == "https":
        tls_started = time.perf_counter()
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host=host, port=port, ssl=ssl.create_default_context(), server_hostname=host),
                timeout=timeout_seconds,
            )
            ssl_object = writer.get_extra_info("ssl_object")
            cipher = ssl_object.cipher() if ssl_object else None
            report["tls"] = {
                "ok": True,
                "elapsed_ms": int((time.perf_counter() - tls_started) * 1000),
                "version": ssl_object.version() if ssl_object else None,
                "cipher": cipher[0] if cipher else None,
            }
            writer.close()
            await writer.wait_closed()
        except Exception as exc:
            report["tls"] = _probe_error(exc)
            report["tls"]["elapsed_ms"] = int((time.perf_counter() - tls_started) * 1000)
            report["ok"] = False
            report["summary"] = _summarize_paic_probe(report)
            return report

    report["http_probe"] = await _run_http_probe(
        url=PAIC_TTS_API_URL,
        trust_env=PAIC_TTS_TRUST_ENV,
        timeout_seconds=timeout_seconds,
    )
    if PAIC_TTS_TRUST_ENV and report.get("proxy_env"):
        report["http_probe_without_env"] = await _run_http_probe(
            url=PAIC_TTS_API_URL,
            trust_env=False,
            timeout_seconds=timeout_seconds,
        )

    report["ok"] = bool(report.get("http_probe", {}).get("reachable"))
    report["summary"] = _summarize_paic_probe(report)
    return report


class OpenAiCompatibleSpeechSynthesizer(SpeechSynthesizer):
    def __init__(self, api_base: str) -> None:
        self.api_base = api_base.rstrip("/")

    async def synthesize(self, *, model: str, text: str) -> str | None:
        speech_text = text.strip()
        if not speech_text or not self.api_base:
            return None

        url = f"{self.api_base}/v1/audio/speech"
        payload = build_openai_tts_payload(model=TTS_MODEL, text=speech_text, response_format=TTS_RESPONSE_FORMAT)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(url, json=payload)
        except Exception as exc:
            logger.error("openai-compatible TTS request failed: %s", exc)
            return None

        if response.status_code >= 400:
            logger.error("openai-compatible TTS error: status=%d body=%s", response.status_code, response.text[:500])
            return None

        if not response.content:
            logger.warning("openai-compatible TTS returned empty body")
            return None
        return _as_audio_data_url(response.content, TTS_RESPONSE_FORMAT)

    async def stream_synthesize(
        self,
        *,
        model: str,
        text: str,
        on_audio_chunk: Callable[[bytes], Any],
    ) -> bool:
        speech_text = text.strip()
        if not speech_text or not self.api_base:
            return False

        url = f"{self.api_base}/v1/audio/speech"
        payload = build_openai_tts_payload(
            model=TTS_MODEL,
            text=speech_text,
            response_format=TTS_STREAM_RESPONSE_FORMAT,
            stream=True,
        )
        sent = False
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        logger.error(
                            "openai-compatible streaming TTS error: status=%d body=%s",
                            response.status_code,
                            body[:500],
                        )
                        return False
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        await on_audio_chunk(chunk)
                        sent = True
        except Exception as exc:
            logger.error("openai-compatible streaming TTS request failed: %s", exc)
            return False
        return sent


class QwenTalkerSpeechSynthesizer(SpeechSynthesizer):
    def __init__(self, model_client: ChatModel) -> None:
        self.model_client = model_client

    async def synthesize(self, *, model: str, text: str) -> str | None:
        speech_text = text.strip()
        if not speech_text:
            return None
        tts_model = resolve_qwen_talker_model(model)

        result, status_code = await self.model_client.complete_text(
            model=tts_model,
            text=speech_text,
            prompt="Please read the user text exactly as provided. Do not explain or rewrite it.",
            history=[],
            max_tokens=FINAL_MAX_TOKENS,
            output_audio=True,
        )
        if status_code >= 400:
            logger.error(
                "qwen_talker TTS failed: model=%s status=%s message=%s",
                tts_model,
                status_code,
                str(result.get("message") or result)[:500],
            )
            return None
        audio_data_url = result.get("audio_data_url")
        return str(audio_data_url) if audio_data_url else None

    async def stream_synthesize(
        self,
        *,
        model: str,
        text: str,
        on_audio_chunk: Callable[[bytes], Any],
    ) -> bool:
        return False


class LocalSpeechSynthesizer(SpeechSynthesizer):
    def __init__(self, backend: WindowsSapiTtsBackend) -> None:
        self.backend = backend

    async def synthesize(self, *, model: str, text: str) -> str | None:
        speech_text = text.strip()
        if not speech_text:
            return None
        try:
            result = await self.backend.synthesize(speech_text, voice=TTS_VOICE)
        except LocalTtsError as exc:
            logger.error("local TTS failed: %s", exc)
            return None
        return _as_audio_data_url(result.wav_bytes, "wav")

    async def stream_synthesize(
        self,
        *,
        model: str,
        text: str,
        on_audio_chunk: Callable[[bytes], Any],
    ) -> bool:
        speech_text = text.strip()
        if not speech_text:
            return False
        try:
            result = await self.backend.synthesize(speech_text, voice=TTS_VOICE)
        except LocalTtsError as exc:
            logger.error("local streaming TTS failed: %s", exc)
            return False

        pcm_bytes = wav_bytes_to_pcm16le_bytes(result.wav_bytes, TTS_SAMPLE_RATE)
        chunk_samples = max(1, int(TTS_SAMPLE_RATE * max(20, TTS_STREAM_CHUNK_MS) / 1000))
        chunk_bytes = chunk_samples * 2
        sent = False
        for start in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[start : start + chunk_bytes]
            if not chunk:
                continue
            await on_audio_chunk(chunk)
            sent = True
            await asyncio.sleep(0)
        return sent


class PaicSpeechSynthesizer(SpeechSynthesizer):
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url.strip()
        self.headers = _parse_json_mapping(PAIC_TTS_HEADERS_JSON, label="PAIC_TTS_HEADERS_JSON")
        self.extra_body = _parse_json_mapping(PAIC_TTS_EXTRA_BODY_JSON, label="PAIC_TTS_EXTRA_BODY_JSON")

    def _build_payload(self, *, text: str, response_format: str, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.extra_body)
        payload[PAIC_TTS_TEXT_FIELD] = text
        if PAIC_TTS_VOICE_FIELD and PAIC_TTS_VOICE:
            payload[PAIC_TTS_VOICE_FIELD] = PAIC_TTS_VOICE
        if PAIC_TTS_SOUND_LIBRARY_ID:
            payload[PAIC_TTS_SOUND_LIBRARY_ID_FIELD] = PAIC_TTS_SOUND_LIBRARY_ID
        if PAIC_TTS_MODEL_FIELD:
            payload[PAIC_TTS_MODEL_FIELD] = TTS_MODEL
        payload.setdefault("pitch", PAIC_TTS_PITCH)
        payload.setdefault("speed", PAIC_TTS_SPEED)
        payload.setdefault("volume", PAIC_TTS_VOLUME)
        payload.setdefault("sample_rate", PAIC_TTS_SAMPLE_RATE)
        payload.setdefault("emotion", PAIC_TTS_EMOTION)
        payload.setdefault("head_silence", PAIC_TTS_HEAD_SILENCE)
        payload.setdefault("tail_silence", PAIC_TTS_TAIL_SILENCE)
        payload.setdefault("response_format", response_format)
        if stream:
            payload["stream"] = True
            payload.setdefault("stream_format", TTS_STREAM_FORMAT)
        return payload

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        for key, value in self.headers.items():
            headers[str(key)] = str(value)
        return headers

    async def synthesize(self, *, model: str, text: str) -> str | None:
        speech_text = text.strip()
        if not speech_text or not self.api_url:
            return None

        payload = self._build_payload(text=speech_text, response_format=TTS_RESPONSE_FORMAT, stream=False)
        try:
            async with httpx.AsyncClient(
                timeout=_paic_http_timeout(PAIC_TTS_TIMEOUT),
                trust_env=PAIC_TTS_TRUST_ENV,
            ) as client:
                response = await client.post(self.api_url, json=payload, headers=self._request_headers())
        except Exception as exc:
            scheme, host, port = _paic_target()
            logger.error(
                "PAIC TTS request failed: type=%s host=%s port=%s scheme=%s trust_env=%s proxies=%s detail=%s",
                type(exc).__name__,
                host,
                port,
                scheme,
                PAIC_TTS_TRUST_ENV,
                json.dumps(_proxy_env_snapshot(), ensure_ascii=False),
                exc,
            )
            return None

        if response.status_code >= 400:
            logger.error("PAIC TTS error: status=%d body=%s", response.status_code, response.text[:500])
            return None

        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type:
            try:
                data = response.json()
            except Exception as exc:
                logger.error("PAIC TTS returned invalid JSON: %s", exc)
                return None
            audio_data = _extract_audio_data_url_from_json(data, response_format=TTS_RESPONSE_FORMAT)
            if audio_data:
                return audio_data
            logger.warning(
                "PAIC TTS JSON response did not contain usable audio. configured_path=%s preview=%s",
                PAIC_TTS_AUDIO_PATH,
                json.dumps(_preview_json_shape(data), ensure_ascii=False),
            )
            return None

        if not response.content:
            logger.warning("PAIC TTS returned empty body")
            return None
        return _as_audio_data_url(response.content, TTS_RESPONSE_FORMAT)

    async def stream_synthesize(
        self,
        *,
        model: str,
        text: str,
        on_audio_chunk: Callable[[bytes], Any],
    ) -> bool:
        speech_text = text.strip()
        if not speech_text or not self.api_url:
            return False

        payload = self._build_payload(text=speech_text, response_format=TTS_STREAM_RESPONSE_FORMAT, stream=True)
        sent = False
        try:
            async with httpx.AsyncClient(
                timeout=_paic_http_timeout(PAIC_TTS_STREAM_TIMEOUT),
                trust_env=PAIC_TTS_TRUST_ENV,
            ) as client:
                async with client.stream("POST", self.api_url, json=payload, headers=self._request_headers()) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        logger.error("PAIC streaming TTS error: status=%d body=%s", response.status_code, body[:500])
                        return False

                    content_type = response.headers.get("content-type", "").lower()
                    if "json" in content_type:
                        body = await response.aread()
                        try:
                            data = json.loads(body.decode("utf-8"))
                        except Exception as exc:
                            logger.error("PAIC streaming TTS returned invalid JSON: %s", exc)
                            return False
                        audio_bytes = _extract_audio_bytes_from_json(data)
                        if audio_bytes:
                            await on_audio_chunk(audio_bytes)
                            return True
                        logger.warning(
                            "PAIC streaming TTS JSON response did not contain usable audio. configured_path=%s preview=%s",
                            PAIC_TTS_AUDIO_PATH,
                            json.dumps(_preview_json_shape(data), ensure_ascii=False),
                        )
                        return False

                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        await on_audio_chunk(chunk)
                        sent = True
        except Exception as exc:
            scheme, host, port = _paic_target()
            logger.error(
                "PAIC streaming TTS request failed: type=%s host=%s port=%s scheme=%s trust_env=%s proxies=%s detail=%s",
                type(exc).__name__,
                host,
                port,
                scheme,
                PAIC_TTS_TRUST_ENV,
                json.dumps(_proxy_env_snapshot(), ensure_ascii=False),
                exc,
            )
            return False
        return sent


def set_speech_backend_status(**kwargs: object) -> None:
    _speech_backend_status.clear()
    _speech_backend_status.update(kwargs)


def speech_backend_status() -> dict[str, object]:
    return dict(_speech_backend_status)


def speech_streaming_supported() -> bool:
    return str(_speech_backend_status.get("backend") or "") in {"openai_compatible", "windows_sapi", "paic_xinxin"}


def create_speech_synthesizer() -> SpeechSynthesizer:
    backend = (TTS_BACKEND or "").strip().lower()

    if backend in {"qwen_talker", "model_audio"}:
        set_speech_backend_status(
            backend="qwen_talker",
            api_base=None,
            model=TTS_MODEL or None,
            voice=None,
            rate=None,
            sample_rate=None,
            stream_response_format=None,
        )
        return QwenTalkerSpeechSynthesizer(model_gateway)

    if backend == "paic_xinxin":
        if PAIC_TTS_API_URL:
            set_speech_backend_status(
                backend="paic_xinxin",
                api_base=PAIC_TTS_API_URL,
                model=TTS_MODEL or None,
                voice=PAIC_TTS_VOICE or None,
                rate=None,
                sample_rate=PAIC_TTS_SAMPLE_RATE,
                stream_response_format=TTS_STREAM_RESPONSE_FORMAT,
                audio_path=PAIC_TTS_AUDIO_PATH,
                sound_library_id=PAIC_TTS_SOUND_LIBRARY_ID or None,
                connect_timeout=PAIC_TTS_CONNECT_TIMEOUT,
                request_timeout=PAIC_TTS_TIMEOUT,
                stream_timeout=PAIC_TTS_STREAM_TIMEOUT,
                trust_env=PAIC_TTS_TRUST_ENV,
                proxy_env=_proxy_env_snapshot(),
            )
            return PaicSpeechSynthesizer(PAIC_TTS_API_URL)
        logger.warning("TTS_BACKEND=paic_xinxin but PAIC_TTS_API_URL is empty, falling back to qwen_talker")

    if backend == "openai_compatible":
        if TTS_API_BASE:
            set_speech_backend_status(
                backend="openai_compatible",
                api_base=TTS_API_BASE,
                model=TTS_MODEL or None,
                voice=TTS_VOICE or None,
                rate=None,
                sample_rate=TTS_SAMPLE_RATE,
                stream_response_format=TTS_STREAM_RESPONSE_FORMAT,
            )
            return OpenAiCompatibleSpeechSynthesizer(TTS_API_BASE)
        logger.warning("TTS_BACKEND=openai_compatible but TTS_API_BASE is empty, falling back to qwen_talker")

    if backend == "windows_sapi":
        set_speech_backend_status(
            backend="windows_sapi",
            api_base=None,
            model=None,
            voice=TTS_VOICE or None,
            rate=TTS_RATE,
            sample_rate=TTS_SAMPLE_RATE,
            stream_response_format=TTS_STREAM_RESPONSE_FORMAT,
        )
        return LocalSpeechSynthesizer(WindowsSapiTtsBackend(voice=TTS_VOICE, rate=TTS_RATE))

    set_speech_backend_status(
        backend="qwen_talker",
        api_base=None,
        model=TTS_MODEL or None,
        voice=None,
        rate=None,
        sample_rate=None,
        stream_response_format=None,
    )
    return QwenTalkerSpeechSynthesizer(model_gateway)


speech_synthesizer = create_speech_synthesizer()
