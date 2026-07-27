from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess
import shutil
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class LocalSpeechResult:
    wav_bytes: bytes
    sample_rate: int


class LocalTtsError(RuntimeError):
    pass


class WindowsSapiTtsBackend:
    def __init__(self, *, voice: str = "", rate: int = 0) -> None:
        self.voice = voice.strip()
        self.rate = max(-10, min(10, rate))

    async def synthesize(self, text: str, *, voice: str | None = None) -> LocalSpeechResult:
        speech_text = text.strip()
        if not speech_text:
            raise LocalTtsError("text is required")

        powershell = shutil.which("powershell") or shutil.which("powershell.exe")
        if not powershell:
            raise LocalTtsError("powershell is not available")

        fd, output_path = tempfile.mkstemp(prefix="carplate_tts_", suffix=".wav")
        os.close(fd)
        path = Path(output_path)
        selected_voice = (voice or self.voice).strip()
        text_b64 = base64.b64encode(speech_text.encode("utf-8")).decode("ascii")
        voice_b64 = base64.b64encode(selected_voice.encode("utf-8")).decode("ascii")

        script = (
            "$ErrorActionPreference='Stop';"
            "Add-Type -AssemblyName System.Speech;"
            f"$text=[System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{text_b64}'));"
            f"$voice=[System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{voice_b64}'));"
            f"$out='{path.as_posix()}';"
            "$synth=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "try {"
            "  if ($voice) {"
            "    try { $synth.SelectVoice($voice) }"
            "    catch { Write-Warning ('voice not found: ' + $voice) }"
            "  }"
            f"  if ({self.rate} -ne 0) {{ $synth.Rate={self.rate}; }}"
            "  $synth.SetOutputToWaveFile($out);"
            "  $synth.Speak($text);"
            "} finally {"
            "  $synth.Dispose();"
            "}"
        )

        try:
            process = await asyncio.to_thread(
                subprocess.run,
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    script,
                ],
                capture_output=True,
                text=False,
            )
            if process.returncode != 0:
                raise LocalTtsError(
                    f"windows sapi failed: {(process.stderr or process.stdout).decode('utf-8', errors='ignore').strip() or process.returncode}"
                )
            wav_bytes = path.read_bytes()
            if not wav_bytes:
                raise LocalTtsError("windows sapi returned empty audio")
            with wave.open(str(path), "rb") as wf:
                sample_rate = int(wf.getframerate())
        except FileNotFoundError as exc:
            raise LocalTtsError("powershell is not available") from exc
        finally:
            path.unlink(missing_ok=True)

        return LocalSpeechResult(wav_bytes=wav_bytes, sample_rate=sample_rate)
