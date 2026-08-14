from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _normalize_char_value(value: Any) -> str:
    text = _clean_text(value)
    replacements = {
        "零": "0",
        "〇": "0",
        "洞": "0",
        "一": "1",
        "幺": "1",
        "么": "1",
        "二": "2",
        "两": "2",
        "三": "3",
        "四": "4",
        "是": "4",
        "五": "5",
        "六": "6",
        "陆": "6",
        "七": "7",
        "拐": "7",
        "八": "8",
        "九": "9",
        "吸": "C",
        "勾": "J",
        "沟儿": "J",
        "圈": "Q",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.upper()
    return text[0] if len(text) == 1 else ""


@dataclass(slots=True)
class PlateConfusion:
    position: int
    value: str
    reason: str
    candidates: list[str] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: Any) -> "PlateConfusion | None":
        if not isinstance(value, dict):
            return None
        try:
            position = int(value.get("position") or 0)
        except (TypeError, ValueError):
            position = 0
        text_value = str(value.get("value") or "").strip()
        reason = str(value.get("reason") or "").strip()
        candidates_raw = value.get("candidates")
        candidates = [str(item).strip() for item in candidates_raw] if isinstance(candidates_raw, list) else []
        if position <= 0 and not text_value and not reason:
            return None
        return cls(position=position, value=text_value, reason=reason, candidates=[item for item in candidates if item])

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "value": self.value,
            "candidates": self.candidates,
            "reason": self.reason,
        }


@dataclass(slots=True)
class PlateCharState:
    position: int
    value: str
    confirmed: bool = False
    needs_confirmation: bool = False
    candidates: list[str] = field(default_factory=list)
    reason: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "PlateCharState | None":
        if not isinstance(value, dict):
            return None
        try:
            position = int(value.get("position") or 0)
        except (TypeError, ValueError):
            position = 0
        text_value = _normalize_char_value(value.get("value") or value.get("char"))
        confirmed = bool(value.get("confirmed"))
        needs_confirmation = bool(value.get("needs_confirmation"))
        candidates_raw = value.get("candidates")
        candidates = [_normalize_char_value(item) for item in candidates_raw] if isinstance(candidates_raw, list) else []
        reason = str(value.get("reason") or "").strip()
        if position <= 0 or not text_value:
            return None
        return cls(
            position=position,
            value=text_value,
            confirmed=confirmed,
            needs_confirmation=needs_confirmation,
            candidates=[item for item in candidates if item],
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "value": self.value,
            "confirmed": self.confirmed,
            "needs_confirmation": self.needs_confirmation,
            "candidates": self.candidates,
            "reason": self.reason,
        }


@dataclass(slots=True)
class PlateAgentState:
    car_plate: str = ""
    plate_chars: list[PlateCharState] = field(default_factory=list)
    confirmed: bool = False
    need_confirm_chars: list[PlateCharState] = field(default_factory=list)
    confirmed_chars: list[PlateCharState] = field(default_factory=list)
    vehicle_type: str = "unknown"
    confusions: list[PlateConfusion] = field(default_factory=list)
    final_car_plate: str = ""
    assistant_reply: str = ""
    ack_sent: bool = False
    turn_summaries: list[str] = field(default_factory=list)
    pending_plate: str = ""
    pending_commands: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_car_plate(self) -> bool:
        return bool(self.car_plate or self.plate_chars)

    @property
    def is_confirmed(self) -> bool:
        return bool(self.confirmed or self.final_car_plate)

    def to_context(self) -> dict[str, Any]:
        return {
            "car_plate": self.car_plate,
            "plate_chars": [item.to_dict() for item in self.plate_chars],
            "confirmed": self.is_confirmed,
            "need_confirm_chars": [item.to_dict() for item in self.need_confirm_chars],
            "confirmed_chars": [item.to_dict() for item in self.confirmed_chars],
            "vehicle_type": self.vehicle_type,
            "confusions": [item.to_dict() for item in self.confusions],
            "final_car_plate": self.final_car_plate,
            "assistant_reply": self.assistant_reply,
            "ack_sent": self.ack_sent,
            "turn_summaries": list(self.turn_summaries),
            "pending_plate": self.pending_plate,
            "pending_commands": list(self.pending_commands),
        }


@dataclass(slots=True)
class PlateAgentResult:
    text: str
    history_text: str
    speech_text: str
    state: PlateAgentState
    latency_ms: int
    debug: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PlateEditCommand:
    action: str
    position: int = 0
    value: str = ""
    old_value: str = ""
    new_value: str = ""
    relation: str = "at"
    occurrence: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "position": self.position,
            "value": self.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "relation": self.relation,
            "occurrence": self.occurrence,
            "raw": self.raw,
        }


@dataclass(slots=True)
class PlateConfirmationAction:
    action: str
    position: int = 0
    value: str = ""
    reason: str = ""
    candidates: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "position": self.position,
            "value": self.value,
            "candidates": self.candidates,
            "reason": self.reason,
            "raw": self.raw,
        }


@dataclass(slots=True)
class PlateUpdateReview:
    confirmed_positions: list[int] = field(default_factory=list)
    needs_more_edit: bool = False
    valid_result: bool = True
    reason: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed_positions": self.confirmed_positions,
            "needs_more_edit": self.needs_more_edit,
            "valid_result": self.valid_result,
            "reason": self.reason,
            "raw": self.raw,
        }


@dataclass(slots=True)
class PlateEditResult:
    car_plate: str
    changed: bool
    command: PlateEditCommand | None = None
    changed_positions: list[int] = field(default_factory=list)
    review: PlateUpdateReview | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "car_plate": self.car_plate,
            "changed": self.changed,
            "command": self.command.to_dict() if self.command else None,
            "changed_positions": self.changed_positions,
            "review": self.review.to_dict() if self.review else None,
            "steps": self.steps,
            "error": self.error,
            "raw": self.raw,
        }


@dataclass(slots=True)
class PendingResponseResult:
    intent: str  # "execute" | "reject" | "new_edit"
    commands: list[PlateEditCommand] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "commands": [cmd.to_dict() for cmd in self.commands],
            "raw": self.raw,
        }
