from __future__ import annotations

from pathlib import Path

CAR_PLATE_EXTRACTION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "car_plate_extraction_prompt.md"
CAR_PLATE_EXTRACTION_PROMPT = CAR_PLATE_EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
