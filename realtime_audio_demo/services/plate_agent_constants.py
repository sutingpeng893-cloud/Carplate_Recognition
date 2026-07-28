from __future__ import annotations

from pathlib import Path


NO_PLATE_REPLY = "我没有听到车牌号内容，请告诉我车牌号。"
INVALID_PLATE_REPLY = "您好，您当前的车牌号并不是有效号码，请重新输入。"
EDIT_UNCLEAR_REPLY = "我没有听清您要修改车牌的哪一处，当前仍保留原来的车牌。请您说明要替换、插入或删除哪一位。"
EDIT_INVALID_REPLY = "我按这次修改后得到的车牌格式不符合规则，当前仍保留原来的车牌。请您重新说明要改哪一处。"

CAR_PLATE_EXTRACTION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "car_plate_extraction_prompt.md"
CAR_PLATE_EXTRACTION_PROMPT = CAR_PLATE_EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
