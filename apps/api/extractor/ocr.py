"""Tesseract OCR with preprocessing for League HUD fields."""

import re

import cv2
import numpy as np
import os

import pytesseract

# Allow overriding Tesseract path via env var (e.g., for Windows installs not in PATH)
_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

from extractor.config import NUMERIC_FIELDS, KDA_FIELDS, HEALTH_BAR_FIELDS


def preprocess_for_ocr(crop: np.ndarray, field_type: str) -> np.ndarray:
    """Preprocess a crop image for better OCR accuracy."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = 4 if field_type in HEALTH_BAR_FIELDS else 3
    scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if field_type in HEALTH_BAR_FIELDS:
        # White text on colored bar needs a high fixed threshold
        _, thresh = cv2.threshold(scaled, 180, 255, cv2.THRESH_BINARY)
    else:
        # Light text on dark background: Otsu works well
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def ocr_field(crop: np.ndarray, field_type: str) -> str | None:
    """OCR a single cropped field with type-appropriate config."""
    processed = preprocess_for_ocr(crop, field_type)

    config = "--psm 7"  # single line mode
    if field_type in NUMERIC_FIELDS:
        config += " -c tessedit_char_whitelist=0123456789:/"
    elif field_type in KDA_FIELDS:
        config += " -c tessedit_char_whitelist=0123456789/"
    elif field_type in HEALTH_BAR_FIELDS:
        config += " -c tessedit_char_whitelist=0123456789/"

    text = pytesseract.image_to_string(processed, config=config).strip()
    return text if text else None


def parse_game_timer(raw: str | None) -> str | None:
    """Parse game timer text like '12:34' or '1:05'."""
    if not raw:
        return None
    match = re.search(r"(\d{1,2}):(\d{2})", raw)
    return f"{match.group(1)}:{match.group(2)}" if match else None


def parse_kda(raw: str | None) -> dict | None:
    """Parse KDA text like '3/1/5' into structured dict."""
    if not raw:
        return None
    # Only match 1-2 digit numbers — no one has 100+ kills/deaths/assists
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{1,2})", raw)
    if match:
        k, d, a = int(match.group(1)), int(match.group(2)), int(match.group(3))
        # Sanity: each component should be <= 50 (generous upper bound)
        if k > 50 or d > 50 or a > 50:
            return None
        return {"kills": k, "deaths": d, "assists": a}
    return None


def parse_int(raw: str | None) -> int | None:
    """Parse a numeric field, stripping non-digit characters."""
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", raw)
    return int(digits) if digits else None


def parse_cs_gold(raw: str | None) -> dict:
    """Parse CS/gold text. Formats vary: '156 CS' or '156 / 4.2k'."""
    result: dict = {"cs": None, "gold": None}
    if not raw:
        return result

    # Try to find CS number
    numbers = re.findall(r"\d+", raw)
    if numbers:
        result["cs"] = int(numbers[0])
    if len(numbers) > 1:
        result["gold"] = int(numbers[1])

    return result


def parse_health_bar(raw: str | None) -> dict:
    """Parse health bar text like '1490/1679' into current, max, and pct."""
    result: dict = {"current": None, "max": None, "pct": None}
    if not raw:
        return result
    numbers = re.findall(r"\d+", raw)
    if len(numbers) >= 2:
        current = int(numbers[0])
        maximum = int(numbers[1])
        # Sanity: max should be reasonable (League HP caps around ~6000),
        # and current should not exceed max
        if maximum <= 0 or maximum > 10000 or current > maximum:
            return result
        result["current"] = current
        result["max"] = maximum
        result["pct"] = round(current / maximum, 3)
    return result


def run_ocr_on_crops(crops: dict[str, np.ndarray]) -> dict:
    """Run OCR on all cropped HUD regions and return structured data."""
    ocr_data: dict = {"raw": {}}

    for field_name, crop in crops.items():
        raw_text = ocr_field(crop, field_name)
        ocr_data["raw"][field_name] = raw_text

    # Parse structured fields from raw OCR
    ocr_data["game_timer"] = parse_game_timer(ocr_data["raw"].get("game_timer"))

    kda = parse_kda(ocr_data["raw"].get("kda"))
    if kda:
        ocr_data["player_kda"] = kda

    level = parse_int(ocr_data["raw"].get("player_level"))
    if level is not None and 1 <= level <= 18:
        ocr_data["player_level"] = level

    cs_gold = parse_cs_gold(ocr_data["raw"].get("cs_gold"))
    if cs_gold["cs"] is not None:
        ocr_data["player_cs"] = cs_gold["cs"]
    if cs_gold["gold"] is not None:
        ocr_data["player_gold"] = cs_gold["gold"]

    hp = parse_health_bar(ocr_data["raw"].get("health_bar"))
    if hp["pct"] is not None:
        ocr_data["player_hp_pct"] = hp["pct"]
        ocr_data["player_hp_current"] = hp["current"]
        ocr_data["player_hp_max"] = hp["max"]

    # Resources bar (mana, energy, rage, ferocity, etc.) — same format as HP: "300/400" or "0/4"
    resources = parse_health_bar(ocr_data["raw"].get("resources_bar"))
    if resources["current"] is not None:
        ocr_data["player_resource_current"] = resources["current"]
        ocr_data["player_resource_max"] = resources["max"]
        ocr_data["player_resource_pct"] = resources["pct"]

    return ocr_data
