"""Post-processing: health bar color correction for YOLO detections.

After YOLO detects entities, it frequently confuses ally vs enemy (e.g.,
labeling an ally minion as enemy_melee_minion). This module checks the
health bar color above each detection and reclassifies based on the
three health bar colors in League:

- Dark green → played_champion (the player)
- Light blue/cyan → ally (freindly_*)
- Red → enemy (enemy_*)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- HSV color ranges for the three health bar colors ---

# Green (player's health bar): H 35-95, S > 50, V > 50
# Player bar is a dark/teal green with hue peaking at H=65-90, so the upper
# boundary must extend past 85 to capture it fully.
_GREEN_LOWER = np.array([35, 50, 50], dtype=np.uint8)
_GREEN_UPPER = np.array([95, 255, 255], dtype=np.uint8)

# Blue/cyan (ally health bars): H 95-130, S > 40, V > 50
# Ally bars peak at H=95-120. Starts at 95 to avoid overlap with player green.
_BLUE_LOWER = np.array([95, 40, 50], dtype=np.uint8)
_BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)

# Red (enemy health bars): H 0-10 or 170-180, S > 50, V > 50
_RED_LOWER_1 = np.array([0, 50, 50], dtype=np.uint8)
_RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
_RED_LOWER_2 = np.array([170, 50, 50], dtype=np.uint8)
_RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

# Minimum fraction of pixels that must match a color to count as confident
_MIN_COLOR_FRACTION = 0.15

# Classes that should never be reclassified by health bar color
_NO_FLIP = {"freindly_super_minion"}

# Champion class names
_CHAMPION_CLASSES = {"played_champion", "freindly_champion", "enemy_champion"}

# Mapping from entity type to (green_class, blue_class, red_class)
# Green = player, Blue = ally, Red = enemy
_CLASS_BY_COLOR: dict[str, dict[str, str]] = {
    # Champions
    "played_champion":      {"green": "played_champion", "blue": "freindly_champion", "red": "enemy_champion"},
    "freindly_champion":    {"green": "played_champion", "blue": "freindly_champion", "red": "enemy_champion"},
    "enemy_champion":       {"green": "played_champion", "blue": "freindly_champion", "red": "enemy_champion"},
    # Melee minions
    "freindly_melee_minion": {"green": "freindly_melee_minion", "blue": "freindly_melee_minion", "red": "enemy_melee_minion"},
    "enemy_melee_minion":    {"green": "freindly_melee_minion", "blue": "freindly_melee_minion", "red": "enemy_melee_minion"},
    # Ranged minions
    "freindly_ranged_minion": {"green": "freindly_ranged_minion", "blue": "freindly_ranged_minion", "red": "enemy_ranged_minion"},
    "enemy_ranged_minion":    {"green": "freindly_ranged_minion", "blue": "freindly_ranged_minion", "red": "enemy_ranged_minion"},
    # Cannon minions
    "freindly_cannon_minion": {"green": "freindly_cannon_minion", "blue": "freindly_cannon_minion", "red": "enemy_cannon_minion"},
    "enemy_cannon_minion":    {"green": "freindly_cannon_minion", "blue": "freindly_cannon_minion", "red": "enemy_cannon_minion"},
    # Towers (note: "friendly_" spelling for structures in dataset)
    "friendly_tower":  {"green": "friendly_tower", "blue": "friendly_tower", "red": "enemy_tower"},
    "enemy_tower":     {"green": "friendly_tower", "blue": "friendly_tower", "red": "enemy_tower"},
    # Inhibitors
    "friendly_inhibitor":  {"green": "friendly_inhibitor", "blue": "friendly_inhibitor", "red": "enemy_inhibitor"},
    "enemy_inhibitor":     {"green": "friendly_inhibitor", "blue": "friendly_inhibitor", "red": "enemy_inhibitor"},
    # Nexus
    "friendly_nexus":  {"green": "friendly_nexus", "blue": "friendly_nexus", "red": "enemy_nexus"},
    "enemy_nexus":     {"green": "friendly_nexus", "blue": "friendly_nexus", "red": "enemy_nexus"},
}


def _detect_health_bar_color(
    frame: np.ndarray,
    bbox: list[float],
) -> str | None:
    """Check the region above a detection bbox for health bar color.

    Returns "green", "blue", "red", or None if inconclusive.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Health bar sits just above the entity
    bar_h = max(8, int((y2 - y1) * 0.15))
    crop_y1 = max(0, y1 - bar_h)
    crop_y2 = y1
    crop_x1 = max(0, x1)
    crop_x2 = min(w_frame, x2)

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    total_pixels = crop.shape[0] * crop.shape[1]
    if total_pixels == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    green_count = int(np.count_nonzero(cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)))
    blue_count = int(np.count_nonzero(cv2.inRange(hsv, _BLUE_LOWER, _BLUE_UPPER)))
    red_count = (
        int(np.count_nonzero(cv2.inRange(hsv, _RED_LOWER_1, _RED_UPPER_1)))
        + int(np.count_nonzero(cv2.inRange(hsv, _RED_LOWER_2, _RED_UPPER_2)))
    )

    green_frac = green_count / total_pixels
    blue_frac = blue_count / total_pixels
    red_frac = red_count / total_pixels

    # Pick the dominant color if it exceeds the threshold
    best_frac = max(green_frac, blue_frac, red_frac)
    if best_frac < _MIN_COLOR_FRACTION:
        return None

    if green_frac == best_frac:
        return "green"
    if blue_frac == best_frac:
        return "blue"
    return "red"


def correct_detections(
    frame: np.ndarray,
    detections: list[dict],
) -> list[dict]:
    """Apply health bar color correction to YOLO detections.

    Three phases:
    1. Color reclassification — use health bar color as ground truth to assign
       the correct ally/enemy/player class.
    2. Enforce max 1 played_champion per frame — keep highest confidence,
       demote the rest to freindly_champion.

    Modifies detections in-place and returns the same list.
    """
    if not detections:
        return detections

    corrected = 0

    # --- Phase 1+2: Color-based reclassification ---
    for det in detections:
        class_name = det.get("class_name", "")

        if class_name in _NO_FLIP or class_name not in _CLASS_BY_COLOR:
            continue

        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        color = _detect_health_bar_color(frame, bbox)
        if color is None:
            continue

        new_class = _CLASS_BY_COLOR[class_name][color]
        if new_class != class_name:
            det["class_name"] = new_class
            det["health_bar_corrected"] = True
            corrected += 1

    # --- Phase 3: Enforce max 1 played_champion ---
    player_dets = [d for d in detections if d.get("class_name") == "played_champion"]
    if len(player_dets) > 1:
        # Keep the highest-confidence one
        player_dets.sort(key=lambda d: d.get("confidence", 0), reverse=True)
        for extra in player_dets[1:]:
            extra["class_name"] = "freindly_champion"
            extra["health_bar_corrected"] = True
            corrected += 1

    if corrected > 0:
        logger.debug(
            "Health bar correction reclassified %d/%d detections",
            corrected, len(detections),
        )

    return detections
