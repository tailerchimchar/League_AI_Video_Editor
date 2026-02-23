#!/usr/bin/env python3
"""Debug tool to calibrate champion portrait positions in the HUD.

Usage:
    python scripts/calibrate_scoreboard.py <image_or_video_path> [--frame N]

Extracts the player portrait and 4 ally portraits from a League screenshot
or video frame, runs champion identification, and saves debug crops to
data/debug/scoreboard_calibration/ for visual inspection.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add apps/api to path so we can import extractor modules
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from extractor.config import (
    BASE_WIDTH,
    BASE_HEIGHT,
    PLAYER_PORTRAIT_REGION,
    ALLY_PORTRAIT_SLOTS,
)
from extractor.champion_id import ChampionIdentifier

DEBUG_DIR = REPO_ROOT / "data" / "debug" / "scoreboard_calibration"


def extract_frame(path: str, frame_num: int = 0) -> np.ndarray | None:
    """Extract a frame from a video file or load an image."""
    p = Path(path)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        return cv2.imread(str(p))

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {path}")
        return None

    if frame_num > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Cannot read frame {frame_num} from {path}")
        return None

    return frame


def save_crop(frame, region, scale_x, scale_y, name, label):
    """Extract a region, save it, and return the crop."""
    rx = int(region["x"] * scale_x)
    ry = int(region["y"] * scale_y)
    rw = int(region["w"] * scale_x)
    rh = int(region["h"] * scale_y)

    h, w = frame.shape[:2]
    rx = max(0, min(rx, w - 1))
    ry = max(0, min(ry, h - 1))
    rw = min(rw, w - rx)
    rh = min(rh, h - ry)

    crop = frame[ry : ry + rh, rx : rx + rw]
    if crop.size > 0:
        cv2.imwrite(str(DEBUG_DIR / f"{name}.png"), crop)
        up = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(DEBUG_DIR / f"{name}_5x.png"), up)
        print(f"  {label}: ({rx},{ry}) {rw}x{rh} -> saved {name}.png")
    else:
        print(f"  {label}: ({rx},{ry}) {rw}x{rh} -> EMPTY!")
    return crop, (rx, ry, rw, rh)


def main():
    parser = argparse.ArgumentParser(description="Calibrate champion portrait positions")
    parser.add_argument("input", help="Path to image or video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number for videos (default: 0)")
    args = parser.parse_args()

    frame = extract_frame(args.input, args.frame)
    if frame is None:
        sys.exit(1)

    height, width = frame.shape[:2]
    print(f"Frame resolution: {width}x{height}")

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_DIR / "00_full_frame.png"), frame)

    scale_x = width / BASE_WIDTH
    scale_y = height / BASE_HEIGHT

    annotated = frame.copy()

    # --- Player portrait ---
    print("\n--- Player Portrait ---")
    _, (px, py, pw, ph) = save_crop(
        frame, PLAYER_PORTRAIT_REGION, scale_x, scale_y,
        "01_player", "Player"
    )
    cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (0, 215, 255), 2)

    # --- Ally portraits ---
    print("\n--- Ally Portraits ---")
    for i, slot in enumerate(ALLY_PORTRAIT_SLOTS):
        _, (ax, ay, aw, ah) = save_crop(
            frame, slot, scale_x, scale_y,
            f"02_ally_{i}", f"Ally {i+1}"
        )
        cv2.rectangle(annotated, (ax, ay), (ax + aw, ay + ah), (255, 100, 0), 2)

    cv2.imwrite(str(DEBUG_DIR / "03_annotated_frame.png"), annotated)

    # --- Champion identification ---
    print("\n--- Champion Identification ---")
    identifier = ChampionIdentifier()
    if not identifier.available:
        print("  Champion icons not downloaded. Run scripts/download_champion_icons.py first.")
        print(f"\nDebug crops saved to: {DEBUG_DIR}")
        return

    roster = identifier.identify_from_portraits(frame, width, height)

    player = roster.get("player")
    if player:
        print(f"\n  Player: {player['name']:20s} (key={player['key']:15s}, conf={player['confidence']:.3f})")
    else:
        print("\n  Player: could not identify")

    print("\n  Allies:")
    for i, ally in enumerate(roster.get("allies", [])):
        print(f"    {i+1}. {ally['name']:20s} (key={ally['key']:15s}, conf={ally['confidence']:.3f})")

    roster_path = DEBUG_DIR / "roster.json"
    roster_path.write_text(json.dumps(roster, indent=2))

    print(f"\nDebug crops saved to: {DEBUG_DIR}")
    print(f"Roster saved to:     {roster_path}")
    print("\nTips:")
    print("  - Check 01_player_5x.png — should show the player's champion face")
    print("  - Check 02_ally_*_5x.png — should show each ally's face")
    print("  - If crops are misaligned, adjust PLAYER_PORTRAIT_REGION and")
    print("    ALLY_PORTRAIT_SLOTS in apps/api/extractor/config.py")


if __name__ == "__main__":
    main()
