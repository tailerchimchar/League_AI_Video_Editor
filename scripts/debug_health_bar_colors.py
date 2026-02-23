"""Debug script: run YOLO detection on a frame and show health bar color analysis.

Usage: python scripts/debug_health_bar_colors.py [frame_path]
Defaults to data/debug/ocr_test/full_frame_50.png
"""
import sys
import os

# Add api dir to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api"))

import cv2
import numpy as np

from extractor.detector import YoloDetector
from extractor.postprocess import (
    _detect_health_bar_color,
    _GREEN_LOWER, _GREEN_UPPER,
    _BLUE_LOWER, _BLUE_UPPER,
    _RED_LOWER_1, _RED_UPPER_1,
    _RED_LOWER_2, _RED_UPPER_2,
)


def debug_health_bar(frame, bbox, label=""):
    """Detailed analysis of the health bar region above a bbox."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    bar_h = max(8, int((y2 - y1) * 0.15))
    crop_y1 = max(0, y1 - bar_h)
    crop_y2 = y1
    crop_x1 = max(0, x1)
    crop_x2 = min(w_frame, x2)

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        print(f"  {label}: empty crop region")
        return

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    total_pixels = crop.shape[0] * crop.shape[1]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    green_count = int(np.count_nonzero(cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)))
    blue_count = int(np.count_nonzero(cv2.inRange(hsv, _BLUE_LOWER, _BLUE_UPPER)))
    red_count = (
        int(np.count_nonzero(cv2.inRange(hsv, _RED_LOWER_1, _RED_UPPER_1)))
        + int(np.count_nonzero(cv2.inRange(hsv, _RED_LOWER_2, _RED_UPPER_2)))
    )

    green_frac = green_count / total_pixels if total_pixels > 0 else 0
    blue_frac = blue_count / total_pixels if total_pixels > 0 else 0
    red_frac = red_count / total_pixels if total_pixels > 0 else 0

    result = _detect_health_bar_color(frame, bbox)

    print(f"\n  {label}")
    print(f"    Bbox: ({x1},{y1})-({x2},{y2}), bar crop: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
    print(f"    Crop size: {crop.shape[1]}x{crop.shape[0]} = {total_pixels}px")
    print(f"    Green: {green_count}/{total_pixels} = {green_frac:.3f}")
    print(f"    Blue:  {blue_count}/{total_pixels} = {blue_frac:.3f}")
    print(f"    Red:   {red_count}/{total_pixels} = {red_frac:.3f}")
    print(f"    => Detected color: {result}")

    # Detailed HSV of all pixels in crop
    flat = hsv.reshape(-1, 3)
    sat_mask = (flat[:, 1] > 30) & (flat[:, 2] > 50)
    if np.any(sat_mask):
        sat_px = flat[sat_mask]
        print(f"    Saturated pixels ({sat_px.shape[0]}):")
        print(f"      H: min={sat_px[:,0].min()} max={sat_px[:,0].max()} mean={sat_px[:,0].mean():.1f}")
        print(f"      S: min={sat_px[:,1].min()} max={sat_px[:,1].max()} mean={sat_px[:,1].mean():.1f}")
        print(f"      V: min={sat_px[:,2].min()} max={sat_px[:,2].max()} mean={sat_px[:,2].mean():.1f}")

        # Hue buckets
        h_vals = sat_px[:, 0]
        for lo in range(0, 180, 15):
            hi = lo + 15
            count = np.sum((h_vals >= lo) & (h_vals < hi))
            if count > 0:
                pct = count * 100 / len(h_vals)
                bar = '#' * min(30, max(1, int(pct)))
                print(f"      H {lo:3d}-{hi:3d}: {count:4d} ({pct:4.1f}%) {bar}")
    else:
        print(f"    No saturated pixels in crop")


def main():
    frame_path = sys.argv[1] if len(sys.argv) > 1 else "data/debug/ocr_test/full_frame_50.png"

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"ERROR: Could not read {frame_path}")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Frame: {frame_path} ({w}x{h})")

    detector = YoloDetector()
    if not detector.available:
        print("YOLO detector not available")
        sys.exit(1)

    detections = detector.detect(frame)
    print(f"\n{len(detections)} detections found:")

    for i, det in enumerate(detections):
        class_name = det["class_name"]
        conf = det.get("confidence", 0)
        bbox = det["bbox"]
        print(f"\n{'='*60}")
        print(f"Detection {i}: {class_name} ({conf:.2f})")
        debug_health_bar(frame, bbox, f"#{i} {class_name}")


if __name__ == "__main__":
    main()
