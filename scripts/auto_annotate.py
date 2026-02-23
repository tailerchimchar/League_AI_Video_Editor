"""Auto-annotate video frames using the current YOLO model.

Samples frames from video clips at 1 fps, runs the ONNX model,
and saves high-confidence detections in YOLO format for manual
correction in Roboflow.

Usage:
    # Annotate a single video:
    python scripts/auto_annotate.py path/to/clip.mp4

    # Custom confidence threshold:
    python scripts/auto_annotate.py path/to/clip.mp4 --conf 0.6

    # Custom output directory:
    python scripts/auto_annotate.py path/to/clip.mp4 -o data/auto_annotated

    # Custom FPS:
    python scripts/auto_annotate.py path/to/clip.mp4 --fps 2

Workflow:
    1. Run this script on your video clips
    2. Upload images + labels to Roboflow
    3. Manually correct annotations in Roboflow's editor
    4. Re-export and merge with merge_datasets.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "auto_annotated"

# Add api dir to path for detector import
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


def parse_args():
    p = argparse.ArgumentParser(description="Auto-annotate video frames with YOLO")
    p.add_argument("video", type=Path, help="Path to video file")
    p.add_argument("--conf", type=float, default=0.5, help="Min confidence threshold (default: 0.5)")
    p.add_argument("--fps", type=float, default=1.0, help="Sampling rate in fps (default: 1.0)")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to ONNX model (default: auto-detect from data/models/)",
    )
    return p.parse_args()


def frame_to_yolo_labels(
    detections: list[dict],
    frame_w: int,
    frame_h: int,
) -> list[str]:
    """Convert detection dicts to YOLO format label lines.

    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    """
    from extractor.detector import CLASS_NAMES

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    lines = []

    for det in detections:
        cls_name = det["class_name"]
        if cls_name not in class_to_idx:
            continue

        cls_id = class_to_idx[cls_name]
        x1, y1, x2, y2 = det["bbox"]

        # Convert to YOLO normalized format
        cx = ((x1 + x2) / 2) / frame_w
        cy = ((y1 + y2) / 2) / frame_h
        w = (x2 - x1) / frame_w
        h = (y2 - y1) / frame_h

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def main():
    args = parse_args()

    if not args.video.exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    from extractor.detector import YoloDetector

    # Load detector
    detector = YoloDetector(model_path=args.model) if args.model else YoloDetector()
    if not detector.available:
        print("ERROR: No YOLO model available. Train one first with train_yolo.py")
        sys.exit(1)
    print(f"Model: {detector.model_path}")

    # Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.video}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"Video: {args.video.name} ({frame_w}x{frame_h}, {video_fps:.1f}fps, {duration:.1f}s)")
    print(f"Sampling at {args.fps} fps, confidence threshold: {args.conf}")

    # Create output dirs
    img_dir = args.output / "images"
    lbl_dir = args.output / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Sample frames
    frame_interval = int(video_fps / args.fps) if video_fps > 0 else 30
    prefix = args.video.stem

    frame_idx = 0
    saved = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            detections = detector.detect(frame, conf_thresh=args.conf)
            labels = frame_to_yolo_labels(detections, frame_w, frame_h)

            # Save even frames with 0 detections (negative examples are useful)
            fname = f"{prefix}_{frame_idx:06d}"
            cv2.imwrite(str(img_dir / f"{fname}.jpg"), frame)
            (lbl_dir / f"{fname}.txt").write_text("\n".join(labels) + "\n" if labels else "")

            saved += 1
            total_detections += len(detections)

            if saved % 10 == 0:
                print(f"  Frame {frame_idx}/{total_frames}: {len(detections)} detections")

        frame_idx += 1

    cap.release()

    print(f"\nDone! Saved {saved} frames with {total_detections} total detections")
    print(f"  Images: {img_dir}")
    print(f"  Labels: {lbl_dir}")
    print(f"\nNext steps:")
    print(f"  1. Upload to Roboflow for manual correction")
    print(f"  2. Re-export and merge with: python scripts/merge_datasets.py ...")
