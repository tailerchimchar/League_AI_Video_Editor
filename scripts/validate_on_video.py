"""Visual validation: run YOLO + champion ID on a video and save annotated frames.

Produces side-by-side comparison images with detection boxes, class labels,
confidence scores, and champion identifications.

Usage:
    # Basic validation:
    python scripts/validate_on_video.py path/to/clip.mp4

    # With champion ID:
    python scripts/validate_on_video.py path/to/clip.mp4 --champion Caitlyn

    # Custom FPS and confidence:
    python scripts/validate_on_video.py path/to/clip.mp4 --fps 2 --conf 0.3

    # Save to specific directory:
    python scripts/validate_on_video.py path/to/clip.mp4 -o data/debug/validation
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "debug" / "validation"

# Add api dir to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

# Colors (BGR)
COLOR_ENEMY = (0, 0, 220)       # red
COLOR_FRIENDLY = (220, 140, 0)  # blue
COLOR_PLAYED = (0, 215, 255)    # gold
COLOR_TEXT_BG = (0, 0, 0)       # black
COLOR_TEXT = (255, 255, 255)     # white


def parse_args():
    p = argparse.ArgumentParser(description="Visual YOLO + Champion ID validation")
    p.add_argument("video", type=Path, help="Path to video file")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p.add_argument("--fps", type=float, default=1.0, help="Sampling fps (default: 1.0)")
    p.add_argument("--champion", type=str, default=None, help="Player's champion name")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    p.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    p.add_argument("--no-champion-id", action="store_true", help="Disable champion identification")
    return p.parse_args()


def get_box_color(class_name: str) -> tuple[int, int, int]:
    """Get box color based on class name."""
    if class_name == "played_champion":
        return COLOR_PLAYED
    if class_name.startswith("enemy"):
        return COLOR_ENEMY
    return COLOR_FRIENDLY


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    champion_info: dict | None = None,
) -> np.ndarray:
    """Draw detection boxes and labels on frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        class_name = det["class_name"]
        color = get_box_color(class_name)

        # Draw box
        thickness = 2
        if class_name == "played_champion":
            thickness = 3
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        label = f"{class_name} {conf:.0%}"
        champ = det.get("champion")
        if champ:
            champ_conf = det.get("champion_confidence", 0)
            label = f"{champ} ({class_name}) {conf:.0%}"

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), COLOR_TEXT_BG, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), font, font_scale, color, 1, cv2.LINE_AA)

    # Draw champion ID info in top-left corner
    if champion_info:
        y_off = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        player = champion_info.get("player")
        if player:
            text = f"Player: {player['name']} ({player['confidence']:.2f})"
            cv2.putText(annotated, text, (10, y_off), font, 0.6, COLOR_PLAYED, 2, cv2.LINE_AA)
            y_off += 25

        allies = champion_info.get("allies", [])
        for i, ally in enumerate(allies):
            text = f"Ally {i+1}: {ally['name']} ({ally['confidence']:.2f})"
            cv2.putText(annotated, text, (10, y_off), font, 0.5, COLOR_FRIENDLY, 1, cv2.LINE_AA)
            y_off += 20

    return annotated


def draw_summary_bar(frame: np.ndarray, detections: list[dict], frame_idx: int) -> np.ndarray:
    """Add a summary bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 35
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)

    # Count detections by category
    enemies = sum(1 for d in detections if d["class_name"].startswith("enemy"))
    friendlies = sum(1 for d in detections
                     if d["class_name"].startswith("freindly") or d["class_name"].startswith("friendly"))
    played = sum(1 for d in detections if d["class_name"] == "played_champion")
    avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0

    text = (
        f"Frame {frame_idx} | "
        f"Detections: {len(detections)} | "
        f"Enemy: {enemies} Friendly: {friendlies} Played: {played} | "
        f"Avg conf: {avg_conf:.0%}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, text, (10, 24), font, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    return np.vstack([frame, bar])


def main():
    args = parse_args()

    if not args.video.exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    from extractor.detector import YoloDetector

    # Load detector
    detector = YoloDetector(model_path=args.model) if args.model else YoloDetector()
    if not detector.available:
        print("ERROR: No YOLO model available")
        sys.exit(1)
    print(f"Model: {detector.model_path}")

    # Load champion identifier
    champ_id = None
    if not args.no_champion_id:
        try:
            from extractor.champion_id import ChampionIdentifier
            champ_id = ChampionIdentifier()
            if not champ_id.available:
                print("Champion ID: not available (no references)")
                champ_id = None
            else:
                print("Champion ID: loaded")
        except Exception as e:
            print(f"Champion ID: failed to load ({e})")

    # Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.video}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {args.video.name} ({frame_w}x{frame_h}, {video_fps:.1f}fps)")
    print(f"Sampling at {args.fps} fps, confidence threshold: {args.conf}")
    if args.champion:
        print(f"Player champion: {args.champion}")

    # Create output
    args.output.mkdir(parents=True, exist_ok=True)

    frame_interval = int(video_fps / args.fps) if video_fps > 0 else 30
    frame_idx = 0
    saved = 0
    all_detections = 0
    conf_sum = 0.0
    conf_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Run detection
            detections = detector.detect(frame, conf_thresh=args.conf)

            # Run champion ID
            champion_info = None
            if champ_id:
                champion_info = champ_id.identify_from_portraits(frame, frame_w, frame_h)

                # If played_champion is specified, label played_champion detections
                if args.champion and champion_info:
                    for det in detections:
                        if det["class_name"] == "played_champion":
                            det["champion"] = args.champion
                            det["champion_confidence"] = 1.0
                        elif det["class_name"] == "freindly_champion":
                            # Try to match from ally portraits
                            for ally in champion_info.get("allies", []):
                                if ally["key"] != "unknown":
                                    det["champion"] = ally["name"]
                                    det["champion_confidence"] = ally["confidence"]
                                    break

            # Draw annotations
            annotated = draw_detections(frame, detections, champion_info)
            annotated = draw_summary_bar(annotated, detections, frame_idx)

            # Save
            out_path = args.output / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), annotated)

            # Stats
            all_detections += len(detections)
            for d in detections:
                conf_sum += d["confidence"]
                conf_count += 1

            saved += 1
            if saved % 5 == 0:
                print(f"  Frame {frame_idx}: {len(detections)} detections")

        frame_idx += 1

    cap.release()

    avg_conf = conf_sum / conf_count if conf_count > 0 else 0
    print(f"\nDone! Saved {saved} annotated frames to {args.output}")
    print(f"Total detections: {all_detections}")
    print(f"Average confidence: {avg_conf:.1%}")
    print(f"\nView results: {args.output}")


if __name__ == "__main__":
    main()
