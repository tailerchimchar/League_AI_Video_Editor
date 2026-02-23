"""OCR debugging tool — test OCR on a single frame or video.

Usage:
    python scripts/test_ocr.py <image_or_video_path> [--frame N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

import cv2
import numpy as np

from extractor.config import get_crop_regions
from extractor.cropper import crop_hud_regions
from extractor.ocr import run_ocr_on_crops, ocr_field


def test_on_frame(frame: np.ndarray, width: int, height: int):
    """Run OCR pipeline on a single frame and print results."""
    crops = crop_hud_regions(frame, width, height)

    print(f"\nFrame size: {width}x{height}")
    print(f"Crop regions detected: {list(crops.keys())}")
    print("-" * 50)

    # Show each crop and its OCR result
    for name, crop in crops.items():
        raw = ocr_field(crop, name)
        print(f"  {name:20s} -> '{raw}'")

    print("-" * 50)
    print("\nParsed OCR data:")
    ocr_data = run_ocr_on_crops(crops)
    for k, v in ocr_data.items():
        if k != "raw":
            print(f"  {k:20s} -> {v}")

    # Optionally save crop images for visual inspection
    output_dir = Path("data/debug/ocr_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, crop in crops.items():
        cv2.imwrite(str(output_dir / f"{name}.png"), crop)
    print(f"\nCrop images saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Test OCR on a frame or video")
    parser.add_argument("path", help="Path to image or video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (for video)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    ext = path.suffix.lower()

    if ext in (".mp4", ".webm", ".avi", ".mkv"):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {path}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        if not ret:
            print(f"ERROR: Cannot read frame {args.frame}")
            sys.exit(1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"Video: {path} (frame {args.frame})")
        test_on_frame(frame, width, height)

    elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"ERROR: Cannot read image: {path}")
            sys.exit(1)

        height, width = frame.shape[:2]
        print(f"Image: {path}")
        test_on_frame(frame, width, height)

    else:
        print(f"ERROR: Unsupported file type: {ext}")
        sys.exit(1)


if __name__ == "__main__":
    main()
