"""Debug endpoints for inspecting detection and postprocessing internals."""

import base64
import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from db.queries import get_video
from storage.local import storage
from extractor.detector import YoloDetector
from extractor.postprocess import _detect_health_bar_color, correct_detections
from extractor.config import DEFAULT_SAMPLE_FPS

router = APIRouter(tags=["debug"])

# Lazy singleton — created on first use
_detector: YoloDetector | None = None


def _get_detector() -> YoloDetector:
    global _detector
    if _detector is None:
        _detector = YoloDetector()
    return _detector


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message})


def _health_bar_crop_b64(frame: np.ndarray, bbox: list[float]) -> str | None:
    """Extract the health bar crop region inside the top of the bbox as base64 JPEG."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    det_h = y2 - y1

    # Health bar is inside the bbox, top 0-10%
    crop_y1 = max(0, y1)
    crop_y2 = min(h_frame, y1 + int(det_h * 0.10))
    crop_x1 = max(0, x1)
    crop_x2 = min(w_frame, x2)

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return base64.b64encode(buf).decode()


def _color_fractions(frame: np.ndarray, bbox: list[float]) -> dict[str, float]:
    """Compute green/blue/red pixel fractions in the health bar region."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    det_h = y2 - y1

    # Health bar is inside the bbox, top 0-10%
    crop_y1 = max(0, y1)
    crop_y2 = min(h_frame, y1 + int(det_h * 0.10))
    crop_x1 = max(0, x1)
    crop_x2 = min(w_frame, x2)

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        return {"green": 0.0, "blue": 0.0, "red": 0.0}

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    total_pixels = crop.shape[0] * crop.shape[1]
    if total_pixels == 0:
        return {"green": 0.0, "blue": 0.0, "red": 0.0}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Same HSV ranges as postprocess.py
    green_count = int(np.count_nonzero(cv2.inRange(
        hsv, np.array([35, 50, 50]), np.array([95, 255, 255])
    )))
    blue_count = int(np.count_nonzero(cv2.inRange(
        hsv, np.array([95, 40, 50]), np.array([130, 255, 255])
    )))
    red_count = (
        int(np.count_nonzero(cv2.inRange(
            hsv, np.array([0, 50, 50]), np.array([10, 255, 255])
        )))
        + int(np.count_nonzero(cv2.inRange(
            hsv, np.array([170, 50, 50]), np.array([180, 255, 255])
        )))
    )

    return {
        "green": round(green_count / total_pixels, 4),
        "blue": round(blue_count / total_pixels, 4),
        "red": round(red_count / total_pixels, 4),
    }


def _context_crop_b64(
    frame: np.ndarray,
    bbox: list[float],
    target_w: int = 400,
    target_h: int = 140,
) -> str | None:
    """Crop a region around the detection bbox, draw the bbox, and return as base64 JPEG.

    Shows a wider gameplay context so you can see what YOLO is looking at.
    """
    h_frame, w_frame = frame.shape[:2]
    bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    det_w = bx2 - bx1
    det_h = by2 - by1
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2

    # Expand the crop to be at least target_w x target_h, centered on detection
    crop_w = max(det_w + 80, target_w)  # padding around the detection
    crop_h = max(det_h + 40, target_h)

    crop_x1 = max(0, cx - crop_w // 2)
    crop_y1 = max(0, cy - crop_h // 2)
    crop_x2 = min(w_frame, crop_x1 + crop_w)
    crop_y2 = min(h_frame, crop_y1 + crop_h)
    # Re-adjust if we hit an edge
    crop_x1 = max(0, crop_x2 - crop_w)
    crop_y1 = max(0, crop_y2 - crop_h)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    # Draw bounding box relative to crop
    rel_x1 = bx1 - crop_x1
    rel_y1 = by1 - crop_y1
    rel_x2 = bx2 - crop_x1
    rel_y2 = by2 - crop_y1
    cv2.rectangle(crop, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 255, 255), 2)

    # Draw health bar scan region (top 0-10% of bbox) as a green box
    hb_y1 = rel_y1
    hb_y2 = rel_y1 + int(det_h * 0.10)
    cv2.rectangle(crop, (rel_x1, hb_y1), (rel_x2, hb_y2), (0, 200, 0), 1)

    # Resize to target dimensions for consistent card display
    crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return base64.b64encode(buf).decode()


def _analyze_frame(video_path: str, frame_index: int, sample_fps: float) -> dict:
    """Synchronous work: open video, seek, detect, analyze colors."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Convert sample frame_index to actual video frame number
    # frame_index is in "sample space" (at sample_fps), map to real frame
    timestamp_s = frame_index / sample_fps
    real_frame_num = int(timestamp_s * video_fps)
    real_frame_num = min(real_frame_num, max(0, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {real_frame_num}")

    # Max frame index based on video duration
    duration_s = total_frames / video_fps if video_fps > 0 else 0
    max_frame_index = max(0, int(duration_s * sample_fps) - 1)

    # Run YOLO detection + color correction (same as real extraction pipeline)
    detector = _get_detector()
    raw_detections = detector.detect(frame, conf_thresh=0.40)
    corrected_detections = correct_detections(frame, raw_detections)

    # Analyze each detection
    results = []
    for det in corrected_detections:
        bbox = det["bbox"]
        color = _detect_health_bar_color(frame, bbox)
        fractions = _color_fractions(frame, bbox)
        crop_b64 = _health_bar_crop_b64(frame, bbox)
        context_b64 = _context_crop_b64(frame, bbox)

        results.append({
            "class_name": det["class_name"],
            "original_class": det.get("original_class", det["class_name"]),
            "corrected": det.get("health_bar_corrected", False),
            "confidence": det["confidence"],
            "bbox": bbox,
            "health_bar_color": color,
            "color_fractions": fractions,
            "crop_b64": crop_b64,
            "context_b64": context_b64,
        })

    return {
        "frame_index": frame_index,
        "timestamp_ms": int(timestamp_s * 1000),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "max_frame_index": max_frame_index,
        "detections": results,
    }


@router.get("/videos/{video_id}/debug/health-bar-colors")
async def debug_health_bar_colors(
    video_id: str,
    frame_index: int = Query(0, ge=0, description="Frame index in sample space"),
    sample_fps: float = Query(DEFAULT_SAMPLE_FPS, gt=0, description="Sampling FPS"),
):
    """Debug endpoint: run YOLO + health bar color analysis on a single frame."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    video_path = storage.base / video["storage_path"]
    if not video_path.exists():
        return _error(404, "FILE_NOT_FOUND", "Video file not found on disk.")

    try:
        result = await asyncio.to_thread(_analyze_frame, str(video_path), frame_index, sample_fps)
    except RuntimeError as e:
        return _error(400, "FRAME_ERROR", str(e))

    return JSONResponse(content=result)
