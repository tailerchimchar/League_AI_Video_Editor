"""Debug endpoints for inspecting detection and postprocessing internals."""

import base64
import asyncio
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from db.queries import get_video
from storage.local import storage
from extractor.detector import YoloDetector
from extractor.champion_id import ChampionIdentifier
from extractor.postprocess import _detect_health_bar_color, correct_detections
from extractor.config import (
    DEFAULT_SAMPLE_FPS,
    PLAYER_PORTRAIT_REGION,
    ALLY_PORTRAIT_SLOTS,
    BASE_WIDTH,
    BASE_HEIGHT,
)

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


# ── Champion ID Debug ────────────────────────────────────────────────

_champion_id: ChampionIdentifier | None = None

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "champions"
_ICONS_DIR = _DATA_DIR / "icons"
_SPRITES_DIR = _DATA_DIR / "sprites"


def _get_champion_id() -> ChampionIdentifier:
    global _champion_id
    if _champion_id is None or not _champion_id.available:
        _champion_id = ChampionIdentifier(icons_only=True)
    return _champion_id


def _img_to_b64(img: np.ndarray, quality: int = 85) -> str | None:
    """Encode a BGR numpy array as base64 JPEG."""
    if img is None or img.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode() if ok else None


def _autocrop_alpha(img: np.ndarray, pad: int = 2) -> np.ndarray:
    """Crop an RGBA image to its non-transparent bounding box, composite onto black.

    This removes transparent padding so the champion fills the display area.
    Falls back to full composite if no alpha or all transparent.
    """
    if img is None or img.size == 0:
        return img
    if len(img.shape) != 3 or img.shape[2] != 4:
        return img  # No alpha channel, return as-is

    alpha = img[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Fully transparent — composite whole image
        a = img[:, :, 3:] / 255.0
        return (img[:, :, :3] * a).astype(np.uint8)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add small padding
    h, w = img.shape[:2]
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)

    cropped = img[rmin:rmax + 1, cmin:cmax + 1]
    a = cropped[:, :, 3:] / 255.0
    return (cropped[:, :, :3] * a).astype(np.uint8)


def _analyze_champion_id(video_path: str, frame_index: int, sample_fps: float) -> dict:
    """Run portrait ID + YOLO detection + match_detection on a single frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamp_s = frame_index / sample_fps
    real_frame_num = min(int(timestamp_s * video_fps), max(0, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {real_frame_num}")

    duration_s = total_frames / video_fps if video_fps > 0 else 0
    max_frame_index = max(0, int(duration_s * sample_fps) - 1)

    cid = _get_champion_id()
    scale_x = frame_width / BASE_WIDTH
    scale_y = frame_height / BASE_HEIGHT

    # 1) Portrait identification — extract HUD portraits and identify
    portrait_result = cid.identify_from_portraits(frame, frame_width, frame_height)
    player_info = portrait_result.get("player")
    allies_info = portrait_result.get("allies", [])

    # Extract portrait crops for display
    def extract_portrait_crop(region: dict) -> str | None:
        rx = int(region["x"] * scale_x)
        ry = int(region["y"] * scale_y)
        rw = int(region["w"] * scale_x)
        rh = int(region["h"] * scale_y)
        fh, fw = frame.shape[:2]
        rx, ry = max(0, min(rx, fw-1)), max(0, min(ry, fh-1))
        rw, rh = min(rw, fw-rx), min(rh, fh-ry)
        crop = frame[ry:ry+rh, rx:rx+rw]
        return _img_to_b64(crop) if crop.size > 0 else None

    # Build portrait strip: actual frame crops with bounding boxes drawn
    def _build_portrait_strip() -> str | None:
        """Crop player + ally HUD areas from the actual frame, draw bbox rectangles."""
        fh, fw = frame.shape[:2]
        PAD = 15  # padding around each portrait region

        # Scale all regions to actual frame resolution
        def scale_region(r: dict) -> tuple[int, int, int, int]:
            rx = int(r["x"] * scale_x)
            ry = int(r["y"] * scale_y)
            rw = int(r["w"] * scale_x)
            rh = int(r["h"] * scale_y)
            return rx, ry, rw, rh

        regions = [PLAYER_PORTRAIT_REGION] + ALLY_PORTRAIT_SLOTS
        labels = ["Player"] + [f"Ally {i+1}" for i in range(len(ALLY_PORTRAIT_SLOTS))]
        colors = [(0, 238, 210)] + [(255, 180, 50)] * 4  # cyan for player, blue-ish for allies

        panels = []
        panel_h = 0
        for region, label, color in zip(regions, labels, colors):
            rx, ry, rw, rh = scale_region(region)
            # Crop area with padding
            cx1 = max(0, rx - PAD)
            cy1 = max(0, ry - PAD)
            cx2 = min(fw, rx + rw + PAD)
            cy2 = min(fh, ry + rh + PAD)
            crop = frame[cy1:cy2, cx1:cx2].copy()
            if crop.size == 0:
                continue
            # Draw bbox rectangle (relative to crop)
            bx1, by1 = rx - cx1, ry - cy1
            bx2, by2 = bx1 + rw, by1 + rh
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 2)
            # Scale panel to uniform height (80px)
            target_h = 80
            aspect = crop.shape[1] / crop.shape[0]
            target_w = int(target_h * aspect)
            panel = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
            # Draw label
            cv2.putText(panel, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            panels.append(panel)
            panel_h = target_h

        if not panels:
            return None
        # Concatenate horizontally with 4px gap
        gap = np.zeros((panel_h, 4, 3), dtype=np.uint8)
        strips = []
        for i, p in enumerate(panels):
            if i > 0:
                strips.append(gap)
            strips.append(p)
        strip = np.concatenate(strips, axis=1)
        return _img_to_b64(strip, quality=90)

    portrait_strip_b64 = _build_portrait_strip()

    player_portrait_b64 = extract_portrait_crop(PLAYER_PORTRAIT_REGION)
    ally_portraits = []
    for i, slot in enumerate(ALLY_PORTRAIT_SLOTS):
        info = allies_info[i] if i < len(allies_info) else {"key": "unknown", "name": "Unknown", "confidence": 0.0}
        icon_b64 = None
        icon_path = _ICONS_DIR / f"{info['key']}.png"
        if icon_path.exists():
            icon_img = cv2.imread(str(icon_path))
            if icon_img is not None:
                icon_b64 = _img_to_b64(icon_img)
        ally_portraits.append({
            "slot": i,
            "key": info["key"],
            "name": info["name"],
            "confidence": info.get("confidence", 0.0),
            "portrait_crop_b64": extract_portrait_crop(slot),
            "icon_b64": icon_b64,
        })

    # Ally candidate keys for match_detection
    ally_keys = [a["key"] for a in allies_info if a.get("key") and a["key"] != "unknown"]

    # 2) YOLO detection + color correction
    detector = _get_detector()
    raw_dets = detector.detect(frame, conf_thresh=0.40) if detector.available else []
    corrected_dets = correct_detections(frame, raw_dets)

    # All champion keys for enemy matching (match against every known champion)
    all_champion_keys = list(cid._champion_names.keys())

    # Helper: build debug info for a detection against a set of candidate keys
    def _build_detection_debug(det: dict, candidate_keys: list[str]) -> dict:
        bbox = det["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        det_h = y2 - y1

        # Body crop (top 10% / bottom 5% trimmed, same as match_detection)
        crop_y1 = max(0, y1 + int(det_h * 0.10))
        crop_y2 = min(frame.shape[0], y2 - int(det_h * 0.05))
        body_crop = frame[crop_y1:crop_y2, x1:x2]
        body_crop_b64 = _img_to_b64(body_crop) if body_crop.size > 0 else None

        # Run match_detection to get the result
        match_result = cid.match_detection(frame, bbox, candidate_keys) if candidate_keys else None

        # Compute per-candidate scores for debug display
        candidate_scores = []
        if candidate_keys and cid.available and body_crop.size > 0:
            crop_hist = cid._compute_spatial_hist(body_crop)
            crop_tmpl = cv2.resize(body_crop, (48, 48))
            candidate_set = set(candidate_keys)
            # Track best score AND which reference label won
            best_per: dict[str, tuple[float, float, float, str]] = {}
            for champ_key, label, ref_hist, ref_tmpl in cid._references:
                if champ_key not in candidate_set:
                    continue
                hist_score = max(0.0, cv2.compareHist(
                    crop_hist.reshape(-1).astype(np.float32),
                    ref_hist.reshape(-1).astype(np.float32),
                    cv2.HISTCMP_CORREL,
                ))
                result = cv2.matchTemplate(crop_tmpl, ref_tmpl, cv2.TM_CCOEFF_NORMED)
                tmpl_score = max(0.0, float(result[0, 0]))
                combined = 0.7 * hist_score + 0.3 * tmpl_score
                if champ_key not in best_per or combined > best_per[champ_key][2]:
                    best_per[champ_key] = (hist_score, tmpl_score, combined, label)

            # For enemies (large candidate list), only return top 10 scores
            if len(candidate_keys) > 20:
                top_keys = sorted(best_per.keys(), key=lambda k: best_per[k][2], reverse=True)[:10]
            else:
                top_keys = candidate_keys

            for key in top_keys:
                if key in best_per:
                    h, t, c, ref_label = best_per[key]
                    # Load the icon for display
                    icon_b64 = None
                    ip = _ICONS_DIR / f"{key}.png"
                    if ip.exists():
                        img = cv2.imread(str(ip))
                        if img is not None:
                            icon_b64 = _img_to_b64(img)

                    # Load the best-matching reference image (sprite or icon)
                    best_ref_b64 = None
                    if "_sprite" in ref_label:
                        sprite_stem = ref_label.replace("_sprite", "")
                        sprite_path = _SPRITES_DIR / f"{sprite_stem}.png"
                        if sprite_path.exists():
                            simg = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
                            if simg is not None:
                                simg = _autocrop_alpha(simg)
                                best_ref_b64 = _img_to_b64(simg)
                    if best_ref_b64 is None:
                        best_ref_b64 = icon_b64  # fallback to icon

                    candidate_scores.append({
                        "key": key,
                        "name": cid.get_champion_name(key),
                        "hist_score": round(h, 4),
                        "tmpl_score": round(t, 4),
                        "combined": round(c, 4),
                        "icon_b64": icon_b64,
                        "best_ref_label": ref_label,
                        "best_ref_b64": best_ref_b64,
                    })

            candidate_scores.sort(key=lambda s: s["combined"], reverse=True)

        context_b64 = _context_crop_b64(frame, bbox, target_w=300, target_h=300)

        return {
            "confidence": det["confidence"],
            "bbox": bbox,
            "body_crop_b64": body_crop_b64,
            "context_b64": context_b64,
            "match_result": match_result,
            "candidate_scores": candidate_scores,
        }

    # 3) Build ally and enemy detection debug info
    ally_detection_results = []
    enemy_detection_results = []
    for det in corrected_dets:
        if det["class_name"] == "freindly_champion":
            ally_detection_results.append(_build_detection_debug(det, ally_keys))
        elif det["class_name"] == "enemy_champion":
            enemy_detection_results.append(_build_detection_debug(det, all_champion_keys))

    return {
        "frame_index": frame_index,
        "timestamp_ms": int(timestamp_s * 1000),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "max_frame_index": max_frame_index,
        "portrait_strip_b64": portrait_strip_b64,
        "player": {
            "key": player_info["key"] if player_info else "unknown",
            "name": player_info["name"] if player_info else "Unknown",
            "confidence": player_info.get("confidence", 0.0) if player_info else 0.0,
            "portrait_crop_b64": player_portrait_b64,
        },
        "ally_portraits": ally_portraits,
        "ally_detections": ally_detection_results,
        "enemy_detections": enemy_detection_results,
    }


@router.get("/videos/{video_id}/debug/champion-id")
async def debug_champion_id(
    video_id: str,
    frame_index: int = Query(0, ge=0, description="Frame index in sample space"),
    sample_fps: float = Query(DEFAULT_SAMPLE_FPS, gt=0, description="Sampling FPS"),
):
    """Debug endpoint: portrait identification + in-game champion matching."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    video_path = storage.base / video["storage_path"]
    if not video_path.exists():
        return _error(404, "FILE_NOT_FOUND", "Video file not found on disk.")

    try:
        result = await asyncio.to_thread(
            _analyze_champion_id, str(video_path), frame_index, sample_fps
        )
    except RuntimeError as e:
        return _error(400, "FRAME_ERROR", str(e))

    return JSONResponse(content=result)
