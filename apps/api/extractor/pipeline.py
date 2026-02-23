"""Extraction pipeline orchestrator.

Runs frame sampling, cropping, OCR, feature computation, and segmentation
as a background task. Processes frames in batches so progress updates are
visible in the UI during extraction (not just during DB inserts).
"""

import asyncio
import logging

import cv2

from extractor.config import BATCH_INSERT_SIZE
from extractor.sampler import sample_frames
from extractor.cropper import crop_hud_regions, save_crops_to_disk
from extractor.ocr import run_ocr_on_crops
from extractor.features import compute_derived_features
from extractor.segmenter import detect_segments
from extractor.detector import YoloDetector
from extractor.champion_id import ChampionIdentifier
from extractor.postprocess import correct_detections
from extractor.tracker import SimpleTracker
from db.queries import (
    get_video,
    update_job_status,
    update_job_progress,
    batch_insert_frames,
    batch_insert_segments,
    get_frames_for_job,
)
from storage.local import storage

logger = logging.getLogger(__name__)

# In-memory extraction progress: job_id -> {"extracted": int, "total": int, "phase": str}
# Updated from the extraction thread, read by the status endpoint.
live_progress: dict[str, dict] = {}

_detector: YoloDetector | None = None
_champion_identifier: ChampionIdentifier | None = None


def _get_champion_identifier() -> ChampionIdentifier:
    """Get or create the ChampionIdentifier singleton (icons-only mode).

    Re-creates if the previous one was unavailable (icons might have been
    downloaded after the API started). Uses icons_only=True to avoid noise
    from skin loading screen images that look nothing like HUD portraits.
    """
    global _champion_identifier
    if _champion_identifier is None or not _champion_identifier.available:
        _champion_identifier = ChampionIdentifier(icons_only=True)
    return _champion_identifier


def _get_detector() -> YoloDetector:
    """Get or create the YOLO detector singleton.

    Re-creates the instance if the previous one was unavailable (model file
    might have been added after the API started).
    """
    global _detector
    if _detector is None or not _detector.available:
        _detector = YoloDetector()
    return _detector


def _estimate_total_frames(video_path: str, sample_fps: float, max_frames: int = 300) -> int:
    """Estimate how many frames the sampler will yield."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_fps <= 0 or total_frames <= 0:
        return 0
    step = max(1, int(video_fps / sample_fps))
    return min(total_frames // step, max_frames)


DETECTION_BATCH_SIZE = 8  # Number of frames to batch for GPU detection


def _process_frame_batch(
    frames: list[tuple],
    video_id: str,
    job_id: str,
    width: int,
    height: int,
    enable_ocr: bool,
    detector,
    tracker: SimpleTracker | None,
    crop_output_dir: str | None,
    prev_ocr: dict | None,
    prev_ts: int,
    game_champions: dict | None = None,
) -> list[dict]:
    """Process a batch of frames: batch YOLO detection on GPU, then sequential OCR/tracking."""
    # Step 1: Batch YOLO detection (single GPU call for all frames)
    if detector and detector.available:
        frame_arrays = [f[2] for f in frames]
        batch_detections = detector.detect_batch(frame_arrays)
    else:
        batch_detections = [[] for _ in frames]

    # Step 2: Process each frame sequentially (OCR, color correction, tracking)
    payloads = []
    for i, (frame_idx, ts_ms, frame_np) in enumerate(frames):
        crops = crop_hud_regions(frame_np, width, height)

        ocr_data: dict = {}
        if enable_ocr and crops:
            ocr_data = run_ocr_on_crops(crops)

        if game_champions:
            ocr_data["game_champions"] = game_champions

        detections = batch_detections[i]
        if detections:
            detections = correct_detections(frame_np, detections)

        # Enrich played_champion detections with champion name BEFORE tracking
        if game_champions and detections:
            player_info = game_champions.get("player")
            if player_info:
                for det in detections:
                    if det["class_name"] == "played_champion":
                        det["champion"] = player_info["name"]
                        det["champion_confidence"] = (
                            1.0 if player_info.get("confidence", 0) >= 1.0
                            else player_info.get("confidence", 0)
                        )

        if tracker is not None and detections:
            detections = tracker.update(detections)

        dt_ms = ts_ms - prev_ts if prev_ocr is not None else 0
        derived = compute_derived_features(ocr_data, prev_ocr, dt_ms)

        crop_paths: dict = {}
        if crop_output_dir:
            crop_paths = save_crops_to_disk(crops, crop_output_dir, frame_idx)

        payload = {
            "job_id": job_id,
            "video_id": video_id,
            "frame_index": frame_idx,
            "timestamp_ms": ts_ms,
            "ocr_data": ocr_data,
            "detections": detections,
            "derived_features": derived,
            "crop_paths": crop_paths,
        }
        payloads.append(payload)
        prev_ocr = ocr_data
        prev_ts = ts_ms

    return payloads


async def run_extraction_pipeline(
    video_id: str,
    job_id: str,
    config: dict,
) -> None:
    """Main extraction orchestrator. Runs as a BackgroundTask."""
    try:
        await update_job_status(job_id, "running")
        logger.info("Pipeline starting for video %s, job %s", video_id, job_id)

        video = await get_video(video_id)
        if not video:
            await update_job_status(job_id, "failed", error_message="Video not found")
            return

        video_path = str(storage.base / video["storage_path"])
        width = video.get("width") or 1920
        height = video.get("height") or 1080
        logger.info("Video path: %s, resolution: %sx%s", video_path, width, height)

        sample_fps = config.get("sample_fps", 2.0)
        enable_ocr = config.get("enable_ocr", True)
        # Auto-enable detector when model file exists (unless explicitly disabled)
        detector_available = _get_detector().available
        enable_detector = config.get("enable_detector", detector_available)
        enable_debug_overlays = config.get("enable_debug_overlays", False)
        played_champion_name = config.get("played_champion")  # user-specified champion name

        detector = _get_detector() if enable_detector else None
        if detector:
            logger.info("Detector enabled (available=%s)", detector.available)
        else:
            logger.info("Detector disabled")
        crop_output_dir = str(storage.crop_dir(video_id)) if enable_debug_overlays else None

        # Estimate total for progress calculation
        estimated_total = await asyncio.to_thread(
            _estimate_total_frames, video_path, sample_fps
        )
        logger.info("Estimated %d frames to extract (fps=%.1f, ocr=%s)", estimated_total, sample_fps, enable_ocr)

        # Publish live progress for the status endpoint
        live_progress[job_id] = {"extracted": 0, "total": estimated_total, "phase": "extracting"}

        # Process all frames with tracking for temporal consistency
        prev_ocr: dict | None = None
        prev_ts: int = 0

        # Champion identification — run on first frame, cache for all frames
        champion_id = _get_champion_identifier()
        enable_champion_id = champion_id.available

        def _extract_all() -> list[dict]:
            nonlocal prev_ocr, prev_ts
            all_payloads = []
            game_champions: dict | None = None
            tracker = SimpleTracker()
            count = 0
            frame_buffer: list[tuple] = []

            for frame_idx, ts_ms, frame_np in sample_frames(video_path, fps=sample_fps):
                # On first frame, identify champions
                if count == 0:
                    game_champions = {"player": None, "allies": []}

                    # User-specified played champion takes priority
                    if played_champion_name:
                        game_champions["player"] = {
                            "key": played_champion_name,
                            "name": played_champion_name,
                            "confidence": 1.0,
                        }
                        logger.info("Player champion (user-specified): %s", played_champion_name)

                    # Auto-identify from HUD portraits (icons-only for accuracy)
                    if enable_champion_id:
                        try:
                            auto = champion_id.identify_from_portraits(
                                frame_np, width, height
                            )
                            # Use auto player only if user didn't specify
                            if not played_champion_name and auto.get("player"):
                                game_champions["player"] = auto["player"]
                            game_champions["allies"] = auto.get("allies", [])
                            logger.info(
                                "Champion roster: player=%s, allies=%s",
                                game_champions["player"]["name"] if game_champions["player"] else "unknown",
                                [a["name"] for a in game_champions["allies"]],
                            )
                        except Exception:
                            logger.exception("Champion identification failed on first frame")

                frame_buffer.append((frame_idx, ts_ms, frame_np))
                count += 1

                # Process batch when full
                if len(frame_buffer) >= DETECTION_BATCH_SIZE:
                    payloads = _process_frame_batch(
                        frame_buffer, video_id, job_id, width, height,
                        enable_ocr, detector, tracker, crop_output_dir,
                        prev_ocr, prev_ts, game_champions=game_champions,
                    )
                    all_payloads.extend(payloads)
                    prev_ocr = payloads[-1]["ocr_data"]
                    prev_ts = payloads[-1]["timestamp_ms"]
                    frame_buffer = []
                    live_progress[job_id] = {"extracted": count, "total": estimated_total, "phase": "extracting"}
                    if count % 10 == 0:
                        logger.info("Extracted %d/%d frames...", count, estimated_total)

            # Flush remaining frames
            if frame_buffer:
                payloads = _process_frame_batch(
                    frame_buffer, video_id, job_id, width, height,
                    enable_ocr, detector, tracker, crop_output_dir,
                    prev_ocr, prev_ts, game_champions=game_champions,
                )
                all_payloads.extend(payloads)
                live_progress[job_id] = {"extracted": count, "total": estimated_total, "phase": "extracting"}

            return all_payloads

        all_payloads = await asyncio.to_thread(_extract_all)
        total = len(all_payloads)
        logger.info("Frame extraction done: %d frames. Inserting into DB...", total)
        live_progress[job_id] = {"extracted": total, "total": total, "phase": "inserting"}

        # Batch insert with progress updates
        for i in range(0, total, BATCH_INSERT_SIZE):
            batch = all_payloads[i:i + BATCH_INSERT_SIZE]
            await batch_insert_frames(batch)
            progress = min((i + len(batch)) / total, 1.0) if total > 0 else 1.0
            await update_job_progress(job_id, progress, i + len(batch))

        # Run segmenter
        all_frames = await get_frames_for_job(job_id)
        frame_dicts = []
        for f in all_frames:
            fd = dict(f)
            for key in ("ocr_data", "derived_features", "detections"):
                if isinstance(fd.get(key), str):
                    import json
                    fd[key] = json.loads(fd[key])
            frame_dicts.append(fd)

        segments = detect_segments(frame_dicts, video_id, job_id)
        if segments:
            await batch_insert_segments(segments)

        await update_job_status(job_id, "completed")
        live_progress.pop(job_id, None)
        logger.info("Extraction completed for video %s, job %s: %d frames, %d segments",
                     video_id, job_id, total, len(segments))

    except Exception as e:
        logger.exception("Extraction pipeline failed for video %s, job %s", video_id, job_id)
        live_progress.pop(job_id, None)
        await update_job_status(job_id, "failed", error_message=str(e))
