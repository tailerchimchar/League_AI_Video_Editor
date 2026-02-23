"""Extraction pipeline endpoints."""

import json

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

from db.queries import (
    get_video,
    insert_job,
    get_job,
    get_latest_job_for_video,
    get_frames_for_video,
    get_segments_for_video,
)
from extractor.pipeline import run_extraction_pipeline
from schemas.extraction import ExtractionConfig

router = APIRouter(tags=["extraction"])


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message})


@router.post("/videos/{video_id}/extract")
async def start_extraction(
    video_id: str,
    config: ExtractionConfig | None = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Start an extraction job for a video."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    if config is None:
        config = ExtractionConfig()

    # Create analysis job
    job = await insert_job(
        video_id=video_id,
        config=config.model_dump(),
    )
    job_id = str(job["id"])

    # Launch pipeline as background task
    background_tasks.add_task(
        run_extraction_pipeline,
        video_id=video_id,
        job_id=job_id,
        config=config.model_dump(),
    )

    return JSONResponse(content={
        "job_id": job_id,
        "video_id": video_id,
        "status": "pending",
        "progress": 0.0,
    })


@router.get("/videos/{video_id}/extract/status")
async def extraction_status(video_id: str):
    """Get the status of the latest extraction job for a video."""
    job = await get_latest_job_for_video(video_id)
    if not job:
        return _error(404, "NOT_FOUND", "No extraction job found for this video.")

    return JSONResponse(content={
        "job_id": str(job["id"]),
        "video_id": str(job["video_id"]),
        "status": job["status"],
        "progress": job["progress"] or 0.0,
        "frame_count": job["frame_count"] or 0,
        "error_message": job["error_message"],
        "started_at": job["started_at"].isoformat() if job["started_at"] else None,
        "completed_at": job["completed_at"].isoformat() if job["completed_at"] else None,
    })


@router.get("/videos/{video_id}/frames")
async def get_frames(
    video_id: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 100,
):
    """Query extracted frame payloads for a video."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    frames = await get_frames_for_video(
        video_id, start_ms=start_ms, end_ms=end_ms, limit=limit
    )

    result = []
    for f in frames:
        fd = dict(f)
        # Ensure JSON fields are properly serialized
        for key in ("ocr_data", "detections", "derived_features", "crop_paths"):
            if isinstance(fd.get(key), str):
                fd[key] = json.loads(fd[key])

        result.append({
            "frame_index": fd["frame_index"],
            "timestamp_ms": fd["timestamp_ms"],
            "ocr_data": fd.get("ocr_data", {}),
            "detections": fd.get("detections", {}),
            "derived_features": fd.get("derived_features", {}),
        })

    return JSONResponse(content={"frames": result, "count": len(result)})


@router.get("/videos/{video_id}/segments")
async def get_segments(video_id: str):
    """Get detected segments for a video."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    segments = await get_segments_for_video(video_id)

    result = []
    for s in segments:
        sd = dict(s)
        features = sd.get("features", {})
        if isinstance(features, str):
            features = json.loads(features)

        result.append({
            "id": str(sd["id"]),
            "segment_type": sd["segment_type"],
            "start_ms": sd["start_ms"],
            "end_ms": sd["end_ms"],
            "confidence": sd["confidence"],
            "features": features,
        })

    return JSONResponse(content={"segments": result, "count": len(result)})
