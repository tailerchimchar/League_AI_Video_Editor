"""Video upload and retrieval endpoints."""

import hashlib
import os
from pathlib import Path

import cv2
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from db.queries import insert_video, get_video, list_videos
from storage.local import storage

router = APIRouter(tags=["videos"])

ALLOWED_EXTENSIONS = {".mp4", ".webm"}
ALLOWED_MIMETYPES = {"video/mp4", "video/webm"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB (increased for extraction pipeline)
CHUNK_SIZE = 1024 * 1024  # 1 MB


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message})


@router.post("/videos")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file. Saves to disk and creates a DB record.

    Deduplication: computes SHA-256 of the file content and checks if a
    video with the same hash already exists. If so, returns the existing
    record without creating a duplicate.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _error(400, "INVALID_EXTENSION", f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are accepted.")

    if file.content_type not in ALLOWED_MIMETYPES:
        return _error(400, "INVALID_MIME_TYPE", f"Invalid MIME type '{file.content_type}'.")

    # Generate a temp ID for the file path, will be replaced by DB UUID
    import uuid
    temp_id = uuid.uuid4().hex
    dest = storage.video_path(temp_id, ext)

    size = 0
    sha256 = hashlib.sha256()
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    f.close()
                    dest.unlink(missing_ok=True)
                    return _error(413, "FILE_TOO_LARGE", f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB.")
                f.write(chunk)
                sha256.update(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    file_hash = sha256.hexdigest()

    # Check for existing video with the same content hash
    from db.engine import get_pool
    pool = get_pool()
    existing = await pool.fetchrow(
        "SELECT * FROM videos WHERE file_hash = $1 LIMIT 1",
        file_hash,
    )
    if existing:
        # Duplicate — remove the temp file and return the existing record
        dest.unlink(missing_ok=True)
        return JSONResponse(content={
            "video_id": str(existing["id"]),
            "filename": existing["filename"],
            "duration_ms": existing["duration_ms"],
            "width": existing["width"],
            "height": existing["height"],
            "status": existing["status"],
            "deduplicated": True,
        })

    # Get video metadata via OpenCV
    cap = cv2.VideoCapture(str(dest))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()

    duration_ms = int((frame_count / fps) * 1000) if fps > 0 and frame_count > 0 else None
    mime_type = "video/mp4" if ext == ".mp4" else "video/webm"

    relative_path = storage.get_relative_path(dest)

    row = await insert_video(
        filename=file.filename or "unknown",
        storage_path=relative_path,
        file_size_bytes=size,
        duration_ms=duration_ms,
        width=width,
        height=height,
        mime_type=mime_type,
        file_hash=file_hash,
    )

    # Rename file to use the DB-generated UUID
    video_id = str(row["id"])
    new_dest = storage.video_path(video_id, ext)
    dest.rename(new_dest)

    # Update storage path in DB
    new_relative = storage.get_relative_path(new_dest)
    await pool.execute(
        "UPDATE videos SET storage_path = $1 WHERE id = $2",
        new_relative, row["id"],
    )

    return JSONResponse(content={
        "video_id": video_id,
        "filename": file.filename,
        "duration_ms": duration_ms,
        "width": width,
        "height": height,
        "status": "uploaded",
    })


@router.get("/videos/{video_id}")
async def serve_video(video_id: str):
    """Serve a video file for playback."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    video_path = storage.base / video["storage_path"]
    if not video_path.exists():
        return _error(404, "FILE_NOT_FOUND", "Video file not found on disk.")

    return FileResponse(
        str(video_path),
        media_type=video["mime_type"],
        filename=video["filename"],
    )


@router.get("/videos")
async def list_all_videos(limit: int = 50, offset: int = 0):
    """List all uploaded videos."""
    rows = await list_videos(limit=limit, offset=offset)
    return JSONResponse(content={
        "videos": [
            {
                "id": str(r["id"]),
                "filename": r["filename"],
                "duration_ms": r["duration_ms"],
                "width": r["width"],
                "height": r["height"],
                "status": r["status"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]
    })
