"""
League AI Video Editor — FastAPI backend.

POST /api/v1/video
  Accepts a multipart upload (field: "file"), validates it, and echoes back
  the same video bytes so the frontend can create a Blob URL and play it.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

app = FastAPI(title="League AI Video Editor API")

# ── Constants ──────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: set[str] = {".mp4", ".webm"}
ALLOWED_MIMETYPES: set[str] = {"video/mp4", "video/webm"}
MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
MAX_DURATION: float = 60.0  # seconds
CHUNK_SIZE: int = 1024 * 1024  # 1 MB stream chunks


# ── Duration helpers ───────────────────────────────────────────────────────────
def _duration_ffprobe(path: str) -> float | None:
    """Try to get duration via ffprobe (most reliable)."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None
    return None


def _duration_opencv(path: str) -> float | None:
    """
    Fallback: estimate duration via OpenCV (frame_count / fps).

    WARNING: This can be imperfect for variable-frame-rate videos or certain
    codecs where OpenCV misreports frame count. Prefer ffprobe when available.
    """
    try:
        import cv2  # noqa: PLC0415 — lazy import so the app starts even without cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return None


def get_duration(path: str) -> float | None:
    """Return video duration in seconds, or None if we can't determine it."""
    return _duration_ffprobe(path) or _duration_opencv(path)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _error(status: int, code: str, message: str, details: object = None) -> JSONResponse:
    body: dict = {"error_code": code, "message": message}
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status, content=body)


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ── Route ──────────────────────────────────────────────────────────────────────
@app.post("/api/v1/video")
async def upload_video(file: UploadFile = File(...)):
    # 1. Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _error(
            400,
            "INVALID_EXTENSION",
            f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are accepted.",
        )

    # 2. Validate MIME type
    if file.content_type not in ALLOWED_MIMETYPES:
        return _error(
            400,
            "INVALID_MIME_TYPE",
            f"Invalid MIME type '{file.content_type}'. Accepted: {', '.join(sorted(ALLOWED_MIMETYPES))}.",
        )

    # 3. Stream to temp file, enforcing max size
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    size = 0
    try:
        while chunk := await file.read(CHUNK_SIZE):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                tmp.close()
                _cleanup(tmp.name)
                return _error(
                    413,
                    "FILE_TOO_LARGE",
                    f"File exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
                )
            tmp.write(chunk)
        tmp.close()
    except Exception:
        tmp.close()
        _cleanup(tmp.name)
        raise

    # 4. Validate duration
    duration = get_duration(tmp.name)
    if duration is not None and duration > MAX_DURATION:
        _cleanup(tmp.name)
        return _error(
            400,
            "VIDEO_TOO_LONG",
            f"Video duration ({duration:.1f}s) exceeds maximum of {MAX_DURATION:.0f}s.",
        )

    # 5. Return the video — FileResponse streams efficiently and the
    #    BackgroundTask cleans up the temp file after the response is sent.
    media_type = "video/mp4" if ext == ".mp4" else "video/webm"
    return FileResponse(
        tmp.name,
        media_type=media_type,
        filename=file.filename,
        background=BackgroundTask(_cleanup, tmp.name),
    )
