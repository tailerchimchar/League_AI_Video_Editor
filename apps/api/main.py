"""
League AI Video Editor — FastAPI backend.

Legacy endpoints (backward compatible):
  POST /api/v1/video       Upload a video (in-memory)
  GET  /api/v1/video/{id}  Serve a previously uploaded video
  GET  /api/v1/analyze/{id} SSE stream of Claude vision-based coaching feedback

New v1 endpoints (DB-backed extraction pipeline):
  POST   /api/v1/videos                          Upload video → DB + disk
  POST   /api/v1/videos/{id}/extract              Start extraction job
  GET    /api/v1/videos/{id}/extract/status        Poll job progress
  GET    /api/v1/videos/{id}/frames                Query frame payloads
  GET    /api/v1/videos/{id}/segments              Get detected segments
  GET    /api/v1/videos/{id}/report                SSE evidence-grounded report
  POST   /api/v1/videos/{id}/labels                Submit manual labels
  GET    /api/v1/videos/{id}                       Serve video file
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Load .env from repo root (two levels up from apps/api/)
_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan: init/close DB pool ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize DB connection pool if DATABASE_URL is set
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        from db.engine import init_db
        await init_db()
        logger.info("Database pool initialized")
    else:
        logger.warning("DATABASE_URL not set — new endpoints will not work")
    yield
    # Shutdown: close DB pool
    if db_url:
        from db.engine import close_db
        await close_db()
        logger.info("Database pool closed")


app = FastAPI(title="League AI Video Editor", lifespan=lifespan)

# ── CORS (needed for direct uploads from Vite dev server) ─────────────────────
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount new routers ─────────────────────────────────────────────────────────
# These require Postgres; they gracefully error if DB is not available.
from routers import videos, extraction, reports, labels

app.include_router(videos.router, prefix="/api/v1")
app.include_router(extraction.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(labels.router, prefix="/api/v1")


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY ENDPOINTS (backward compatible — kept for existing frontend)
# ══════════════════════════════════════════════════════════════════════════════

ALLOWED_EXTENSIONS: set[str] = {".mp4", ".webm"}
ALLOWED_MIMETYPES: set[str] = {"video/mp4", "video/webm"}
MAX_FILE_SIZE: int = 50 * 1024 * 1024
MAX_DURATION: float = 60.0
CHUNK_SIZE: int = 1024 * 1024

STORAGE_DIR = tempfile.mkdtemp(prefix="league_ai_")
_legacy_videos: dict[str, dict] = {}


def _duration_ffprobe(path: str) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None
    return None


def _duration_opencv(path: str) -> float | None:
    try:
        import cv2
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


def _get_duration(path: str) -> float | None:
    return _duration_ffprobe(path) or _duration_opencv(path)


def _extract_frames(video_path: str, max_frames: int = 12) -> list[str]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    num = min(total_frames, max_frames)
    indices = [int(i * total_frames / num) for i in range(num)]
    frames: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                frames.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return frames


def _error(status: int, code: str, message: str, details: object = None) -> JSONResponse:
    body: dict = {"error_code": code, "message": message}
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status, content=body)


@app.post("/api/v1/video")
async def legacy_upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _error(400, "INVALID_EXTENSION", f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are accepted.")
    if file.content_type not in ALLOWED_MIMETYPES:
        return _error(400, "INVALID_MIME_TYPE", f"Invalid MIME type '{file.content_type}'.")

    video_id = uuid.uuid4().hex
    dest = os.path.join(STORAGE_DIR, f"{video_id}{ext}")
    size = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    f.close()
                    os.unlink(dest)
                    return _error(413, "FILE_TOO_LARGE", f"File exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)} MB.")
                f.write(chunk)
    except Exception:
        if os.path.exists(dest):
            os.unlink(dest)
        raise

    duration = _get_duration(dest)
    if duration is not None and duration > MAX_DURATION:
        os.unlink(dest)
        return _error(400, "VIDEO_TOO_LONG", f"Video duration ({duration:.1f}s) exceeds maximum of {MAX_DURATION:.0f}s.")

    media_type = "video/mp4" if ext == ".mp4" else "video/webm"
    _legacy_videos[video_id] = {
        "path": dest, "filename": file.filename, "duration": duration, "media_type": media_type,
    }
    return JSONResponse(content={"video_id": video_id, "filename": file.filename, "duration": duration})


@app.get("/api/v1/video/{video_id}")
async def legacy_get_video(video_id: str):
    meta = _legacy_videos.get(video_id)
    if not meta:
        return _error(404, "NOT_FOUND", "Video not found.")
    return FileResponse(meta["path"], media_type=meta["media_type"], filename=meta["filename"])


@app.get("/api/v1/analyze/{video_id}")
async def legacy_analyze_video(video_id: str):
    meta = _legacy_videos.get(video_id)
    if not meta:
        return _error(404, "NOT_FOUND", "Video not found.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        return _error(500, "MISSING_API_KEY", "ANTHROPIC_API_KEY is not configured.")

    async def event_stream():
        try:
            from anthropic import AsyncAnthropic

            frames = await asyncio.to_thread(_extract_frames, meta["path"])
            if not frames:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Could not extract frames from video.'})}\n\n"
                return

            content: list[dict] = []
            for frame_b64 in frames:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64},
                })
            content.append({
                "type": "text",
                "text": (
                    "These are frames extracted from a League of Legends gameplay clip. "
                    "Analyze the gameplay and provide specific, actionable coaching feedback. "
                    "Cover positioning, map awareness, ability usage, CS/farming, "
                    "team fighting, objective control, and any mistakes you spot."
                ),
            })

            client = AsyncAnthropic(api_key=api_key)
            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=(
                    "You are an expert League of Legends coach. "
                    "Analyze gameplay frames and provide specific, actionable coaching advice. "
                    "Be encouraging but honest about mistakes. "
                    "Structure your feedback with clear sections and priorities."
                ),
                messages=[{"role": "user", "content": content}],
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
