# League AI Video Editor

Upload a short gameplay clip (.mp4 / .webm, max 60 s, max 50 MB), get it played back in the browser.

## File tree

```
League_AI_Video_Editor/
├── package.json              # root — runs both servers via `bun run dev`
├── .gitignore
├── README.md
├── apps/
│   ├── api/                  # FastAPI backend (Python)
│   │   ├── .gitignore
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   └── .venv/            # Python virtual environment (gitignored)
│   └── web/                  # React + Vite frontend (TypeScript)
│       ├── package.json
│       ├── tsconfig.json
│       ├── vite.config.ts
│       ├── index.html
│       └── src/
│           ├── main.tsx
│           ├── App.tsx
│           ├── App.css
│           └── vite-env.d.ts
```

## Prerequisites

- **Node.js** (v18+)
- **Bun** — install via `npm install -g bun`
- **Python 3.11+**
- **ffprobe** (optional, from FFmpeg) — improves duration detection accuracy

## Setup

```bash
# 1. Clone & cd into the repo
cd League_AI_Video_Editor

# 2. Create & activate a Python venv, install backend deps
python -m venv apps/api/.venv
# Windows (Git Bash):
source apps/api/.venv/Scripts/activate
# macOS / Linux:
# source apps/api/.venv/bin/activate
pip install -r apps/api/requirements.txt
deactivate

# 3. Install JS dependencies (root + frontend)
bun install
cd apps/web && bun install && cd ../..
```

## Run

```bash
bun run dev
```

This starts:
- **FastAPI** on http://localhost:8000 (with hot-reload)
- **Vite dev server** on http://localhost:8080 (proxies `/api/*` to :8000)

Open http://localhost:8080 in your browser.

## API

### `POST /api/v1/video`

Upload a video file via multipart form data.

**Success** — returns the video bytes with the correct `Content-Type` header.

**Failure** — returns JSON:

```json
{
  "error_code": "INVALID_EXTENSION | INVALID_MIME_TYPE | FILE_TOO_LARGE | VIDEO_TOO_LONG",
  "message": "Human-readable description",
  "details": null
}
```

### curl example

```bash
curl -X POST http://localhost:8000/api/v1/video \
  -F "file=@my-clip.mp4;type=video/mp4" \
  --output returned.mp4
```

## Tricky parts

### Duration validation

The backend tries **ffprobe** first (most accurate, handles VFR and all codecs). If ffprobe isn't installed, it falls back to **OpenCV** (`frame_count / fps`), which can be imperfect for variable-frame-rate files. If neither works, duration validation is skipped (the file is still accepted).

### Streaming response

`FileResponse` from Starlette streams the file in chunks rather than loading it all into memory. A `BackgroundTask` deletes the temp file after the response finishes.

### No CORS needed

Vite's dev server proxies all `/api/*` requests to FastAPI on :8000, so both frontend and API appear on the same origin (:8080).

---

## OPTIONAL — Future extensions (not implemented)

### Step A: Object storage

Store uploaded videos in **Supabase Storage** or **Cloudflare R2** instead of temp files.

- Backend generates a signed upload URL → client uploads directly to storage.
- Backend receives a webhook or the client confirms upload, then stores the object key in the DB.
- Playback uses a signed download URL with a short TTL.

### Step B: LLM frame analysis

Extract frames at ~1 fps using ffmpeg/OpenCV, then send them to an **LLM vision model** (e.g., Claude) for coaching feedback.

- `ffmpeg -i clip.mp4 -vf fps=1 frame_%04d.jpg`
- Batch frames into a single multi-image prompt: *"Analyze this League of Legends gameplay. What mistakes did the player make? What could they improve?"*
- Stream the LLM response back to the frontend via SSE or WebSocket.

### Step C: Persist analysis results

Save analysis results to **Supabase Postgres** keyed by a unique video ID.

- Schema: `videos(id uuid PK, storage_key text, uploaded_at timestamptz)` + `analyses(id uuid PK, video_id uuid FK, model text, result jsonb, created_at timestamptz)`.
- Frontend can fetch past analyses by video ID.
- Add a simple listing page showing upload history + analysis status.
