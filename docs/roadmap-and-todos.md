# League AI Video Editor — Roadmap & TODO Checklist

## The Vision

A coaching platform where:
1. **Pro player VODs** are ingested and processed through a deterministic extraction pipeline, building a massive labeled dataset of "what great players do"
2. A **machine learning model** is trained on this pro data to understand what good play looks like — not through vibes, but through measurable features (CS/min, trading patterns, positioning, objective timing)
3. A **user uploads their own clip** and gets specific, evidence-grounded coaching that compares their play against pro patterns — "you traded at 3:24 and lost 40% HP for 15 CS; in a similar situation, Faker maintained spacing and farmed safely"
4. The system **scales to thousands of concurrent users** with low latency, high availability, and consistent quality

**The core difference from existing tools:** Everything is deterministic and evidence-based. No LLM hallucination. The AI doesn't guess what happened — it knows, because the data pipeline extracted it. The LLM's job is to **explain and coach**, not to **perceive**.

---

## Phase Overview

| Phase | Focus | Timeline | Status |
|-------|-------|----------|--------|
| **Phase 1** | Core extraction pipeline + evidence-grounded reports | Weeks 1-2 | **Complete** |
| **Phase 1.5** | YOLO object detection + OCR calibration | Week 3 | **Complete** |
| **Phase 2A** | Champion identification (portrait matching + user input) | Week 3 | **Complete** |
| **Phase 2B** | Health bar color tuning + admin debug tools + filter toggles | Week 4 | **Complete** |
| **Phase 2** | Lightweight learning (retrieval + GRU classifier) | Weeks 4-5 | Not Started |
| **Phase 3** | Pro VOD data pipeline + model training | Weeks 5-8 | Not Started |
| **Phase 4** | Scale, auth, production deployment | Weeks 9-12 | Not Started |

---

## Phase 1: Core Pipeline (Weeks 1-2)

**Goal:** Upload a video → extract structured game data → generate evidence-grounded coaching report.

### Infrastructure
- [x] Docker Compose + PostgreSQL 16
- [x] Database schema (6 tables: videos, analysis_jobs, frame_payloads, segments, labels, reports)
- [x] Migration runner
- [x] asyncpg connection pool + typed queries
- [x] Local disk storage abstraction
- [ ] **Install Tesseract OCR on dev machine**
  - Download from https://github.com/UB-Mannheim/tesseract/wiki
  - Add `C:\Program Files\Tesseract-OCR` to PATH
  - Verify: `tesseract --version`
- [ ] **Start Postgres and run migrations**
  - `docker compose up -d postgres`
  - `cd apps/api; .venv/Scripts/python -m migrations.run`
- [ ] **Install Python dependencies**
  - `cd apps/api; .venv/Scripts/pip install -r requirements.txt`

### Extraction Pipeline
- [x] Frame sampler (OpenCV, 2 fps)
- [x] Deterministic HUD cropper (9 regions)
- [x] Tesseract OCR with preprocessing
- [x] Feature computer (HP/CS/gold deltas)
- [x] Rule-based segmenter (fight + death)
- [x] Pipeline orchestrator with BackgroundTask
- [x] YOLO detector (complete — see Phase 1.5 section)
- [ ] **Graceful fallback when Tesseract not installed** (pipeline runs but OCR returns empty)
- [ ] **Test full pipeline end-to-end with a real League clip**
- [ ] **Calibrate OCR crop coordinates against actual League 1080p screenshots**
  - Use `scripts/test_ocr.py <screenshot.png>` to debug
  - Adjust values in `extractor/config.py`

### API Endpoints
- [x] POST /api/v1/videos (upload)
- [x] POST /api/v1/videos/{id}/extract (start extraction)
- [x] GET /api/v1/videos/{id}/extract/status (poll progress)
- [x] GET /api/v1/videos/{id}/frames (query frame data)
- [x] GET /api/v1/videos/{id}/segments (get segments)
- [x] GET /api/v1/videos/{id}/report (evidence-grounded SSE report)
- [x] POST /api/v1/videos/{id}/labels (manual labeling)
- [x] Legacy endpoints preserved (/api/v1/video, /api/v1/analyze)

### Frontend
- [x] Mode toggle (Extraction Pipeline vs Quick Analysis)
- [x] VideoUploader component
- [x] ExtractionStatus with progress bar
- [x] FrameTimeline data table
- [x] SegmentList with type badges
- [x] ReportView with SSE streaming
- [ ] **Test full UI flow end-to-end**
- [ ] **Add error recovery** (retry extraction, handle disconnects)

### Report Generation
- [x] Evidence payload builder (assembles OCR timeline, segments, feature summary)
- [x] System prompt for evidence-grounded coaching
- [x] SSE streaming + DB storage
- [ ] **Add evidence citations** (link report claims to specific frames/timestamps)
- [ ] **Test report quality with real extraction data**

---

## Phase 1.5: Object Detection + OCR Polish (Week 3) — COMPLETE

**Goal:** Add YOLO-based detection for champions, minions, towers. Calibrate OCR for accuracy.

### OCR Calibration
- [x] Crop coordinates calibrated for Outplayed recordings at 1080p
- [x] `scripts/test_ocr.py` used to debug and verify OCR accuracy
- [x] Timer, KDA, CS, level, HP% all reading correctly
- [x] Added `health_bar` and `resources_bar` crop regions
- [ ] Test at 1440p and 720p — verify proportional scaling works
- [ ] Add support for different HUD skins / UI scales
- [ ] Measure OCR accuracy per field systematically

### YOLO Detector
- [x] Found vasyz's "LeagueOfLegends" Roboflow dataset (v4, 435 images, 16 classes)
- [x] Downloaded dataset in YOLOv8 format to `data/datasets/league-yolo/`
- [x] Trained YOLOv8n for 50 epochs at 640px (transfer from COCO pretrained)
  - mAP50 = 0.845, mAP50-95 = 0.614
  - Best: played_champion (0.995), friendly_tower (0.995), enemy_tower (0.955)
  - Weakest: enemy_melee_minion (0.117 — only 2 val samples)
- [x] Exported to ONNX at `data/models/yolov8n_lol.onnx` (11.7 MB)
- [x] Completed `detector.py` postprocessing (tensor decode, per-class NMS, box scaling)
- [x] Detector auto-enables when ONNX model exists on disk
- [x] Detection results stored in `frame_payloads.detections` JSONB column
- [x] Created `scripts/train_yolo.py` for reproducible retraining
- [ ] Add detection-based features (champion proximity, ward coverage)

### Frontend Overlay
- [x] AI Vision overlay draws OCR crop regions with live values
- [x] YOLO detection boxes drawn with dashed borders + class labels
- [x] Color-coded: gold (played champion), red (enemies), blue (friendlies)
- [x] Boxes move frame-to-frame as video plays (synced via binary search)
- [ ] Save debug overlay images to `data/debug/`
- [ ] Toggle via `enable_debug_overlays` in extraction config

### Documentation
- [x] Created `docs/yolo-detection-guide.md` — comprehensive step-by-step guide

---

## Phase 2A: Champion Identification (Week 3) — COMPLETE

**Goal:** Identify *which* champion each YOLO detection represents (e.g., "Caitlyn" instead of "played_champion"). Two-stage approach: YOLO detects bounding boxes, then a separate system identifies the champion.

### Data Dragon Icon Download
- [x] `scripts/download_champion_icons.py` fetches latest champion data from Riot CDN
- [x] 172 champion square icons (120x120 PNG) saved to `data/champions/icons/`
- [x] 2054 skin loading screen images (308x560 JPG) via `--skins` flag to `data/champions/skins/`
- [x] `manifest.json` generated with champion key/name mapping + skin lists
- [x] `data/champions/` added to `.gitignore`

### Portrait Matching (Auto-Identification)
- [x] `extractor/champion_id.py` — ChampionIdentifier class with HSV histogram matching
- [x] Extracts player portrait from bottom-center HUD (x:650, y:1005, 60x52)
- [x] Extracts 4 ally portraits above minimap (x:1635-1856, y:688, 50x38 each)
- [x] Matches against both default icons and skin loading screen face crops
- [x] Best-per-champion scoring (highest score across all skin variants wins)
- [x] Follows same singleton pattern as YoloDetector
- [x] `scripts/calibrate_scoreboard.py` debug tool for visualizing crops and match results
- [ ] Accuracy is limited (~50% for allies) — skins change appearance too drastically for template matching
- [ ] Enemy identification not possible from HUD (would need Tab screen or YOLO Phase B)

### User-Specified Champion Input
- [x] `played_champion` field added to `ExtractionConfig` schema (Python + TypeScript)
- [x] Text input in ExtractionStatus component ("Your champion (e.g. Caitlyn)")
- [x] User-specified name takes priority over auto-matching (confidence=1.0)
- [x] Pipeline enriches `played_champion` YOLO detections with the champion name
- [x] Frontend overlay shows champion name instead of generic class label

### Key Finding
Image matching (template, histogram, ORB features) fundamentally cannot reliably identify champions across skins. Pool Party Caitlyn scores higher as LeBlanc than as Caitlyn in every metric tested. Future work: train a CNN embedding classifier on cropped YOLO bounding boxes (Phase B).

---

## Phase 2B: Health Bar Detection Tuning & Admin Tools (Week 4) — COMPLETE

**Goal:** Fix health bar color correction accuracy, build debug tooling for rapid iteration, add UI polish.

*Full writeup: [docs/phase-2b-health-bar-detection-and-admin-tools.md](./phase-2b-health-bar-detection-and-admin-tools.md)*

### Health Bar Crop Region Fix
- [x] Discovered crop was scanning **above** YOLO bbox (landing on name text, not health bar)
- [x] Fixed to scan **inside** top 0-10% of bbox (where health bar actually sits)
- [x] Iterated through values (5-25% → 0-22% → 0-10%) using admin tool to avoid mana bar contamination
- [x] Fixed in 3 files: `postprocess.py`, `debug.py` (crop + fractions + context visualization)
- [x] `correct_detections()` now stores `original_class` for debug visibility

### Admin Debug Tool (`/admin`)
- [x] `GET /api/v1/videos/{id}/debug/health-bar-colors` — runs YOLO + color correction live on any frame
- [x] Returns per-detection: class, confidence, bbox, color fractions, health bar crop (base64), context crop (base64)
- [x] Context crop (400x140) shows gameplay with yellow bbox + green health bar scan region drawn
- [x] Frontend: video selector, frame stepper (prev/next + slider + number input), detection cards
- [x] Shows "was X" badge when a detection was reclassified by color correction
- [x] Built with Tailwind CSS (v4)
- [x] React Router integration: "Editor" (`/`) and "Admin Tools" (`/admin`) nav tabs

### Detection Filter Toggles
- [x] 4 toggle chips (Player / Allies / Enemies / Minions) on AI Vision overlay
- [x] Ref-based filtering to avoid re-renders in rAF draw loop
- [x] Chips appear to the left of "Hide AI Vision" button
- [x] Category-colored with CSS custom properties

### Live Extraction Progress
- [x] In-memory `live_progress` dict updated from extraction thread
- [x] Status endpoint returns `extracted_frames`, `total_frames`, `phase`
- [x] Frontend shows "Extracting 30/140 frames..." with smooth progress bar
- [x] Two-phase progress: extraction (0-80%) + DB insert (80-100%)
- [x] Polling interval reduced from 3s to 1.5s

### Key Lessons
- Health bar scan region placement matters more than HSV ranges — wrong crop = wrong color
- Mana bars are blue and will cause false ally classifications if the scan region is too tall
- Build debug visualization tools **before** tuning ML post-processing
- YOLO bounding boxes in our dataset include name text + health bar + mana bar + body

---

## Phase 2: Lightweight Learning (Weeks 4-5)

**Goal:** Move beyond rule-based analysis. Use retrieval and a small classifier to provide smarter coaching.

### Retrieval-Based Coaching (No Training Required)
- [ ] Add `pgvector` extension to Postgres
- [ ] Create `segment_embeddings` table (segment_id, embedding VECTOR(512))
- [ ] Integrate OpenCLIP for frame embedding
  - `pip install open-clip-torch`
  - Use ViT-B/32 model (fast, good quality)
- [ ] Embed segment keyframes (middle frame of each segment)
- [ ] Store embeddings in DB
- [ ] Add API endpoint: GET /api/v1/videos/{id}/similar
  - Input: segment from user's video
  - Output: K-nearest pro segments with timestamps + context
- [ ] Bootstrap with 50 manually-labeled pro VOD segments
- [ ] Add "Similar Pro Plays" section to report

### GRU Segment Classifier
- [ ] Define label taxonomy:
  - `good_trade` / `bad_trade`
  - `missed_cs` / `good_cs`
  - `good_roam` / `bad_roam`
  - `overextend`
  - `good_objective_play`
- [ ] Create training data format:
  ```json
  {
    "segment_id": "uuid",
    "features": [[hp_delta, cs_delta, gold_delta, ...], ...],
    "label": "good_trade",
    "score": 0.8,
    "source": "manual"
  }
  ```
- [ ] Collect training data:
  - [ ] 100+ labeled segments from rule-based detection + manual review
  - [ ] Use the labels endpoint (POST /api/v1/videos/{id}/labels) to build dataset
- [ ] Implement GRU model:
  - [ ] 1-layer GRU, hidden_dim=64
  - [ ] Linear classification head
  - [ ] PyTorch implementation
  - [ ] Training script with train/val split
- [ ] Train model (<1 hour CPU)
- [ ] Export to ONNX
- [ ] Integrate into pipeline (classify segments automatically)
- [ ] Add classifier confidence to segment features
- [ ] Update report builder to include ML-classified events

### Manual Labeling UI
- [ ] Frame scrubbing interface (click timeline → see frame)
- [ ] Segment editing (adjust start/end timestamps)
- [ ] Label tagging dropdown (micro tags from taxonomy)
- [ ] Quality score slider (1-10)
- [ ] Correction input (free text for what should have happened)
- [ ] Bulk export labeled data (JSON/CSV)

---

## Phase 3: Pro VOD Data Pipeline (Weeks 5-8)

**Goal:** Build the "expert demonstration dataset" — what pro players do in various situations.

### VOD Ingestion Pipeline
- [ ] YouTube VOD downloader (yt-dlp integration)
  - Input: YouTube URL or channel
  - Output: Downloaded video files
  - Respect rate limits and ToS
- [ ] Twitch VOD archiver (optional)
- [ ] Pro player channel database:
  - [ ] Top 20 pro players per role (Top, Jungle, Mid, ADC, Support)
  - [ ] Channel URLs, typical upload schedule
  - [ ] Region tags (KR, CN, EU, NA)
- [ ] Batch extraction runner:
  - [ ] Queue system for processing multiple VODs
  - [ ] Auto-detect League gameplay vs non-gameplay sections
  - [ ] Extract at 1 fps (slower but sufficient for dataset)
  - [ ] Store all frame payloads and segments
- [ ] Dataset statistics dashboard:
  - [ ] Total hours processed
  - [ ] Segments by type
  - [ ] Coverage by champion, role, elo
  - [ ] Feature distributions

### Dataset Quality
- [ ] Auto-validation rules:
  - Game timer must be monotonically increasing
  - CS can only increase (except for in-store views)
  - KDA deaths can only increase
- [ ] Outlier detection (flag frames where OCR likely failed)
- [ ] Manual spot-checking workflow (sample N frames, verify)
- [ ] Version tracking (which pipeline version extracted each frame)

### Model Training Pipeline
- [ ] Feature normalization (z-score or min-max across dataset)
- [ ] Train/val/test split (by video, not by frame — prevent leakage)
- [ ] GRU training with pro data:
  - [ ] Pre-train on pro data (behavioral cloning)
  - [ ] Fine-tune with manual labels
  - [ ] Evaluate: accuracy, F1 per class, confusion matrix
- [ ] Embedding model training (contrastive learning):
  - [ ] Positive pairs: similar game states (same champion, same phase)
  - [ ] Negative pairs: different game states
  - [ ] Train embedding model to cluster similar plays
- [ ] Model evaluation:
  - [ ] Retrieval quality (is the "similar pro play" actually similar?)
  - [ ] Classification accuracy (does the model correctly tag good/bad trades?)
  - [ ] A/B test: evidence-grounded report vs report with ML insights

---

## Phase 4: Scale & Production (Weeks 9-12)

**Goal:** Multi-user platform with low latency, high availability, production-grade infrastructure.

### Supabase Migration
- [ ] Create Supabase project
- [ ] Migrate schema from Docker Postgres to Supabase
- [ ] Enable pgvector extension
- [ ] Set up Row Level Security (users see only their videos)
- [ ] Configure Supabase Auth (email + Google OAuth)
- [ ] Migrate file storage to Supabase Storage
- [ ] Set up Realtime subscriptions (replace polling for extraction status)
- [ ] Update all DATABASE_URL references
- [ ] Test all queries against Supabase

### Authentication & Multi-User
- [ ] Add auth to all API endpoints
- [ ] User registration / login flow in frontend
- [ ] User-scoped video list
- [ ] Usage quotas (free tier: 5 videos/month, paid: unlimited)
- [ ] API key support for programmatic access

### Task Queue (Replace BackgroundTask)
- [ ] Evaluate options:
  - Supabase Edge Functions + pgmq
  - Celery + Redis
  - BullMQ + Redis (if Node workers)
- [ ] Implement worker pool for extraction jobs
- [ ] Job priority (paid users first)
- [ ] Job cancellation support
- [ ] Dead letter queue for failed jobs
- [ ] Retry logic with exponential backoff

### Horizontal Scaling
- [ ] Containerize API (Dockerfile)
- [ ] Deploy to cloud (Fly.io, Railway, or AWS ECS)
- [ ] Autoscaling based on queue depth
- [ ] CDN for video serving (Cloudflare R2)
- [ ] Connection pooling (PgBouncer or Supabase built-in)
- [ ] Redis cache for hot data (recent extraction results)

### Performance Optimization
- [ ] GPU inference for YOLO + embeddings (if needed)
- [ ] Batch ONNX inference (multiple frames per forward pass)
- [ ] Frame sampling optimization (adaptive FPS based on game state)
- [ ] Lazy loading of frame data in frontend
- [ ] Pagination for all list endpoints
- [ ] Database query optimization (EXPLAIN ANALYZE, index tuning)

### Monitoring & Observability
- [ ] Structured logging (JSON format)
- [ ] Request tracing (correlation IDs)
- [ ] Extraction pipeline metrics (frames/sec, OCR accuracy, job duration)
- [ ] Error alerting (Sentry or similar)
- [ ] Dashboard (Grafana or Supabase dashboard)
- [ ] Health check endpoint

### Frontend Polish
- [ ] Landing page with product explanation
- [ ] Video library (list all uploaded videos with status)
- [ ] Side-by-side view: video playback synced with data timeline
- [ ] Interactive segment timeline (click to seek video)
- [ ] Report history (view past reports)
- [ ] Share reports (public links)
- [ ] Mobile responsive layout
- [ ] Dark/light theme
- [ ] Loading skeletons
- [ ] Toast notifications

---

## What the Final Product Looks Like

### User Flow
1. User signs up / logs in
2. Uploads a League gameplay clip (up to 10 minutes)
3. System extracts game state: CS, gold, KDA, HP, champion positions, items, game timer — all deterministic, no guessing
4. System detects key moments: fights, deaths, trades, recalls, objective plays
5. System compares each moment against pro player data:
   - "At 4:30, you had 32 CS. Pros average 38 CS at this game time on this champion."
   - "Your trade at 6:15 lost 45% HP for 1 kill. In similar situations, Faker maintains spacing and only commits when jungle is visible on minimap."
6. System generates an evidence-grounded coaching report with citations to specific timestamps
7. User can browse frame-by-frame data, see OCR output, view detected segments
8. User can label segments to improve the model (optional, gamified)

### Technical Differentiators
- **Evidence-grounded**: Every coaching claim cites a specific timestamp and data point
- **Deterministic extraction**: Same video always produces same data (reproducible)
- **Pro comparison**: Not just "you did X wrong" but "here's what the best players do instead"
- **Scalable**: Extraction pipeline is parallelizable, models are ONNX-optimized
- **Improving**: More pro data + more user labels = better models over time

---

## Quick Reference: Running the Project

```bash
# Start database
docker compose up -d postgres

# Install Python deps
cd apps/api; .venv/Scripts/pip install -r requirements.txt; cd ../..

# Run migrations
cd apps/api; .venv/Scripts/python -m migrations.run; cd ../..

# Start dev servers (API + Web)
bun run dev

# Open browser
# http://localhost:8080

# Test OCR on a screenshot
cd apps/api; .venv/Scripts/python ../../scripts/test_ocr.py <path-to-screenshot>
```

---

*Last updated: February 2026*