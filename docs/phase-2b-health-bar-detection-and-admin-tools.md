# Phase 2B: Health Bar Color Detection Tuning & Admin Debug Tools

*Completed: February 2026*

## Table of Contents

1. [What We Built](#what-we-built)
2. [The Health Bar Color Problem](#health-bar-problem)
3. [How Health Bar Color Correction Works](#how-it-works)
4. [The Crop Region Bug & How We Fixed It](#crop-region-bug)
5. [Admin Debug Tool](#admin-debug-tool)
6. [Detection Filter Toggles](#filter-toggles)
7. [Live Extraction Progress](#live-progress)
8. [ML Concepts at Work](#ml-concepts)
9. [Lessons Learned](#lessons-learned)
10. [What's Next](#whats-next)

---

## 1. What We Built <a name="what-we-built"></a>

Phase 2B added three major features:

- **Admin Debug Tool** — A browser-based page where you can select a video, step through frames, and see exactly what YOLO detects + how the health bar color correction classifies each entity. Shows the raw bounding box, the health bar scan region, color fractions, and whether the detection was reclassified.
- **Detection Filter Toggles** — Clickable chips (Player / Allies / Enemies / Minions) on the video player that hide/show detection categories. Useful for debugging noisy frames.
- **Live Extraction Progress** — Real-time "Extracting 30/140 frames..." progress bar during extraction, replacing the old "please wait" message.

### Files Changed

| File | Description |
|------|-------------|
| `apps/api/routers/debug.py` | **NEW** — Debug endpoint, runs YOLO + color analysis on a single frame |
| `apps/api/main.py` | Mount debug router |
| `apps/api/extractor/postprocess.py` | Fixed health bar crop region (was above bbox → now inside top 0-10%) |
| `apps/api/extractor/pipeline.py` | Added `live_progress` dict for real-time extraction tracking |
| `apps/api/routers/extraction.py` | Status endpoint now includes `extracted_frames`, `total_frames`, `phase` |
| `apps/web/src/components/AdminPage.tsx` | **NEW** — Admin page with video selector, frame stepper, detection cards |
| `apps/web/src/components/AIVisionOverlay.tsx` | Added `DetectionFilters` type, `filtersRef` prop, filter logic in draw loop |
| `apps/web/src/components/ExtractionStatus.tsx` | Live progress display with phase-aware progress bar |
| `apps/web/src/App.tsx` | React Router (`/` Editor, `/admin` Admin Tools), filter chips, EditorPage extraction |
| `apps/web/src/api/client.ts` | Added `listVideos()`, `getHealthBarDebug()`, new interfaces |
| `apps/web/src/App.css` | Nav links, filter chip styles |

---

## 2. The Health Bar Color Problem <a name="health-bar-problem"></a>

### Why YOLO Alone Isn't Enough

YOLO detects objects and classifies them (e.g., `freindly_champion`, `enemy_melee_minion`), but it frequently confuses allies and enemies because the visual appearance of champion models is similar regardless of team. A fed Caitlyn and an enemy Caitlyn look identical — the only reliable differentiator is the **health bar color**:

| Health Bar Color | Meaning |
|-----------------|---------|
| Dark green/teal (H 35-95) | **You** (the played champion) |
| Light blue/cyan (H 95-130) | **Your allies** |
| Red (H 0-10 or 170-180) | **Enemies** |

This is a form of **post-processing heuristic** — using domain knowledge (League's color system) to correct ML model outputs.

### The Three-Color System

League uses exactly three health bar colors, visible from any camera angle and at any game state. This makes it a perfect signal for reclassification. The HSV (Hue-Saturation-Value) color space is used because hue is robust to brightness changes (shadows, ability effects, time of day).

**HSV ranges calibrated from actual game frames:**

```python
# Green (player): dark teal-green, hue peaks at 65-90
GREEN = H[35-95], S>50, V>50

# Blue (ally): light cyan-blue
BLUE  = H[95-130], S>40, V>50

# Red (enemy): wraps around hue circle
RED   = H[0-10] or H[170-180], S>50, V>50
```

The boundary at H=95 is critical — the player's green bar extends to ~H=90, ally blue starts at ~H=95. Getting this wrong causes the player to be classified as an ally.

---

## 3. How Health Bar Color Correction Works <a name="how-it-works"></a>

The correction pipeline in `extractor/postprocess.py`:

```
YOLO detection → crop health bar region → convert to HSV → count color pixels
→ pick dominant color → reclassify if needed → enforce max 1 played_champion
```

### Step-by-step:

1. **YOLO outputs a bounding box** with a class guess (e.g., `freindly_champion`)
2. **Crop the top 0-10% of the bbox** — this is where the health bar sits
3. **Convert to HSV color space** — isolates hue from brightness
4. **Count pixels matching each color range** — green, blue, red
5. **Compute fractions** — what % of the crop is green vs blue vs red
6. **Pick the dominant color** if it exceeds 15% of total pixels
7. **Reclassify** using the `_CLASS_BY_COLOR` mapping:
   - `freindly_champion` + red health bar → `enemy_champion`
   - `enemy_melee_minion` + blue health bar → `freindly_melee_minion`
   - etc.
8. **Enforce max 1 `played_champion`** per frame — keep highest confidence, demote extras to `freindly_champion`

### The Reclassification Mapping

Every entity type has a mapping from detected color to correct class:

```python
"freindly_champion": {
    "green": "played_champion",  # If green bar, it's actually YOU
    "blue": "freindly_champion", # Blue = ally, correct
    "red": "enemy_champion"      # Red = enemy, YOLO was wrong
}
```

This pattern applies to all 16 classes (champions, minions, towers, inhibitors, nexus).

---

## 4. The Crop Region Bug & How We Fixed It <a name="crop-region-bug"></a>

### The Bug

The original code assumed health bars sit **above** the YOLO bounding box:

```python
# OLD (WRONG): crop ABOVE the bbox
bar_h = max(8, int((y2 - y1) * 0.15))
crop_y1 = y1 - bar_h  # Above the bbox top
crop_y2 = y1           # At the bbox top
```

This landed the scan region on the **champion name text** (e.g., "Noodlz", "Milio") instead of the actual health bar. Name text often contains blues and greens from the font rendering, causing false classifications.

### The Fix

The YOLO bounding box in our dataset **includes** the name + health bar + champion body. The health bar is inside the top portion of the bbox:

```python
# NEW (CORRECT): crop INSIDE the top 0-10% of the bbox
crop_y1 = y1                          # Top of bbox
crop_y2 = y1 + int(det_h * 0.10)     # 10% down
```

### Why 10%?

We iterated through several values using the Admin Debug Tool:

| Range | Result |
|-------|--------|
| Above bbox (original) | Scanning name text — many false blues/greens |
| 5-25% | Health bar captured, but mana bars included (blue pixels from mana confused ally detection) |
| 0-22% | Better, but mana bars still leaked in for tall champions |
| **0-10%** | Tight crop on just the health bar, no mana bar contamination |

The Admin Debug Tool's green bounding box visualization made this tuning process visual and fast — change the percentage, restart the API, and immediately see where the scan region lands on real frames.

### Where the fix lives

The crop region is defined in **three places** (all must stay in sync):

1. `extractor/postprocess.py` — the actual correction used during extraction
2. `routers/debug.py` `_health_bar_crop_b64()` — the crop thumbnail shown in admin
3. `routers/debug.py` `_color_fractions()` — the color analysis shown in admin
4. `routers/debug.py` `_context_crop_b64()` — the green box drawn on the context image

---

## 5. Admin Debug Tool <a name="admin-debug-tool"></a>

### What It Does

The admin page (`/admin`) provides a visual debugging interface for the detection + color correction pipeline. It does **not** use pre-extracted data — it runs YOLO + color analysis live on the raw video file, so changes to postprocessing are immediately visible without re-extracting.

### Architecture

```
Browser (AdminPage.tsx)
  → GET /api/v1/videos (list available videos)
  → GET /api/v1/videos/{id}/debug/health-bar-colors?frame_index=N
      → Backend: open video → seek to frame → run YOLO → run correct_detections()
      → For each detection: compute color fractions + crop thumbnails
      → Return JSON with base64-encoded images
```

### What Each Detection Card Shows

- **Class name** — the corrected class (after health bar reclassification)
- **"was X" badge** — if the detection was reclassified, shows the original YOLO guess
- **Context image** (400x140) — a crop of the gameplay around the detection with:
  - Yellow bounding box = YOLO detection
  - Green bounding box = health bar scan region (where color is checked)
- **Color fraction bars** — G/B/R percentages in the health bar region
- **Health bar crop** — the tiny actual pixel region being analyzed
- **Bounding box coordinates** — [x1, y1, x2, y2] in pixels

### Why It Matters

Without this tool, debugging color correction required:
1. Run a full extraction (~2 minutes)
2. Check the database for detection results
3. Guess why a classification was wrong
4. Change code, re-extract, repeat

With the admin tool:
1. Select video, navigate to the problem frame
2. See exactly where the scan region lands
3. Adjust percentages, restart API
4. Refresh the page — instant feedback

This reduced iteration time from **minutes to seconds**.

---

## 6. Detection Filter Toggles <a name="filter-toggles"></a>

### What They Do

Four toggle chips below the video player (visible when AI Vision is enabled):

- **Player** — show/hide `played_champion` detections
- **Allies** — show/hide `freindly_*` / `friendly_*` detections
- **Enemies** — show/hide `enemy_*` detections
- **Minions** — sub-filter that hides minion detections from both ally/enemy categories

### Implementation

Filter state uses a **ref** (not state) to avoid re-renders in the `requestAnimationFrame` draw loop. A separate React state triggers UI re-renders for the chip active/inactive styling. Both are kept in sync via a `toggleFilter` callback.

```typescript
// Ref-based for the rAF loop (no re-renders)
const filtersRef = useRef<DetectionFilters>({ player: true, allies: true, ... });

// State-based for chip UI re-renders
const [filterState, setFilterState] = useState<DetectionFilters>({ ... });

// Toggle syncs both
const toggleFilter = (key) => {
  setFilterState(prev => {
    const next = { ...prev, [key]: !prev[key] };
    filtersRef.current = next;
    return next;
  });
};
```

The filter check happens inside the canvas draw loop:

```typescript
if (filters && !shouldShowDetection(det.class_name, filters)) continue;
```

---

## 7. Live Extraction Progress <a name="live-progress"></a>

### The Problem

Previously, extraction showed 0% progress for the entire frame extraction phase (which takes 1-2 minutes), then jumped to 100% during the fast DB insert phase. Users had no idea if extraction was working.

### The Fix

An in-memory dict (`live_progress`) in `pipeline.py` is updated from the extraction thread:

```python
# In the extraction loop (runs in a background thread)
live_progress[job_id] = {
    "extracted": count,      # e.g., 30
    "total": estimated_total, # e.g., 140
    "phase": "extracting"
}
```

The status endpoint reads this dict and includes it in the response. The frontend maps this to a progress bar:

- **Extracting phase** (0-80% of bar): `extracted / total * 0.8`
- **DB insert phase** (80-100% of bar): `0.8 + db_progress * 0.2`

Polling interval was reduced from 3s to 1.5s for snappier updates.

---

## 8. ML Concepts at Work <a name="ml-concepts"></a>

### Object Detection (YOLO)

YOLOv8 ("You Only Look Once") is a single-shot detector — it processes the entire image in one forward pass and outputs all bounding boxes simultaneously. This is in contrast to two-stage detectors like Faster R-CNN which first propose regions then classify them.

Key concepts:
- **Anchor-free detection**: YOLOv8 predicts center points + width/height directly (no anchor boxes)
- **Non-Maximum Suppression (NMS)**: When multiple overlapping boxes detect the same object, NMS keeps the highest-confidence one and removes duplicates (IoU threshold = 0.45)
- **Transfer learning**: We fine-tune from COCO pretrained weights rather than training from scratch — the lower layers already know edges, textures, and shapes
- **ONNX Runtime**: The model runs as an ONNX graph, not PyTorch — faster inference, no GPU required, cross-platform

### Color Space Analysis (HSV)

RGB (Red-Green-Blue) is how screens display color, but it's terrible for analysis because brightness changes affect all three channels. HSV (Hue-Saturation-Value) separates:
- **Hue**: The actual color (0-180 in OpenCV, representing 0-360 degrees)
- **Saturation**: How vivid the color is (0 = gray, 255 = pure color)
- **Value**: How bright (0 = black, 255 = bright)

This means we can detect "red health bar" regardless of whether the scene is dark (nighttime/fog) or bright (dragon pit explosion).

### Post-Processing Heuristics

Not everything needs to be learned. The health bar color system is a **fixed game mechanic** — it never changes across patches, champions, or game modes. Using a hardcoded HSV-based classifier is more reliable than training an ML model for this specific task. This is a common pattern in production ML systems: **use ML where the problem is hard (object detection), use rules where the domain is simple and fixed (color classification)**.

### IoU-Based Object Tracking

The `SimpleTracker` in `tracker.py` uses Intersection-over-Union (IoU) to match detections across frames:

```
IoU = Area of Overlap / Area of Union
```

If a detection in frame N overlaps significantly (IoU > 0.3) with one in frame N-1, they're the same object. This enables:
- Smooth bounding box interpolation between keyframes
- Consistent track IDs (same champion keeps the same color)
- Majority-vote class labels (prevents a champion from flickering between ally/enemy)

---

## 9. Lessons Learned <a name="lessons-learned"></a>

### Build debug tools early

The Admin Debug Tool was built to diagnose health bar detection issues, but it became indispensable for all detection work. Being able to see exactly what the model detects + what the post-processor sees + how the color analysis interprets it — all on the same page — makes the difference between guessing and knowing.

### YOLO bounding boxes include more than you think

Our initial assumption was that YOLO bbox = the champion body. In reality, the training data's bounding boxes include the name text, health bar, mana bar, and the champion model. This is actually good (more context for detection), but it means post-processing needs to know where within the bbox to look for specific signals.

### Mana bars are blue too

When the health bar scan region was too tall (0-22%), it captured mana bars below the health bar. Mana bars are blue, which caused enemies to be classified as allies. Tightening to 0-10% fixed this — a reminder that in computer vision, your crop region is as important as your algorithm.

### In-memory progress for background tasks

FastAPI's `BackgroundTasks` doesn't natively support progress reporting. A simple module-level dict, written from the background thread and read by the API endpoint, is a lightweight solution that avoids adding Redis or a message queue just for progress tracking.

---

## 10. What's Next <a name="whats-next"></a>

### Short-term improvements

- **Retrain YOLO with more data** — The current model (YOLOv8n, 435 images) misses many minions and struggles with overlapping champions. Expanding the dataset to 1000+ images and using YOLOv8s (small) instead of YOLOv8n (nano) should significantly improve recall.
- **GPU training** — Current model was trained on CPU. GPU training (CUDA) enables larger models, more epochs, and data augmentation.
- **Resolution-adaptive crop regions** — Currently calibrated for 1080p only. Need proportional scaling for 1440p/720p.
- **Confidence threshold tuning** — Currently fixed at 0.40. Lower thresholds catch more detections but increase false positives. Could be adaptive per class.

### Medium-term (Phase 2)

- **Retrieval-based coaching** — Use OpenCLIP embeddings + pgvector to find "similar pro plays" in a database of pro VOD segments
- **GRU segment classifier** — A small recurrent network that classifies segments (good trade, bad trade, missed CS) based on feature sequences
- **Manual labeling UI** — Click-to-label interface for building training data

### Long-term vision

- **Pro VOD pipeline** — Ingest thousands of hours of pro gameplay, extract structured data, build the "what good looks like" dataset
- **Multi-user platform** — Auth, usage quotas, Supabase migration
- **Real-time coaching** — Eventually, screen capture + live inference = coaching overlay during actual gameplay

---

*This document was written during Phase 2B development (February 2026).*
