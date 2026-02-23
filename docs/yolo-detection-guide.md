# YOLO Champion/Object Detection — How It Works

This document explains the YOLO detection system (Phase 1.5) step by step: what it is, why we need it, how it was built, and how all the pieces connect.

---

## Table of Contents

1. [The Big Picture: Why Detection?](#big-picture)
2. [What is YOLO?](#what-is-yolo)
3. [Step 1: Getting Training Data](#step-1)
4. [Step 2: Training the Model](#step-2)
5. [Step 3: Exporting to ONNX](#step-3)
6. [Step 4: Inference in the Pipeline](#step-4)
7. [Step 5: Frontend Visualization](#step-5)
8. [How Everything Connects End-to-End](#end-to-end)
9. [Training Results & What the Numbers Mean](#results)
10. [Files Reference](#files)
11. [How to Retrain or Improve the Model](#retrain)
12. [Champion Identification (Phase 2A)](#champion-id)

---

## 1. The Big Picture: Why Detection? <a name="big-picture"></a>

Before YOLO, our pipeline could only read **text from the HUD** (game timer, CS, KDA, HP). That's useful, but it's blind to what's actually happening in the **gameplay viewport** — the big area where champions fight, minions walk, and towers shoot.

```
+------------------------------------------------------------------+
|  [Scoreboard] [KDA] [CS] [Timer]    <-- OCR reads this (Phase 1) |
|                                                                    |
|                                                                    |
|         THE ACTUAL GAMEPLAY                                        |
|         Champions, minions, towers                                 |
|         are all moving here                                        |
|         OCR can't help with this!     <-- YOLO reads this (1.5)   |
|                                                                    |
|                                                                    |
|  [Abilities] [Items] [HP Bar]        <-- OCR reads this (Phase 1) |
|  [Minimap]                                                         |
+------------------------------------------------------------------+
```

**YOLO gives us spatial awareness.** Now we can answer questions like:
- Where are the champions on screen? (positioning)
- How many enemy minions are nearby? (wave state)
- Is the player near a tower? (safety)
- Are enemies clustering? (gank incoming)

This data feeds into better segment detection (fights, trades) and richer coaching reports.

---

## 2. What is YOLO? <a name="what-is-yolo"></a>

**YOLO** stands for **"You Only Look Once"** — it's an object detection model that can find and classify multiple objects in a single image in one pass.

### What object detection means

Given an image (a video frame), the model outputs a list of **detections**, each containing:
- **class_name**: What the object is (e.g., `"enemy_champion"`, `"friendly_tower"`)
- **confidence**: How sure the model is (0.0 to 1.0, higher = more confident)
- **bbox**: A bounding box `[x1, y1, x2, y2]` — the pixel coordinates of a rectangle around the object

Example output for one frame:
```json
[
  {"class_name": "played_champion", "confidence": 0.94, "bbox": [850.2, 520.1, 920.5, 610.3]},
  {"class_name": "enemy_champion",  "confidence": 0.87, "bbox": [1100.0, 480.5, 1170.2, 570.8]},
  {"class_name": "enemy_tower",     "confidence": 0.92, "bbox": [1300.1, 350.0, 1380.4, 500.2]},
  {"class_name": "freindly_melee_minion", "confidence": 0.71, "bbox": [950.0, 540.0, 980.0, 570.0]}
]
```

### Why YOLOv8 specifically?

| Feature | Why it matters |
|---------|---------------|
| **Fast** | ~43ms per frame on CPU — works at real-time-ish speed without a GPU |
| **Small** | YOLOv8**n** ("nano") is only 6MB, exports to 12MB ONNX — easily fits in memory |
| **Accurate** | State-of-the-art detection accuracy, even the nano variant |
| **Easy to train** | `ultralytics` library makes training a one-liner |
| **ONNX export** | Runs anywhere via ONNX Runtime — no PyTorch needed at inference time |

### Why ONNX?

ONNX (Open Neural Network Exchange) is a standard format for ML models. Think of it like a "compiled" version of the model:

- **Training** uses PyTorch (big framework, GPU-friendly, flexible)
- **Inference** uses ONNX Runtime (small library, CPU-optimized, fast)

We train once, export to ONNX, and then the API server only needs `onnxruntime` (15MB pip package) instead of PyTorch (2GB+ pip package). This is why `requirements.txt` has `onnxruntime` and NOT `torch` or `ultralytics` — those are only needed for training.

---

## 3. Step 1: Getting Training Data <a name="step-1"></a>

A YOLO model needs to be **trained** on labeled images. You can't just give it a random model and expect it to know what a League champion looks like — it needs examples.

### The dataset

We used **vasyz's "LeagueOfLegends" dataset** from [Roboflow](https://universe.roboflow.com/vasyz-xhrmx/leagueoflegends-kvjwx):
- **435 annotated images** of League gameplay
- **16 object classes** (see below)
- **YOLOv8 format**: images + text files with bounding box coordinates
- **Pre-split**: 411 train / 16 validation / 8 test

### The 16 classes

The dataset can detect these objects:

| # | Class Name | What it is |
|---|-----------|-----------|
| 0 | `enemy_cannon_minion` | Enemy siege/cannon minion |
| 1 | `enemy_champion` | Any enemy champion |
| 2 | `enemy_inhibitor` | Enemy inhibitor structure |
| 3 | `enemy_melee_minion` | Enemy melee minion |
| 4 | `enemy_nexus` | Enemy nexus |
| 5 | `enemy_ranged_minion` | Enemy caster/ranged minion |
| 6 | `enemy_tower` | Enemy turret |
| 7 | `freindly_cannon_minion` | Allied cannon minion (note: "freindly" is misspelled in the dataset) |
| 8 | `freindly_champion` | Any allied champion |
| 9 | `freindly_melee_minion` | Allied melee minion |
| 10 | `freindly_ranged_minion` | Allied caster/ranged minion |
| 11 | `freindly_super_minion` | Allied super minion |
| 12 | `friendly_inhibitor` | Allied inhibitor (note: correctly spelled here!) |
| 13 | `friendly_nexus` | Allied nexus |
| 14 | `friendly_tower` | Allied turret |
| 15 | `played_champion` | The player's own champion |

The misspelling (`freindly` vs `friendly`) is in the original dataset and we preserve it exactly — if we "fixed" it, the class indices wouldn't match the model's training and detections would be wrong.

### Dataset format (YOLOv8)

The dataset has this structure:
```
data/datasets/league-yolo/
  data.yaml           # Class names + paths
  train/
    images/           # 411 .jpg files
    labels/           # 411 .txt files (one per image)
  valid/
    images/           # 16 .jpg files
    labels/           # 16 .txt files
  test/
    images/           # 8 .jpg files
    labels/           # 8 .txt files
```

Each label `.txt` file has one line per object in the image:
```
1 0.45 0.52 0.08 0.12
```
This means: class 1 (`enemy_champion`), center at (45%, 52%) of the image, width 8%, height 12%.

### How the download works

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")  # stored in .env as ROBOFLOW_API_KEY
project = rf.workspace("vasyz-xhrmx").project("leagueoflegends-kvjwx")
version = project.version(4)       # v4 has the most images (435)
dataset = version.download("yolov8", location="data/datasets/league-yolo")
```

---

## 4. Step 2: Training the Model <a name="step-2"></a>

Training means showing the model thousands of labeled examples until it learns to recognize the objects on its own.

### What happens during training

1. **Load a pretrained model** (`yolov8n.pt`) — this already knows basic shapes, edges, and textures from being trained on the COCO dataset (everyday objects like cars, people, dogs)
2. **Fine-tune on our data** — we show it League screenshots with labeled bounding boxes, and it adjusts its weights to recognize League-specific objects
3. **Repeat for 50 epochs** — one "epoch" = one pass through all 411 training images. 50 passes means the model sees each image 50 times (with random augmentations like flipping, color shifts, scaling)
4. **Validate after each epoch** — check performance on the 16 validation images to track improvement

### The training command (inside `scripts/train_yolo.py`)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained on COCO
results = model.train(
    data="data/datasets/league-yolo/data.yaml",
    epochs=50,
    imgsz=640,    # resize images to 640x640 for training
    batch=16,     # process 16 images at a time
)
```

### Why "fine-tuning" instead of training from scratch

Starting from a pretrained model (transfer learning) is hugely important:
- **From scratch**: The model knows nothing. It would need thousands of images and days of training.
- **Fine-tuning**: The model already understands edges, shapes, textures. It just needs to learn "this particular shape is a League champion." 435 images and 50 epochs is enough.

### Training timeline

On our CPU (i7-13700K, no GPU):
- ~70 seconds per epoch
- 50 epochs total
- **~1 hour total training time**

With a GPU, this would take ~5 minutes.

---

## 5. Step 3: Exporting to ONNX <a name="step-3"></a>

After training, the best model weights are saved as a PyTorch `.pt` file. We export to ONNX for deployment:

```python
model = YOLO("runs/detect/league-yolo/weights/best.pt")
model.export(format="onnx", imgsz=640)
# Copies to: data/models/yolov8n_lol.onnx (11.7 MB)
```

The ONNX model is what the API server loads. It contains:
- The neural network architecture (layers, connections)
- The trained weights (numbers the model learned)
- Input/output format specification

**Key detail**: The ONNX model's output shape is `(1, 20, 8400)`:
- `1` = batch size (one image at a time)
- `20` = 4 box coordinates + 16 class probabilities
- `8400` = number of candidate detection boxes the model considers

---

## 6. Step 4: Inference in the Pipeline <a name="step-4"></a>

This is where the model actually runs on video frames during extraction. The code lives in `apps/api/extractor/detector.py`.

### The three stages of inference

#### Stage 1: Preprocessing (`_preprocess`)

The model expects input as a specific format: a 640x640 pixel image, normalized to 0-1, in NCHW format.

```python
def _preprocess(self, frame):
    # frame is a 1920x1080 numpy array (the video frame)
    resized = cv2.resize(frame, (640, 640))    # Shrink to model input size
    blob = resized / 255.0                      # Normalize: [0,255] -> [0,1]
    blob = blob.transpose(2, 0, 1)              # HWC -> CHW (channels first)
    blob = np.expand_dims(blob, 0)              # Add batch dimension
    return blob  # Shape: (1, 3, 640, 640)
```

#### Stage 2: Model inference

```python
outputs = self.session.run(None, {self.input_name: blob})
# outputs[0] shape: (1, 20, 8400) — 8400 candidate boxes, each with 4 coords + 16 class scores
```

#### Stage 3: Postprocessing (`_postprocess`)

The raw model output is a big tensor of numbers. Postprocessing turns it into usable detections:

1. **Decode boxes**: Extract center-x, center-y, width, height for each of the 8400 candidates
2. **Get best class**: For each box, find which of the 16 classes has the highest probability
3. **Filter by confidence**: Throw away anything below 25% confidence (most of the 8400 boxes are garbage)
4. **Non-Maximum Suppression (NMS)**: If multiple boxes overlap on the same object, keep only the best one
5. **Scale back to original size**: The model thinks in 640x640 space, but we need pixel coordinates in the original 1920x1080 frame

**NMS explained simply**: The model might detect the same champion with 5 slightly different boxes. NMS picks the highest-confidence one and removes the rest. It does this by checking how much boxes overlap (IoU = Intersection over Union). If two boxes overlap more than 45%, the lower-confidence one is removed.

### Auto-enabling in the pipeline

In `apps/api/extractor/pipeline.py`:
```python
# Old behavior: detector disabled by default
enable_detector = config.get("enable_detector", False)

# New behavior: auto-enable when model file exists
detector_available = _get_detector().available
enable_detector = config.get("enable_detector", detector_available)
```

This means: if `data/models/yolov8n_lol.onnx` exists, detection runs automatically. No config change needed.

### Where detections are stored

Each frame's detections go into the `frame_payloads` table as a JSONB column:
```sql
frame_payloads.detections = [
  {"class_name": "enemy_champion", "confidence": 0.87, "bbox": [1100.0, 480.5, 1170.2, 570.8]},
  ...
]
```

The API serves them at `GET /api/v1/videos/{id}/frames`, and the frontend reads them.

---

## 7. Step 5: Frontend Visualization <a name="step-5"></a>

The AI Vision overlay (`apps/web/src/components/AIVisionOverlay.tsx`) draws detection boxes on top of the video.

### How it works

The overlay is a `<canvas>` element positioned exactly on top of the `<video>` element. On every animation frame (~60fps):

1. **Get current video time** → find the closest extracted frame in the data
2. **Draw OCR crop regions** (the static HUD boxes, always the same position)
3. **Draw YOLO detections** (dynamic boxes that move every frame!)

### Detection box styling

Different classes get different colors so you can instantly tell friend from foe:

| Class pattern | Color | Example |
|--------------|-------|---------|
| `played_champion` | Gold (#ffd700) | Your champion |
| `enemy_*` | Red (#ef4444) | Enemy champions, minions, towers |
| `freindly_*` / `friendly_*` | Blue (#3b82f6) | Allied champions, minions, towers |

Detection boxes are drawn with:
- **Dashed borders** (to distinguish from the solid OCR crop region borders)
- **Semi-transparent fill**
- **Label pill** showing class name + confidence percentage (e.g., "enemy_champion 87%")

### Scaling

The video might be rendered at any size on screen (depending on browser window). The overlay scales detection coordinates from native resolution (1920x1080) to whatever size the video is displayed at, using the same scale factors as the OCR crop regions.

---

## 8. How Everything Connects End-to-End <a name="end-to-end"></a>

Here's the complete flow from video upload to seeing detection boxes:

```
1. User uploads video
   └─> POST /api/v1/videos
       └─> Saved to data/videos/

2. User clicks "Extract"
   └─> POST /api/v1/videos/{id}/extract
       └─> BackgroundTask starts pipeline

3. Pipeline processes each frame (2 fps):
   ┌─────────────────────────────────────────────┐
   │  For each frame:                             │
   │                                              │
   │  a) Crop HUD regions ──> OCR ──> game data   │
   │  b) Run YOLO detector ──> detection boxes    │
   │  c) Compute features (deltas)                │
   │  d) Store in frame_payloads table            │
   └─────────────────────────────────────────────┘

4. After all frames: run segmenter (detect fights, deaths)

5. User toggles "AI Vision" in the frontend
   └─> GET /api/v1/videos/{id}/frames
       └─> Returns all frame_payloads (including detections)

6. Canvas overlay renders per-frame:
   └─> Current video time ──> find closest frame
       └─> Draw OCR regions (static, with live values)
       └─> Draw YOLO detections (dynamic, move each frame!)
```

**The key insight**: Detection boxes are **per-frame data**, not a live model running in the browser. The heavy computation happened during extraction. The frontend just reads pre-computed results and draws them.

---

## 9. Training Results & What the Numbers Mean <a name="results"></a>

### Overall metrics

| Metric | Value | What it means |
|--------|-------|---------------|
| **mAP50** | **0.845** | 84.5% average precision when a detection is "correct" if it overlaps >50% with the real box |
| **mAP50-95** | **0.614** | 61.4% average precision across stricter overlap thresholds (50% to 95%) |

**mAP** = Mean Average Precision. It's the standard metric for object detection. Higher is better.
- **mAP50 > 0.8** is considered good for a custom dataset
- **mAP50-95 > 0.5** is solid, especially for a small dataset and nano model

### Per-class results

| Class | Precision | Recall | mAP50 | Notes |
|-------|-----------|--------|-------|-------|
| `played_champion` | 0.91 | 1.00 | 0.995 | Excellent — always finds the player's champ |
| `enemy_tower` | 0.78 | 0.90 | 0.955 | Strong — towers are big and distinctive |
| `friendly_tower` | 0.88 | 1.00 | 0.995 | Excellent |
| `enemy_cannon_minion` | 0.59 | 1.00 | 0.995 | Good recall, moderate precision |
| `enemy_champion` | 0.66 | 0.86 | 0.821 | Good — champions vary a lot in appearance |
| `freindly_champion` | 0.79 | 0.50 | 0.586 | Weakest champ class — only 10 samples in val |
| `enemy_melee_minion` | 1.00 | 0.00 | 0.117 | Poor — only 2 samples in validation! |

### What "Precision" and "Recall" mean

- **Precision**: Of the boxes the model draws, how many are actually correct? High precision = few false alarms.
- **Recall**: Of all the real objects in the image, how many did the model find? High recall = few missed objects.
- **mAP**: Combines both into a single score across different confidence thresholds.

### Why some classes are weak

`enemy_melee_minion` has mAP of only 0.117 — but this isn't a model failure. The validation set only has **2 enemy melee minion instances**. With so few examples to test against, the metric is unreliable. The model likely detects melee minions fine in practice, but we'd need more validation data to confirm.

### Inference speed

- **Preprocessing**: ~1ms
- **Model inference**: ~43ms
- **Postprocessing**: ~1ms
- **Total**: ~45ms per frame

At 2 FPS sampling, we process one frame every 500ms. Detection adds only 45ms to that — well within budget.

---

## 10. Files Reference <a name="files"></a>

### Core files

| File | Purpose |
|------|---------|
| `apps/api/extractor/detector.py` | YOLO inference: load model, preprocess, run, postprocess, NMS |
| `apps/api/extractor/pipeline.py` | Pipeline orchestrator — calls detector on each frame |
| `apps/api/requirements.txt` | `onnxruntime>=1.17.0` for inference |
| `apps/web/src/components/AIVisionOverlay.tsx` | Frontend canvas overlay — draws detection boxes |
| `apps/web/src/api/client.ts` | `Detection` and `FramePayload` TypeScript types |

### Training files

| File | Purpose |
|------|---------|
| `scripts/train_yolo.py` | Training script — download dataset, train, export ONNX |
| `data/datasets/league-yolo/` | Dataset (gitignored) |
| `data/models/yolov8n_lol.onnx` | Trained model (gitignored) |
| `runs/detect/league-yolo/` | Training logs, charts, val images (gitignored) |

### Configuration

| Setting | Location | Value |
|---------|----------|-------|
| Model path | `detector.py` | `data/models/yolov8n_lol.onnx` (resolved from repo root) |
| Confidence threshold | `detector.py` `detect()` | 0.25 (25%) |
| NMS IoU threshold | `detector.py` `NMS_IOU_THRESHOLD` | 0.45 |
| Auto-enable | `pipeline.py` | `True` when model file exists |
| Roboflow API key | `.env` | `ROBOFLOW_API_KEY` |

---

## 11. How to Retrain or Improve the Model <a name="retrain"></a>

### If you want to retrain from scratch

```bash
# 1. Make sure you have the training dependencies
cd apps/api
.venv/Scripts/pip install ultralytics roboflow

# 2. Run the training script
.venv/Scripts/python ../../scripts/train_yolo.py

# This will:
# - Verify dataset exists at data/datasets/league-yolo/
# - Train for 50 epochs
# - Evaluate on validation set
# - Export to ONNX at data/models/yolov8n_lol.onnx
```

### If you want to improve accuracy

1. **More data**: Add more annotated images to the dataset. Even 200 more images would help.
   - Go to the Roboflow project, add images, re-export
   - Or use a different/additional dataset

2. **More epochs**: Change `EPOCHS = 50` to `EPOCHS = 100` in `scripts/train_yolo.py`

3. **Larger model**: Change `BASE_MODEL = "yolov8n.pt"` to `"yolov8s.pt"` (small) or `"yolov8m.pt"` (medium). Bigger = more accurate but slower inference.

4. **GPU training**: If you have an NVIDIA GPU, training will be 10-20x faster. The `ultralytics` library auto-detects CUDA.

5. **Custom augmentation**: Add more aggressive augmentation (rotation, mosaic, mixup) in the training config for better generalization.

### If the model seems wrong

- Check `runs/detect/league-yolo/` for training plots (`results.png` shows loss curves)
- Check `runs/detect/league-yolo/val_batch0_pred.jpg` to see what the model predicts on validation images
- Run `scripts/train_yolo.py` again — the script uses `exist_ok=True` so it overwrites the previous run

---

## 12. Champion Identification (Phase 2A) <a name="champion-id"></a>

YOLO detects generic classes — `played_champion`, `enemy_champion`, `freindly_champion`. Phase 2A adds identification of *which* champion each detection is.

### The approach

A two-part system:

1. **User-specified player champion**: Before starting extraction, the user types their champion name (e.g., "Caitlyn"). This is stored in the extraction config and applied to all `played_champion` detections with 100% confidence.

2. **Auto-matching for allies**: On the first frame, the system crops 4 ally portraits from above the minimap and matches them against Riot's Data Dragon champion icons using HSV color histogram comparison.

### How portrait matching works

```
Frame (1920x1080)
                                      ┌────────────────────┐
                                      │  Ally 1  Ally 2    │
                                      │  Ally 3  Ally 4    │
                                      │                    │
                                      │     [Minimap]      │
                                      └────────────────────┘
     ┌──────┐
     │Player│  ← Bottom-center HUD
     │ Icon │
     └──────┘
```

1. Extract portrait crops from known HUD positions (calibrated for 1080p Outplayed recordings)
2. Compute HSV histogram for the crop (Hue + Saturation, ignore Value for lighting robustness)
3. Compare against all 2000+ reference histograms (default icons + skin loading screens)
4. Best score per champion wins (so if Caitlyn has 20 skins, the best-matching skin determines Caitlyn's score)

### Why it's imperfect

**Skins break image matching.** We tested every reasonable matching approach:

| Method | Result |
|--------|--------|
| HSV histogram correlation | Pool Party Caitlyn → LeBlanc (wrong) |
| LAB histogram correlation | Pool Party Caitlyn → LeBlanc (wrong) |
| Grayscale template matching | Pool Party Caitlyn → LeBlanc (wrong) |
| BGR template matching | Pool Party Caitlyn → LeBlanc (wrong) |
| ORB feature matching | 0 keypoints detected (portrait too small at 50x48px) |

The fundamental problem: skins change color palette, silhouette, and visual features so dramatically that a different champion's default icon can look more similar than the same champion in a skin.

**Current accuracy**: ~100% for player (user-specified), ~50% for allies (auto-matched).

### Future: Phase B — CNN Classifier

The real solution is a learned embedding model:
1. Crop YOLO bounding boxes from gameplay frames
2. Train a CNN to embed champion crops into a vector space where same-champion crops cluster together regardless of skin
3. At inference: embed the crop, find nearest champion cluster
4. Phase 2A's game roster data provides bootstrap labels for training

### Files

| File | Purpose |
|------|---------|
| `apps/api/extractor/champion_id.py` | ChampionIdentifier class — histogram matching |
| `apps/api/extractor/config.py` | Portrait crop positions (PLAYER_PORTRAIT_REGION, ALLY_PORTRAIT_SLOTS) |
| `scripts/download_champion_icons.py` | Downloads icons + skin loading screens from Data Dragon |
| `scripts/calibrate_scoreboard.py` | Debug tool — visualize portrait crops and match results |
| `data/champions/icons/` | 172 champion square icons (120x120 PNG) |
| `data/champions/skins/` | 2054 skin loading screen images (308x560 JPG) |
| `data/champions/manifest.json` | Champion key/name mapping + skin lists |

### Setup

```bash
# Download champion reference data
cd apps/api
.venv/Scripts/python ../../scripts/download_champion_icons.py          # icons only
.venv/Scripts/python ../../scripts/download_champion_icons.py --skins   # + all skins (~500 MB)

# Test portrait matching on a screenshot
.venv/Scripts/python ../../scripts/calibrate_scoreboard.py path/to/screenshot.png
```

---

*Last updated: February 2026*
