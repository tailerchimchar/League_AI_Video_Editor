"""YOLO ONNX inference for champion/object detection (Phase 1.5).

Loads the best available model: yolov8s_lol.onnx > yolov8n_lol.onnx.
Uses CUDA GPU acceleration when available, falls back to CPU.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np


def _add_nvidia_dll_dirs() -> None:
    """Add pip-installed nvidia package DLL dirs so onnxruntime can find CUDA."""
    if sys.platform != "win32":
        return
    site_pkgs = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "nvidia"
    if not site_pkgs.is_dir():
        return
    for sub in site_pkgs.iterdir():
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_add_nvidia_dll_dirs()

logger = logging.getLogger(__name__)

# 16 classes from vasyz's LeagueOfLegends Roboflow dataset (v4).
# Order matches data.yaml alphabetical indices exactly.
# Note: dataset has inconsistent spelling ("freindly_" vs "friendly_").
CLASS_NAMES: list[str] = [
    "enemy_cannon_minion",     # 0
    "enemy_champion",          # 1
    "enemy_inhibitor",         # 2
    "enemy_melee_minion",      # 3
    "enemy_nexus",             # 4
    "enemy_ranged_minion",     # 5
    "enemy_tower",             # 6
    "freindly_cannon_minion",  # 7  (sic)
    "freindly_champion",       # 8  (sic)
    "freindly_melee_minion",   # 9  (sic)
    "freindly_ranged_minion",  # 10 (sic)
    "freindly_super_minion",   # 11 (sic)
    "friendly_inhibitor",      # 12
    "friendly_nexus",          # 13
    "friendly_tower",          # 14
    "played_champion",         # 15
]

# IOU threshold for Non-Maximum Suppression
NMS_IOU_THRESHOLD = 0.45

# Resolve model path relative to repo root (same as storage/local.py).
# Prefer yolov8s (better accuracy) and fall back to yolov8n.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "data" / "models"
_MODEL_CANDIDATES = [
    _MODEL_DIR / "yolov8s_lol.onnx",
    _MODEL_DIR / "yolov8n_lol.onnx",
]


def _resolve_model_path() -> str:
    """Return the path to the best available ONNX model."""
    for candidate in _MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    # Default to first candidate (will fail gracefully in __init__)
    return str(_MODEL_CANDIDATES[0])


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS. boxes: (N,4) as x1,y1,x2,y2. Returns kept indices."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask = iou <= iou_thresh
        order = order[1:][mask]

    return keep


class YoloDetector:
    """YOLO object detector using ONNX Runtime."""

    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path) if model_path else Path(_resolve_model_path())
        self.session = None
        self._available = False
        self.class_names = CLASS_NAMES

        if self.model_path.exists():
            try:
                import onnxruntime as ort

                # Session-level optimizations
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_opts.enable_mem_pattern = True

                # Prefer GPU; onnxruntime-gpu provides CUDAExecutionProvider.
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.session = ort.InferenceSession(
                    str(self.model_path), sess_options=sess_opts, providers=providers,
                )
                active = self.session.get_providers()
                logger.info("ONNX Runtime providers: %s", active)
                self.input_name = self.session.get_inputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape  # [batch, 3, H, W]
                self._available = True
                logger.info(
                    "YOLO detector loaded: %s (input %s)",
                    self.model_path, self.input_shape,
                )
            except ImportError:
                logger.warning("onnxruntime not installed — detector disabled")
            except Exception:
                logger.exception("Failed to load YOLO detector")

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.32) -> list[dict]:
        """Run detection on a single frame. Returns list of detection dicts."""
        if not self._available:
            return []

        blob = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, frame.shape, conf_thresh)

    def detect_batch(
        self, frames: list[np.ndarray], conf_thresh: float = 0.40,
    ) -> list[list[dict]]:
        """Run detection on multiple frames. Processes sequentially on GPU.

        Returns a list of detection lists, one per input frame.
        """
        if not self._available or not frames:
            return [[] for _ in frames]

        return [self.detect(f, conf_thresh) for f in frames]

    def _preprocess_single(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess one frame: resize, normalize, NCHW, add batch dim."""
        import cv2

        h, w = self.input_shape[2], self.input_shape[3]
        resized = cv2.resize(frame, (w, h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(blob, 0)   # (1, 3, H, W)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize to model input size, normalize to [0,1], NCHW format."""
        return self._preprocess_single(frame)

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        orig_shape: tuple,
        conf_thresh: float,
    ) -> list[dict]:
        """Decode YOLOv8 output tensor, apply NMS, scale boxes to original coords.

        YOLOv8 ONNX output shape: (1, 4+num_classes, num_boxes)
        - First 4 rows: cx, cy, w, h (in model input coords)
        - Remaining rows: class probabilities
        """
        output = outputs[0]  # shape: (1, 4+C, N)

        # Squeeze batch dimension
        if output.ndim == 3:
            output = output[0]  # (4+C, N)

        # Transpose to (N, 4+C) for easier handling
        output = output.T  # (N, 4+C)

        num_classes = output.shape[1] - 4
        if num_classes <= 0:
            return []

        # Split boxes and class scores
        cx = output[:, 0]
        cy = output[:, 1]
        w = output[:, 2]
        h = output[:, 3]
        class_scores = output[:, 4:]  # (N, C)

        # Best class per box
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask = confidences >= conf_thresh
        if not mask.any():
            return []

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Convert cx,cy,w,h -> x1,y1,x2,y2 (still in model input coords)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Per-class NMS
        keep_indices: list[int] = []
        unique_classes = np.unique(class_ids)
        for cls_id in unique_classes:
            cls_mask = class_ids == cls_id
            cls_indices = np.where(cls_mask)[0]
            cls_boxes = boxes[cls_mask]
            cls_scores = confidences[cls_mask]
            kept = _nms(cls_boxes, cls_scores, NMS_IOU_THRESHOLD)
            keep_indices.extend(cls_indices[kept].tolist())

        if not keep_indices:
            return []

        boxes = boxes[keep_indices]
        confidences = confidences[keep_indices]
        class_ids = class_ids[keep_indices]

        # Scale boxes from model input coords to original frame coords
        model_h, model_w = self.input_shape[2], self.input_shape[3]
        orig_h, orig_w = orig_shape[0], orig_shape[1]
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h

        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y

        # Clip to frame bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        # Build result list
        detections: list[dict] = []
        for i in range(len(boxes)):
            cls_id = int(class_ids[i])
            class_name = (
                self.class_names[cls_id]
                if cls_id < len(self.class_names)
                else f"class_{cls_id}"
            )
            detections.append({
                "class_name": class_name,
                "confidence": round(float(confidences[i]), 3),
                "bbox": [
                    round(float(boxes[i, 0]), 1),
                    round(float(boxes[i, 1]), 1),
                    round(float(boxes[i, 2]), 1),
                    round(float(boxes[i, 3]), 1),
                ],
            })

        return detections
