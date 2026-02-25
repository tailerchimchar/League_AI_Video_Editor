"""IoU-based object tracker for smoothing YOLO detections across frames.

Provides temporal consistency by:
- Matching detections across frames via IoU (intersection over union)
- Smoothing bounding boxes with exponential moving average (EMA)
- Stabilizing class labels via majority vote over recent history
- Preserving champion identity once assigned to a track
"""

from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)

# --- Tracker hyperparameters ---
IOU_THRESHOLD = 0.3      # Minimum IoU to consider a match
MAX_AGE = 6             # Frames without a match before killing a track
HISTORY_LEN = 10         # Number of frames for class majority vote
BBOX_ALPHA = 0.4         # EMA smoothing factor (higher = more responsive)


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute intersection-over-union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class _Track:
    """Internal state for a single tracked object."""

    __slots__ = (
        "track_id", "bbox", "class_history", "confidence_history",
        "champion", "champion_confidence", "champion_votes",
        "champion_vote_names", "frames_since_seen",
    )

    def __init__(self, track_id: int, det: dict):
        self.track_id = track_id
        self.bbox = list(det["bbox"])
        self.class_history: list[str] = [det["class_name"]]
        self.confidence_history: list[float] = [det.get("confidence", 0.0)]
        self.champion: str | None = det.get("champion")
        self.champion_confidence: float | None = det.get("champion_confidence")
        self.champion_votes: dict[str, float] = {}
        self.champion_vote_names: dict[str, str] = {}  # key -> display name
        self.frames_since_seen = 0

    def update(self, det: dict) -> None:
        """Update track with a new matched detection."""
        # EMA bbox smoothing
        det_bbox = det["bbox"]
        for i in range(4):
            self.bbox[i] = BBOX_ALPHA * det_bbox[i] + (1 - BBOX_ALPHA) * self.bbox[i]

        # Append class + confidence to history (bounded)
        self.class_history.append(det["class_name"])
        if len(self.class_history) > HISTORY_LEN:
            self.class_history.pop(0)

        self.confidence_history.append(det.get("confidence", 0.0))
        if len(self.confidence_history) > HISTORY_LEN:
            self.confidence_history.pop(0)

        # Preserve champion identity once assigned (for played_champion from pipeline)
        if det.get("champion") and not det.get("champion_vote_key"):
            if not self.champion:
                self.champion = det["champion"]
                self.champion_confidence = det.get("champion_confidence")
            elif det.get("champion_confidence", 0) > (self.champion_confidence or 0):
                self.champion = det["champion"]
                self.champion_confidence = det.get("champion_confidence")

        # Accumulate champion match votes (for ally/enemy identification)
        vote_key = det.get("champion_vote_key")
        vote_score = det.get("champion_vote_score", 0.0)
        if vote_key and vote_score > 0:
            self.champion_votes[vote_key] = self.champion_votes.get(vote_key, 0.0) + vote_score
            if det.get("champion_vote_name"):
                self.champion_vote_names[vote_key] = det["champion_vote_name"]
            self._resolve_champion_from_votes()

        self.frames_since_seen = 0

    def _resolve_champion_from_votes(self) -> None:
        """Derive champion identity from accumulated votes if confident enough."""
        if not self.champion_votes:
            return
        total = sum(self.champion_votes.values())
        if total < 3.0:
            return  # Not enough evidence yet
        best_key = max(self.champion_votes, key=self.champion_votes.get)
        best_score = self.champion_votes[best_key]
        fraction = best_score / total
        if fraction >= 0.5:
            self.champion = self.champion_vote_names.get(best_key, best_key)
            self.champion_confidence = round(fraction, 4)

    def clear_champion_votes(self) -> None:
        """Clear accumulated votes (used when deduplication reassigns a champion)."""
        self.champion_votes.clear()
        self.champion_vote_names.clear()
        self.champion = None
        self.champion_confidence = None

    @property
    def voted_class(self) -> str:
        """Majority vote over recent class history."""
        counts = Counter(self.class_history)
        return counts.most_common(1)[0][0]

    @property
    def avg_confidence(self) -> float:
        """Average confidence over recent history."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def to_detection(self) -> dict:
        """Export track state as a detection dict."""
        det = {
            "class_name": self.voted_class,
            "confidence": round(self.avg_confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "track_id": self.track_id,
        }
        if self.champion:
            det["champion"] = self.champion
        if self.champion_confidence is not None:
            det["champion_confidence"] = self.champion_confidence
        return det


class SimpleTracker:
    """Frame-to-frame IoU tracker with EMA smoothing and majority-vote class labels."""

    def __init__(self):
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """Match new detections to existing tracks and return stabilized output.

        Args:
            detections: List of detection dicts with "bbox", "class_name", "confidence".

        Returns:
            List of tracked detection dicts with smoothed bbox, voted class,
            and track_id field added.
        """
        if not detections and not self._tracks:
            return []

        # Step 1: Compute IoU matrix and do greedy matching
        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        if self._tracks and detections:
            # Build (iou, track_idx, det_idx) pairs
            pairs: list[tuple[float, int, int]] = []
            for ti, track in enumerate(self._tracks):
                for di, det in enumerate(detections):
                    bbox = det.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    score = _iou(track.bbox, bbox)
                    if score >= IOU_THRESHOLD:
                        pairs.append((score, ti, di))

            # Greedy assignment: highest IoU first
            pairs.sort(key=lambda p: p[0], reverse=True)
            for score, ti, di in pairs:
                if ti in matched_track_indices or di in matched_det_indices:
                    continue
                self._tracks[ti].update(detections[di])
                matched_track_indices.add(ti)
                matched_det_indices.add(di)

        # Step 2: Age unmatched existing tracks, remove dead ones
        num_existing = len(self._tracks)
        surviving: list[_Track] = []
        for ti in range(num_existing):
            if ti in matched_track_indices:
                surviving.append(self._tracks[ti])
            else:
                self._tracks[ti].frames_since_seen += 1
                if self._tracks[ti].frames_since_seen <= MAX_AGE:
                    surviving.append(self._tracks[ti])

        # Step 3: Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            new_track = _Track(self._next_id, det)
            self._next_id += 1
            surviving.append(new_track)
        self._tracks = surviving

        # Step 4: Deduplicate ally champion assignments
        # If two freindly_champion tracks claim the same champion, keep the higher confidence one
        champ_owners: dict[str, _Track] = {}
        for track in self._tracks:
            if track.champion and track.voted_class == "freindly_champion":
                existing = champ_owners.get(track.champion)
                if existing is None:
                    champ_owners[track.champion] = track
                elif (track.champion_confidence or 0) > (existing.champion_confidence or 0):
                    existing.clear_champion_votes()
                    champ_owners[track.champion] = track
                else:
                    track.clear_champion_votes()

        # Step 5: Export all active tracks as detections
        return [t.to_detection() for t in self._tracks if t.frames_since_seen == 0]
