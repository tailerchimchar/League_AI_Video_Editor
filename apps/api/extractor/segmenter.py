"""Rule-based segment detection from frame-level features."""

from extractor.config import FIGHT_HP_DELTA_THRESHOLD, FIGHT_MIN_DURATION_MS


def detect_segments(
    frames: list[dict],
    video_id: str,
    job_id: str,
) -> list[dict]:
    """Detect game segments from frame-level features.

    Args:
        frames: List of frame payload dicts with derived_features and ocr_data.
        video_id: UUID string for the video.
        job_id: UUID string for the analysis job.

    Returns:
        List of segment dicts ready for DB insertion.
    """
    segments: list[dict] = []

    segments.extend(_detect_fights(frames, video_id, job_id))
    segments.extend(_detect_deaths(frames, video_id, job_id))

    return segments


def _detect_fights(frames: list[dict], video_id: str, job_id: str) -> list[dict]:
    """Detect fight windows based on sustained HP delta spikes."""
    segments = []
    fight_start_ms: int | None = None
    fight_frames: list[dict] = []

    for frame in frames:
        features = frame.get("derived_features", {})
        if isinstance(features, str):
            import json
            features = json.loads(features)

        hp_delta = features.get("hp_delta", 0) or 0

        if abs(hp_delta) > FIGHT_HP_DELTA_THRESHOLD:
            if fight_start_ms is None:
                fight_start_ms = frame["timestamp_ms"]
                fight_frames = []
            fight_frames.append(frame)
        else:
            if fight_start_ms is not None:
                duration = frame["timestamp_ms"] - fight_start_ms
                if duration >= FIGHT_MIN_DURATION_MS:
                    # Compute aggregate features for the fight
                    total_hp_loss = sum(
                        abs(f.get("derived_features", {}).get("hp_delta", 0) or 0)
                        for f in fight_frames
                    )
                    segments.append({
                        "video_id": video_id,
                        "job_id": job_id,
                        "segment_type": "fight",
                        "start_ms": fight_start_ms,
                        "end_ms": frame["timestamp_ms"],
                        "confidence": min(0.5 + total_hp_loss, 1.0),
                        "features": {
                            "total_hp_loss": total_hp_loss,
                            "frame_count": len(fight_frames),
                        },
                    })
                fight_start_ms = None
                fight_frames = []

    # Handle fight at end of video
    if fight_start_ms is not None and frames:
        duration = frames[-1]["timestamp_ms"] - fight_start_ms
        if duration >= FIGHT_MIN_DURATION_MS:
            segments.append({
                "video_id": video_id,
                "job_id": job_id,
                "segment_type": "fight",
                "start_ms": fight_start_ms,
                "end_ms": frames[-1]["timestamp_ms"],
                "confidence": 0.6,
                "features": {"frame_count": len(fight_frames)},
            })

    return segments


def _detect_deaths(frames: list[dict], video_id: str, job_id: str) -> list[dict]:
    """Detect death events based on HP dropping to 0."""
    segments = []

    for i, frame in enumerate(frames):
        ocr = frame.get("ocr_data", {})
        if isinstance(ocr, str):
            import json
            ocr = json.loads(ocr)

        hp_pct = ocr.get("player_hp_pct")
        if hp_pct is not None and hp_pct <= 0.01:
            # Check if previous frame had HP > 0
            if i > 0:
                prev_ocr = frames[i - 1].get("ocr_data", {})
                if isinstance(prev_ocr, str):
                    import json
                    prev_ocr = json.loads(prev_ocr)
                prev_hp = prev_ocr.get("player_hp_pct")
                if prev_hp is not None and prev_hp > 0.01:
                    segments.append({
                        "video_id": video_id,
                        "job_id": job_id,
                        "segment_type": "death",
                        "start_ms": frame["timestamp_ms"] - 500,
                        "end_ms": frame["timestamp_ms"] + 500,
                        "confidence": 0.8,
                        "features": {"hp_before": prev_hp},
                    })

    return segments
