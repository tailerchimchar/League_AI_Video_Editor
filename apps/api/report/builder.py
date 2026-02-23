"""Evidence assembly and prompt construction for LLM reports."""

import json

from db.queries import (
    get_video,
    get_latest_job_for_video,
    get_frames_for_job,
    get_segments_for_video,
)
from report.prompts import REPORT_SYSTEM_PROMPT, REPORT_USER_TEMPLATE


def _ms_to_display(ms: int) -> str:
    """Convert milliseconds to M:SS display format."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def _format_ocr_timeline(frames: list[dict]) -> str:
    """Format OCR timeline data for the prompt."""
    lines = []
    for f in frames:
        ocr = f.get("ocr_data", {})
        if isinstance(ocr, str):
            ocr = json.loads(ocr)

        parts = [f"t={_ms_to_display(f['timestamp_ms'])}"]

        if ocr.get("game_timer"):
            parts.append(f"game={ocr['game_timer']}")
        if ocr.get("player_cs") is not None:
            parts.append(f"CS={ocr['player_cs']}")
        if ocr.get("player_gold") is not None:
            parts.append(f"gold={ocr['player_gold']}")
        if ocr.get("player_kda"):
            kda = ocr["player_kda"]
            parts.append(f"KDA={kda.get('kills',0)}/{kda.get('deaths',0)}/{kda.get('assists',0)}")
        if ocr.get("player_hp_pct") is not None:
            parts.append(f"HP={ocr['player_hp_pct']:.0%}")
        if ocr.get("player_level") is not None:
            parts.append(f"Lv={ocr['player_level']}")

        lines.append(" | ".join(parts))

    return "\n".join(lines) if lines else "No OCR data available."


def _format_segments(segments: list[dict]) -> str:
    """Format segments for the prompt."""
    if not segments:
        return "No segments detected."

    lines = []
    for s in segments:
        features = s.get("features", {})
        if isinstance(features, str):
            features = json.loads(features)

        line = (
            f"- {s['segment_type'].upper()} "
            f"({_ms_to_display(s['start_ms'])} → {_ms_to_display(s['end_ms'])})"
            f" confidence={s['confidence']:.1f}"
        )
        if features:
            line += f" {features}"
        lines.append(line)

    return "\n".join(lines)


def _compute_feature_summary(frames: list[dict]) -> str:
    """Compute aggregate feature summary across all frames."""
    cs_values = []
    gold_values = []
    hp_deltas = []
    cs_deltas = []

    for f in frames:
        ocr = f.get("ocr_data", {})
        if isinstance(ocr, str):
            ocr = json.loads(ocr)
        features = f.get("derived_features", {})
        if isinstance(features, str):
            features = json.loads(features)

        if ocr.get("player_cs") is not None:
            cs_values.append(ocr["player_cs"])
        if ocr.get("player_gold") is not None:
            gold_values.append(ocr["player_gold"])
        if features.get("hp_delta") is not None:
            hp_deltas.append(features["hp_delta"])
        if features.get("cs_delta") is not None:
            cs_deltas.append(features["cs_delta"])

    lines = []
    if cs_values:
        lines.append(f"CS range: {min(cs_values)} → {max(cs_values)}")
    if gold_values:
        lines.append(f"Gold range: {min(gold_values)} → {max(gold_values)}")
    if hp_deltas:
        total_dmg = sum(abs(d) for d in hp_deltas if d < 0)
        lines.append(f"Total HP% lost: {total_dmg:.1%}")
        big_hits = sum(1 for d in hp_deltas if d < -0.1)
        lines.append(f"Large HP drops (>10%): {big_hits}")
    if cs_deltas:
        avg_cspm = sum(cs_deltas) / len(cs_deltas) if cs_deltas else 0
        lines.append(f"Average CS/min: {avg_cspm:.1f}")

    return "\n".join(lines) if lines else "Insufficient data for feature summary."


async def build_evidence_payload(video_id: str) -> dict | None:
    """Assemble structured evidence for LLM prompt.

    Returns dict with system_prompt and user_prompt, or None if no data.
    """
    video = await get_video(video_id)
    if not video:
        return None

    job = await get_latest_job_for_video(video_id)
    if not job or job["status"] != "completed":
        return None

    job_id = str(job["id"])
    frames = await get_frames_for_job(job_id)
    segments = await get_segments_for_video(video_id)

    if not frames:
        return None

    # Convert records to dicts
    frame_dicts = [dict(f) for f in frames]
    segment_dicts = [dict(s) for s in segments]

    # Sample every 5th frame for prompt size management
    sampled_frames = frame_dicts[::5]

    duration_ms = video.get("duration_ms") or 0
    duration_display = _ms_to_display(duration_ms) if duration_ms else "unknown"

    user_prompt = REPORT_USER_TEMPLATE.format(
        duration_display=duration_display,
        total_frames=len(frame_dicts),
        ocr_timeline=_format_ocr_timeline(sampled_frames),
        segments_text=_format_segments(segment_dicts),
        feature_summary=_compute_feature_summary(frame_dicts),
    )

    return {
        "system_prompt": REPORT_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "job_id": job_id,
        "frame_count": len(frame_dicts),
        "segment_count": len(segment_dicts),
    }
