"""Typed query functions for all database tables."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import asyncpg

from db.engine import get_pool


# ── Videos ─────────────────────────────────────────────────────────────────────

async def insert_video(
    *,
    filename: str,
    storage_path: str,
    file_size_bytes: int,
    duration_ms: int | None,
    width: int | None,
    height: int | None,
    mime_type: str,
    file_hash: str | None = None,
) -> dict:
    pool = get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO videos (filename, storage_path, file_size_bytes, duration_ms, width, height, mime_type, file_hash)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING *
        """,
        filename, storage_path, file_size_bytes, duration_ms, width, height, mime_type, file_hash,
    )
    return dict(row)


async def get_video(video_id: str) -> dict | None:
    pool = get_pool()
    row = await pool.fetchrow("SELECT * FROM videos WHERE id = $1", uuid.UUID(video_id))
    return dict(row) if row else None


async def update_video_status(video_id: str, status: str) -> None:
    pool = get_pool()
    await pool.execute(
        "UPDATE videos SET status = $1, updated_at = now() WHERE id = $2",
        status, uuid.UUID(video_id),
    )


async def list_videos(limit: int = 50, offset: int = 0) -> list[dict]:
    pool = get_pool()
    rows = await pool.fetch(
        "SELECT * FROM videos ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        limit, offset,
    )
    return [dict(r) for r in rows]


# ── Analysis Jobs ──────────────────────────────────────────────────────────────

async def insert_job(
    *,
    video_id: str,
    config: dict | None = None,
) -> dict:
    pool = get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO analysis_jobs (video_id, config)
        VALUES ($1, $2::jsonb)
        RETURNING *
        """,
        uuid.UUID(video_id), json.dumps(config or {}),
    )
    return dict(row)


async def get_job(job_id: str) -> dict | None:
    pool = get_pool()
    row = await pool.fetchrow("SELECT * FROM analysis_jobs WHERE id = $1", uuid.UUID(job_id))
    return dict(row) if row else None


async def get_latest_job_for_video(video_id: str) -> dict | None:
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT * FROM analysis_jobs
        WHERE video_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        uuid.UUID(video_id),
    )
    return dict(row) if row else None


async def update_job_status(
    job_id: str,
    status: str,
    *,
    error_message: str | None = None,
) -> None:
    pool = get_pool()
    if status == "running":
        await pool.execute(
            "UPDATE analysis_jobs SET status = $1, started_at = now() WHERE id = $2",
            status, uuid.UUID(job_id),
        )
    elif status in ("completed", "failed"):
        await pool.execute(
            """
            UPDATE analysis_jobs
            SET status = $1, completed_at = now(), error_message = $2
            WHERE id = $3
            """,
            status, error_message, uuid.UUID(job_id),
        )
    else:
        await pool.execute(
            "UPDATE analysis_jobs SET status = $1 WHERE id = $2",
            status, uuid.UUID(job_id),
        )


async def update_job_progress(job_id: str, progress: float, frame_count: int = 0) -> None:
    pool = get_pool()
    await pool.execute(
        "UPDATE analysis_jobs SET progress = $1, frame_count = $2 WHERE id = $3",
        progress, frame_count, uuid.UUID(job_id),
    )


# ── Frame Payloads ─────────────────────────────────────────────────────────────

async def batch_insert_frames(frames: list[dict]) -> None:
    if not frames:
        return
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO frame_payloads
                (job_id, video_id, frame_index, timestamp_ms, ocr_data, detections, derived_features, crop_paths)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb)
            ON CONFLICT (job_id, frame_index) DO UPDATE SET
                ocr_data = EXCLUDED.ocr_data,
                detections = EXCLUDED.detections,
                derived_features = EXCLUDED.derived_features,
                crop_paths = EXCLUDED.crop_paths
            """,
            [
                (
                    uuid.UUID(f["job_id"]),
                    uuid.UUID(f["video_id"]),
                    f["frame_index"],
                    f["timestamp_ms"],
                    json.dumps(f.get("ocr_data", {})),
                    json.dumps(f.get("detections", {})),
                    json.dumps(f.get("derived_features", {})),
                    json.dumps(f.get("crop_paths", {})),
                )
                for f in frames
            ],
        )


async def get_frames_for_job(job_id: str) -> list[dict]:
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT * FROM frame_payloads
        WHERE job_id = $1
        ORDER BY frame_index
        """,
        uuid.UUID(job_id),
    )
    return [dict(r) for r in rows]


async def get_frames_for_video(
    video_id: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Return frames from the latest completed extraction job for a video."""
    pool = get_pool()
    vid = uuid.UUID(video_id)

    # Find the latest completed job for this video
    latest_job = await pool.fetchrow(
        "SELECT id FROM analysis_jobs WHERE video_id = $1 AND status = 'completed' "
        "ORDER BY created_at DESC LIMIT 1",
        vid,
    )

    conditions = ["video_id = $1"]
    params: list = [vid]
    idx = 2

    if latest_job:
        conditions.append(f"job_id = ${idx}")
        params.append(latest_job["id"])
        idx += 1

    if start_ms is not None:
        conditions.append(f"timestamp_ms >= ${idx}")
        params.append(start_ms)
        idx += 1

    if end_ms is not None:
        conditions.append(f"timestamp_ms <= ${idx}")
        params.append(end_ms)
        idx += 1

    where = " AND ".join(conditions)
    params.append(limit)

    rows = await pool.fetch(
        f"SELECT * FROM frame_payloads WHERE {where} ORDER BY timestamp_ms LIMIT ${idx}",
        *params,
    )
    return [dict(r) for r in rows]


# ── Segments ───────────────────────────────────────────────────────────────────

async def batch_insert_segments(segments: list[dict]) -> None:
    if not segments:
        return
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO segments (video_id, job_id, segment_type, start_ms, end_ms, confidence, features)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            """,
            [
                (
                    uuid.UUID(s["video_id"]),
                    uuid.UUID(s["job_id"]),
                    s["segment_type"],
                    s["start_ms"],
                    s["end_ms"],
                    s.get("confidence", 1.0),
                    json.dumps(s.get("features", {})),
                )
                for s in segments
            ],
        )


async def get_segments_for_video(video_id: str) -> list[dict]:
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT * FROM segments
        WHERE video_id = $1
        ORDER BY start_ms
        """,
        uuid.UUID(video_id),
    )
    return [dict(r) for r in rows]


# ── Labels ─────────────────────────────────────────────────────────────────────

async def insert_label(
    *,
    video_id: str,
    segment_id: str | None = None,
    frame_index: int | None = None,
    label_type: str,
    value: dict,
    source: str = "manual",
) -> dict:
    pool = get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO labels (video_id, segment_id, frame_index, label_type, value, source)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6)
        RETURNING *
        """,
        uuid.UUID(video_id),
        uuid.UUID(segment_id) if segment_id else None,
        frame_index,
        label_type,
        json.dumps(value),
        source,
    )
    return dict(row)


async def get_labels_for_video(video_id: str) -> list[dict]:
    pool = get_pool()
    rows = await pool.fetch(
        "SELECT * FROM labels WHERE video_id = $1 ORDER BY created_at",
        uuid.UUID(video_id),
    )
    return [dict(r) for r in rows]


# ── Reports ────────────────────────────────────────────────────────────────────

async def insert_report(
    *,
    video_id: str,
    job_id: str,
    summary_text: str,
    evidence_refs: list[dict],
    model_used: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict:
    pool = get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO reports (video_id, job_id, summary_text, evidence_refs, model_used, prompt_tokens, completion_tokens)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
        RETURNING *
        """,
        uuid.UUID(video_id),
        uuid.UUID(job_id),
        summary_text,
        json.dumps(evidence_refs),
        model_used,
        prompt_tokens,
        completion_tokens,
    )
    return dict(row)


async def get_latest_report_for_video(video_id: str) -> dict | None:
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT * FROM reports
        WHERE video_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        uuid.UUID(video_id),
    )
    return dict(row) if row else None
