-- 001_initial_schema.sql
-- League AI Video Editor — initial database schema

-- Videos table
CREATE TABLE IF NOT EXISTS videos (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    storage_path    TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    duration_ms     INTEGER,
    width           INTEGER,
    height          INTEGER,
    mime_type       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'uploaded'
                    CHECK (status IN ('uploaded','extracting','extracted','failed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_created ON videos(created_at DESC);

-- Analysis jobs table
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id        UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','running','completed','failed','cancelled')),
    pipeline_version TEXT NOT NULL DEFAULT 'v1',
    config          JSONB NOT NULL DEFAULT '{}',
    progress        REAL DEFAULT 0.0,
    frame_count     INTEGER DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_jobs_video ON analysis_jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status);

-- Per-frame extraction payloads
CREATE TABLE IF NOT EXISTS frame_payloads (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    video_id        UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frame_index     INTEGER NOT NULL,
    timestamp_ms    INTEGER NOT NULL,
    ocr_data        JSONB NOT NULL DEFAULT '{}',
    detections      JSONB NOT NULL DEFAULT '{}',
    derived_features JSONB NOT NULL DEFAULT '{}',
    crop_paths      JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(job_id, frame_index)
);

CREATE INDEX IF NOT EXISTS idx_frames_job ON frame_payloads(job_id);
CREATE INDEX IF NOT EXISTS idx_frames_video_ts ON frame_payloads(video_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_frames_ocr ON frame_payloads USING GIN (ocr_data);

-- Detected segments (fights, trades, lane phases)
CREATE TABLE IF NOT EXISTS segments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id        UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    job_id          UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    segment_type    TEXT NOT NULL
                    CHECK (segment_type IN ('fight','trade','lane','roam','objective','death','recall')),
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,
    confidence      REAL DEFAULT 1.0,
    features        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (end_ms > start_ms)
);

CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_segments_type ON segments(video_id, segment_type);
CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(video_id, start_ms, end_ms);

-- Manual labels (for training data bootstrapping)
CREATE TABLE IF NOT EXISTS labels (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id        UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    segment_id      UUID REFERENCES segments(id) ON DELETE SET NULL,
    frame_index     INTEGER,
    label_type      TEXT NOT NULL,
    value           JSONB NOT NULL,
    source          TEXT NOT NULL DEFAULT 'manual'
                    CHECK (source IN ('manual','model','rule','import')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_labels_video ON labels(video_id);
CREATE INDEX IF NOT EXISTS idx_labels_type ON labels(label_type);

-- LLM-generated reports (evidence-grounded)
CREATE TABLE IF NOT EXISTS reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id        UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    job_id          UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    summary_text    TEXT NOT NULL,
    evidence_refs   JSONB NOT NULL DEFAULT '[]',
    model_used      TEXT NOT NULL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reports_video ON reports(video_id);
