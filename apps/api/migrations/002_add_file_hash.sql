-- 002_add_file_hash.sql
-- Add file_hash column for video deduplication

ALTER TABLE videos ADD COLUMN IF NOT EXISTS file_hash TEXT;
CREATE INDEX IF NOT EXISTS idx_videos_file_hash ON videos(file_hash);
