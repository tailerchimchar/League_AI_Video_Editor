/**
 * Typed API client for the League AI Video Editor backend.
 */

const BASE = "/api/v1";

/**
 * Direct URL to FastAPI for file uploads. The Vite dev proxy can't handle
 * large multipart uploads reliably, so uploads go directly to the API.
 * In production this would be the same origin.
 */
const UPLOAD_BASE = import.meta.env.DEV ? "http://localhost:8000/api/v1" : "/api/v1";

export interface VideoUploadResult {
  video_id: string;
  filename: string;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  status: string;
}

export interface ExtractionConfig {
  sample_fps?: number;
  enable_ocr?: boolean;
  enable_detector?: boolean;
  enable_debug_overlays?: boolean;
  crop_preset?: string;
  played_champion?: string;
}

export interface ExtractionStatus {
  job_id: string;
  video_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  frame_count: number;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
  /** Live extraction progress (only present during extraction) */
  extracted_frames?: number;
  total_frames?: number;
  phase?: "extracting" | "inserting";
}

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] in pixel coords
  champion?: string | null;
  champion_confidence?: number | null;
  track_id?: number;
}

export interface FramePayload {
  frame_index: number;
  timestamp_ms: number;
  ocr_data: Record<string, unknown>;
  detections: Detection[];
  derived_features: Record<string, unknown>;
}

export interface Segment {
  id: string;
  segment_type: string;
  start_ms: number;
  end_ms: number;
  confidence: number;
  features: Record<string, unknown>;
}

export interface ApiError {
  error_code: string;
  message: string;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body: ApiError = await res.json();
    throw body;
  }
  return res.json();
}

/** Upload a video to the new DB-backed endpoint.
 *  Uses direct API URL to bypass Vite proxy (which can't handle large uploads).
 */
export async function uploadVideo(file: File): Promise<VideoUploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${UPLOAD_BASE}/videos`, { method: "POST", body: form });
  return handleResponse<VideoUploadResult>(res);
}

/** Start extraction pipeline for a video. */
export async function startExtraction(
  videoId: string,
  config?: ExtractionConfig
): Promise<{ job_id: string; video_id: string; status: string; progress: number }> {
  const res = await fetch(`${BASE}/videos/${videoId}/extract`, {
    method: "POST",
    headers: config ? { "Content-Type": "application/json" } : {},
    body: config ? JSON.stringify(config) : undefined,
  });
  return handleResponse(res);
}

/** Poll extraction job status. */
export async function getExtractionStatus(videoId: string): Promise<ExtractionStatus> {
  const res = await fetch(`${BASE}/videos/${videoId}/extract/status`);
  return handleResponse<ExtractionStatus>(res);
}

/** Get extracted frames for a video. */
export async function getFrames(
  videoId: string,
  params?: { start_ms?: number; end_ms?: number; limit?: number }
): Promise<{ frames: FramePayload[]; count: number }> {
  const query = new URLSearchParams();
  if (params?.start_ms != null) query.set("start_ms", String(params.start_ms));
  if (params?.end_ms != null) query.set("end_ms", String(params.end_ms));
  if (params?.limit != null) query.set("limit", String(params.limit));
  const qs = query.toString();
  const res = await fetch(`${BASE}/videos/${videoId}/frames${qs ? `?${qs}` : ""}`);
  return handleResponse(res);
}

/** Get detected segments for a video. */
export async function getSegments(
  videoId: string
): Promise<{ segments: Segment[]; count: number }> {
  const res = await fetch(`${BASE}/videos/${videoId}/segments`);
  return handleResponse(res);
}

/** Get video serve URL. */
export function getVideoUrl(videoId: string): string {
  return `${BASE}/videos/${videoId}`;
}

/** Get report SSE URL. */
export function getReportUrl(videoId: string): string {
  return `${BASE}/videos/${videoId}/report`;
}

// ── Debug / Admin endpoints ─────────────────────────────────────────

export interface VideoListItem {
  id: string;
  filename: string;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  status: string;
  created_at: string;
}

export interface HealthBarDetection {
  class_name: string;
  original_class?: string;
  corrected?: boolean;
  confidence: number;
  bbox: [number, number, number, number];
  health_bar_color: string | null;
  color_fractions: { green: number; blue: number; red: number };
  crop_b64: string | null;
  context_b64: string | null;
}

export interface HealthBarDebugResponse {
  frame_index: number;
  timestamp_ms: number;
  frame_width: number;
  frame_height: number;
  max_frame_index: number;
  detections: HealthBarDetection[];
}

/** List all uploaded videos. */
export async function listVideos(
  params?: { limit?: number; offset?: number }
): Promise<{ videos: VideoListItem[] }> {
  const query = new URLSearchParams();
  if (params?.limit != null) query.set("limit", String(params.limit));
  if (params?.offset != null) query.set("offset", String(params.offset));
  const qs = query.toString();
  const res = await fetch(`${BASE}/videos${qs ? `?${qs}` : ""}`);
  return handleResponse(res);
}

/** Get health bar debug analysis for a single frame. */
export async function getHealthBarDebug(
  videoId: string,
  frameIndex: number,
  sampleFps?: number,
): Promise<HealthBarDebugResponse> {
  const query = new URLSearchParams();
  query.set("frame_index", String(frameIndex));
  if (sampleFps != null) query.set("sample_fps", String(sampleFps));
  const res = await fetch(`${BASE}/videos/${videoId}/debug/health-bar-colors?${query}`);
  return handleResponse(res);
}

// ── Champion ID Debug ───────────────────────────────────────────────

export interface ChampionIdAllyPortrait {
  slot: number;
  key: string;
  name: string;
  confidence: number;
  portrait_crop_b64: string | null;
  icon_b64: string | null;
}

export interface ChampionIdCandidateScore {
  key: string;
  name: string;
  hist_score: number;
  tmpl_score: number;
  combined: number;
  icon_b64: string | null;
  best_ref_label: string | null;
  best_ref_b64: string | null;
}

export interface ChampionIdDetection {
  confidence: number;
  bbox: [number, number, number, number];
  body_crop_b64: string | null;
  context_b64: string | null;
  match_result: { key: string; name: string; score: number } | null;
  candidate_scores: ChampionIdCandidateScore[];
}

export interface ChampionIdDebugResponse {
  frame_index: number;
  timestamp_ms: number;
  frame_width: number;
  frame_height: number;
  max_frame_index: number;
  portrait_strip_b64: string | null;
  player: {
    key: string;
    name: string;
    confidence: number;
    portrait_crop_b64: string | null;
  };
  ally_portraits: ChampionIdAllyPortrait[];
  ally_detections: ChampionIdDetection[];
  enemy_detections: ChampionIdDetection[];
}

/** Get champion ID debug analysis for a single frame. */
export async function getChampionIdDebug(
  videoId: string,
  frameIndex: number,
  sampleFps?: number,
): Promise<ChampionIdDebugResponse> {
  const query = new URLSearchParams();
  query.set("frame_index", String(frameIndex));
  if (sampleFps != null) query.set("sample_fps", String(sampleFps));
  const res = await fetch(`${BASE}/videos/${videoId}/debug/champion-id?${query}`);
  return handleResponse(res);
}
