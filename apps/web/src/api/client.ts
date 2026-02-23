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
