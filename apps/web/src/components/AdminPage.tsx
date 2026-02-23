import { useState, useEffect, useCallback } from "react";
import {
  listVideos,
  getHealthBarDebug,
  type VideoListItem,
  type HealthBarDebugResponse,
  type HealthBarDetection,
} from "../api/client";

export default function AdminPage() {
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [frameIndex, setFrameIndex] = useState(0);
  const [maxFrame, setMaxFrame] = useState(0);
  const [debugData, setDebugData] = useState<HealthBarDebugResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listVideos({ limit: 100 }).then((data) => {
      setVideos(data.videos);
    }).catch(() => {
      setError("Failed to load videos");
    });
  }, []);

  const fetchDebug = useCallback(async (videoId: string, frame: number) => {
    if (!videoId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getHealthBarDebug(videoId, frame);
      setDebugData(data);
      setMaxFrame(data.max_frame_index);
    } catch (e: unknown) {
      const msg = (e as { message?: string })?.message ?? "Failed to fetch debug data";
      setError(msg);
      setDebugData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleVideoSelect = useCallback((id: string) => {
    setSelectedVideoId(id);
    setFrameIndex(0);
    setDebugData(null);
    if (id) fetchDebug(id, 0);
  }, [fetchDebug]);

  const handleFrameChange = useCallback((frame: number) => {
    const clamped = Math.max(0, Math.min(frame, maxFrame));
    setFrameIndex(clamped);
    if (selectedVideoId) fetchDebug(selectedVideoId, clamped);
  }, [selectedVideoId, maxFrame, fetchDebug]);

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-lg font-semibold text-white">Health Bar Color Debug Tool</h2>
      <p className="text-sm text-gray-500 -mt-2">
        Select a video and step through frames to inspect YOLO detections and health bar HSV analysis.
      </p>

      {/* Video selector */}
      <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-2">
        <label className="text-sm font-semibold text-gray-300">Video</label>
        <select
          className="px-3 py-2 border border-[#333] rounded-md bg-[#0f1115] text-gray-200 text-sm font-sans"
          value={selectedVideoId}
          onChange={(e) => handleVideoSelect(e.target.value)}
        >
          <option value="">-- Select a video --</option>
          {videos.map((v) => (
            <option key={v.id} value={v.id}>
              {v.filename} ({v.status})
            </option>
          ))}
        </select>
      </div>

      {/* Frame stepper */}
      {selectedVideoId && (
        <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-2">
          <label className="text-sm font-semibold text-gray-300">
            Frame {frameIndex} / {maxFrame}
            {debugData && (
              <span className="font-normal text-gray-500"> ({debugData.timestamp_ms} ms)</span>
            )}
          </label>
          <div className="flex items-center gap-2">
            <button
              className="px-3 py-1.5 border border-[#444] rounded-md bg-transparent text-gray-300 text-sm cursor-pointer shrink-0 hover:border-[#666] hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              onClick={() => handleFrameChange(frameIndex - 1)}
              disabled={frameIndex <= 0 || loading}
            >
              &lt; Prev
            </button>
            <input
              type="range"
              className="flex-1 h-1 accent-[#646cff] cursor-pointer"
              min={0}
              max={maxFrame || 1}
              value={frameIndex}
              onChange={(e) => handleFrameChange(Number(e.target.value))}
              disabled={loading}
            />
            <button
              className="px-3 py-1.5 border border-[#444] rounded-md bg-transparent text-gray-300 text-sm cursor-pointer shrink-0 hover:border-[#666] hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              onClick={() => handleFrameChange(frameIndex + 1)}
              disabled={frameIndex >= maxFrame || loading}
            >
              Next &gt;
            </button>
            <input
              type="number"
              className="w-16 px-2 py-1 border border-[#333] rounded bg-[#0f1115] text-gray-200 text-sm text-center font-sans focus:outline-none focus:border-[#646cff]"
              min={0}
              max={maxFrame}
              value={frameIndex}
              onChange={(e) => handleFrameChange(Number(e.target.value))}
              disabled={loading}
            />
          </div>
        </div>
      )}

      {loading && <p className="text-sm text-gray-500">Analyzing frame...</p>}
      {error && <p className="text-sm text-red-500">{error}</p>}

      {/* Detection cards */}
      {debugData && debugData.detections.length > 0 && (
        <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-3">
          <label className="text-sm font-semibold text-gray-300">
            {debugData.detections.length} detection{debugData.detections.length !== 1 ? "s" : ""}
            {" "}({debugData.frame_width}x{debugData.frame_height})
          </label>
          <div className="flex flex-col gap-3">
            {debugData.detections.map((det, i) => (
              <DetectionCard key={i} det={det} />
            ))}
          </div>
        </div>
      )}

      {debugData && debugData.detections.length === 0 && !loading && (
        <p className="text-sm text-gray-500">No detections on this frame.</p>
      )}
    </div>
  );
}

function DetectionCard({ det }: { det: HealthBarDetection }) {
  const confPct = Math.round(det.confidence * 100);
  const colorLabel = det.health_bar_color ?? "none";
  const colorHex = det.health_bar_color === "green" ? "#22c55e"
    : det.health_bar_color === "blue" ? "#3b82f6"
    : det.health_bar_color === "red" ? "#ef4444"
    : "#666";

  return (
    <div className="bg-[#0f1115] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-3">
      {/* Header row: class + confidence + color */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-white">{det.class_name}</span>
          {det.corrected && det.original_class && (
            <span className="text-[10px] text-yellow-500 bg-yellow-500/10 px-1.5 py-0.5 rounded">
              was {det.original_class}
            </span>
          )}
          <span className="text-xs text-gray-500">{confPct}%</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="w-2.5 h-2.5 rounded-full shrink-0"
            style={{ background: colorHex }}
          />
          <span className="text-xs text-gray-400">{colorLabel}</span>
        </div>
      </div>

      {/* Context crop — the main image showing gameplay with bbox */}
      {det.context_b64 && (
        <img
          className="w-full rounded-md border border-[#2a2d35]"
          src={`data:image/jpeg;base64,${det.context_b64}`}
          alt={`${det.class_name} context`}
          style={{ imageRendering: "auto" }}
        />
      )}

      {/* Bottom row: color bars + health bar crop + bbox */}
      <div className="flex gap-4 items-start">
        {/* Color fraction bars */}
        <div className="flex flex-col gap-1 flex-1 min-w-0">
          <ColorBar label="G" value={det.color_fractions.green} color="#22c55e" />
          <ColorBar label="B" value={det.color_fractions.blue} color="#3b82f6" />
          <ColorBar label="R" value={det.color_fractions.red} color="#ef4444" />
        </div>

        {/* Health bar crop (small) */}
        {det.crop_b64 && (
          <div className="flex flex-col gap-1 shrink-0">
            <span className="text-[10px] text-gray-500">Health bar crop</span>
            <img
              className="h-5 w-auto rounded border border-[#2a2d35]"
              src={`data:image/jpeg;base64,${det.crop_b64}`}
              alt="Health bar"
              style={{ imageRendering: "pixelated" }}
            />
          </div>
        )}

        {/* Bbox coords */}
        <span className="text-[10px] text-gray-600 font-mono shrink-0 self-end">
          [{det.bbox.map(Math.round).join(", ")}]
        </span>
      </div>
    </div>
  );
}

function ColorBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold text-gray-500 w-3 text-center shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-[#1a1d24] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-[width] duration-200"
          style={{ width: `${Math.min(100, pct)}%`, background: color }}
        />
      </div>
      <span className="text-[10px] text-gray-500 w-7 text-right shrink-0 tabular-nums">{pct}%</span>
    </div>
  );
}
