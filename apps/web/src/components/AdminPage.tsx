import { useState, useEffect, useCallback } from "react";
import {
  listVideos,
  getHealthBarDebug,
  getChampionIdDebug,
  type VideoListItem,
  type HealthBarDebugResponse,
  type HealthBarDetection,
  type ChampionIdDebugResponse,
  type ChampionIdDetection,
  type ChampionIdAllyPortrait,
  type ChampionIdCandidateScore,
} from "../api/client";

// ── Shared components ───────────────────────────────────────────────

function VideoSelector({
  videos, selected, onSelect,
}: { videos: VideoListItem[]; selected: string; onSelect: (id: string) => void }) {
  return (
    <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-2">
      <label className="text-sm font-semibold text-gray-300">Video</label>
      <select
        className="px-3 py-2 border border-[#333] rounded-md bg-[#0f1115] text-gray-200 text-sm font-sans"
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
      >
        <option value="">-- Select a video --</option>
        {videos.map((v) => (
          <option key={v.id} value={v.id}>
            {v.filename} ({v.status})
          </option>
        ))}
      </select>
    </div>
  );
}

function FrameStepper({
  frameIndex, maxFrame, timestampMs, loading, onChange,
}: {
  frameIndex: number; maxFrame: number; timestampMs?: number;
  loading: boolean; onChange: (f: number) => void;
}) {
  const btnCls = "px-3 py-1.5 border border-[#444] rounded-md bg-transparent text-gray-300 text-sm cursor-pointer shrink-0 hover:border-[#666] hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors";
  return (
    <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-2">
      <label className="text-sm font-semibold text-gray-300">
        Frame {frameIndex} / {maxFrame}
        {timestampMs != null && (
          <span className="font-normal text-gray-500"> ({timestampMs} ms)</span>
        )}
      </label>
      <div className="flex items-center gap-2">
        <button className={btnCls} onClick={() => onChange(frameIndex - 1)} disabled={frameIndex <= 0 || loading}>
          &lt; Prev
        </button>
        <input
          type="range" className="flex-1 h-1 accent-[#646cff] cursor-pointer"
          min={0} max={maxFrame || 1} value={frameIndex}
          onChange={(e) => onChange(Number(e.target.value))} disabled={loading}
        />
        <button className={btnCls} onClick={() => onChange(frameIndex + 1)} disabled={frameIndex >= maxFrame || loading}>
          Next &gt;
        </button>
        <input
          type="number"
          className="w-16 px-2 py-1 border border-[#333] rounded bg-[#0f1115] text-gray-200 text-sm text-center font-sans focus:outline-none focus:border-[#646cff]"
          min={0} max={maxFrame} value={frameIndex}
          onChange={(e) => onChange(Number(e.target.value))} disabled={loading}
        />
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

function ScoreBar({ score, maxScore = 1 }: { score: number; maxScore?: number }) {
  const pct = Math.min(100, Math.round((score / maxScore) * 100));
  const hue = score > 0.3 ? 120 : score > 0.1 ? 60 : 0; // green/yellow/red
  return (
    <div className="flex-1 h-1.5 bg-[#1a1d24] rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-[width] duration-200"
        style={{ width: `${pct}%`, background: `hsl(${hue}, 70%, 50%)` }}
      />
    </div>
  );
}

// ── Health Bar Debug Tab ────────────────────────────────────────────

function HealthBarDebugTab({ videos }: { videos: VideoListItem[] }) {
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [frameIndex, setFrameIndex] = useState(0);
  const [maxFrame, setMaxFrame] = useState(0);
  const [debugData, setDebugData] = useState<HealthBarDebugResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDebug = useCallback(async (videoId: string, frame: number) => {
    if (!videoId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getHealthBarDebug(videoId, frame);
      setDebugData(data);
      setMaxFrame(data.max_frame_index);
    } catch (e: unknown) {
      setError((e as { message?: string })?.message ?? "Failed to fetch");
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
      <p className="text-[13px] text-gray-500 -mt-1">
        Step through frames to inspect YOLO detections and health bar HSV color analysis.
      </p>
      <VideoSelector videos={videos} selected={selectedVideoId} onSelect={handleVideoSelect} />
      {selectedVideoId && (
        <FrameStepper
          frameIndex={frameIndex} maxFrame={maxFrame}
          timestampMs={debugData?.timestamp_ms} loading={loading}
          onChange={handleFrameChange}
        />
      )}
      {loading && <p className="text-sm text-gray-500">Analyzing frame...</p>}
      {error && <p className="text-sm text-red-500">{error}</p>}
      {debugData && debugData.detections.length > 0 && (
        <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-3">
          <label className="text-sm font-semibold text-gray-300">
            {debugData.detections.length} detection{debugData.detections.length !== 1 ? "s" : ""}
            {" "}({debugData.frame_width}x{debugData.frame_height})
          </label>
          <div className="flex flex-col gap-3">
            {debugData.detections.map((det, i) => (
              <HealthBarDetectionCard key={i} det={det} />
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

function HealthBarDetectionCard({ det }: { det: HealthBarDetection }) {
  const confPct = Math.round(det.confidence * 100);
  const colorHex = det.health_bar_color === "green" ? "#22c55e"
    : det.health_bar_color === "blue" ? "#3b82f6"
    : det.health_bar_color === "red" ? "#ef4444" : "#666";
  return (
    <div className="bg-[#0f1115] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-3">
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
          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: colorHex }} />
          <span className="text-xs text-gray-400">{det.health_bar_color ?? "none"}</span>
        </div>
      </div>
      {det.context_b64 && (
        <img className="w-full rounded-md border border-[#2a2d35]"
          src={`data:image/jpeg;base64,${det.context_b64}`} alt="context" />
      )}
      <div className="flex gap-4 items-start">
        <div className="flex flex-col gap-1 flex-1 min-w-0">
          <ColorBar label="G" value={det.color_fractions.green} color="#22c55e" />
          <ColorBar label="B" value={det.color_fractions.blue} color="#3b82f6" />
          <ColorBar label="R" value={det.color_fractions.red} color="#ef4444" />
        </div>
        {det.crop_b64 && (
          <div className="flex flex-col gap-1 shrink-0">
            <span className="text-[10px] text-gray-500">Health bar crop</span>
            <img className="h-5 w-auto rounded border border-[#2a2d35]"
              src={`data:image/jpeg;base64,${det.crop_b64}`} alt="Health bar"
              style={{ imageRendering: "pixelated" }} />
          </div>
        )}
        <span className="text-[10px] text-gray-600 font-mono shrink-0 self-end">
          [{det.bbox.map(Math.round).join(", ")}]
        </span>
      </div>
    </div>
  );
}

// ── Champion ID Debug Tab ───────────────────────────────────────────

function ChampionIdDebugTab({ videos }: { videos: VideoListItem[] }) {
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [frameIndex, setFrameIndex] = useState(0);
  const [maxFrame, setMaxFrame] = useState(0);
  const [debugData, setDebugData] = useState<ChampionIdDebugResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDebug = useCallback(async (videoId: string, frame: number) => {
    if (!videoId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getChampionIdDebug(videoId, frame);
      setDebugData(data);
      setMaxFrame(data.max_frame_index);
    } catch (e: unknown) {
      setError((e as { message?: string })?.message ?? "Failed to fetch");
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
      <p className="text-[13px] text-gray-500 -mt-1">
        Inspect HUD portrait identification and in-game ally/enemy detection matching.
      </p>
      <VideoSelector videos={videos} selected={selectedVideoId} onSelect={handleVideoSelect} />
      {selectedVideoId && (
        <FrameStepper
          frameIndex={frameIndex} maxFrame={maxFrame}
          timestampMs={debugData?.timestamp_ms} loading={loading}
          onChange={handleFrameChange}
        />
      )}
      {loading && <p className="text-sm text-gray-500">Analyzing frame...</p>}
      {error && <p className="text-sm text-red-500">{error}</p>}

      {debugData && (
        <>
          {/* Portrait identification results */}
          <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-5 flex flex-col gap-4">
            <label className="text-sm font-semibold text-white tracking-wide">HUD Portrait Identification</label>

            {/* Portrait strip: actual game frame with bboxes drawn */}
            {debugData.portrait_strip_b64 && (
              <div className="flex flex-col gap-1">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider">Crop regions on actual frame</span>
                <img
                  className="w-full rounded-md border border-[#2a2d35]"
                  src={`data:image/jpeg;base64,${debugData.portrait_strip_b64}`}
                  alt="Portrait crop regions"
                  style={{ imageRendering: "auto" }}
                />
              </div>
            )}

            {/* Matched results */}
            <div className="flex gap-3 flex-wrap">
              {/* Player portrait */}
              <PortraitCard
                label="Player"
                name={debugData.player.name}
                confidence={debugData.player.confidence}
                cropB64={debugData.player.portrait_crop_b64}
                borderColor="#22d3ee"
              />
              {/* Ally portraits */}
              {debugData.ally_portraits.map((a) => (
                <AllyPortraitCard key={a.slot} ally={a} />
              ))}
            </div>
          </div>

          {/* In-game ally detections + matching */}
          {debugData.ally_detections.length > 0 ? (
            <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-3">
              <label className="text-sm font-semibold text-gray-300">
                {debugData.ally_detections.length} Ally Detection{debugData.ally_detections.length !== 1 ? "s" : ""}
              </label>
              <div className="flex flex-col gap-4">
                {debugData.ally_detections.map((det, i) => (
                  <AllyDetectionCard key={i} det={det} />
                ))}
              </div>
            </div>
          ) : !loading && (
            <p className="text-sm text-gray-500">No ally (freindly_champion) detections on this frame.</p>
          )}

          {/* In-game enemy detections + matching */}
          {debugData.enemy_detections.length > 0 ? (
            <div className="bg-[#1a1d24] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-3">
              <label className="text-sm font-semibold text-gray-300">
                {debugData.enemy_detections.length} Enemy Detection{debugData.enemy_detections.length !== 1 ? "s" : ""}
              </label>
              <div className="flex flex-col gap-4">
                {debugData.enemy_detections.map((det, i) => (
                  <EnemyDetectionCard key={i} det={det} />
                ))}
              </div>
            </div>
          ) : !loading && (
            <p className="text-sm text-gray-500">No enemy (enemy_champion) detections on this frame.</p>
          )}
        </>
      )}
    </div>
  );
}

function PortraitCard({
  label, name, confidence, cropB64, borderColor,
}: { label: string; name: string; confidence: number; cropB64: string | null; borderColor: string }) {
  const confPct = Math.round(confidence * 100);
  return (
    <div className="bg-[#0f1115] border border-[#2a2d35] rounded-lg p-3 flex flex-col items-center gap-1.5 w-[110px]">
      <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: borderColor }}>{label}</span>
      {cropB64 ? (
        <img className="w-16 h-16 rounded-md border-2 object-cover" style={{ borderColor }}
          src={`data:image/jpeg;base64,${cropB64}`} alt={name} />
      ) : (
        <div className="w-16 h-16 rounded-md border-2 border-[#333] bg-[#1a1d24] flex items-center justify-center">
          <span className="text-xs text-gray-600">?</span>
        </div>
      )}
      <span className="text-xs text-white font-semibold leading-tight text-center">{name}</span>
      <span className={`text-[10px] font-medium ${confPct >= 50 ? "text-green-400" : "text-gray-500"}`}>
        {confPct}%
      </span>
    </div>
  );
}

function AllyPortraitCard({ ally }: { ally: ChampionIdAllyPortrait }) {
  const confPct = Math.round(ally.confidence * 100);
  return (
    <div className="bg-[#0f1115] border border-[#2a2d35] rounded-lg p-3 flex flex-col items-center gap-1.5 w-[140px]">
      <span className="text-[10px] font-bold uppercase tracking-widest text-blue-400">Ally {ally.slot + 1}</span>
      <div className="flex gap-2 items-center">
        {/* HUD portrait crop */}
        {ally.portrait_crop_b64 ? (
          <img className="w-11 h-11 rounded-md border border-[#444] object-cover"
            src={`data:image/jpeg;base64,${ally.portrait_crop_b64}`} alt="HUD" />
        ) : (
          <div className="w-11 h-11 rounded-md border border-[#333] bg-[#1a1d24]" />
        )}
        <svg className="w-3 h-3 text-gray-600 shrink-0" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M2 6h8M7 3l3 3-3 3" />
        </svg>
        {/* Matched icon */}
        {ally.icon_b64 ? (
          <img className="w-11 h-11 rounded-md border border-blue-500/40 object-cover"
            src={`data:image/jpeg;base64,${ally.icon_b64}`} alt={ally.name} />
        ) : (
          <div className="w-11 h-11 rounded-md border border-[#333] bg-[#1a1d24] flex items-center justify-center">
            <span className="text-[10px] text-gray-600">?</span>
          </div>
        )}
      </div>
      <span className="text-xs text-white font-semibold leading-tight text-center">{ally.name}</span>
      <span className={`text-[10px] font-medium ${confPct >= 50 ? "text-green-400" : "text-gray-500"}`}>
        {confPct}%
      </span>
    </div>
  );
}

function AllyDetectionCard({ det }: { det: ChampionIdDetection }) {
  const confPct = Math.round(det.confidence * 100);
  const topMatch = det.candidate_scores.length > 0 ? det.candidate_scores[0] : null;
  const topRef = topMatch ? refLabelDisplay(topMatch.best_ref_label) : null;
  return (
    <div className="bg-[#0f1115] border border-[#2a2d35] rounded-lg p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-blue-400">freindly_champion</span>
          <span className="text-xs text-gray-500">{confPct}%</span>
        </div>
        {det.match_result ? (
          <span className="text-xs font-semibold text-green-400 bg-green-500/10 px-2 py-0.5 rounded">
            Match: {det.match_result.name} ({Math.round(det.match_result.score * 100)}%)
          </span>
        ) : (
          <span className="text-xs text-gray-600">No confident match</span>
        )}
      </div>

      {/* Side-by-side: context crop + body crop + best matching sprite */}
      <div className="flex gap-3">
        {det.context_b64 && (
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-gray-500">Context (bbox)</span>
            <img className="w-40 h-40 rounded border border-[#2a2d35] object-cover"
              src={`data:image/jpeg;base64,${det.context_b64}`} alt="context" />
          </div>
        )}
        {det.body_crop_b64 && (
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-gray-500">Body crop (matched region)</span>
            <img className="h-40 w-auto rounded border border-blue-500/30 object-contain"
              src={`data:image/jpeg;base64,${det.body_crop_b64}`} alt="body" />
          </div>
        )}
        {topMatch?.best_ref_b64 && (
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-purple-400">
              Best match: {topMatch.name} — {topRef?.type}{topRef?.skin ? ` (${topRef.skin})` : ""}
            </span>
            <img className="h-40 w-40 rounded border-2 border-purple-500/50 object-contain bg-black"
              src={`data:image/jpeg;base64,${topMatch.best_ref_b64}`} alt="best match sprite"
              title={topMatch.best_ref_label ?? ""} />
            <span className="text-[10px] text-gray-500">Score: {(topMatch.combined * 100).toFixed(1)}%</span>
          </div>
        )}
      </div>

      {/* Candidate match scores */}
      {det.candidate_scores.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider">Match Scores vs References</span>
          <div className="flex flex-col gap-1.5">
            {det.candidate_scores.map((cs) => (
              <CandidateScoreRow key={cs.key} cs={cs} isBest={det.match_result?.key === cs.key} />
            ))}
          </div>
        </div>
      )}

      <span className="text-[10px] text-gray-600 font-mono">
        bbox: [{det.bbox.map(Math.round).join(", ")}]
      </span>
    </div>
  );
}

function EnemyDetectionCard({ det }: { det: ChampionIdDetection }) {
  const confPct = Math.round(det.confidence * 100);
  const topMatch = det.candidate_scores.length > 0 ? det.candidate_scores[0] : null;
  const topRef = topMatch ? refLabelDisplay(topMatch.best_ref_label) : null;
  return (
    <div className="bg-[#0f1115] border border-red-500/20 rounded-lg p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-red-400">enemy_champion</span>
          <span className="text-xs text-gray-500">{confPct}%</span>
        </div>
        {det.match_result ? (
          <span className="text-xs font-semibold text-green-400 bg-green-500/10 px-2 py-0.5 rounded">
            Match: {det.match_result.name} ({Math.round(det.match_result.score * 100)}%)
          </span>
        ) : (
          <span className="text-xs text-gray-600">No confident match</span>
        )}
      </div>

      {/* Side-by-side: context crop + body crop + best matching sprite */}
      <div className="flex gap-3">
        {det.context_b64 && (
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-gray-500">Context (bbox)</span>
            <img className="w-40 h-40 rounded border border-[#2a2d35] object-cover"
              src={`data:image/jpeg;base64,${det.context_b64}`} alt="context" />
          </div>
        )}
        {det.body_crop_b64 && (
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-gray-500">Body crop (matched region)</span>
            <img className="h-40 w-auto rounded border border-red-500/30 object-contain"
              src={`data:image/jpeg;base64,${det.body_crop_b64}`} alt="body" />
          </div>
        )}
        {topMatch?.best_ref_b64 && (
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-purple-400">
              Best match: {topMatch.name} — {topRef?.type}{topRef?.skin ? ` (${topRef.skin})` : ""}
            </span>
            <img className="h-40 w-40 rounded border-2 border-purple-500/50 object-contain bg-black"
              src={`data:image/jpeg;base64,${topMatch.best_ref_b64}`} alt="best match sprite"
              title={topMatch.best_ref_label ?? ""} />
            <span className="text-[10px] text-gray-500">Score: {(topMatch.combined * 100).toFixed(1)}%</span>
          </div>
        )}
      </div>

      {/* Top candidate match scores */}
      {det.candidate_scores.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider">Top 10 Match Scores (all champions)</span>
          <div className="flex flex-col gap-1.5">
            {det.candidate_scores.map((cs) => (
              <CandidateScoreRow key={cs.key} cs={cs} isBest={det.match_result?.key === cs.key} />
            ))}
          </div>
        </div>
      )}

      <span className="text-[10px] text-gray-600 font-mono">
        bbox: [{det.bbox.map(Math.round).join(", ")}]
      </span>
    </div>
  );
}

function refLabelDisplay(label: string | null): { type: string; skin: string | null } {
  if (!label) return { type: "icon", skin: null };
  if (label.endsWith("_icon")) return { type: "icon", skin: null };
  if (label.endsWith("_sprite")) {
    const stem = label.replace("_sprite", "");
    const skinMatch = stem.match(/_skin(\d+)$/);
    if (skinMatch) return { type: "sprite", skin: `skin ${skinMatch[1]}` };
    return { type: "sprite", skin: "default" };
  }
  return { type: label, skin: null };
}

function CandidateScoreRow({ cs, isBest }: { cs: ChampionIdCandidateScore; isBest: boolean }) {
  const ref = refLabelDisplay(cs.best_ref_label);
  return (
    <div className={`flex items-center gap-2 p-1.5 rounded ${isBest ? "bg-green-500/10 border border-green-500/20" : ""}`}>
      {/* Icon */}
      {cs.icon_b64 ? (
        <img className="w-10 h-10 rounded border border-[#444] shrink-0 object-cover"
          src={`data:image/jpeg;base64,${cs.icon_b64}`} alt={cs.name} />
      ) : (
        <div className="w-10 h-10 rounded border border-[#333] bg-[#1a1d24] shrink-0" />
      )}
      {/* Best matching reference (sprite) if different from icon */}
      {cs.best_ref_b64 && cs.best_ref_b64 !== cs.icon_b64 ? (
        <div className="flex flex-col items-center shrink-0">
          <img className="w-10 h-10 rounded border border-purple-500/40 shrink-0 object-contain bg-black"
            src={`data:image/jpeg;base64,${cs.best_ref_b64}`} alt="matched ref"
            title={cs.best_ref_label ?? ""} />
          <span className="text-[8px] text-purple-400 leading-none mt-0.5">
            {ref.skin ?? ref.type}
          </span>
        </div>
      ) : null}
      <div className="flex flex-col shrink-0 w-20">
        <span className={`text-xs ${isBest ? "text-green-400 font-semibold" : "text-gray-300"}`}>
          {cs.name}
        </span>
        <span className="text-[8px] text-gray-600">
          via {ref.type}{ref.skin ? ` (${ref.skin})` : ""}
        </span>
      </div>
      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-gray-600 w-8 shrink-0">Hist</span>
          <ScoreBar score={cs.hist_score} />
          <span className="text-[9px] text-gray-500 w-8 text-right tabular-nums">{(cs.hist_score * 100).toFixed(1)}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-gray-600 w-8 shrink-0">Tmpl</span>
          <ScoreBar score={cs.tmpl_score} />
          <span className="text-[9px] text-gray-500 w-8 text-right tabular-nums">{(cs.tmpl_score * 100).toFixed(1)}</span>
        </div>
      </div>
      <span className={`text-xs font-mono w-10 text-right shrink-0 tabular-nums ${isBest ? "text-green-400" : "text-gray-400"}`}>
        {(cs.combined * 100).toFixed(1)}
      </span>
    </div>
  );
}

// ── Admin Page (Tab Shell) ──────────────────────────────────────────

type AdminTab = "health-bar" | "champion-id";

const TABS: { key: AdminTab; label: string }[] = [
  { key: "health-bar", label: "Health Bar Colors" },
  { key: "champion-id", label: "Champion ID" },
];

export default function AdminPage() {
  const [tab, setTab] = useState<AdminTab>("health-bar");
  const [videos, setVideos] = useState<VideoListItem[]>([]);

  useEffect(() => {
    listVideos({ limit: 100 }).then((data) => setVideos(data.videos)).catch(() => {});
  }, []);

  return (
    <div className="flex flex-col gap-5">
      {/* Tab bar */}
      <div className="flex gap-0 border-b border-[#2a2d35]">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-5 py-2.5 text-sm font-medium transition-colors relative ${
              tab === t.key
                ? "text-white"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            {t.label}
            {tab === t.key && (
              <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[#646cff] rounded-t" />
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "health-bar" && <HealthBarDebugTab videos={videos} />}
      {tab === "champion-id" && <ChampionIdDebugTab videos={videos} />}
    </div>
  );
}
