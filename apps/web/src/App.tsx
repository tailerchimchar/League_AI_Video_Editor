import { useState, useRef, useCallback, useEffect } from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import Markdown from "react-markdown";
import VideoUploader from "./components/VideoUploader";
import ExtractionStatus from "./components/ExtractionStatus";
import FrameTimeline from "./components/FrameTimeline";
import SegmentList from "./components/SegmentList";
import ReportView from "./components/ReportView";
import AIVisionOverlay, { useAIVision, type DetectionFilters } from "./components/AIVisionOverlay";
import VideoControls from "./components/VideoControls";
import AdminPage from "./components/AdminPage";

type Mode = "pipeline" | "legacy";
type Status =
  | "idle"
  | "uploading"
  | "uploaded"
  | "extracting"
  | "extracted"
  | "analyzing"
  | "analyzed"
  | "error";

interface ApiError {
  error_code: string;
  message: string;
}

const FILTER_CHIP_META: { key: keyof DetectionFilters; label: string; color: string }[] = [
  { key: "player",  label: "Player",  color: "#22d3ee" },
  { key: "allies",  label: "Allies",  color: "#3b82f6" },
  { key: "enemies", label: "Enemies", color: "#ef4444" },
  { key: "minions", label: "Minions", color: "#f59e0b" },
];

function EditorPage() {
  const [mode, setMode] = useState<Mode>("pipeline");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<ApiError | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [forceExtract, setForceExtract] = useState(false);
  const [videoMeta, setVideoMeta] = useState<{ width: number; height: number } | null>(null);
  const videoElRef = useRef<HTMLVideoElement>(null);
  const aiVision = useAIVision(videoId ?? "");

  // Filter state (ref for rAF loop, state for chip UI re-renders)
  const filtersRef = useRef<DetectionFilters>({ player: true, allies: true, enemies: true, minions: true });
  const [filterState, setFilterState] = useState<DetectionFilters>({ player: true, allies: true, enemies: true, minions: true });

  const toggleFilter = useCallback((key: keyof DetectionFilters) => {
    setFilterState((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      filtersRef.current = next;
      return next;
    });
  }, []);

  // Legacy mode state
  const [legacyAnalysis, setLegacyAnalysis] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const closeEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => closeEventSource();
  }, [closeEventSource]);

  // ── Pipeline mode handlers ─────────────────────────────────
  const handlePipelineUpload = useCallback((id: string, _filename: string, meta: { width: number | null; height: number | null; status?: string }) => {
    setVideoId(id);
    setVideoMeta(meta.width && meta.height ? { width: meta.width, height: meta.height } : null);
    // If the video was already extracted (e.g., deduplicated upload), skip to extracted state
    setStatus(meta.status === "extracted" ? "extracted" : "uploaded");
    setError(null);
  }, []);

  const handlePipelineError = useCallback((err: ApiError) => {
    setError(err);
    setStatus("error");
  }, []);

  const handleExtractionComplete = useCallback(() => {
    setStatus("extracted");
    aiVision.invalidateFrames();
  }, [aiVision.invalidateFrames]);

  const handleExtractionError = useCallback((message: string) => {
    setError({ error_code: "EXTRACTION_ERROR", message });
    setStatus("error");
  }, []);

  // ── Legacy mode handlers ───────────────────────────────────
  const legacyUpload = useCallback(
    async (file: File) => {
      closeEventSource();
      setError(null);
      setVideoId(null);
      setLegacyAnalysis("");

      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!ext || !["mp4", "webm"].includes(ext)) {
        setStatus("error");
        setError({ error_code: "INVALID_EXTENSION", message: "Only .mp4 and .webm files are accepted." });
        return;
      }

      setStatus("uploading");
      const form = new FormData();
      form.append("file", file);

      try {
        const res = await fetch("/api/v1/video", { method: "POST", body: form });
        if (!res.ok) {
          const body: ApiError = await res.json();
          setStatus("error");
          setError(body);
          return;
        }
        const data = await res.json();
        setVideoId(data.video_id);
        setStatus("uploaded");
      } catch {
        setStatus("error");
        setError({ error_code: "NETWORK_ERROR", message: "Could not reach the server." });
      }
    },
    [closeEventSource]
  );

  const legacyAnalyze = useCallback(() => {
    if (!videoId) return;
    closeEventSource();
    setLegacyAnalysis("");
    setError(null);
    setStatus("analyzing");

    const es = new EventSource(`/api/v1/analyze/${videoId}`);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "text") {
        setLegacyAnalysis((prev) => prev + data.text);
      } else if (data.type === "done") {
        setStatus("analyzed");
        es.close();
        eventSourceRef.current = null;
      } else if (data.type === "error") {
        setStatus("error");
        setError({ error_code: "ANALYSIS_ERROR", message: data.message });
        es.close();
        eventSourceRef.current = null;
      }
    };

    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) return;
      setStatus("error");
      setError({ error_code: "SSE_ERROR", message: "Lost connection to the analysis stream." });
      es.close();
      eventSourceRef.current = null;
    };
  }, [videoId, closeEventSource]);

  const reset = useCallback(() => {
    closeEventSource();
    setStatus("idle");
    setError(null);
    setVideoId(null);
    setVideoMeta(null);
    setLegacyAnalysis("");
  }, [closeEventSource]);

  // ── Video URL ──────────────────────────────────────────────
  const videoUrl = videoId
    ? mode === "pipeline"
      ? `/api/v1/videos/${videoId}`
      : `/api/v1/video/${videoId}`
    : null;

  return (
    <>
      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === "pipeline" ? "active" : ""}`}
          onClick={() => { reset(); setMode("pipeline"); }}
        >
          Extraction Pipeline
        </button>
        <button
          className={`mode-btn ${mode === "legacy" ? "active" : ""}`}
          onClick={() => { reset(); setMode("legacy"); }}
        >
          Quick Analysis
        </button>
      </div>

      <p className="subtitle">
        {mode === "pipeline"
          ? "Upload a gameplay clip for structured data extraction + evidence-grounded coaching"
          : "Upload a gameplay clip for quick frame-based analysis"}
      </p>

      {/* ── Upload ── */}
      {status === "idle" && mode === "pipeline" && (
        <VideoUploader
          onUploadComplete={handlePipelineUpload}
          onError={handlePipelineError}
        />
      )}

      {status === "idle" && mode === "legacy" && (
        <div className="dropzone" onClick={() => inputRef.current?.click()}>
          <p className="dropzone-label">Drag &amp; drop a video here, or click to browse</p>
          <p className="dropzone-hint">.mp4 or .webm &middot; max 60 s &middot; max 50 MB</p>
          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.webm"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) legacyUpload(file);
            }}
            hidden
          />
        </div>
      )}

      {/* ── Error display ── */}
      {status === "error" && error && (
        <div className="status-bar error">
          <span className="error-msg">
            {error.message} <code className="error-code">{error.error_code}</code>
          </span>
          <button className="reset-btn" onClick={reset}>Try Again</button>
        </div>
      )}

      {/* ── Video player ── */}
      {videoUrl && (
        <div className="player-wrapper">
          <div className="player-container">
            <video ref={videoElRef} className="player" src={videoUrl} autoPlay />
            {mode === "pipeline" && videoId && (
              <AIVisionOverlay
                videoRef={videoElRef}
                videoId={videoId}
                nativeWidth={videoMeta?.width ?? 1920}
                nativeHeight={videoMeta?.height ?? 1080}
                enabled={aiVision.enabled}
                frameVersion={aiVision.frameVersion}
                filtersRef={filtersRef}
              />
            )}
          </div>
          <VideoControls
            videoRef={videoElRef}
            extraControls={
              mode === "pipeline" ? (
                <>
                  {aiVision.enabled && (
                    <div className="filter-chips">
                      {FILTER_CHIP_META.map(({ key, label, color }) => (
                        <button
                          key={key}
                          className={`filter-chip${filterState[key] ? " active" : ""}`}
                          style={{
                            "--chip-color": color,
                          } as React.CSSProperties}
                          onClick={() => toggleFilter(key)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  )}
                  <button
                    className={`ai-vision-toggle${aiVision.enabled ? " active" : ""}`}
                    onClick={aiVision.toggle}
                    disabled={aiVision.loading}
                  >
                    {aiVision.loading ? "Loading..." : aiVision.enabled ? "Hide AI Vision" : "Show AI Vision"}
                  </button>
                </>
              ) : undefined
            }
          />
        </div>
      )}

      {/* ── Pipeline mode: extraction → data → report ── */}
      {mode === "pipeline" && videoId && status !== "error" && (
        <>
          {(status === "uploaded" || status === "extracting") && (
            <ExtractionStatus
              videoId={videoId}
              force={forceExtract}
              onComplete={() => { setForceExtract(false); handleExtractionComplete(); }}
              onError={handleExtractionError}
            />
          )}

          {status === "extracted" && (
            <>
              <div className="reextract-row">
                <button
                  className="reextract-btn"
                  onClick={() => { setForceExtract(true); setStatus("uploaded"); }}
                >
                  Re-extract
                </button>
              </div>
              <FrameTimeline videoId={videoId} />
              <SegmentList videoId={videoId} />
              <ReportView videoId={videoId} />
            </>
          )}
        </>
      )}

      {/* ── Legacy mode: direct analysis ── */}
      {mode === "legacy" && videoId && status !== "error" && (
        <>
          {(status === "uploaded" || status === "analyzed") && (
            <button className="analyze-btn" onClick={legacyAnalyze}>
              {status === "analyzed" ? "Re-Analyze" : "Analyze Gameplay"}
            </button>
          )}

          {(status === "analyzing" || status === "analyzed" || legacyAnalysis) && (
            <div className="analysis-panel">
              <h2 className="analysis-title">Coaching Feedback</h2>
              <div className="analysis-text">
                {legacyAnalysis ? (
                  <Markdown>{legacyAnalysis}</Markdown>
                ) : (
                  "Extracting frames and analyzing\u2026"
                )}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── New upload button when not idle ── */}
      {status !== "idle" && (
        <button className="reset-btn secondary" onClick={reset}>
          Upload New Video
        </button>
      )}
    </>
  );
}

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>League AI Video Editor</h1>
        <nav className="nav-links">
          <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
            Editor
          </NavLink>
          <NavLink to="/admin" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
            Admin Tools
          </NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<EditorPage />} />
        <Route path="/admin" element={<AdminPage />} />
      </Routes>
    </div>
  );
}
