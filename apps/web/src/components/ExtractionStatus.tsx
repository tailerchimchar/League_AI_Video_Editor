import { useState, useEffect, useCallback, useRef } from "react";
import type { ExtractionStatus as ExtStatus } from "../api/client";

interface Props {
  videoId: string;
  force?: boolean;
  onComplete: () => void;
  onError: (message: string) => void;
}

export default function ExtractionStatus({
  videoId,
  force = false,
  onComplete,
  onError,
}: Props) {
  const [status, setStatus] = useState<ExtStatus | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [playedChampion, setPlayedChampion] = useState("");
  const [checked, setChecked] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  // Check for existing completed extraction on mount (skip if force re-extract)
  useEffect(() => {
    if (force) {
      setChecked(true);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/api/v1/videos/${videoId}/extract/status`);
        if (!res.ok) {
          setChecked(true);
          return;
        }
        const data: ExtStatus = await res.json();
        if (cancelled) return;
        if (data.status === "completed") {
          setStatus(data);
          onComplete();
        } else if (data.status === "running" || data.status === "pending") {
          // Extraction in progress — start polling
          setStatus(data);
          setExtracting(true);
          intervalRef.current = setInterval(async () => {
            try {
              const pollRes = await fetch(`/api/v1/videos/${videoId}/extract/status`);
              if (!pollRes.ok) return;
              const pollData: ExtStatus = await pollRes.json();
              setStatus(pollData);
              if (pollData.status === "completed") {
                stopPolling();
                setExtracting(false);
                onComplete();
              } else if (pollData.status === "failed") {
                stopPolling();
                setExtracting(false);
                onError(pollData.error_message || "Extraction failed");
              }
            } catch { /* ignore */ }
          }, 1500);
        }
      } catch { /* no existing job */ }
      if (!cancelled) setChecked(true);
    })();
    return () => { cancelled = true; };
  }, [videoId, onComplete, onError, stopPolling]);

  const startExtraction = useCallback(async (force = false) => {
    setExtracting(true);
    setStatus(null);
    try {
      const config = playedChampion.trim()
        ? { played_champion: playedChampion.trim() }
        : undefined;
      const url = `/api/v1/videos/${videoId}/extract${force ? "?force=true" : ""}`;
      const res = await fetch(url, {
        method: "POST",
        ...(config
          ? {
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(config),
            }
          : {}),
      });
      if (!res.ok) {
        const body = await res.json();
        onError(body.message || "Failed to start extraction");
        setExtracting(false);
        return;
      }

      const data = await res.json();
      // If backend returned cached completed result, skip straight to done
      if (data.cached && data.status === "completed") {
        setStatus(data as ExtStatus);
        setExtracting(false);
        onComplete();
        return;
      }

      // Start polling
      intervalRef.current = setInterval(async () => {
        try {
          const pollRes = await fetch(
            `/api/v1/videos/${videoId}/extract/status`
          );
          if (!pollRes.ok) return;
          const pollData: ExtStatus = await pollRes.json();
          setStatus(pollData);

          if (pollData.status === "completed") {
            stopPolling();
            setExtracting(false);
            onComplete();
          } else if (pollData.status === "failed") {
            stopPolling();
            setExtracting(false);
            onError(pollData.error_message || "Extraction failed");
          }
        } catch {
          // Ignore transient poll errors
        }
      }, 1500);
    } catch {
      onError("Network error starting extraction");
      setExtracting(false);
    }
  }, [videoId, playedChampion, onComplete, onError, stopPolling]);

  const progress = status?.progress ?? 0;
  const frameCount = status?.frame_count ?? 0;
  const extractedFrames = status?.extracted_frames;
  const totalFrames = status?.total_frames;
  const phase = status?.phase;

  // Compute a combined progress: extraction phase = 0-80%, insert phase = 80-100%
  let combinedProgress = progress; // DB progress (0-1) used for insert phase
  if (phase === "extracting" && totalFrames && totalFrames > 0) {
    combinedProgress = ((extractedFrames ?? 0) / totalFrames) * 0.8;
  } else if (phase === "inserting") {
    combinedProgress = 0.8 + progress * 0.2;
  }

  function getStatusText(): string {
    if (status?.status === "completed") {
      return `Extraction complete — ${frameCount} frames analyzed`;
    }
    if (status?.status === "failed") {
      return `Extraction failed: ${status.error_message}`;
    }
    if (phase === "extracting" && totalFrames) {
      return `Extracting ${extractedFrames ?? 0}/${totalFrames} frames...`;
    }
    if (phase === "inserting") {
      return `Saving to database... ${Math.round(progress * 100)}%`;
    }
    if (progress > 0) {
      return `Inserting data... ${Math.round(progress * 100)}% (${frameCount} frames)`;
    }
    return "Starting extraction...";
  }

  return (
    <div className="extraction-panel">
      {!extracting && !status && checked && (
        <div className="extraction-form">
          <div className="champion-input-row">
            <input
              type="text"
              className="champion-input"
              placeholder="Your champion (e.g. Caitlyn)"
              value={playedChampion}
              onChange={(e) => setPlayedChampion(e.target.value)}
            />
          </div>
          <button className="extract-btn" onClick={() => startExtraction(force)}>
            {force ? "Re-extract Game Data" : "Extract Game Data"}
          </button>
        </div>
      )}

      {(extracting || status) && (
        <div className="extraction-status">
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill"
              style={{ width: `${Math.round(Math.min(combinedProgress, 1) * 100)}%` }}
            />
          </div>
          <p className="extraction-info">{getStatusText()}</p>
        </div>
      )}
    </div>
  );
}
