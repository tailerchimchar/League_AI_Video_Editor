import { useState, useEffect, useCallback, useRef } from "react";
import type { ExtractionStatus as ExtStatus } from "../api/client";

interface Props {
  videoId: string;
  onComplete: () => void;
  onError: (message: string) => void;
}

export default function ExtractionStatus({
  videoId,
  onComplete,
  onError,
}: Props) {
  const [status, setStatus] = useState<ExtStatus | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [playedChampion, setPlayedChampion] = useState("");
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

  const startExtraction = useCallback(async () => {
    setExtracting(true);
    try {
      const config = playedChampion.trim()
        ? { played_champion: playedChampion.trim() }
        : undefined;
      const res = await fetch(`/api/v1/videos/${videoId}/extract`, {
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

      // Start polling
      intervalRef.current = setInterval(async () => {
        try {
          const pollRes = await fetch(
            `/api/v1/videos/${videoId}/extract/status`
          );
          if (!pollRes.ok) return;
          const data: ExtStatus = await pollRes.json();
          setStatus(data);

          if (data.status === "completed") {
            stopPolling();
            setExtracting(false);
            onComplete();
          } else if (data.status === "failed") {
            stopPolling();
            setExtracting(false);
            onError(data.error_message || "Extraction failed");
          }
        } catch {
          // Ignore transient poll errors
        }
      }, 3000);
    } catch {
      onError("Network error starting extraction");
      setExtracting(false);
    }
  }, [videoId, playedChampion, onComplete, onError, stopPolling]);

  const progress = status?.progress ?? 0;
  const frameCount = status?.frame_count ?? 0;

  return (
    <div className="extraction-panel">
      {!extracting && !status && (
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
          <button className="extract-btn" onClick={startExtraction}>
            Extract Game Data
          </button>
        </div>
      )}

      {(extracting || status) && (
        <div className="extraction-status">
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>
          <p className="extraction-info">
            {status?.status === "completed"
              ? `Extraction complete — ${frameCount} frames analyzed`
              : status?.status === "failed"
                ? `Extraction failed: ${status.error_message}`
                : progress === 0
                  ? "Extracting frames & running OCR (this may take a minute)..."
                  : `Inserting data... ${Math.round(progress * 100)}% (${frameCount} frames)`}
          </p>
        </div>
      )}
    </div>
  );
}
