import { useState, useEffect } from "react";
import type { Segment } from "../api/client";

interface Props {
  videoId: string;
}

function msToDisplay(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${sec.toString().padStart(2, "0")}`;
}

const TYPE_COLORS: Record<string, string> = {
  fight: "#e74c3c",
  trade: "#e67e22",
  death: "#c0392b",
  lane: "#2ecc71",
  roam: "#3498db",
  objective: "#9b59b6",
  recall: "#95a5a6",
};

export default function SegmentList({ videoId }: Props) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/api/v1/videos/${videoId}/segments`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setSegments(data.segments || []);
      } catch {
        // ignore
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [videoId]);

  if (loading) return <p className="loading-text">Loading segments...</p>;
  if (segments.length === 0) return null;

  return (
    <div className="segments-panel">
      <h2 className="panel-title">Detected Segments</h2>
      <div className="segment-list">
        {segments.map((s) => (
          <div key={s.id} className="segment-item">
            <span
              className="segment-badge"
              style={{ backgroundColor: TYPE_COLORS[s.segment_type] || "#666" }}
            >
              {s.segment_type.toUpperCase()}
            </span>
            <span className="segment-time">
              {msToDisplay(s.start_ms)} → {msToDisplay(s.end_ms)}
            </span>
            <span className="segment-confidence">
              {Math.round(s.confidence * 100)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
