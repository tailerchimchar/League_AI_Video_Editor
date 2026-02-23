import { useState, useEffect } from "react";
import type { FramePayload } from "../api/client";

interface Props {
  videoId: string;
}

function msToDisplay(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${sec.toString().padStart(2, "0")}`;
}

export default function FrameTimeline({ videoId }: Props) {
  const [frames, setFrames] = useState<FramePayload[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/api/v1/videos/${videoId}/frames?limit=50`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setFrames(data.frames || []);
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

  if (loading) return <p className="loading-text">Loading frame data...</p>;
  if (frames.length === 0) return null;

  return (
    <div className="frame-timeline-panel">
      <h2 className="panel-title">Extracted Data Timeline</h2>
      <div className="frame-timeline">
        <table className="frame-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Timer</th>
              <th>CS</th>
              <th>Gold</th>
              <th>KDA</th>
              <th>Level</th>
              <th>HP Delta</th>
            </tr>
          </thead>
          <tbody>
            {frames.map((f) => {
              const ocr = f.ocr_data as Record<string, unknown>;
              const feat = f.derived_features as Record<string, unknown>;
              const kda = ocr.player_kda as
                | { kills: number; deaths: number; assists: number }
                | undefined;
              const hpDelta = feat.hp_delta as number | undefined;

              return (
                <tr key={f.frame_index}>
                  <td>{msToDisplay(f.timestamp_ms)}</td>
                  <td>{(ocr.game_timer as string) || "—"}</td>
                  <td>{ocr.player_cs != null ? String(ocr.player_cs) : "—"}</td>
                  <td>{ocr.player_gold != null ? String(ocr.player_gold) : "—"}</td>
                  <td>
                    {kda
                      ? `${kda.kills}/${kda.deaths}/${kda.assists}`
                      : "—"}
                  </td>
                  <td>
                    {ocr.player_level != null
                      ? String(ocr.player_level)
                      : "—"}
                  </td>
                  <td
                    className={
                      hpDelta != null && hpDelta < -0.05
                        ? "hp-loss"
                        : hpDelta != null && hpDelta > 0.05
                          ? "hp-gain"
                          : ""
                    }
                  >
                    {hpDelta != null ? `${(hpDelta * 100).toFixed(1)}%` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
