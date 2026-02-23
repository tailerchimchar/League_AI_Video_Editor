import { useRef, useState, useEffect, useCallback } from "react";
import { getFrames, type FramePayload } from "../api/client";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  videoId: string;
  nativeWidth: number;
  nativeHeight: number;
  enabled: boolean;
  /** Bump this number to trigger a re-fetch of frame data (e.g., after extraction completes). */
  frameVersion: number;
}

/** Crop regions at 1920x1080 base — must match apps/api/extractor/config.py */
const CROP_REGIONS = [
  { key: "game_timer",   x: 1830, y: 2,    w: 85,  h: 28,  color: "#f1c40f", label: "Timer",   field: "game_timer" },
  { key: "scoreboard",   x: 1525, y: 2,    w: 118, h: 28,  color: "#e67e22", label: "Score",   field: "scoreboard" },
  { key: "kda",          x: 1650, y: 2,    w: 90,  h: 28,  color: "#e74c3c", label: "KDA",     field: "kda" },
  { key: "cs_gold",      x: 1780, y: 4,    w: 45,  h: 22,  color: "#2ecc71", label: "CS",      field: "cs_gold" },
  { key: "minimap",      x: 1630, y: 805,  w: 280, h: 275, color: "#3498db", label: "Minimap", field: "minimap" },
  { key: "player_stats", x: 480,  y: 970,  w: 138, h: 108, color: "#9b59b6", label: "Stats",   field: "player_stats" },
  { key: "player_level", x: 688,  y: 1043, w: 28,  h: 30,  color: "#1abc9c", label: "Level",   field: "player_level" },
  { key: "items",        x: 1085,  y: 960,  w: 170, h: 115,  color: "#e84393", label: "Items",   field: "items" },
  { key: "ability_bar",  x: 730,  y: 970,  w: 250, h: 60,  color: "#00cec9", label: "Abilities", field: "ability_bar" },
  { key: "health_bar",   x: 860,  y: 1032, w: 100, h: 20,  color: "#ff6b6b", label: "HP",      field: "health_bar" },
  { key: "resources_bar",   x: 860,  y: 1050, w: 100, h: 20,  color: "#123456", label: "mana",      field: "resources_bar" },
  { key: "summoner_spells", x: 990, y: 970, w: 90, h: 50, color: "#ffff00", label: "SummonersSpells", field: "summoner_spells" },
] as const;

const BASE_W = 1920;
const BASE_H = 1080;

/** Palette for assigning consistent colors to tracked objects via track_id. */
const TRACK_COLORS = [
  "#22d3ee", "#3b82f6", "#ef4444", "#f59e0b", "#10b981",
  "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#6366f1",
];

/** Color mapping for YOLO detection classes, with optional track_id consistency. */
function getDetectionColor(className: string, trackId?: number): string {
  // Class-based colors take priority for key categories
  if (className === "played_champion") return "#22d3ee"; // cyan for you
  if (className.startsWith("enemy_")) return "#ef4444";  // red
  // Dataset has both "freindly_" (sic) and "friendly_" spellings
  if (className.startsWith("freindly_") || className.startsWith("friendly_")) return "#3b82f6";
  // Use track_id for consistent coloring of other/unknown entities
  if (trackId != null) return TRACK_COLORS[trackId % TRACK_COLORS.length];
  return "#a3a3a3"; // neutral gray
}

/** Human-readable label for detection display. */
function getDisplayLabel(det: { class_name: string; champion?: string | null; champion_confidence?: number | null; confidence: number }): string {
  const confPct = Math.round(det.confidence * 100);
  const cn = det.class_name;

  // Champions
  if (cn === "played_champion") {
    if (det.champion && det.champion_confidence && det.champion_confidence >= 1.0) {
      // User-specified name — definitive
      return `You: ${det.champion} ${confPct}%`;
    }
    if (det.champion) {
      // Auto-detected name — show as a guess
      return `You (${det.champion}?) ${confPct}%`;
    }
    return `You ${confPct}%`;
  }
  if (cn === "freindly_champion") {
    return `Ally ${confPct}%`;
  }
  if (cn === "enemy_champion") {
    return `Enemy ${confPct}%`;
  }

  // Structures: clean up
  if (cn.includes("tower")) return `${cn.startsWith("enemy") ? "Enemy" : "Ally"} Tower ${confPct}%`;
  if (cn.includes("inhibitor")) return `${cn.startsWith("enemy") ? "Enemy" : "Ally"} Inhib ${confPct}%`;
  if (cn.includes("nexus")) return `${cn.startsWith("enemy") ? "Enemy" : "Ally"} Nexus ${confPct}%`;

  // Minions: short labels
  const side = cn.startsWith("enemy") ? "E" : "A";
  if (cn.includes("cannon")) return `${side} Cannon ${confPct}%`;
  if (cn.includes("super")) return `${side} Super ${confPct}%`;
  if (cn.includes("melee")) return `${side} Melee ${confPct}%`;
  if (cn.includes("ranged")) return `${side} Ranged ${confPct}%`;

  return `${cn} ${confPct}%`;
}

/** Binary search for closest frame by timestamp. */
function findClosestFrame(frames: FramePayload[], timeMs: number): FramePayload | null {
  if (frames.length === 0) return null;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].timestamp_ms < timeMs) lo = mid + 1;
    else hi = mid;
  }
  // lo is the first frame >= timeMs; check if lo-1 is closer
  if (lo > 0) {
    const diffLo = Math.abs(frames[lo].timestamp_ms - timeMs);
    const diffPrev = Math.abs(frames[lo - 1].timestamp_ms - timeMs);
    if (diffPrev < diffLo) lo = lo - 1;
  }
  return frames[lo];
}

function formatOcrValue(field: string, ocr: Record<string, unknown>): string {
  // OCR data has a "raw" sub-object with the direct text per region,
  // plus top-level parsed fields. Use raw for display labels.
  const raw = ocr.raw as Record<string, unknown> | undefined;
  const val = raw?.[field];
  if (val === undefined || val === null || val === "") return "—";
  return String(val);
}

export default function AIVisionOverlay({ videoRef, videoId, nativeWidth, nativeHeight, enabled, frameVersion }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const framesRef = useRef<FramePayload[] | null>(null);
  const rafRef = useRef<number>(0);
  const loadedVersionRef = useRef<number>(0);

  // Keep a ref to CROP_REGIONS so the animation loop always reads the latest
  // values after Vite HMR hot-swaps this module.
  const regionsRef = useRef(CROP_REGIONS);
  regionsRef.current = CROP_REGIONS;

  // Load frames when enabled, or re-fetch when frameVersion bumps
  useEffect(() => {
    if (!enabled) return;
    // Skip if already loaded for this version
    if (framesRef.current && loadedVersionRef.current === frameVersion) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await getFrames(videoId, { limit: 9999 });
        if (!cancelled) {
          framesRef.current = data.frames;
          loadedVersionRef.current = frameVersion;
        }
      } catch {
        // silently fail
      }
    })();
    return () => { cancelled = true; };
  }, [enabled, videoId, frameVersion]);

  useEffect(() => {
    if (!enabled) {
      // Clear canvas when disabled
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      cancelAnimationFrame(rafRef.current);
      return;
    }

    function draw() {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const dpr = window.devicePixelRatio || 1;
      const elemW = video.clientWidth;
      const elemH = video.clientHeight;

      // Resize canvas to match video element size (HiDPI-aware)
      if (canvas.width !== elemW * dpr || canvas.height !== elemH * dpr) {
        canvas.width = elemW * dpr;
        canvas.height = elemH * dpr;
        canvas.style.width = `${elemW}px`;
        canvas.style.height = `${elemH}px`;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, elemW, elemH);

      // Compute actual video rendering rect within the element.
      // The <video> uses object-fit:contain by default, so the video
      // is letterboxed inside the element (controls bar eats into height).
      const intrinsicW = video.videoWidth || BASE_W;
      const intrinsicH = video.videoHeight || BASE_H;
      const videoAspect = intrinsicW / intrinsicH;
      const elemAspect = elemW / elemH;

      let renderW: number, renderH: number, offsetX: number, offsetY: number;
      if (elemAspect > videoAspect) {
        // Element wider than video — pillarboxed (black bars on sides)
        renderH = elemH;
        renderW = renderH * videoAspect;
        offsetX = (elemW - renderW) / 2;
        offsetY = 0;
      } else {
        // Element taller than video — letterboxed (black bars top/bottom or controls)
        renderW = elemW;
        renderH = renderW / videoAspect;
        offsetX = 0;
        offsetY = (elemH - renderH) / 2;
      }

      // Scale factors from 1920x1080 base to actual rendered video area
      const scaleX = renderW / BASE_W;
      const scaleY = renderH / BASE_H;

      // Find closest frame for OCR labels
      const currentTimeMs = video.currentTime * 1000;
      const frames = framesRef.current;
      const closestFrame = frames ? findClosestFrame(frames, currentTimeMs) : null;

      for (const region of regionsRef.current) {
        const rx = offsetX + region.x * scaleX;
        const ry = offsetY + region.y * scaleY;
        const rw = region.w * scaleX;
        const rh = region.h * scaleY;

        // Semi-transparent fill
        ctx.fillStyle = region.color + "18";
        ctx.fillRect(rx, ry, rw, rh);

        // Colored border
        ctx.strokeStyle = region.color;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(rx, ry, rw, rh);

        // Label pill above box
        const ocrValue = closestFrame
          ? formatOcrValue(region.field, closestFrame.ocr_data)
          : "—";
        const labelText = `${region.label}: ${ocrValue}`;
        const fontSize = Math.max(10, Math.round(12 * scaleX));
        ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
        const measured = ctx.measureText(labelText);
        const pillW = measured.width + 8;
        const pillH = fontSize + 6;
        const pillX = rx;
        const pillY = ry - pillH - 2;

        // Pill background
        ctx.fillStyle = region.color + "cc";
        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, 3);
        ctx.fill();

        // Pill text
        ctx.fillStyle = "#fff";
        ctx.fillText(labelText, pillX + 4, pillY + fontSize);
      }

      // --- Draw YOLO detection boxes ---
      const detections = closestFrame?.detections;
      if (detections && Array.isArray(detections) && detections.length > 0) {
        // Scale from native video resolution to rendered area
        const nativeW = video.videoWidth || BASE_W;
        const nativeH = video.videoHeight || BASE_H;
        const detScaleX = renderW / nativeW;
        const detScaleY = renderH / nativeH;

        for (const det of detections) {
          if (!det.bbox || det.bbox.length < 4) continue;
          const [bx1, by1, bx2, by2] = det.bbox;
          const dx = offsetX + bx1 * detScaleX;
          const dy = offsetY + by1 * detScaleY;
          const dw = (bx2 - bx1) * detScaleX;
          const dh = (by2 - by1) * detScaleY;

          const color = getDetectionColor(det.class_name, det.track_id);

          // Dashed bounding box
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 3]);
          ctx.strokeRect(dx, dy, dw, dh);
          ctx.setLineDash([]);

          // Semi-transparent fill
          ctx.fillStyle = color + "15";
          ctx.fillRect(dx, dy, dw, dh);

          // Label pill — human-readable labels
          const detLabel = getDisplayLabel(det);
          const detFontSize = Math.max(9, Math.round(11 * detScaleX));
          ctx.font = `600 ${detFontSize}px Inter, system-ui, sans-serif`;
          const detMeasured = ctx.measureText(detLabel);
          const detPillW = detMeasured.width + 8;
          const detPillH = detFontSize + 4;
          const detPillX = dx;
          const detPillY = dy - detPillH - 1;

          ctx.fillStyle = color + "dd";
          ctx.beginPath();
          ctx.roundRect(detPillX, detPillY, detPillW, detPillH, 3);
          ctx.fill();

          ctx.fillStyle = "#fff";
          ctx.fillText(detLabel, detPillX + 4, detPillY + detFontSize);
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [enabled, videoRef, nativeWidth, nativeHeight]);

  return (
    <canvas
      ref={canvasRef}
      className="ai-vision-canvas"
      style={{ display: enabled ? "block" : "none" }}
    />
  );
}

/** Hook to control AI Vision overlay state from outside the component. */
export function useAIVision(videoId: string) {
  const [enabled, setEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [frameVersion, setFrameVersion] = useState(0);
  const framesLoadedRef = useRef(false);

  const loadFrames = useCallback(async () => {
    if (framesLoadedRef.current) return;
    setLoading(true);
    try {
      await getFrames(videoId, { limit: 9999 });
      framesLoadedRef.current = true;
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [videoId]);

  const toggle = useCallback(() => {
    setEnabled((prev) => {
      const next = !prev;
      if (next) loadFrames();
      return next;
    });
  }, [loadFrames]);

  /** Call this when new frame data is available (e.g., extraction completed). */
  const invalidateFrames = useCallback(() => {
    framesLoadedRef.current = false;
    setFrameVersion((v) => v + 1);
  }, []);

  return { enabled, loading, toggle, frameVersion, invalidateFrames };
}
