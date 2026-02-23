import { useState, useEffect, useRef, useCallback } from "react";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  extraControls?: React.ReactNode;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function VideoControls({ videoRef, extraControls }: Props) {
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const [dragging, setDragging] = useState(false);
  const seekBarRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef(0);

  // Sync state from video element
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onDurationChange = () => setDuration(video.duration || 0);
    const onVolumeChange = () => {
      setVolume(video.volume);
      setMuted(video.muted);
    };

    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("durationchange", onDurationChange);
    video.addEventListener("volumechange", onVolumeChange);

    // Init
    if (video.duration) setDuration(video.duration);
    setVolume(video.volume);
    setMuted(video.muted);

    return () => {
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("durationchange", onDurationChange);
      video.removeEventListener("volumechange", onVolumeChange);
    };
  }, [videoRef]);

  // Update currentTime via rAF for smooth scrubbing
  useEffect(() => {
    function tick() {
      const video = videoRef.current;
      if (video && !dragging) {
        setCurrentTime(video.currentTime);
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [videoRef, dragging]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) video.play();
    else video.pause();
  }, [videoRef]);

  const seek = useCallback(
    (clientX: number) => {
      const bar = seekBarRef.current;
      const video = videoRef.current;
      if (!bar || !video || !duration) return;
      const rect = bar.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      video.currentTime = pct * duration;
      setCurrentTime(pct * duration);
    },
    [videoRef, duration]
  );

  const onSeekDown = useCallback(
    (e: React.MouseEvent) => {
      setDragging(true);
      seek(e.clientX);

      const onMove = (ev: MouseEvent) => seek(ev.clientX);
      const onUp = () => {
        setDragging(false);
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [seek]
  );

  const toggleMute = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !video.muted;
  }, [videoRef]);

  const onVolumeChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const video = videoRef.current;
      if (!video) return;
      const v = parseFloat(e.target.value);
      video.volume = v;
      if (v > 0 && video.muted) video.muted = false;
    },
    [videoRef]
  );

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="video-controls">
      <button className="vc-play" onClick={togglePlay} title={playing ? "Pause" : "Play"}>
        {playing ? (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <rect x="3" y="2" width="4" height="12" rx="1" />
            <rect x="9" y="2" width="4" height="12" rx="1" />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M4 2.5v11l9-5.5z" />
          </svg>
        )}
      </button>

      <span className="vc-time">{formatTime(currentTime)}</span>

      <div className="vc-seek" ref={seekBarRef} onMouseDown={onSeekDown}>
        <div className="vc-seek-track">
          <div className="vc-seek-fill" style={{ width: `${progress}%` }} />
          <div className="vc-seek-thumb" style={{ left: `${progress}%` }} />
        </div>
      </div>

      <span className="vc-time">{formatTime(duration)}</span>

      <button className="vc-mute" onClick={toggleMute} title={muted ? "Unmute" : "Mute"}>
        {muted || volume === 0 ? (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 2L4.5 5.5H2v5h2.5L8 14V2z" />
            <path d="M11 5.5l4 5M15 5.5l-4 5" stroke="currentColor" strokeWidth="1.5" fill="none" />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 2L4.5 5.5H2v5h2.5L8 14V2z" />
            <path d="M11 5.5c.8.8 1.2 1.8 1.2 2.5s-.4 1.7-1.2 2.5" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
          </svg>
        )}
      </button>

      <input
        className="vc-volume"
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={muted ? 0 : volume}
        onChange={onVolumeChange}
      />

      {extraControls}
    </div>
  );
}
