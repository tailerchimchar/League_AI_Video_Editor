import { useState, useRef, useCallback } from "react";

type Status = "idle" | "uploading" | "processing" | "done" | "error";

interface ApiError {
  error_code: string;
  message: string;
  details?: unknown;
}

export default function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<ApiError | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = useCallback(
    async (file: File) => {
      // Reset
      setError(null);
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      setVideoUrl(null);

      // Client-side guard
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!ext || !["mp4", "webm"].includes(ext)) {
        setStatus("error");
        setError({
          error_code: "INVALID_EXTENSION",
          message: "Only .mp4 and .webm files are accepted.",
        });
        return;
      }

      setStatus("uploading");
      const form = new FormData();
      form.append("file", file);

      try {
        const res = await fetch("/api/v1/video", {
          method: "POST",
          body: form,
        });

        if (!res.ok) {
          const body: ApiError = await res.json();
          setStatus("error");
          setError(body);
          return;
        }

        setStatus("processing");
        const blob = await res.blob();
        setVideoUrl(URL.createObjectURL(blob));
        setStatus("done");
      } catch {
        setStatus("error");
        setError({
          error_code: "NETWORK_ERROR",
          message: "Could not reach the server. Is the API running on :8000?",
        });
      }
    },
    [videoUrl],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) upload(file);
    },
    [upload],
  );

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) upload(file);
    },
    [upload],
  );

  return (
    <div className="app">
      <h1>League AI Video Editor</h1>
      <p className="subtitle">Upload a gameplay clip for analysis</p>

      <div
        className={`dropzone${dragOver ? " drag-over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <p className="dropzone-label">
          Drag &amp; drop a video here, or click to browse
        </p>
        <p className="dropzone-hint">.mp4 or .webm &middot; max 60 s &middot; max 50 MB</p>
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.webm,video/mp4,video/webm"
          onChange={onFileChange}
          hidden
        />
      </div>

      {status !== "idle" && (
        <div className={`status-bar ${status}`}>
          {status === "uploading" && "Uploading\u2026"}
          {status === "processing" && "Processing\u2026"}
          {status === "done" && "Done!"}
          {status === "error" && error && (
            <span className="error-msg">
              {error.message}{" "}
              <code className="error-code">{error.error_code}</code>
            </span>
          )}
        </div>
      )}

      {videoUrl && (
        <video className="player" src={videoUrl} controls autoPlay />
      )}
    </div>
  );
}
