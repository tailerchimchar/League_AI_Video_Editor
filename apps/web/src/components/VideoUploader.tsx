import { useRef, useCallback, useState } from "react";

/** Direct URL to FastAPI for uploads (Vite proxy can't handle large files). */
const UPLOAD_URL = import.meta.env.DEV
  ? "http://localhost:8000/api/v1/videos"
  : "/api/v1/videos";

interface Props {
  onUploadComplete: (videoId: string, filename: string, meta: { width: number | null; height: number | null; status?: string }) => void;
  onError: (error: { error_code: string; message: string }) => void;
}

export default function VideoUploader({ onUploadComplete, onError }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = useCallback(
    async (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!ext || !["mp4", "webm"].includes(ext)) {
        onError({
          error_code: "INVALID_EXTENSION",
          message: "Only .mp4 and .webm files are accepted.",
        });
        return;
      }

      setUploading(true);
      const form = new FormData();
      form.append("file", file);

      try {
        const res = await fetch(UPLOAD_URL, {
          method: "POST",
          body: form,
        });

        if (!res.ok) {
          const body = await res.json();
          onError(body);
          return;
        }

        const data = await res.json();
        onUploadComplete(data.video_id, data.filename, {
          width: data.width ?? null,
          height: data.height ?? null,
          status: data.status ?? "uploaded",
        });
      } catch {
        onError({
          error_code: "NETWORK_ERROR",
          message: "Could not reach the server. Is the API running on :8000?",
        });
      } finally {
        setUploading(false);
      }
    },
    [onUploadComplete, onError]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) upload(file);
    },
    [upload]
  );

  return (
    <div
      className={`dropzone${dragOver ? " drag-over" : ""}${uploading ? " uploading" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <p className="dropzone-label">
        {uploading
          ? "Uploading..."
          : "Drag & drop a video here, or click to browse"}
      </p>
      <p className="dropzone-hint">
        .mp4 or .webm &middot; max 200 MB
      </p>
      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.webm,video/mp4,video/webm"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) upload(file);
        }}
        hidden
      />
    </div>
  );
}
