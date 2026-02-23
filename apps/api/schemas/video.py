from pydantic import BaseModel


class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    status: str


class VideoMeta(BaseModel):
    id: str
    filename: str
    storage_path: str
    file_size_bytes: int
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    mime_type: str
    status: str
    created_at: str
    updated_at: str
