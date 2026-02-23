from pydantic import BaseModel


class LabelCreate(BaseModel):
    segment_id: str | None = None
    frame_index: int | None = None
    label_type: str
    value: dict
    source: str = "manual"


class LabelResponse(BaseModel):
    id: str
    video_id: str
    segment_id: str | None
    frame_index: int | None
    label_type: str
    value: dict
    source: str
    created_at: str
