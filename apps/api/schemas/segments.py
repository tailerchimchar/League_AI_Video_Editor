from pydantic import BaseModel


class Segment(BaseModel):
    id: str
    segment_type: str
    start_ms: int
    end_ms: int
    confidence: float
    features: dict


class SegmentList(BaseModel):
    segments: list[Segment]
    count: int
