from pydantic import BaseModel


class ExtractionConfig(BaseModel):
    sample_fps: float = 5.0
    enable_ocr: bool = True
    enable_detector: bool = True
    enable_debug_overlays: bool = False
    crop_preset: str = "auto"
    played_champion: str | None = None


class ExtractionJobResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    progress: float
    frame_count: int = 0
    error_message: str | None = None


class OcrData(BaseModel):
    game_timer: str | None = None
    player_level: int | None = None
    player_cs: int | None = None
    player_gold: int | None = None
    player_kda: dict | None = None
    player_hp_pct: float | None = None
    player_mana_pct: float | None = None
    raw: dict = {}


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]
    champion: str | None = None
    champion_confidence: float | None = None


class DerivedFeatures(BaseModel):
    hp_delta: float | None = None
    cs_delta: float | None = None
    gold_delta: float | None = None
    spacing_proxy: float | None = None


class FramePayloadResponse(BaseModel):
    frame_index: int
    timestamp_ms: int
    ocr_data: dict
    detections: list[dict]
    derived_features: dict
