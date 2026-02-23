from pydantic import BaseModel


class ReportResponse(BaseModel):
    id: str
    summary_text: str
    evidence_refs: list[dict]
    model_used: str
    created_at: str
