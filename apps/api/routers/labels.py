"""Manual labeling endpoints for training data bootstrapping."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from db.queries import get_video, insert_label, get_labels_for_video
from schemas.labels import LabelCreate

router = APIRouter(tags=["labels"])


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message})


@router.post("/videos/{video_id}/labels")
async def create_label(video_id: str, label: LabelCreate):
    """Submit a manual label for a video frame or segment."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    row = await insert_label(
        video_id=video_id,
        segment_id=label.segment_id,
        frame_index=label.frame_index,
        label_type=label.label_type,
        value=label.value,
        source=label.source,
    )

    return JSONResponse(content={
        "id": str(row["id"]),
        "video_id": str(row["video_id"]),
        "label_type": row["label_type"],
        "source": row["source"],
        "created_at": row["created_at"].isoformat(),
    })


@router.get("/videos/{video_id}/labels")
async def list_labels(video_id: str):
    """List all labels for a video."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    labels = await get_labels_for_video(video_id)

    result = []
    for lb in labels:
        ld = dict(lb)
        import json
        value = ld.get("value", {})
        if isinstance(value, str):
            value = json.loads(value)

        result.append({
            "id": str(ld["id"]),
            "video_id": str(ld["video_id"]),
            "segment_id": str(ld["segment_id"]) if ld["segment_id"] else None,
            "frame_index": ld["frame_index"],
            "label_type": ld["label_type"],
            "value": value,
            "source": ld["source"],
            "created_at": ld["created_at"].isoformat(),
        })

    return JSONResponse(content={"labels": result, "count": len(result)})
