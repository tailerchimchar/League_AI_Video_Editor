"""Evidence-grounded report generation endpoints."""

import json
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from db.queries import get_video, insert_report, get_latest_report_for_video
from report.builder import build_evidence_payload

router = APIRouter(tags=["reports"])


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message})


@router.get("/videos/{video_id}/report")
async def generate_report(video_id: str):
    """Generate an evidence-grounded coaching report via Claude.

    Streams the response as SSE events, then stores in DB.
    """
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        return _error(500, "MISSING_API_KEY", "ANTHROPIC_API_KEY is not configured.")

    evidence = await build_evidence_payload(video_id)
    if not evidence:
        return _error(400, "NO_EXTRACTION", "No completed extraction found. Run extraction first.")

    async def event_stream():
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=api_key)
            model = "claude-sonnet-4-6"
            full_text = ""

            async with client.messages.stream(
                model=model,
                max_tokens=4096,
                system=evidence["system_prompt"],
                messages=[{"role": "user", "content": evidence["user_prompt"]}],
            ) as stream:
                async for text in stream.text_stream:
                    full_text += text
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

            # Get usage info
            message = await stream.get_final_message()
            prompt_tokens = message.usage.input_tokens if message.usage else None
            completion_tokens = message.usage.output_tokens if message.usage else None

            # Store report in DB
            report = await insert_report(
                video_id=video_id,
                job_id=evidence["job_id"],
                summary_text=full_text,
                evidence_refs=[],
                model_used=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            yield f"data: {json.dumps({'type': 'done', 'report_id': str(report['id'])})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/videos/{video_id}/report/latest")
async def get_latest_report(video_id: str):
    """Get the most recent stored report for a video."""
    video = await get_video(video_id)
    if not video:
        return _error(404, "NOT_FOUND", "Video not found.")

    report = await get_latest_report_for_video(video_id)
    if not report:
        return _error(404, "NOT_FOUND", "No report found for this video.")

    evidence_refs = report.get("evidence_refs", [])
    if isinstance(evidence_refs, str):
        evidence_refs = json.loads(evidence_refs)

    return JSONResponse(content={
        "id": str(report["id"]),
        "summary_text": report["summary_text"],
        "evidence_refs": evidence_refs,
        "model_used": report["model_used"],
        "created_at": report["created_at"].isoformat(),
    })
