"""
MA-XAI API — Pipeline routes (status + SSE stream).
"""
import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.state import pipeline, run_pipeline_async
from api.models import PipelineStatus

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/status", response_model=PipelineStatus)
async def get_status():
    """Return current pipeline status, progress and cached results."""
    return PipelineStatus(
        status=pipeline.status,
        progress=pipeline.progress,
        current_step=pipeline.current_step,
        metrics=pipeline.model_metrics if pipeline.is_ready() else None,
        ate_table=pipeline.ate_table if pipeline.is_ready() else None,
        error=pipeline.error,
    )


@router.post("/run")
async def run_pipeline():
    """Trigger full pipeline run (idempotent — skips if already running/ready)."""
    if pipeline.status in ("running", "ready"):
        return {"message": "Pipeline already running or complete", "status": pipeline.status}
    asyncio.create_task(run_pipeline_async())
    return {"message": "Pipeline started", "status": "running"}


@router.get("/stream")
async def stream_progress():
    """
    Server-Sent Events stream — frontend subscribes to get live progress updates.
    Closes automatically when pipeline reaches ready/error state.
    """
    async def event_generator():
        while True:
            data = {
                "status": pipeline.status,
                "progress": pipeline.progress,
                "current_step": pipeline.current_step,
                "error": pipeline.error,
            }
            yield f"data: {json.dumps(data)}\n\n"

            if pipeline.status in ("ready", "error"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/results")
async def get_results():
    """Return all cached pipeline results (metrics, ATE, SHAP importance)."""
    if not pipeline.is_ready():
        return {"error": "Pipeline not ready", "status": pipeline.status}
    return {
        "status": "ready",
        "model_metrics": pipeline.model_metrics,
        "ate_table": pipeline.ate_table,
        "global_shap": pipeline.global_shap,
    }
