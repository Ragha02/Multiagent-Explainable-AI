"""
MA-XAI API — FastAPI application entry point.
Run with: python3 -m uvicorn api.main:app --reload --port 8000
"""
import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.pipeline import router as pipeline_router
from api.routes.predict import router as predict_router
from api.routes.advisory import router as advisory_router
from api.routes.locations import router as locations_router
from api.routes.compare import router as compare_router
from api.routes.recommend import router as recommend_router
from api.state import pipeline, run_pipeline_async

# ── Output dir ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "ma_xai_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MA-XAI API",
    description="Multi-Agent Explainable AI for Crop Yield Prediction",
    version="1.0.0",
)

# ── CORS — allow Next.js dev server ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving — PNGs from ma_xai_outputs/ ─────────────────────────
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(pipeline_router,  prefix="/api")
app.include_router(predict_router,   prefix="/api")
app.include_router(advisory_router,  prefix="/api")
app.include_router(locations_router, prefix="/api")
app.include_router(compare_router,   prefix="/api")
app.include_router(recommend_router, prefix="/api")


# ── Startup: kick off pipeline automatically ─────────────────────────────────
@app.on_event("startup")
async def on_startup():
    print("\n🌾 MA-XAI API starting — launching pipeline in background…")
    asyncio.create_task(run_pipeline_async())


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "pipeline": pipeline.status,
        "progress": pipeline.progress,
    }
