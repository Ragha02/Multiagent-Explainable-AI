"""
MA-XAI API — Crop Recommendation routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.state import pipeline

router = APIRouter(prefix="/recommend", tags=["recommend"])


class RecommendInput(BaseModel):
    N: float = 80.0
    P: float = 40.0
    K: float = 40.0
    ph: float = 6.5
    humidity: float = 65.0
    rainfall: float = 700.0
    temperature: float = 27.0
    top_k: int = 3


@router.post("")
async def recommend_crops(params: RecommendInput):
    """Return top-k crop recommendations for given soil/climate inputs."""
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")
    if pipeline.agent6_recommend is None:
        raise HTTPException(status_code=503, detail="Crop recommender not trained (CSV may be missing)")

    results = pipeline.agent6_recommend.recommend(params.dict(), top_k=params.top_k)
    return {"status": "ok", "recommendations": results}


@router.get("/crops")
async def list_crops():
    """Return list of crops the model can recommend."""
    if not pipeline.is_ready() or pipeline.agent6_recommend is None:
        return {"crops": []}
    return {"crops": pipeline.agent6_recommend.classes_}
