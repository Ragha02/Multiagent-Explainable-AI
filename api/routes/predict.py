"""
MA-XAI API — Prediction routes.
Accepts a custom farm input and returns yield prediction + SHAP values.
"""
import numpy as np
from fastapi import APIRouter, HTTPException
from api.state import pipeline
from api.models import FarmInput, PredictionResult

router = APIRouter(prefix="/predict", tags=["predict"])


def _farm_to_feature_vector(farm: FarmInput) -> np.ndarray:
    """
    Build a full feature vector from the farm inputs.
    Unspecified features are filled with dataset medians.
    """
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    clean_df = pipeline.clean_df
    feat_names = pipeline.feat_names
    scaler = pipeline.scaler

    # Start from column medians (numeric only, excluding target)
    medians = clean_df.select_dtypes(include=np.number).median().to_dict()
    medians.pop("yield_q_ha", None)

    # Override with user-supplied values
    user_vals = {
        "irrigation_coverage_pct": farm.irrigation_coverage_pct,
        "rainfall_annual": farm.rainfall_annual,
        "soil_moisture": farm.soil_moisture,
        "soil_ph": farm.soil_ph,
        "organic_carbon": farm.organic_carbon,
        "npk_dosage_kg_ha": farm.npk_dosage_kg_ha,
        "temp_mean": farm.temp_mean,
        "variety_improved": float(farm.variety_improved),
        "sowing_week": float(farm.sowing_week),
        "prev_year_yield": farm.prev_year_yield,
    }
    medians.update(user_vals)

    # Build in model feature order
    vec = np.array([medians.get(f, 0.0) for f in feat_names], dtype=float).reshape(1, -1)
    return scaler.transform(vec)


@router.post("", response_model=PredictionResult)
async def predict(farm: FarmInput):
    """Predict crop yield for a given farm configuration."""
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet")

    X = _farm_to_feature_vector(farm)
    agent2 = pipeline.agent2_pred

    pred_yield = float(agent2.predict(X)[0])

    # SHAP for this instance
    shap_exp = agent2.explain_instance(X[0])
    top_shap = sorted(
        shap_exp["shap_values"].items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]
    shap_list = [{"feature": f, "shap_value": round(v, 4)} for f, v in top_shap]

    return PredictionResult(
        farm_label=farm.farm_label,
        predicted_yield=round(pred_yield, 2),
        model_metrics=pipeline.model_metrics,
        top_shap_features=shap_list,
    )


@router.get("/sample/{index}")
async def predict_sample(index: int = 0):
    """Predict yield for sample index from the test dataset (for demo)."""
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {"message": f"Sample {index} prediction endpoint — connect to full pipeline"}
