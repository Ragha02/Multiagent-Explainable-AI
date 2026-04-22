"""
MA-XAI API — Advisory routes.
Generates full priority-ranked advisory + traceability chain for a farm.
"""
import numpy as np
from fastapi import APIRouter, HTTPException
from api.state import pipeline
from api.models import FarmInput, AdvisoryResult, Recommendation
from api.routes.predict import _farm_to_feature_vector

router = APIRouter(prefix="/advisory", tags=["advisory"])


@router.post("", response_model=AdvisoryResult)
async def generate_advisory(farm: FarmInput):
    """Generate a full advisory report for the given farm input."""
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet")

    X = _farm_to_feature_vector(farm)
    agent5 = pipeline.agent5_advisory

    farm_dict = {
        "irrigation_coverage_pct": farm.irrigation_coverage_pct,
        "rainfall_annual": farm.rainfall_annual,
        "soil_moisture": farm.soil_moisture,
        "soil_ph": farm.soil_ph,
        "organic_carbon": farm.organic_carbon,
        "npk_dosage_kg_ha": farm.npk_dosage_kg_ha,
        "temp_mean": farm.temp_mean,
        "variety_improved": int(farm.variety_improved),
        "sowing_week": int(farm.sowing_week),
        "prev_year_yield": farm.prev_year_yield,
        # Fill medians for other keys advisory engine may need
        **{f: float(pipeline.clean_df[f].median())
           for f in pipeline.num_cols
           if f not in ["irrigation_coverage_pct", "rainfall_annual", "soil_moisture",
                         "soil_ph", "organic_carbon", "npk_dosage_kg_ha", "temp_mean",
                         "variety_improved", "sowing_week", "prev_year_yield",
                         "yield_q_ha"]},
    }

    advisory = agent5.generate_advisory(X[0], farm_dict, farm.farm_label)
    chain = agent5.traceability_chain(advisory, farm_dict)

    recs = [
        Recommendation(
            phase=r["phase"],
            priority=r["priority"],
            recommendation=r["recommendation"],
            basis=r["basis"],
            confidence=r["confidence"],
            delta_yield=r.get("delta_yield"),
        )
        for r in advisory["recommendations"]
    ]

    return AdvisoryResult(
        farm_label=advisory["farm_label"],
        predicted_yield=advisory["predicted_yield"],
        n_critical=advisory["n_critical"],
        n_high=advisory["n_high"],
        recommendations=recs,
        traceability=chain,
    )
