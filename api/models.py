"""
MA-XAI API — Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class FarmInput(BaseModel):
    irrigation_coverage_pct: float = Field(60.0, ge=0, le=100, description="Irrigation coverage %")
    rainfall_annual: float = Field(700.0, ge=50, le=2500, description="Annual rainfall in mm")
    soil_moisture: float = Field(45.0, ge=5, le=95, description="Soil moisture %")
    soil_ph: float = Field(6.5, ge=4.0, le=9.5, description="Soil pH")
    organic_carbon: float = Field(0.55, ge=0.05, le=3.5, description="Organic carbon %")
    npk_dosage_kg_ha: float = Field(120.0, ge=10, le=400, description="NPK dosage kg/ha")
    temp_mean: float = Field(27.0, ge=5, le=50, description="Mean temperature °C")
    variety_improved: int = Field(1, ge=0, le=1, description="1=Improved variety, 0=Traditional")
    sowing_week: int = Field(20, ge=1, le=52, description="Sowing week of the year")
    prev_year_yield: float = Field(25.0, ge=0, le=100, description="Previous year yield q/ha")
    farm_label: str = Field("Custom Farm", description="Label for this farm")


class PredictionResult(BaseModel):
    farm_label: str
    predicted_yield: float
    model_metrics: Optional[Dict[str, Any]] = None
    top_shap_features: Optional[List[Dict[str, Any]]] = None


class ATERow(BaseModel):
    treatment: str
    t_low: str
    t_high: str
    ate_qha: float
    ate_pct: float
    ci_low: float
    ci_high: float


class Recommendation(BaseModel):
    phase: str
    priority: str
    recommendation: str
    basis: str
    confidence: str
    delta_yield: Optional[float] = None


class AdvisoryResult(BaseModel):
    farm_label: str
    predicted_yield: float
    n_critical: int
    n_high: int
    recommendations: List[Recommendation]
    traceability: Optional[str] = None


class PipelineStatus(BaseModel):
    status: str          # "idle" | "running" | "ready" | "error"
    progress: int        # 0–100
    current_step: str
    metrics: Optional[Dict[str, Any]] = None
    ate_table: Optional[List[ATERow]] = None
    error: Optional[str] = None


# ── Comparison models ──────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    farm_a: FarmInput
    farm_b: FarmInput


class ShapEntry(BaseModel):
    feature: str
    shap_value: float


class DeltaShapEntry(BaseModel):
    feature: str
    shap_a: float
    shap_b: float
    delta: float           # shap_a - shap_b
    direction_flip: bool   # True when sign(shap_a) != sign(shap_b)


class LimeContrastEntry(BaseModel):
    feature_condition: str
    contribution_a: float
    contribution_b: float


class DiceAction(BaseModel):
    feature: str
    from_val: float
    to_val: float
    unit: str
    estimated_gain: float  # q/ha


class FarmSummary(BaseModel):
    label: str
    predicted_yield: float
    shap: List[ShapEntry]


class CompareResult(BaseModel):
    farm_a: FarmSummary
    farm_b: FarmSummary
    delta_yield: float
    delta_shap: List[DeltaShapEntry]
    lime_contrast: List[LimeContrastEntry]
    dice_actions: List[DiceAction]
