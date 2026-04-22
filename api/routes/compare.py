"""
MA-XAI API — XAI Farm Comparison routes.
POST /api/compare  →  Run two farms through the full XAI stack side-by-side.

Computes:
  • Yield prediction for both farms
  • SHAP vectors for both  →  ΔSHAP = shap_a − shap_b per feature
  • LIME local approximation contrast
  • DiCE minimal-change counterfactual actions (how to upgrade Farm B → Farm A level)
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.state import pipeline
from api.models import (
    CompareRequest, CompareResult, FarmSummary,
    ShapEntry, DeltaShapEntry, LimeContrastEntry, DiceAction,
)
from api.routes.predict import _farm_to_feature_vector

router = APIRouter(prefix="/compare", tags=["compare"])

# ── Feature units for DiCE display ────────────────────────────────────────────
FEATURE_UNITS: dict[str, str] = {
    "irrigation_coverage_pct": "%",
    "rainfall_annual":         "mm",
    "soil_moisture":           "%",
    "soil_ph":                 "",
    "organic_carbon":          "%",
    "npk_dosage_kg_ha":        "kg/ha",
    "temp_mean":               "°C",
    "variety_improved":        "",
    "sowing_week":             "wk",
    "prev_year_yield":         "q/ha",
}

# Farmer-controllable features (used as DiCE actionable set)
ACTIONABLE_FEATURES = [
    "irrigation_coverage_pct",
    "npk_dosage_kg_ha",
    "soil_moisture",
    "organic_carbon",
    "variety_improved",
    "sowing_week",
    "prev_year_yield",
]


def _build_user_dict(farm) -> dict:
    """Return the raw user-supplied values as a dict (before scaling)."""
    return {
        "irrigation_coverage_pct": farm.irrigation_coverage_pct,
        "rainfall_annual":         farm.rainfall_annual,
        "soil_moisture":           farm.soil_moisture,
        "soil_ph":                 farm.soil_ph,
        "organic_carbon":          farm.organic_carbon,
        "npk_dosage_kg_ha":        farm.npk_dosage_kg_ha,
        "temp_mean":               farm.temp_mean,
        "variety_improved":        float(farm.variety_improved),
        "sowing_week":             float(farm.sowing_week),
        "prev_year_yield":         farm.prev_year_yield,
    }


def _run_dice(X_b_scaled: np.ndarray, user_b: dict, user_a: dict,
              target_yield: float) -> list[DiceAction]:
    """
    Generate DiCE minimal-change counterfactuals for Farm B aimed at
    reaching Farm A's yield level. Falls back gracefully if dice-ml is
    not installed or the model is not compatible.
    """
    try:
        import dice_ml
        from dice_ml import Dice
    except ImportError:
        return []

    agent2   = pipeline.agent2_pred
    feat_names = pipeline.feat_names
    clean_df   = pipeline.clean_df
    scaler     = pipeline.scaler

    # ── Build a small background DataFrame in the model's feature space ───────
    # Sample 400 rows from clean_df and run them through the feature pipeline
    # to get scaled vectors identical to what the model sees.
    sample_df = clean_df.sample(n=min(400, len(clean_df)), random_state=42)

    medians = clean_df.select_dtypes(include=np.number).median().to_dict()
    medians.pop("yield_q_ha", None)

    rows = []
    for _, row in sample_df.iterrows():
        r = medians.copy()
        for col in ACTIONABLE_FEATURES:
            if col in row:
                r[col] = float(row[col])
        row_vec = np.array([r.get(f, 0.0) for f in feat_names], dtype=float)
        rows.append(row_vec)
    bg_array = np.vstack(rows)
    bg_scaled = scaler.transform(bg_array)

    # Add target column (yield) — DiCE needs it to know it's a regression task
    yields = agent2.predict(bg_scaled)
    bg_df = pd.DataFrame(bg_scaled, columns=feat_names)
    bg_df["yield_q_ha"] = yields

    # Build DiCE data and model wrappers
    feature_types = {f: "continuous" for f in feat_names}
    d = dice_ml.Data(
        dataframe=bg_df,
        continuous_features=feat_names,
        outcome_name="yield_q_ha",
    )

    # Sklearn predict wrapper (DiCE needs predict, not predict_proba)
    def predict_fn(X_df):
        arr = X_df[feat_names].values.astype(float)
        return agent2.predict(arr)

    m = dice_ml.Model(model=agent2.rf, backend="sklearn")

    exp = Dice(d, m, method="random")

    # Farm B query point
    query_df = pd.DataFrame([X_b_scaled[0]], columns=feat_names)

    # Keep only actionable features in DiCE's feature list
    actionable_idxs = [feat_names.index(f) for f in ACTIONABLE_FEATURES if f in feat_names]
    permitted = [feat_names[i] for i in actionable_idxs]

    try:
        cf = exp.generate_counterfactuals(
            query_df,
            total_CFs=3,
            desired_range=[target_yield * 0.9, target_yield * 1.2],
            features_to_vary=permitted,
        )
        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            return []
    except Exception:
        return []

    # Best CF = the one with highest predicted yield
    best_cf = cf_df.iloc[0]

    # Convert scaled CF values back to original space via inverse_transform
    cf_vec = best_cf[feat_names].values.astype(float).reshape(1, -1)
    cf_unscaled = scaler.inverse_transform(cf_vec)[0]
    b_unscaled  = scaler.inverse_transform(X_b_scaled)[0]
    cf_yield    = float(agent2.predict(cf_vec)[0])
    b_yield     = float(agent2.predict(X_b_scaled)[0])
    gain_total  = cf_yield - b_yield

    actions: list[DiceAction] = []
    for i, feat in enumerate(feat_names):
        if feat not in ACTIONABLE_FEATURES:
            continue
        from_v = round(float(b_unscaled[i]), 2)
        to_v   = round(float(cf_unscaled[i]), 2)
        if abs(to_v - from_v) < 0.01:
            continue
        # Proportional gain estimate
        ratio = abs(to_v - from_v) / (
            sum(abs(cf_unscaled[feat_names.index(f)] - b_unscaled[feat_names.index(f)])
                for f in ACTIONABLE_FEATURES if f in feat_names) + 1e-9
        )
        actions.append(DiceAction(
            feature=feat,
            from_val=from_v,
            to_val=to_v,
            unit=FEATURE_UNITS.get(feat, ""),
            estimated_gain=round(gain_total * ratio, 2),
        ))

    # Sort by |estimated_gain| descending
    return sorted(actions, key=lambda x: abs(x.estimated_gain), reverse=True)[:6]


# ── Main comparison endpoint ───────────────────────────────────────────────────

@router.post("", response_model=CompareResult)
async def compare_farms(req: CompareRequest):
    """
    Compare two farms side-by-side using the full MA-XAI explanation stack.
    Returns SHAP vectors, ΔSHAP divergence table, LIME contrast, and DiCE actions.
    """
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    agent2 = pipeline.agent2_pred
    agent4 = pipeline.agent4_explain

    # ── 1. Feature vectors ────────────────────────────────────────────────────
    Xa = _farm_to_feature_vector(req.farm_a)   # shape (1, n_features), scaled
    Xb = _farm_to_feature_vector(req.farm_b)

    # ── 2. Predictions ────────────────────────────────────────────────────────
    yield_a = float(agent2.predict(Xa)[0])
    yield_b = float(agent2.predict(Xb)[0])

    # ── 3. SHAP vectors ───────────────────────────────────────────────────────
    sv_a = agent2.explainer.shap_values(Xa)[0]   # shape (n_features,)
    sv_b = agent2.explainer.shap_values(Xb)[0]

    feat_names = pipeline.feat_names

    shap_a_list = [ShapEntry(feature=f, shap_value=round(float(v), 4))
                   for f, v in zip(feat_names, sv_a)]
    shap_b_list = [ShapEntry(feature=f, shap_value=round(float(v), 4))
                   for f, v in zip(feat_names, sv_b)]

    # ── 4. ΔSHAP — sorted by |delta| ─────────────────────────────────────────
    delta_shap: list[DeltaShapEntry] = []
    for f, va, vb in zip(feat_names, sv_a, sv_b):
        d = float(va) - float(vb)
        flip = (va > 0) != (vb > 0) and (abs(va) > 0.001 or abs(vb) > 0.001)
        delta_shap.append(DeltaShapEntry(
            feature=f,
            shap_a=round(float(va), 4),
            shap_b=round(float(vb), 4),
            delta=round(d, 4),
            direction_flip=bool(flip),
        ))
    delta_shap.sort(key=lambda x: abs(x.delta), reverse=True)

    # ── 5. LIME contrast (top 6 features) ────────────────────────────────────
    lime_contrast: list[LimeContrastEntry] = []
    try:
        lime_exp_a = agent4.lime_explainer.explain_instance(
            Xa[0], predict_fn=agent2.predict, num_features=8, num_samples=300
        )
        lime_exp_b = agent4.lime_explainer.explain_instance(
            Xb[0], predict_fn=agent2.predict, num_features=8, num_samples=300
        )
        lime_a = dict(lime_exp_a.as_list())
        lime_b = dict(lime_exp_b.as_list())
        all_conditions = set(lime_a) | set(lime_b)
        lime_entries = [
            LimeContrastEntry(
                feature_condition=cond,
                contribution_a=round(float(lime_a.get(cond, 0.0)), 4),
                contribution_b=round(float(lime_b.get(cond, 0.0)), 4),
            )
            for cond in all_conditions
        ]
        # Sort by max absolute contribution across both farms
        lime_entries.sort(
            key=lambda x: max(abs(x.contribution_a), abs(x.contribution_b)),
            reverse=True
        )
        lime_contrast = lime_entries[:8]
    except Exception:
        pass  # LIME is bonus — never block the response

    # ── 6. DiCE counterfactual actions (Farm B → Farm A level) ───────────────
    # Only compute if Farm A is better; otherwise reverse
    if yield_a >= yield_b:
        dice_actions = _run_dice(Xb, _build_user_dict(req.farm_b),
                                 _build_user_dict(req.farm_a), yield_a)
    else:
        dice_actions = _run_dice(Xa, _build_user_dict(req.farm_a),
                                 _build_user_dict(req.farm_b), yield_b)

    # ── 7. Assemble response ──────────────────────────────────────────────────
    return CompareResult(
        farm_a=FarmSummary(
            label=req.farm_a.farm_label,
            predicted_yield=round(yield_a, 2),
            shap=shap_a_list[:10],
        ),
        farm_b=FarmSummary(
            label=req.farm_b.farm_label,
            predicted_yield=round(yield_b, 2),
            shap=shap_b_list[:10],
        ),
        delta_yield=round(yield_a - yield_b, 2),
        delta_shap=delta_shap[:12],
        lime_contrast=lime_contrast,
        dice_actions=dice_actions,
    )
