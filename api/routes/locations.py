"""
MA-XAI API — Location routes.
Provides state/district/crop lists and per-district-crop farm parameter
defaults derived from the real Kaggle dataset already loaded in memory.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import numpy as np
from api.state import pipeline

router = APIRouter(prefix="/locations", tags=["locations"])


def _normalised_df():
    """Return clean_df with state/district/crop stripped of whitespace."""
    df = pipeline.clean_df.copy()
    df["state"]    = df["state"].str.strip()
    df["district"] = df["district"].str.strip()
    if "crop" in df.columns:
        df["crop"] = df["crop"].str.strip()
    if "season" in df.columns:
        df["season"] = df["season"].str.strip()
    return df


# ── GET /api/locations ─────────────────────────────────────────────────────────
@router.get("")
async def get_locations():
    """
    Return:
      • states:           sorted list of 33 Indian states
      • districts:        { "State": ["DIST1", ...] }
      • crops_by_district: { "State||DISTRICT": ["Crop1", ...] }

    crops_by_district is keyed "State||DISTRICT" to stay flat and easy
    to look up from the frontend without extra round-trips.
    """
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    df = _normalised_df()

    districts:        dict[str, list[str]] = {}
    crops_by_district: dict[str, list[str]] = {}

    for state, s_grp in df.groupby("state"):
        districts[state] = sorted(s_grp["district"].dropna().unique().tolist())
        for dist, d_grp in s_grp.groupby("district"):
            key = f"{state}||{dist}"
            crops_by_district[key] = sorted(d_grp["crop"].dropna().unique().tolist())

    return {
        "states":             sorted(districts.keys()),
        "districts":          districts,
        "crops_by_district":  crops_by_district,
    }


# ── GET /api/locations/defaults/{state}/{district}?crop=Rice ──────────────────
@router.get("/defaults/{state}/{district}")
async def get_location_defaults(
    state:    str,
    district: str,
    crop:     Optional[str] = None,   # ← optional crop filter
):
    """
    Return median farm parameters for state + district, optionally filtered to
    a specific crop.  If crop= is supplied, medians are computed only over rows
    for that crop — giving crop-specific soil/NPK/yield values.

    Falls back to district-wide medians if the crop is not found, and then to
    state-wide medians if the district is not found.
    """
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    df = _normalised_df()

    state_s    = state.strip()
    district_s = district.strip()
    crop_s     = crop.strip() if crop else None

    # ── Filter: state + district + crop ───────────────────────────────────────
    base_mask = (df["state"] == state_s) & (df["district"] == district_s)
    sub       = df[base_mask & (df["crop"] == crop_s)] if crop_s else df[base_mask]

    # Fallback 1: district without crop filter
    if sub.empty and crop_s:
        sub = df[base_mask]

    # Fallback 2: whole state
    if sub.empty:
        sub = df[df["state"] == state_s]

    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for state='{state_s}' district='{district_s}'"
        )

    def med(col: str, fallback: float) -> float:
        if col not in sub.columns:
            return fallback
        v = sub[col].dropna().median()
        return round(float(v), 2) if not np.isnan(v) else fallback

    def mode_int(col: str, fallback: int) -> int:
        if col not in sub.columns:
            return fallback
        vc = sub[col].dropna().value_counts()
        return int(vc.index[0]) if not vc.empty else fallback

    # Available crops at this district (for convenience)
    all_crops = sorted(df[base_mask]["crop"].dropna().unique().tolist())

    defaults = {
        # ── Real district/crop data ──────────────────────────────────────────
        "prev_year_yield":         med("prev_year_yield",         25.0),
        "rainfall_annual":         med("rainfall_annual",         700.0),
        "temp_mean":               med("temp_mean",               27.0),
        # ── Crop-type soil medians (DS3) ──────────────────────────────────────
        "soil_ph":                 med("soil_ph",                 6.5),
        "nitrogen_kg_ha":          med("nitrogen_kg_ha",          220.0),
        # ── Distribution-filled (no district-level source) ───────────────────
        "irrigation_coverage_pct": med("irrigation_coverage_pct", 55.0),
        "soil_moisture":           med("soil_moisture",           45.0),
        "organic_carbon":          med("organic_carbon",          0.55),
        "npk_dosage_kg_ha":        med("npk_dosage_kg_ha",        120.0),
        "sowing_week":             mode_int("sowing_week",         21),
        "variety_improved":        mode_int("variety_improved",     1),
        # ── Metadata ─────────────────────────────────────────────────────────
        "n_records":               int(len(sub)),
        "available_crops":         all_crops,
        "filtered_by_crop":        crop_s or None,
    }

    return {
        "state":    state_s,
        "district": district_s,
        "crop":     crop_s,
        "defaults": defaults,
    }
