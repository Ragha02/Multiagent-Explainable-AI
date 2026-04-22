"""
MA-XAI Framework — Agent 1: Data Agent
========================================
Responsibilities:
  • Real data loading & merging from 3 Kaggle datasets (primary path)
  • Synthetic dataset generation as fallback (mirrors paper's 51-feature schema)
  • Data ingestion & format validation
  • Missing value imputation with logged confidence scores
  • Outlier detection & capping
  • Data quality grading (A / B / C)
  • Feature engineering (temporal rainfall decomposition, interaction terms)

Real-data mode activates automatically when the following files exist in data/:
  data/crop_production.csv      — India govt production data (246k rows)
  data/yield_df.csv             — FAO climate features (rainfall, temperature)
  data/Crop_recommendation.csv  — Soil / NPK medians by crop type
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATA GENERATION
#     Mimics ICAR / IMD / Soil Health Card / Agricultural Census schema
# ══════════════════════════════════════════════════════════════════════════════

CROPS   = ["Rice", "Wheat", "Maize", "Soybean", "Cotton",
           "Sugarcane", "Groundnut", "Bajra", "Jowar", "Mustard"]
STATES  = ["Andhra Pradesh", "Punjab", "Haryana", "Maharashtra",
           "Madhya Pradesh", "Uttar Pradesh", "Karnataka", "Rajasthan",
           "Gujarat", "Telangana"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
SOIL_TYPES = ["Sandy", "Loamy", "Clay", "Silt", "Sandy Loam"]
IRR_TYPES  = ["Canal", "Borewell", "Rainfed", "Drip", "Sprinkler"]

# Agro-climate zone per state (simplified)
STATE_ZONE = {
    "Andhra Pradesh": "Semi-Arid", "Punjab": "Sub-Humid", "Haryana": "Sub-Humid",
    "Maharashtra": "Semi-Arid", "Madhya Pradesh": "Sub-Humid",
    "Uttar Pradesh": "Sub-Humid", "Karnataka": "Semi-Arid",
    "Rajasthan": "Arid", "Gujarat": "Arid", "Telangana": "Semi-Arid"
}

# ── Real Kaggle datasets expected at these paths ──────────────────────────────
_REAL_DATA_FILES = [
    "data/crop_production.csv",
    "data/yield_df.csv",
    "data/Crop_recommendation.csv",
]

# ── Crop-name normalisation (FAO & vernacular → common name used in DS1) ──────
_CROP_NORM = {
    # Dataset 2 FAO names
    "rice, paddy":             "Rice",
    "rice":                    "Rice",
    "wheat":                   "Wheat",
    "maize (corn)":            "Maize",
    "maize":                   "Maize",
    "soybeans":                "Soybean",
    "soya bean":               "Soybean",
    "soybean":                 "Soybean",
    "sorghum":                 "Jowar",
    "jowar":                   "Jowar",
    # Dataset 3 lowercase labels
    "cotton":                  "Cotton",
    "cottonseed":              "Cotton",
    "cotton(lint)":            "Cotton",
    "sugarcane":               "Sugarcane",
    "sugar cane":              "Sugarcane",
    "groundnut":               "Groundnut",
    "groundnuts, with shell":  "Groundnut",
    "mustard":                 "Mustard",
    "rapeseed and mustard":    "Mustard",
    "rapeseed &mustard":       "Mustard",
    "bajra":                   "Bajra",
    "pearl millet":            "Bajra",
    "potatoes":                "Potato",
    "potato":                  "Potato",
    "cassava":                 "Cassava",
    "sweet potatoes":          "Sweet Potato",
}


def _norm_crop(name: str) -> str:
    """Normalise a crop name to the common set used throughout the pipeline."""
    if not isinstance(name, str):
        return "Other"
    key = name.strip().lower()
    return _CROP_NORM.get(key, name.strip().title())


# ══════════════════════════════════════════════════════════════════════════════
# 1-B.  REAL DATA LOADER
#     Merges 3 Kaggle CSVs into the SAME schema used by
#     generate_synthetic_dataset().  Nothing downstream changes.
# ══════════════════════════════════════════════════════════════════════════════

def load_real_dataset(data_dir: str = "data", n_samples: int = 50_000) -> pd.DataFrame:
    """
    Load and merge the 3 Kaggle datasets into a DataFrame whose column
    schema is compatible with clean_data() → engineer_features() →
    encode_and_split().  Columns absent from real data are filled with
    domain-appropriate random draws so feature engineering never fails.

    Args:
        data_dir:  folder containing the 3 CSV files
        n_samples: stratified sample size (0 = load all rows).
                   Default 50,000 keeps training under ~60 s.
    """
    print(f"[DataAgent] Loading real Kaggle datasets (target n={n_samples:,}) …")
    rng = np.random.default_rng(SEED)

    # ── Dataset 1: India crop production ─────────────────────────────────────
    df1 = pd.read_csv(os.path.join(data_dir, "crop_production.csv"),
                      low_memory=False)
    df1 = df1.dropna(subset=["Production", "Area"])
    df1 = df1[df1["Area"] > 0].copy()
    df1["yield_q_ha"] = (df1["Production"] / df1["Area"]) * 10   # tonnes/ha → q/ha
    df1 = df1.rename(columns={
        "State_Name":    "state",
        "District_Name": "district",
        "Crop_Year":     "year",
        "Season":        "season",
        "Crop":          "crop",
        "Area":          "area_ha",
    })
    df1["crop"]   = df1["crop"].apply(_norm_crop)
    df1["season"] = df1["season"].str.strip()
    # Remove physically implausible yields
    df1 = df1[df1["yield_q_ha"].between(0.5, 500)].copy()
    df1 = df1.drop(columns=["Production"], errors="ignore")
    print(f"[DataAgent]   DS1 (production): {df1.shape}")

    # ── Dataset 2: FAO climate — India rows only ──────────────────────────────
    df2 = pd.read_csv(os.path.join(data_dir, "yield_df.csv"), low_memory=False)
    df2_india = df2[df2["Area"].str.strip() == "India"].copy()
    df2_india = df2_india.rename(columns={
        "Item":                          "crop",
        "Year":                          "year",
        "average_rain_fall_mm_per_year": "rainfall_annual",
        "avg_temp":                      "temp_mean",
    })
    df2_india["crop"] = df2_india["crop"].apply(_norm_crop)
    df2_india = (df2_india[["crop", "year", "rainfall_annual", "temp_mean"]]
                 .drop_duplicates())
    print(f"[DataAgent]   DS2 (climate/India): {df2_india.shape}")

    # ── Dataset 3: Crop recommendation — soil / NPK medians by crop ───────────
    df3 = pd.read_csv(os.path.join(data_dir, "Crop_recommendation.csv"),
                      low_memory=False)
    df3 = df3.rename(columns={"label": "crop"})
    df3["crop"] = df3["crop"].apply(_norm_crop)
    soil_med = (
        df3.groupby("crop")[["N", "P", "K", "ph", "humidity"]]
           .median()
           .reset_index()
           .rename(columns={
               "N":        "nitrogen_kg_ha",
               "P":        "phosphorus",
               "K":        "potassium",
               "ph":       "soil_ph",
               "humidity": "humidity",
           })
    )
    print(f"[DataAgent]   DS3 (soil medians): {soil_med.shape}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = df1.merge(df2_india, on=["crop", "year"], how="left")
    df = df.merge(soil_med,   on="crop",           how="left")

    # ── Compute prev_year_yield (shift-1 within state × district × crop) ──────
    df = df.sort_values(["state", "district", "crop", "year"])
    df["prev_year_yield"] = (
        df.groupby(["state", "district", "crop"])["yield_q_ha"]
          .shift(1)
    )

    # ── Fill columns that are absent from Kaggle data ─────────────────────────
    # These are required by engineer_features(); we use domain-appropriate
    # random draws so the pipeline schema stays intact.
    n = len(df)
    _FILL: dict = {
        "soil_moisture":           lambda: rng.normal(45, 15, n).clip(10, 85),
        "organic_carbon":          lambda: rng.normal(0.55, 0.2, n).clip(0.1, 2.5),
        "irrigation_coverage_pct": lambda: rng.normal(55, 25, n).clip(0, 100),
        "irrigation_reliability":  lambda: rng.uniform(0.4, 1.0, n),
        "npk_dosage_kg_ha":        lambda: rng.normal(120, 40, n).clip(20, 300),
        "urea_kg_ha":              lambda: rng.normal(80, 25, n).clip(10, 200),
        "dap_kg_ha":               lambda: rng.normal(50, 20, n).clip(0, 150),
        "organic_manure_t_ha":     lambda: rng.normal(2.5, 1.5, n).clip(0, 10),
        "plant_density":           lambda: rng.normal(180, 40, n).clip(60, 400),
        "mechanization_index":     lambda: rng.uniform(0, 1, n),
        "variety_improved":        lambda: rng.choice([0, 1], n, p=[0.35, 0.65]),
        "solar_radiation":         lambda: rng.normal(18, 4, n).clip(8, 30),
        "sowing_week":             lambda: rng.integers(15, 30, n).astype(float),
        "agro_climate_zone":       lambda: ["Semi-Arid"] * n,
        "latitude":                lambda: rng.uniform(8.0, 35.0, n),
        "longitude":               lambda: rng.uniform(68.0, 97.0, n),
        "irrigation_type":         lambda: rng.choice(IRR_TYPES, n),
        "soil_type":               lambda: rng.choice(SOIL_TYPES, n),
    }
    for col, fn in _FILL.items():
        if col not in df.columns:
            df[col] = fn()

    # Fill NaN for merged columns that didn't join
    df["rainfall_annual"]  = df["rainfall_annual"].fillna(
        df.groupby("crop")["rainfall_annual"].transform("median")).fillna(700.0)
    df["temp_mean"]        = df["temp_mean"].fillna(
        df.groupby("state")["temp_mean"].transform("median")).fillna(27.0)
    df["prev_year_yield"]  = df["prev_year_yield"].fillna(df["yield_q_ha"].median())
    df["nitrogen_kg_ha"]   = df["nitrogen_kg_ha"].fillna(220.0)
    df["phosphorus"]       = df["phosphorus"].fillna(18.0)
    df["potassium"]        = df["potassium"].fillna(175.0)
    df["soil_ph"]          = df["soil_ph"].fillna(6.5)
    df["humidity"]         = df["humidity"].fillna(65.0)

    # Derived columns used by engineer_features
    df["temp_max"] = df["temp_mean"] + rng.uniform(4, 10, n)
    df["temp_min"] = df["temp_mean"] - rng.uniform(4, 10, n)
    df["rainfall_early_season"] = df["rainfall_annual"] * rng.uniform(0.25, 0.35, n)
    df["rainfall_mid_season"]   = df["rainfall_annual"] * rng.uniform(0.35, 0.45, n)
    df["rainfall_late_season"]  = (df["rainfall_annual"]
                                    - df["rainfall_early_season"]
                                    - df["rainfall_mid_season"])

    df = df.dropna(subset=["yield_q_ha"]).reset_index(drop=True)

    # ── Stratified sample (by state × crop) to cap size ──────────────────
    if 0 < n_samples < len(df):
        # Compute per-group quota proportional to group size
        grp_key = ["state", "crop"]
        grp = df.groupby(grp_key, observed=True)
        sizes = grp.size()
        quota = (sizes / sizes.sum() * n_samples).round().astype(int).clip(lower=1)

        sampled_idx = []
        for keys, idx_group in grp.groups.items():
            k = quota.get(keys, 1)
            take = min(k, len(idx_group))
            sampled_idx.extend(
                df.index[idx_group].to_series().sample(take, random_state=SEED).tolist()
            )

        df = df.loc[sampled_idx].reset_index(drop=True)

        # Exact trim/top-up to n_samples
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)
        elif len(df) < n_samples:
            extra = df.sample(n=n_samples - len(df), replace=True,
                              random_state=SEED + 1)
            df = pd.concat([df, extra], ignore_index=True)

    print(f"[DataAgent] Real dataset ready: {df.shape}")
    return df


def generate_synthetic_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Primary dataset entry-point.  When the 3 Kaggle CSVs exist in data/
    this function delegates to load_real_dataset() so the entire pipeline
    runs on real data with zero changes to any downstream agent.
    Falls back to synthetic generation only when the files are absent.
    """
    # ── Auto-detect real data ──────────────────────────────────────────────
    if all(os.path.exists(f) for f in _REAL_DATA_FILES):
        return load_real_dataset(n_samples=n_samples)

    print(f"[DataAgent] Generating {n_samples} synthetic records …")

    rng = np.random.default_rng(SEED)

    # ── Spatial & categorical ────────────────────────────────────────────────
    states    = rng.choice(STATES, n_samples)
    districts = [f"{s[:3].upper()}_D{rng.integers(1, 15):02d}" for s in states]
    zones     = [STATE_ZONE[s] for s in states]
    lat       = rng.uniform(8.0, 35.0, n_samples)
    lon       = rng.uniform(68.0, 97.0, n_samples)

    # ── Temporal ─────────────────────────────────────────────────────────────
    years   = rng.integers(2010, 2024, n_samples)
    seasons = rng.choice(SEASONS, n_samples)
    crops   = rng.choice(CROPS, n_samples)
    sowing_week = rng.integers(1, 52, n_samples)

    # ── Climate features (Layer 1 — exogenous) ───────────────────────────────
    # Rainfall varies by zone & season
    base_rain = np.where(np.isin(zones, ["Semi-Arid", "Arid"]), 400, 700)
    rainfall_annual   = rng.normal(base_rain, 120, n_samples).clip(80, 2500)
    rainfall_early    = rainfall_annual * rng.uniform(0.25, 0.35, n_samples)
    rainfall_mid      = rainfall_annual * rng.uniform(0.35, 0.45, n_samples)
    rainfall_late     = rainfall_annual - rainfall_early - rainfall_mid

    temp_mean  = rng.normal(27, 5, n_samples).clip(10, 45)
    temp_max   = temp_mean + rng.uniform(4, 10, n_samples)
    temp_min   = temp_mean - rng.uniform(4, 10, n_samples)
    humidity   = rng.normal(65, 15, n_samples).clip(20, 100)
    solar_rad  = rng.normal(18, 4, n_samples).clip(8, 30)   # MJ/m²/day

    # ── Soil features (Layer 2 — intermediate) ───────────────────────────────
    soil_type      = rng.choice(SOIL_TYPES, n_samples)
    soil_ph        = rng.normal(6.8, 0.8, n_samples).clip(4.5, 9.0)
    organic_carbon = rng.normal(0.55, 0.25, n_samples).clip(0.1, 2.5)
    nitrogen_kg_ha = rng.normal(220, 60, n_samples).clip(50, 500)
    phosphorus     = rng.normal(18, 8, n_samples).clip(2, 60)
    potassium      = rng.normal(175, 50, n_samples).clip(50, 400)
    soil_moisture  = (rainfall_annual / 2500) * 100 + rng.normal(0, 5, n_samples)
    soil_moisture  = soil_moisture.clip(10, 90)

    # ── Irrigation (Layer 3 — controllable) ──────────────────────────────────
    irr_type        = rng.choice(IRR_TYPES, n_samples)
    irr_coverage    = np.where(irr_type == "Rainfed",
                               rng.uniform(0, 20, n_samples),
                               rng.uniform(40, 100, n_samples))
    irr_reliability = rng.uniform(0.3, 1.0, n_samples)

    # ── Fertilizer (Layer 3 — controllable) ──────────────────────────────────
    npk_dosage     = rng.normal(120, 40, n_samples).clip(20, 300)   # kg/ha
    urea_kg_ha     = rng.normal(80, 25, n_samples).clip(10, 200)
    dap_kg_ha      = rng.normal(50, 20, n_samples).clip(0, 150)
    organic_manure = rng.normal(2.5, 1.5, n_samples).clip(0, 10)    # t/ha

    # ── Crop management (Layer 3) ─────────────────────────────────────────────
    plant_density   = rng.normal(180, 40, n_samples).clip(60, 400)  # plants/m²
    mechanization   = rng.uniform(0, 1, n_samples)
    variety_improved= rng.choice([0, 1], n_samples, p=[0.35, 0.65])

    # ── Previous-year yield (temporal feature) ───────────────────────────────
    prev_year_yield = rng.normal(25, 12, n_samples).clip(5, 80)

    # ── Yield simulation (causal model) ──────────────────────────────────────
    # Domain-informed yield formula — reproduces agronomic logic
    base_yield = (
          0.018 * rainfall_annual
        + 0.30  * irr_coverage
        + 0.12  * npk_dosage
        + 0.08  * organic_carbon * 10
        + 0.05  * soil_moisture
        - 0.20  * np.abs(soil_ph - 6.5)       # penalty away from optimal pH
        - 0.015 * np.abs(temp_mean - 25) ** 2  # quadratic temp penalty
        + 3.5   * variety_improved
        + 2.0   * mechanization
        + 0.05  * prev_year_yield
    )
    # Introduce realistic noise & heterogeneity
    noise = rng.normal(0, 3.5, n_samples)
    yield_q_ha = (base_yield + noise).clip(3, 85)

    # ─────────────────────────────────────────────────────────────────────────
    # Assemble DataFrame (51 features + 1 target)
    # ─────────────────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        # Spatial
        "district": districts, "state": states,
        "agro_climate_zone": zones, "latitude": lat, "longitude": lon,
        # Temporal
        "year": years, "season": seasons, "crop": crops,
        "sowing_week": sowing_week, "prev_year_yield": prev_year_yield,
        # Climate (7)
        "rainfall_annual": rainfall_annual,
        "rainfall_early_season": rainfall_early,
        "rainfall_mid_season": rainfall_mid,
        "rainfall_late_season": rainfall_late,
        "temp_mean": temp_mean, "temp_max": temp_max, "temp_min": temp_min,
        "humidity": humidity, "solar_radiation": solar_rad,
        # Soil (8)
        "soil_type": soil_type, "soil_ph": soil_ph,
        "organic_carbon": organic_carbon,
        "nitrogen_kg_ha": nitrogen_kg_ha, "phosphorus": phosphorus,
        "potassium": potassium, "soil_moisture": soil_moisture,
        # Irrigation (3)
        "irrigation_type": irr_type,
        "irrigation_coverage_pct": irr_coverage,
        "irrigation_reliability": irr_reliability,
        # Fertilizer (4)
        "npk_dosage_kg_ha": npk_dosage, "urea_kg_ha": urea_kg_ha,
        "dap_kg_ha": dap_kg_ha, "organic_manure_t_ha": organic_manure,
        # Crop management (5)
        "plant_density": plant_density, "mechanization_index": mechanization,
        "variety_improved": variety_improved,
        # Target
        "yield_q_ha": yield_q_ha,
    })

    print(f"[DataAgent] Dataset shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MISSING VALUE INJECTION (simulates real-world incompleteness)
# ══════════════════════════════════════════════════════════════════════════════

def inject_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic missing values matching paper's stated completeness."""
    df = df.copy()
    rng = np.random.default_rng(SEED + 1)

    # Climate: <5% missing
    for col in ["rainfall_annual", "humidity", "solar_radiation"]:
        idx = rng.choice(len(df), int(0.03 * len(df)), replace=False)
        df.loc[idx, col] = np.nan

    # Soil: ~10% missing
    for col in ["organic_carbon", "phosphorus", "potassium"]:
        idx = rng.choice(len(df), int(0.08 * len(df)), replace=False)
        df.loc[idx, col] = np.nan

    # Agricultural practices: 60-80% completeness → ~25% missing
    for col in ["npk_dosage_kg_ha", "organic_manure_t_ha", "mechanization_index"]:
        idx = rng.choice(len(df), int(0.25 * len(df)), replace=False)
        df.loc[idx, col] = np.nan

    print(f"[DataAgent] Missing values injected. Total NaNs: {df.isna().sum().sum()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA CLEANING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class DataQualityReport:
    def __init__(self):
        self.imputation_log   = []
        self.outlier_log      = []
        self.quality_grades   = {}

    def log_imputation(self, col, method, n_imputed, confidence):
        self.imputation_log.append({
            "column": col, "method": method,
            "n_imputed": n_imputed, "confidence": confidence
        })

    def log_outlier(self, col, n_capped, low_cap, high_cap):
        self.outlier_log.append({
            "column": col, "n_capped": n_capped,
            "low_cap": round(low_cap, 3), "high_cap": round(high_cap, 3)
        })

    def assign_grade(self, col, missing_pct):
        if missing_pct < 5:
            grade = "A"
        elif missing_pct < 15:
            grade = "B"
        else:
            grade = "C"
        self.quality_grades[col] = grade


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Full cleaning pipeline as specified in Agent 1 (Section 3.2):
      1. Grade columns by missing %
      2. Impute missing values (temporal → spatial → district-median)
      3. Cap outliers at domain-justified limits
    """
    df = df.copy()
    report = DataQualityReport()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if "yield_q_ha" in numeric_cols:
        numeric_cols.remove("yield_q_ha")

    # ── Grade & impute ───────────────────────────────────────────────────────
    DOMAIN_LIMITS = {
        "rainfall_annual":        (50, 3000),
        "temp_mean":              (5, 50),
        "humidity":               (10, 100),
        "solar_radiation":        (5, 35),
        "soil_ph":                (4.0, 9.5),
        "organic_carbon":         (0.05, 3.5),
        "nitrogen_kg_ha":         (20, 600),
        "phosphorus":             (1, 100),
        "potassium":              (20, 600),
        "soil_moisture":          (5, 95),
        "irrigation_coverage_pct":(0, 100),
        "npk_dosage_kg_ha":       (10, 400),
        "yield_q_ha":             (2, 100),
    }

    for col in numeric_cols:
        missing_pct = df[col].isna().mean() * 100
        report.assign_grade(col, missing_pct)

        if df[col].isna().any():
            n_missing = df[col].isna().sum()
            # Temporal interpolation first (sort by year), then district median
            df_sorted = df.sort_values("year")
            fill_vals = (df_sorted
                         .groupby("district")[col]
                         .transform(lambda x: x.interpolate(method="linear",
                                                             limit_direction="both")))
            still_missing = fill_vals.isna()
            if still_missing.any():
                fill_vals[still_missing] = (df_sorted
                                            .groupby("state")[col]
                                            .transform("median"))[still_missing]
            # Last resort: global median
            fill_vals = fill_vals.fillna(df[col].median())
            df[col] = fill_vals

            confidence = "High" if missing_pct < 5 else ("Medium" if missing_pct < 15 else "Low")
            report.log_imputation(col, "temporal+spatial_interpolation", n_missing, confidence)

    # ── Outlier capping ──────────────────────────────────────────────────────
    for col, (lo, hi) in DOMAIN_LIMITS.items():
        if col not in df.columns:
            continue
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out > 0:
            df[col] = df[col].clip(lo, hi)
            report.log_outlier(col, int(n_out), lo, hi)

    print(f"[DataAgent] Cleaning done. Remaining NaNs: {df.isna().sum().sum()}")
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features including:
      - Rainfall adequacy ratio
      - Temperature stress index
      - Nutrient balance score
      - Irrigation sufficiency
    """
    df = df.copy()

    # Rainfall adequacy (ratio to seasonal need — Rice needs ~1200mm, Wheat ~450mm)
    crop_water_need = {"Rice": 1200, "Wheat": 450, "Maize": 600, "Soybean": 500,
                       "Cotton": 700, "Sugarcane": 1800, "Groundnut": 550,
                       "Bajra": 350, "Jowar": 400, "Mustard": 350}
    df["crop_water_need"] = df["crop"].map(crop_water_need).fillna(600)
    df["rainfall_adequacy"] = (df["rainfall_annual"] / df["crop_water_need"]).clip(0, 3)

    # Temperature stress index (deviation from optimal 25°C, squared)
    df["temp_stress_index"] = np.abs(df["temp_mean"] - 25) ** 2 / 100

    # Nutrient balance score (N-P-K normalised sum)
    df["nutrient_balance"] = (
        df["nitrogen_kg_ha"] / 400 +
        df["phosphorus"] / 60 +
        df["potassium"] / 400
    ) / 3

    # Irrigation × rainfall interaction
    df["water_supply_index"] = (
        df["irrigation_coverage_pct"] * df["irrigation_reliability"] * 0.5 +
        df["rainfall_adequacy"] * 50
    ).clip(0, 100)

    # Soil quality composite
    df["soil_quality_index"] = (
        df["organic_carbon"] / 2.5 * 40 +
        (1 - np.abs(df["soil_ph"] - 6.5) / 3) * 30 +
        df["nutrient_balance"] * 30
    ).clip(0, 100)

    print(f"[DataAgent] Feature engineering done. Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENCODING & TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def encode_and_split(df: pd.DataFrame):
    """
    Label-encode categoricals, apply StandardScaler, split 70/15/15
    with temporal boundary (paper Section 4.3).
    Returns: X_train, X_val, X_test, y_train, y_val, y_test,
             feature_names, scaler, encoders
    """
    df = df.copy().sort_values("year").reset_index(drop=True)

    # Drop non-feature columns
    drop_cols = ["district", "crop_water_need"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target = "yield_q_ha"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols].values
    y = df[target].values

    # Temporal split
    n = len(df)
    n_train = int(0.70 * n)
    n_val   = int(0.85 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:], y[n_val:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"[DataAgent] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler, encoders


# ══════════════════════════════════════════════════════════════════════════════
# 6.  QUALITY REPORT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_quality_report(report: DataQualityReport):
    print("\n" + "═"*60)
    print("  DATA QUALITY REPORT")
    print("═"*60)

    grade_counts = pd.Series(report.quality_grades).value_counts()
    print(f"\n  Column Grades  →  A:{grade_counts.get('A',0)}  "
          f"B:{grade_counts.get('B',0)}  C:{grade_counts.get('C',0)}")

    print(f"\n  Imputation Log ({len(report.imputation_log)} columns affected):")
    for entry in report.imputation_log[:6]:
        print(f"    {entry['column']:<30} method={entry['method'][:30]}"
              f"  n={entry['n_imputed']}  confidence={entry['confidence']}")

    print(f"\n  Outlier Caps ({len(report.outlier_log)} columns capped):")
    for entry in report.outlier_log[:6]:
        print(f"    {entry['column']:<30} n_capped={entry['n_capped']}"
              f"  range=[{entry['low_cap']}, {entry['high_cap']}]")
    print("═"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — run this file standalone to verify Agent 1
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # generate_synthetic_dataset() auto-detects real data when data/ CSVs exist
    raw_df   = generate_synthetic_dataset(n_samples=5000)

    # Skip synthetic missing-value injection for real data
    # (the data has its own natural missingness handled by clean_data)
    if all(os.path.exists(f) for f in _REAL_DATA_FILES):
        dirty_df = raw_df
    else:
        dirty_df = inject_missing_values(raw_df)

    clean_df, qreport = clean_data(dirty_df)
    eng_df   = engineer_features(clean_df)
    print_quality_report(qreport)

    splits   = encode_and_split(eng_df)
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names, scaler, enc = splits

    print("Agent 1 complete ✓")
    print(f"Features used: {feat_names[:8]} … ({len(feat_names)} total)")
