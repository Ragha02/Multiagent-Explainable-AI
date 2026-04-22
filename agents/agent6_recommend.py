"""
MA-XAI Framework — Agent 6: Crop Recommendation Agent
=======================================================
Responsibilities:
  • Train a multi-class classifier on Crop_recommendation.csv
  • Given soil + climate parameters, recommend top-3 crops with confidence
  • Provide expected yield range for each recommended crop from real data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Features used from Crop_recommendation.csv
RECOMMEND_FEATURES = ["N", "P", "K", "ph", "humidity", "rainfall", "temperature"]

# Friendly labels
FEAT_LABELS = {
    "N": "Nitrogen (kg/ha)",
    "P": "Phosphorus (kg/ha)",
    "K": "Potassium (kg/ha)",
    "ph": "Soil pH",
    "humidity": "Humidity (%)",
    "rainfall": "Rainfall (mm)",
    "temperature": "Temperature (°C)",
}

# Agronomy notes per crop (displayed on frontend)
CROP_NOTES = {
    "Rice":       "High water demand (>1200mm). Best in Kharif. Heavy clay soils preferred.",
    "Wheat":      "Rabi crop. Requires cool dry weather. pH 6.0–7.5 optimal.",
    "Maize":      "Versatile — Kharif/Rabi. Well-drained loamy soils. Moderate NPK.",
    "Soybean":    "Nitrogen-fixing legume. Semi-arid zones. pH 6.0–6.8.",
    "Cotton":     "Long duration Kharif. Deep black soils. High irrigation needs.",
    "Sugarcane":  "12–18 month crop. Very high water (1800mm). Rich heavy soils.",
    "Groundnut":  "Drought-tolerant legume. Sandy loam. pH 6.0–6.5.",
    "Bajra":      "Most drought-tolerant cereal. Arid/semi-arid zones. Low NPK.",
    "Jowar":      "Dual-purpose (grain+fodder). Semi-arid. Moderate water needs.",
    "Mustard":    "Rabi oilseed. Cool dry climate. Low water requirement.",
    "Chickpea":   "Rabi legume. Nitrogen-fixing. Low water. pH 6.0–8.0.",
    "Lentil":     "Rabi pulse. Cool temperatures. Drought-tolerant.",
    "Mungbean":   "Short-duration Kharif/Zaid. Sandy loam. Low NPK.",
    "Blackgram":  "Kharif pulse. Loamy soils. Moderate rainfall.",
    "Pomegranate":"Deep well-drained soils. Semi-arid. Drip irrigation profitable.",
    "Banana":     "Tropical perennial. High NPK demand. >1200mm rainfall.",
    "Grapes":     "Well-drained sandy loam. Dry climate. Trellis system needed.",
    "Watermelon": "Sandy soils. Hot dry climate. Short Zaid crop.",
    "Muskmelon":  "Light sandy soils. Dry hot Zaid. Low water after fruit set.",
    "Apple":      "Temperate zones only (Kashmir, HP). Chill hours required.",
    "Orange":     "Sub-tropical. Well-drained loamy. pH 5.5–7.",
    "Papaya":     "Year-round tropical. Well-drained. Waterlogging fatal.",
    "Coconut":    "Coastal humid zones. Sandy loam. High humidity.",
    "Cotton":     "Black heavy soils. Kharif. High irrigation/NPK.",
    "Jute":       "Alluvial river deltas. High rainfall (>1500mm). Kharif.",
    "Coffee":     "Shaded hill slopes. Acidic pH 5.0–6.5. High humidity.",
}


class CropRecommendAgent:
    """
    Trains RF + GB ensemble on Crop_recommendation.csv.
    Predicts top-k crops with probability scores.
    Also computes expected yield ranges from the real production data.
    """

    def __init__(self):
        self.rf_model  = None
        self.gb_model  = None
        self.scaler    = StandardScaler()
        self.encoder   = LabelEncoder()
        self.classes_  = []
        self.yield_stats: dict = {}   # crop → {p25, median, p75} from real data

    def train(self, recommend_csv_path: str, clean_df: pd.DataFrame = None):
        """
        Train classifier on Crop_recommendation.csv.
        Optionally compute yield stats from the real production clean_df.
        """
        df = pd.read_csv(recommend_csv_path)
        df = df.rename(columns={"label": "crop"})

        # Use available features only
        feat_cols = [f for f in RECOMMEND_FEATURES if f in df.columns]
        X = df[feat_cols].values
        y = df["crop"].values

        X_scaled = self.scaler.fit_transform(X)
        y_enc    = self.encoder.fit_transform(y)
        self.classes_ = list(self.encoder.classes_)

        self.rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42
        )
        self.rf_model.fit(X_scaled, y_enc)
        self.gb_model.fit(X_scaled, y_enc)

        # Cross-val accuracy
        cv_rf = cross_val_score(self.rf_model, X_scaled, y_enc, cv=5, scoring="accuracy")
        print(f"[CropRecommendAgent] RF accuracy: {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")

        # Build yield stats from real production data
        if clean_df is not None and "crop" in clean_df.columns and "yield_q_ha" in clean_df.columns:
            stats = (
                clean_df.groupby("crop")["yield_q_ha"]
                .describe(percentiles=[0.25, 0.5, 0.75])
                .rename(columns={"25%": "p25", "50%": "median", "75%": "p75"})
                [["p25", "median", "p75", "count"]]
            )
            self.yield_stats = stats.to_dict(orient="index")
        print("[CropRecommendAgent] Training complete ✓")

    def recommend(self, params: dict, top_k: int = 3) -> list:
        """
        Return top-k crop recommendations with confidence and yield stats.

        params keys: N, P, K, ph, humidity, rainfall, temperature
        """
        feat_cols = [f for f in RECOMMEND_FEATURES
                     if f in params or f in ["N","P","K","ph","humidity","rainfall","temperature"]]

        # Map frontend-friendly keys if needed
        key_map = {
            "nitrogen_kg_ha": "N",
            "phosphorus_kg_ha": "P",
            "potassium_kg_ha": "K",
            "soil_ph": "ph",
            "humidity": "humidity",
            "rainfall_annual": "rainfall",
            "temp_mean": "temperature",
        }
        mapped = {key_map.get(k, k): v for k, v in params.items()}

        x_raw = np.array([[
            float(mapped.get("N",          80)),
            float(mapped.get("P",          40)),
            float(mapped.get("K",          40)),
            float(mapped.get("ph",          6.5)),
            float(mapped.get("humidity",   65)),
            float(mapped.get("rainfall",   700)),
            float(mapped.get("temperature", 27)),
        ]])

        x_scaled = self.scaler.transform(x_raw)

        # Ensemble probabilities (equal weight)
        prob_rf = self.rf_model.predict_proba(x_scaled)[0]
        prob_gb = self.gb_model.predict_proba(x_scaled)[0]
        prob    = (prob_rf + prob_gb) / 2

        top_idx = np.argsort(prob)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx):
            crop = self.classes_[idx]
            conf = float(prob[idx]) * 100
            ys   = self.yield_stats.get(crop, {})
            results.append({
                "rank":        rank + 1,
                "crop":        crop,
                "confidence":  round(conf, 1),
                "yield_p25":   round(float(ys.get("p25", 10)), 1),
                "yield_median": round(float(ys.get("median", 20)), 1),
                "yield_p75":   round(float(ys.get("p75", 35)), 1),
                "sample_size": int(ys.get("count", 0)),
                "note":        CROP_NOTES.get(crop, ""),
            })

        return results
