"""
MA-XAI Framework — Agent 3: Causal Agent
==========================================
Responsibilities:
  • Build the 4-layer causal DAG (Figure 3.2 from paper)
  • Estimate Average Treatment Effects (ATE) for controllable variables
  • Generate counterfactual scenarios ("What if irrigation +40mm?")
  • Visualise the DAG and ATE results
  
Tools: DoWhy (causal inference), NetworkX (DAG), statsmodels (regression)
All open-source, no paid APIs.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CAUSAL DAG DEFINITION  (paper Figure 3.2 — 4-layer structure)
# ══════════════════════════════════════════════════════════════════════════════

CAUSAL_EDGES = [
    # Layer 1 (exogenous) → Layer 2 (environmental state)
    ("rainfall",          "soil_moisture"),
    ("temperature",       "soil_moisture"),
    ("rainfall",          "nutrient_availability"),
    ("soil_type",         "nutrient_availability"),
    ("soil_type",         "soil_moisture"),

    # Layer 2 → Layer 3 (controllable actions)
    ("soil_moisture",     "irrigation_applied"),
    ("nutrient_availability", "fertilizer_applied"),

    # Layer 1 → Layer 3 (direct)
    ("rainfall",          "sowing_date"),
    ("temperature",       "sowing_date"),

    # Layer 3 → Layer 4 (outcome)
    ("irrigation_applied",  "crop_growth"),
    ("fertilizer_applied",  "crop_growth"),
    ("sowing_date",         "crop_growth"),
    ("soil_moisture",       "crop_growth"),
    ("nutrient_availability","crop_growth"),
    ("temperature",         "crop_growth"),
    ("crop_growth",         "yield"),

    # Direct paths to yield
    ("irrigation_applied",  "yield"),
    ("fertilizer_applied",  "yield"),
    ("soil_moisture",       "yield"),
]

LAYER_COLORS = {
    "rainfall":             "#4e9af1",   # Layer 1 — exogenous
    "temperature":          "#4e9af1",
    "soil_type":            "#4e9af1",
    "soil_moisture":        "#5cb85c",   # Layer 2 — environmental state
    "nutrient_availability":"#5cb85c",
    "irrigation_applied":   "#f0ad4e",   # Layer 3 — controllable
    "fertilizer_applied":   "#f0ad4e",
    "sowing_date":          "#f0ad4e",
    "crop_growth":          "#9b59b6",   # Layer 4 — intermediate outcome
    "yield":                "#e74c3c",   # Layer 4 — final outcome
}

LAYER_LABELS = {
    1: "Layer 1: Exogenous (Climate & Soil)",
    2: "Layer 2: Environmental State",
    3: "Layer 3: Controllable Actions",
    4: "Layer 4: Outcome",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CAUSAL AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class CausalAgent:
    """
    Builds causal DAG, estimates ATEs, generates counterfactuals.
    Uses regression-based causal inference (no data randomisation needed).
    """

    def __init__(self):
        self.dag        = None
        self.ate_table  = None
        self._cf_model  = None   # outcome model for counterfactuals
        self._cf_scaler = None
        self._cf_features = None

    # ── 1. Build DAG ──────────────────────────────────────────────────────

    def build_dag(self):
        G = nx.DiGraph()
        G.add_edges_from(CAUSAL_EDGES)
        self.dag = G
        print(f"[CausalAgent] DAG built — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # ── 2. Estimate ATE via linear regression with confounders ────────────

    def estimate_ate(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        For each controllable variable (irrigation, fertilizer, sowing_week),
        estimate ATE on yield using regression adjustment (backdoor criterion).
        
        ATE(T) = E[Y | do(T=t_high)] - E[Y | do(T=t_low)]
        """
        print("[CausalAgent] Estimating ATEs …")

        TREATMENTS = {
            "irrigation_coverage_pct": {
                "confounders": ["rainfall_annual", "soil_moisture", "temp_mean"],
                "t_low":  20.0,   # 20% coverage  (rainfed)
                "t_high": 80.0,   # 80% coverage  (irrigated)
                "unit": "%",
                "label": "Irrigation Coverage",
            },
            "npk_dosage_kg_ha": {
                "confounders": ["soil_ph", "organic_carbon", "nitrogen_kg_ha"],
                "t_low":  60.0,   # low fertiliser
                "t_high": 180.0,  # recommended dose
                "unit": "kg/ha",
                "label": "NPK Fertiliser Dose",
            },
            "sowing_week": {
                "confounders": ["rainfall_annual", "temp_mean"],
                "t_low":  10,
                "t_high": 22,
                "unit": "week",
                "label": "Sowing Week (early vs. late)",
            },
        }

        rows = []
        for treatment, cfg in TREATMENTS.items():
            if treatment not in df_clean.columns:
                continue
            confounders = [c for c in cfg["confounders"] if c in df_clean.columns]
            cols = [treatment] + confounders + ["yield_q_ha"]
            sub  = df_clean[cols].dropna()

            # Outcome model: Y ~ T + confounders
            from sklearn.linear_model import LinearRegression
            Xreg = sub[[treatment] + confounders].values
            Yreg = sub["yield_q_ha"].values
            sc   = StandardScaler()
            Xreg = sc.fit_transform(Xreg)
            model = LinearRegression().fit(Xreg, Yreg)

            # Predict under do(T=t_low) and do(T=t_high)
            def predict_do(t_val):
                X_do = sub[[treatment] + confounders].copy()
                X_do[treatment] = t_val
                return model.predict(sc.transform(X_do.values)).mean()

            ate     = predict_do(cfg["t_high"]) - predict_do(cfg["t_low"])
            ate_pct = ate / sub["yield_q_ha"].mean() * 100

            # Bootstrap CI (100 reps)
            ates = []
            rng  = np.random.default_rng(42)
            for _ in range(100):
                idx   = rng.integers(0, len(sub), len(sub))
                boot  = sub.iloc[idx]
                Xb    = boot[[treatment] + confounders].values
                Yb    = boot["yield_q_ha"].values
                Xb    = sc.transform(Xb)
                mb    = LinearRegression().fit(Xb, Yb)
                X_lo  = boot[[treatment] + confounders].copy(); X_lo[treatment] = cfg["t_low"]
                X_hi  = boot[[treatment] + confounders].copy(); X_hi[treatment] = cfg["t_high"]
                ates.append(mb.predict(sc.transform(X_hi.values)).mean() -
                             mb.predict(sc.transform(X_lo.values)).mean())
            ci_lo, ci_hi = np.percentile(ates, [2.5, 97.5])

            rows.append({
                "treatment":  cfg["label"],
                "t_low":      f"{cfg['t_low']} {cfg['unit']}",
                "t_high":     f"{cfg['t_high']} {cfg['unit']}",
                "ATE (q/ha)": round(ate, 3),
                "ATE (%)":    round(ate_pct, 2),
                "CI_low":     round(ci_lo, 3),
                "CI_high":    round(ci_hi, 3),
            })

            print(f"  {cfg['label']:<35} ATE={ate:+.3f} q/ha "
                  f"({ate_pct:+.1f}%)  95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

        self.ate_table = pd.DataFrame(rows)
        return self.ate_table

    # ── 3. Fit counterfactual outcome model ───────────────────────────────

    def fit_counterfactual_model(self, df_clean: pd.DataFrame,
                                  feature_cols: list, scaler):
        """Fit a RF outcome model for counterfactual queries."""
        valid_features = [f for f in feature_cols if f in df_clean.columns]
        X = df_clean[valid_features].fillna(df_clean[valid_features].median()).values
        y = df_clean["yield_q_ha"].values
        self._cf_model    = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        self._cf_scaler   = StandardScaler()
        X_sc = self._cf_scaler.fit_transform(X)
        self._cf_model.fit(X_sc, y)
        self._cf_features = valid_features
        print("[CausalAgent] Counterfactual model fitted ✓")

    def counterfactual_query(self, instance_dict: dict,
                              intervention: dict) -> dict:
        """
        Answer: "What yield would be expected if <intervention>?"
        
        instance_dict: {feature_name: value} for a specific farm
        intervention:  {feature_name: new_value}  e.g. {"irrigation_coverage_pct": 80}
        """
        # Factual prediction
        fact = {f: instance_dict.get(f, 0) for f in self._cf_features}
        cf   = fact.copy()
        cf.update(intervention)

        X_fact = np.array([[fact[f] for f in self._cf_features]])
        X_cf   = np.array([[cf[f]   for f in self._cf_features]])

        y_fact = float(self._cf_model.predict(self._cf_scaler.transform(X_fact))[0])
        y_cf   = float(self._cf_model.predict(self._cf_scaler.transform(X_cf))[0])

        result = {
            "factual_yield":        round(y_fact, 2),
            "counterfactual_yield": round(y_cf, 2),
            "delta_yield":          round(y_cf - y_fact, 2),
            "delta_pct":            round((y_cf - y_fact) / y_fact * 100, 1),
            "intervention":         intervention,
        }
        return result

    # ── 4. Visualise DAG ──────────────────────────────────────────────────

    def plot_dag(self, save_path: str = "causal_dag.png"):
        G = self.dag
        fig, ax = plt.subplots(figsize=(14, 9))

        # Manual layout (4 layers)
        pos = {
            # Layer 1
            "rainfall":             (0, 3.5),
            "temperature":          (0, 2.0),
            "soil_type":            (0, 0.5),
            # Layer 2
            "soil_moisture":        (3, 3.0),
            "nutrient_availability":(3, 1.0),
            # Layer 3
            "irrigation_applied":   (6, 4.0),
            "fertilizer_applied":   (6, 2.0),
            "sowing_date":          (6, 0.5),
            # Layer 4
            "crop_growth":          (9, 2.5),
            "yield":                (12, 2.0),
        }

        node_colors = [LAYER_COLORS.get(n, "#aaaaaa") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200,
                                alpha=0.92, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold",
                                 font_color="white", ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20,
                                edge_color="#555555", width=1.5,
                                connectionstyle="arc3,rad=0.05", ax=ax)

        # Legend
        handles = [
            mpatches.Patch(color="#4e9af1", label="Layer 1 — Exogenous"),
            mpatches.Patch(color="#5cb85c", label="Layer 2 — Environmental State"),
            mpatches.Patch(color="#f0ad4e", label="Layer 3 — Controllable Actions"),
            mpatches.Patch(color="#9b59b6", label="Layer 4 — Intermediate"),
            mpatches.Patch(color="#e74c3c", label="Layer 4 — Yield (Outcome)"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_title("MA-XAI — Causal DAG (4-Layer Structure)", fontsize=13, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[CausalAgent] DAG plot saved → {save_path}")

    # ── 5. Plot ATE results ───────────────────────────────────────────────

    def plot_ate(self, save_path: str = "causal_ate.png"):
        df = self.ate_table.copy()
        fig, ax = plt.subplots(figsize=(10, 5))

        x     = np.arange(len(df))
        bars  = ax.bar(x, df["ATE (q/ha)"], width=0.5, color=["#5b7be9","#5cb85c","#f0ad4e"],
                       edgecolor="black", alpha=0.85)
        ax.errorbar(x, df["ATE (q/ha)"],
                    yerr=[df["ATE (q/ha)"] - df["CI_low"],
                          df["CI_high"]    - df["ATE (q/ha)"]],
                    fmt="none", color="black", capsize=6, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["treatment"], fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("ATE (q/ha)")
        ax.set_title("Average Treatment Effects (ATE) with 95% CI", fontsize=12, fontweight="bold")
        for bar, row in zip(bars, df.itertuples()):
            ypos = bar.get_height() + (0.1 if bar.get_height() >= 0 else -0.4)
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{row._4:+.2f}\n({row._5:+.1f}%)", ha="center", va="bottom", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[CausalAgent] ATE plot saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from agents.agent1_data import (generate_synthetic_dataset, inject_missing_values,
                                    clean_data, engineer_features, encode_and_split)

    raw_df   = generate_synthetic_dataset(5000)
    dirty_df = inject_missing_values(raw_df)
    clean_df, _ = clean_data(dirty_df)
    eng_df   = engineer_features(clean_df)
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names, scaler, enc = encode_and_split(eng_df)

    agent3 = CausalAgent()
    agent3.build_dag()
    agent3.estimate_ate(clean_df)
    agent3.plot_dag(save_path="causal_dag.png")
    agent3.plot_ate(save_path="causal_ate.png")

    # Fit CF model on clean (pre-encode) data — use numeric subset
    num_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
    if "yield_q_ha" in num_cols: num_cols.remove("yield_q_ha")
    agent3.fit_counterfactual_model(clean_df, num_cols, scaler)

    # Example counterfactual query
    sample_farm = clean_df[num_cols].iloc[0].to_dict()
    query = {"irrigation_coverage_pct": 80.0}  # intervention: increase irrigation
    cf_result = agent3.counterfactual_query(sample_farm, query)
    print(f"\n[CausalAgent] Counterfactual Query:")
    print(f"  Intervention: {cf_result['intervention']}")
    print(f"  Factual yield:         {cf_result['factual_yield']} q/ha")
    print(f"  Counterfactual yield:  {cf_result['counterfactual_yield']} q/ha")
    print(f"  Expected change:       {cf_result['delta_yield']:+.2f} q/ha  ({cf_result['delta_pct']:+.1f}%)")
    print("\nAgent 3 complete ✓")
