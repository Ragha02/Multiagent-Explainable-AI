"""
MA-XAI Framework — Agent 2: Prediction Agent
==============================================
Responsibilities:
  • Train Random Forest + XGBoost ensemble
  • Weighted ensemble prediction: ŷ = w_RF × ŷ_RF + w_XGB × ŷ_XGB
  • Compute SHAP values for every prediction (feature-level explanations)
  • Evaluate: RMSE, MAE, R², MAPE
  • Compare vs baselines (MLR, RF alone, XGBoost alone)
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor
from sklearn.metrics       import mean_squared_error, mean_absolute_error, r2_score
from xgboost               import XGBRegressor
from scipy.optimize        import minimize_scalar
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# METRICS HELPER
# ══════════════════════════════════════════════════════════════════════════════

def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, label="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    print(f"  [{label}]  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}  MAPE={mp:.2f}%")
    return {"model": label, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mp}


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class PredictionAgent:
    """
    Trains baselines + MA-XAI ensemble, computes SHAP explanations.
    Paper formula: ŷ = w_RF × ŷ_RF + w_XGB × ŷ_XGB
    Weights optimised on validation RMSE.
    """

    def __init__(self, feature_names: list, seed: int = 42):
        self.feature_names = feature_names
        self.seed          = seed
        self.rf            = None
        self.xgb           = None
        self.mlr           = None
        self.w_rf          = 0.40    # initial weight; refined on val set
        self.w_xgb         = 0.60
        self.shap_values_train = None
        self.explainer         = None
        self.results_df        = None

    # ── 1. Train all models ────────────────────────────────────────────────

    def train(self, X_train, y_train, X_val, y_val):
        print("\n[PredictionAgent] Training models …")

        # --- Baseline: MLR ---
        self.mlr = LinearRegression()
        self.mlr.fit(X_train, y_train)

        # --- Baseline: Random Forest (500 trees, paper spec) ---
        self.rf = RandomForestRegressor(
            n_estimators=500, max_depth=None,
            min_samples_leaf=5, n_jobs=-1, random_state=self.seed
        )
        self.rf.fit(X_train, y_train)

        # --- Baseline: XGBoost (300 estimators, paper spec) ---
        self.xgb = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=self.seed,
            eval_metric="rmse", verbosity=0
        )
        self.xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # --- Optimise ensemble weights on validation set ---
        self._optimise_weights(X_val, y_val)
        print(f"[PredictionAgent] Optimal weights → RF={self.w_rf:.3f}, XGB={self.w_xgb:.3f}")

        # --- SHAP values on training set (TreeExplainer) ---
        print("[PredictionAgent] Computing SHAP values (XGBoost) …")
        self.explainer = shap.TreeExplainer(self.xgb)
        self.shap_values_train = self.explainer.shap_values(X_train)

    def _optimise_weights(self, X_val, y_val):
        """Find w_RF that minimises val RMSE; w_XGB = 1 - w_RF."""
        rf_pred  = self.rf.predict(X_val)
        xgb_pred = self.xgb.predict(X_val)

        def neg_rmse(w):
            combo = w * rf_pred + (1 - w) * xgb_pred
            return np.sqrt(mean_squared_error(y_val, combo))

        res = minimize_scalar(neg_rmse, bounds=(0.1, 0.9), method="bounded")
        self.w_rf  = res.x
        self.w_xgb = 1.0 - res.x

    # ── 2. Predict (ensemble) ──────────────────────────────────────────────

    def predict(self, X):
        return self.w_rf * self.rf.predict(X) + self.w_xgb * self.xgb.predict(X)

    # ── 3. Explain a single prediction (SHAP) ─────────────────────────────

    def explain_instance(self, X_instance: np.ndarray) -> dict:
        """
        Return a dictionary of {feature: shap_value} for one sample.
        Used by the Explanation Agent (Agent 4).
        """
        sv = self.explainer.shap_values(X_instance.reshape(1, -1))[0]
        base = self.explainer.expected_value
        return {
            "base_value": float(base),
            "prediction": float(self.predict(X_instance.reshape(1, -1))[0]),
            "shap_values": dict(zip(self.feature_names, sv.tolist()))
        }

    # ── 4. Full evaluation vs baselines ───────────────────────────────────

    def evaluate_all(self, X_train, y_train, X_val, y_val, X_test, y_test) -> pd.DataFrame:
        print("\n[PredictionAgent] Evaluation Results (Test Set)")
        print("─" * 65)

        rows = []
        rows.append(evaluate(y_test, self.mlr.predict(X_test),      "MLR (baseline)"))
        rows.append(evaluate(y_test, self.rf.predict(X_test),       "Random Forest"))
        rows.append(evaluate(y_test, self.xgb.predict(X_test),      "XGBoost"))
        ensemble_pred = self.predict(X_test)
        rows.append(evaluate(y_test, ensemble_pred,                  "MA-XAI Ensemble"))
        print("─" * 65)

        self.results_df = pd.DataFrame(rows)
        return self.results_df

    # ── 5. SHAP feature importance (global) ───────────────────────────────

    def global_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        mean_abs = np.abs(self.shap_values_train).mean(axis=0)
        fi = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)
        return fi

    # ── 6. Visualise ──────────────────────────────────────────────────────

    def plot_results(self, X_test, y_test, save_path: str = "prediction_results.png"):
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

        # (a) Baseline comparison bar chart
        ax0 = fig.add_subplot(gs[0, 0])
        df  = self.results_df.set_index("model")
        colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
        df["RMSE"].plot(kind="bar", ax=ax0, color=colors, edgecolor="black", width=0.6)
        ax0.set_title("RMSE Comparison — All Models", fontsize=11, fontweight="bold")
        ax0.set_ylabel("RMSE (q/ha)")
        ax0.set_xlabel("")
        ax0.tick_params(axis="x", rotation=30)
        for bar, val in zip(ax0.patches, df["RMSE"]):
            ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        # (b) Actual vs Predicted
        ax1 = fig.add_subplot(gs[0, 1])
        ens_pred = self.predict(X_test)
        ax1.scatter(y_test, ens_pred, alpha=0.35, s=12, color="#5cb85c", label="Predictions")
        lims = [min(y_test.min(), ens_pred.min()), max(y_test.max(), ens_pred.max())]
        ax1.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
        ax1.set_title(f"Actual vs Predicted — MA-XAI Ensemble\n(R²={r2_score(y_test,ens_pred):.4f})",
                      fontsize=11, fontweight="bold")
        ax1.set_xlabel("Actual Yield (q/ha)")
        ax1.set_ylabel("Predicted Yield (q/ha)")
        ax1.legend(fontsize=8)

        # (c) SHAP global importance (top 15)
        ax2 = fig.add_subplot(gs[1, :])
        fi  = self.global_feature_importance(top_n=15)
        bars = ax2.barh(fi["feature"][::-1], fi["mean_abs_shap"][::-1],
                        color="#5b7be9", edgecolor="white")
        ax2.set_title("SHAP Feature Importance (Global) — Top 15", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Mean |SHAP value|")
        for bar, val in zip(bars, fi["mean_abs_shap"][::-1]):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=7.5)

        plt.suptitle("MA-XAI — Prediction Agent Results", fontsize=14, fontweight="bold", y=0.98)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PredictionAgent] Plot saved → {save_path}")


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

    agent2 = PredictionAgent(feature_names=feat_names)
    agent2.train(X_train, y_train, X_val, y_val)
    agent2.evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test)
    agent2.plot_results(X_test, y_test, save_path="prediction_results.png")

    # Demo: explain instance 0
    exp = agent2.explain_instance(X_test[0])
    top5 = sorted(exp["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"\n[PredictionAgent] Instance explanation  →  predicted={exp['prediction']:.2f} q/ha")
    print("  Top-5 SHAP contributors:")
    for feat, val in top5:
        sign = "▲" if val > 0 else "▼"
        print(f"    {sign} {feat:<35} SHAP={val:+.4f}")

    print("\nAgent 2 complete ✓")
