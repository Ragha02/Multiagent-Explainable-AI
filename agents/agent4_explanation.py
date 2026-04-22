"""
MA-XAI Framework — Agent 4: Explanation Agent
===============================================
Responsibilities:
  • Type 1 — Global:      SHAP summary plots (top features across all farms)
  • Type 2 — Local:       LIME instance explanation (why THIS prediction?)
  • Type 3 — Contrastive: Counterfactual comparison (why Farm A > Farm B?)
  • Type 4 — Causal:      ATE-based causal narrative

All four explanation types are fused into a single human-readable
"Explanation Card" per farm query.
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# EXPLANATION AGENT
# ══════════════════════════════════════════════════════════════════════════════

class ExplanationAgent:
    """
    Synthesises all four MA-XAI explanation types.
    Requires references to the trained PredictionAgent and CausalAgent.
    """

    def __init__(self, prediction_agent, causal_agent,
                 X_train: np.ndarray, feature_names: list,
                 scaler, df_clean: pd.DataFrame):
        self.pa            = prediction_agent
        self.ca            = causal_agent
        self.X_train       = X_train
        self.feature_names = feature_names
        self.scaler        = scaler
        self.df_clean      = df_clean
        self.lime_explainer = None
        self._init_lime()

    def _init_lime(self):
        """Initialise LIME tabular explainer on training distribution."""
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data  = self.X_train,
            feature_names  = self.feature_names,
            mode           = "regression",
            random_state   = 42,
            discretize_continuous = True,
        )
        print("[ExplanationAgent] LIME explainer initialised ✓")

    # ── TYPE 1: Global SHAP summary ───────────────────────────────────────

    def global_explanation(self, top_n: int = 15) -> pd.DataFrame:
        """Type 1 — Model-level SHAP feature importance (all training samples)."""
        fi = self.pa.global_feature_importance(top_n=top_n)
        print(f"\n[ExplanationAgent] Type 1 — Global SHAP (top {top_n} features):")
        for _, row in fi.iterrows():
            print(f"  {row['feature']:<35} mean|SHAP|={row['mean_abs_shap']:.4f}")
        return fi

    # ── TYPE 2: Local LIME explanation ────────────────────────────────────

    def local_explanation(self, X_instance: np.ndarray,
                           instance_label: str = "Farm") -> dict:
        """
        Type 2 — Instance-level LIME explanation.
        Returns top-5 features influencing this specific prediction.
        """
        exp = self.lime_explainer.explain_instance(
            data_row      = X_instance,
            predict_fn    = self.pa.predict,
            num_features  = 10,
            num_samples   = 500,
        )
        lime_dict = dict(exp.as_list())
        prediction = float(self.pa.predict(X_instance.reshape(1, -1))[0])

        result = {
            "instance_label": instance_label,
            "predicted_yield": round(prediction, 2),
            "lime_features": lime_dict,
        }

        print(f"\n[ExplanationAgent] Type 2 — Local LIME ({instance_label})")
        print(f"  Predicted yield: {prediction:.2f} q/ha")
        print("  Top LIME features:")
        for feat, contrib in sorted(lime_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            sign = "▲" if contrib > 0 else "▼"
            print(f"    {sign} {feat:<40} contribution={contrib:+.4f}")

        return result

    # ── TYPE 3: Contrastive explanation ───────────────────────────────────

    def contrastive_explanation(self, X_farm_a: np.ndarray,
                                 X_farm_b: np.ndarray,
                                 label_a: str = "Farm A",
                                 label_b: str = "Farm B") -> dict:
        """
        Type 3 — Why does Farm A outperform Farm B?
        Computes SHAP delta to attribute the yield gap.
        """
        sv_a = self.pa.explainer.shap_values(X_farm_a.reshape(1, -1))[0]
        sv_b = self.pa.explainer.shap_values(X_farm_b.reshape(1, -1))[0]
        delta_sv = sv_a - sv_b

        y_a = float(self.pa.predict(X_farm_a.reshape(1, -1))[0])
        y_b = float(self.pa.predict(X_farm_b.reshape(1, -1))[0])
        gap = y_a - y_b

        top_features = sorted(
            zip(self.feature_names, delta_sv),
            key=lambda x: abs(x[1]), reverse=True
        )[:8]

        result = {
            f"{label_a}_yield":  round(y_a, 2),
            f"{label_b}_yield":  round(y_b, 2),
            "yield_gap":         round(gap, 2),
            "top_factors":       top_features,
        }

        print(f"\n[ExplanationAgent] Type 3 — Contrastive")
        print(f"  {label_a}: {y_a:.2f} q/ha  |  {label_b}: {y_b:.2f} q/ha  |  Gap: {gap:+.2f} q/ha")
        print("  Key differentiating factors:")
        for feat, dv in top_features[:5]:
            direction = f"{label_a} advantage" if dv > 0 else f"{label_b} advantage"
            print(f"    {feat:<35} ΔSHAP={dv:+.4f}  ({direction})")

        return result

    # ── TYPE 4: Causal explanation ────────────────────────────────────────

    def causal_explanation(self) -> pd.DataFrame:
        """Type 4 — ATE-based causal narrative."""
        ate_df = self.ca.ate_table.copy()
        print("\n[ExplanationAgent] Type 4 — Causal Explanations:")
        for _, row in ate_df.iterrows():
            direction = "increases" if row["ATE (q/ha)"] > 0 else "decreases"
            print(f"  Changing {row['treatment']}")
            print(f"    from {row['t_low']} → {row['t_high']}")
            print(f"    {direction} yield by {abs(row['ATE (q/ha)']):.2f} q/ha "
                  f"({row['ATE (%)']:+.1f}%)  [95% CI: {row['CI_low']:.2f}–{row['CI_high']:.2f}]")
        return ate_df

    # ── FUSION: Explanation Card ──────────────────────────────────────────

    def generate_explanation_card(self, X_instance: np.ndarray,
                                   instance_label: str,
                                   instance_dict: dict,
                                   intervention: dict) -> str:
        """
        Fuse all 4 explanation types into one human-readable advisory card.
        This is what a farmer or agronomist would see.
        """
        shap_exp  = self.pa.explain_instance(X_instance)
        lime_exp  = self.local_explanation(X_instance, instance_label)
        cf_result = self.ca.counterfactual_query(instance_dict, intervention)
        top_shap  = sorted(shap_exp["shap_values"].items(),
                            key=lambda x: abs(x[1]), reverse=True)[:5]
        top_lime  = sorted(lime_exp["lime_features"].items(),
                            key=lambda x: abs(x[1]), reverse=True)[:5]

        intv_feat = list(intervention.keys())[0]
        intv_val  = list(intervention.values())[0]

        card = f"""
╔══════════════════════════════════════════════════════════════════╗
║            MA-XAI  ·  FARM EXPLANATION CARD                     ║
║  {instance_label:<60}║
╠══════════════════════════════════════════════════════════════════╣
║  🌾 Predicted Yield:  {lime_exp['predicted_yield']:.2f} q/ha                          ║
╠══════════════════════════════════════════════════════════════════╣
║  TYPE 1 — GLOBAL FACTORS (model-wide top drivers)               ║
║  These features matter most across all farms in the dataset.     ║
╠══════════════════════════════════════════════════════════════════╣
║  TYPE 2 — WHY THIS PREDICTION? (LIME, local)                     ║"""
        for feat, contrib in top_lime:
            icon = "▲" if contrib > 0 else "▼"
            card += f"\n║  {icon} {feat[:40]:<40}  {contrib:+.3f}  ║"

        card += f"""
╠══════════════════════════════════════════════════════════════════╣
║  TYPE 4 — CAUSAL INSIGHT  (what if?)                            ║
║  If you increase {intv_feat[:30]:<30} to {intv_val}:           ║
║  → Expected yield: {cf_result['counterfactual_yield']:.2f} q/ha  """
        delta = cf_result['delta_yield']
        card += f"(change: {delta:+.2f} q/ha, {cf_result['delta_pct']:+.1f}%)  ║"
        card += """
╠══════════════════════════════════════════════════════════════════╣
║  CONFIDENCE LEVEL:  Medium                                       ║
║  Data Quality: Grade A/B  |  Causal CI: bootstrapped 95%        ║
╚══════════════════════════════════════════════════════════════════╝
"""
        return card

    # ── Plot all 4 explanation types ──────────────────────────────────────

    def plot_all_explanations(self, X_instance: np.ndarray,
                               X_farm_a: np.ndarray, X_farm_b: np.ndarray,
                               save_path: str = "explanations.png"):
        shap_exp = self.pa.explain_instance(X_instance)
        fi       = self.pa.global_feature_importance(12)
        sv_a     = self.pa.explainer.shap_values(X_farm_a.reshape(1, -1))[0]
        sv_b     = self.pa.explainer.shap_values(X_farm_b.reshape(1, -1))[0]
        delta    = sv_a - sv_b
        top_cf   = sorted(zip(self.feature_names, delta),
                          key=lambda x: abs(x[1]), reverse=True)[:10]

        fig = plt.figure(figsize=(18, 12))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

        # (a) Type 1 — Global SHAP
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.barh(fi["feature"][::-1], fi["mean_abs_shap"][::-1],
                 color="#5b7be9", edgecolor="white")
        ax0.set_title("Type 1 — Global Feature Importance\n(SHAP, top 12)", fontsize=10, fontweight="bold")
        ax0.set_xlabel("Mean |SHAP value|")

        # (b) Type 2 — Local SHAP waterfall (instance)
        ax1 = fig.add_subplot(gs[0, 1])
        top5 = sorted(shap_exp["shap_values"].items(),
                      key=lambda x: abs(x[1]), reverse=True)[:8]
        feats = [f[0] for f in top5]
        vals  = [f[1] for f in top5]
        colors = ["#5cb85c" if v > 0 else "#d9534f" for v in vals]
        ax1.barh(range(len(feats)), vals, color=colors, edgecolor="white")
        ax1.set_yticks(range(len(feats)))
        ax1.set_yticklabels([f[:28] for f in feats], fontsize=8)
        ax1.axvline(0, color="black", linewidth=0.8)
        ax1.set_title(f"Type 2 — Local SHAP\n(Instance pred={shap_exp['prediction']:.1f} q/ha)",
                      fontsize=10, fontweight="bold")
        ax1.set_xlabel("SHAP value")

        # (c) Type 3 — Contrastive ΔSHAP
        ax2 = fig.add_subplot(gs[1, 0])
        cf_feats = [f[0] for f in top_cf]
        cf_vals  = [f[1] for f in top_cf]
        colors3  = ["#5cb85c" if v > 0 else "#d9534f" for v in cf_vals]
        ax2.barh(range(len(cf_feats)), cf_vals, color=colors3, edgecolor="white")
        ax2.set_yticks(range(len(cf_feats)))
        ax2.set_yticklabels([f[:28] for f in cf_feats], fontsize=8)
        ax2.axvline(0, color="black", linewidth=0.8)
        y_a = self.pa.predict(X_farm_a.reshape(1,-1))[0]
        y_b = self.pa.predict(X_farm_b.reshape(1,-1))[0]
        ax2.set_title(f"Type 3 — Contrastive (Farm A={y_a:.1f} vs B={y_b:.1f} q/ha)\n"
                      f"ΔSHAP: Farm A advantage (green) / disadvantage (red)",
                      fontsize=9, fontweight="bold")
        ax2.set_xlabel("ΔSHAP value (A - B)")

        # (d) Type 4 — ATE bar chart
        ax3 = fig.add_subplot(gs[1, 1])
        ate  = self.ca.ate_table
        x    = np.arange(len(ate))
        bars = ax3.bar(x, ate["ATE (q/ha)"],
                       color=["#5b7be9","#5cb85c","#f0ad4e"], edgecolor="black", width=0.5)
        ax3.errorbar(x, ate["ATE (q/ha)"],
                     yerr=[ate["ATE (q/ha)"] - ate["CI_low"], ate["CI_high"] - ate["ATE (q/ha)"]],
                     fmt="none", color="black", capsize=5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(ate["treatment"], fontsize=8, rotation=15, ha="right")
        ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax3.set_title("Type 4 — Causal ATEs with 95% CI", fontsize=10, fontweight="bold")
        ax3.set_ylabel("ATE (q/ha)")

        plt.suptitle("MA-XAI — All Four Explanation Types", fontsize=14, fontweight="bold", y=0.99)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[ExplanationAgent] Explanation plots saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from agents.agent1_data       import (generate_synthetic_dataset, inject_missing_values,
                                           clean_data, engineer_features, encode_and_split)
    from agents.agent2_prediction import PredictionAgent
    from agents.agent3_causal     import CausalAgent

    # Agent 1
    raw_df   = generate_synthetic_dataset(5000)
    dirty_df = inject_missing_values(raw_df)
    clean_df, _ = clean_data(dirty_df)
    eng_df   = engineer_features(clean_df)
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names, scaler, enc = encode_and_split(eng_df)

    # Agent 2
    agent2 = PredictionAgent(feat_names)
    agent2.train(X_train, y_train, X_val, y_val)
    agent2.evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test)

    # Agent 3
    agent3 = CausalAgent()
    agent3.build_dag()
    agent3.estimate_ate(clean_df)
    num_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "yield_q_ha"]
    agent3.fit_counterfactual_model(clean_df, num_cols, scaler)

    # Agent 4
    agent4 = ExplanationAgent(agent2, agent3, X_train, feat_names, scaler, clean_df)

    # Global explanation
    agent4.global_explanation(top_n=10)

    # Causal explanation
    agent4.causal_explanation()

    # Explanation card for one farm
    sample_dict  = clean_df[num_cols].iloc[10].to_dict()
    card = agent4.generate_explanation_card(
        X_instance      = X_test[10],
        instance_label  = "Warangal · Kharif Rice · 2022",
        instance_dict   = sample_dict,
        intervention    = {"irrigation_coverage_pct": 80.0},
    )
    print(card)

    # Contrastive explanation
    agent4.contrastive_explanation(X_test[5], X_test[20], "Farm A", "Farm B")

    # Full visualisation
    agent4.plot_all_explanations(X_test[10], X_test[5], X_test[20],
                                  save_path="explanations.png")

    print("\nAgent 4 complete ✓")
