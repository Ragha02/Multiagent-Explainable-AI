"""
MA-XAI Framework — Orchestrator
=================================
Wires all 5 agents into the single pipeline described in Figure 3.1:

  Data Agent → Prediction Agent → Causal Agent → Explanation Agent → Advisory Agent

Run this file to execute the full framework end-to-end and produce
all outputs and visualisations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os, sys

# ── Agent imports ─────────────────────────────────────────────────────────────
from agents.agent1_data        import (generate_synthetic_dataset, inject_missing_values,
                                       clean_data, engineer_features, encode_and_split,
                                       print_quality_report, _REAL_DATA_FILES)
from agents.agent2_prediction  import PredictionAgent
from agents.agent3_causal      import CausalAgent
from agents.agent4_explanation import ExplanationAgent
from agents.agent5_advisory    import AdvisoryAgent

OUTPUT_DIR = "ma_xai_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class MAXAIOrchestrator:
    """
    Central controller: instantiates, sequences, and monitors all 5 agents.
    """

    def __init__(self, n_samples: int = 5000, seed: int = 42):
        self.n_samples = n_samples
        self.seed      = seed
        self.timings   = {}

        # Agents
        self.agent1_data    = None
        self.agent2_pred    = None
        self.agent3_causal  = None
        self.agent4_explain = None
        self.agent5_advisory= None

        # Data artefacts
        self.clean_df   = None
        self.eng_df     = None
        self.feat_names = None
        self.scaler     = None
        self.enc        = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.num_cols   = None

    def _timed(self, label, fn, *args, **kwargs):
        t0  = time.time()
        res = fn(*args, **kwargs)
        self.timings[label] = round(time.time() - t0, 2)
        print(f"   ⏱  {label}: {self.timings[label]}s")
        return res

    # ── PHASE 1 — Data Agent ─────────────────────────────────────────────────

    def run_phase1(self):
        print("\n" + "═"*65)
        print("  PHASE 1 — DATA AGENT")
        print("═"*65)

        raw_df = self._timed("generate", generate_synthetic_dataset, self.n_samples)

        # Skip synthetic noise injection when real Kaggle data is loaded
        if all(os.path.exists(f) for f in _REAL_DATA_FILES):
            dirty_df = raw_df   # real data has natural missingness; clean_data handles it
        else:
            dirty_df = self._timed("inject_missing", inject_missing_values, raw_df)

        self.clean_df, qreport = self._timed("clean", clean_data, dirty_df)
        self.eng_df  = self._timed("feature_eng", engineer_features, self.clean_df)
        print_quality_report(qreport)

        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test,
         self.feat_names, self.scaler, self.enc
        ) = self._timed("encode_split", encode_and_split, self.eng_df)

        self.num_cols = [c for c in self.clean_df.select_dtypes(include=np.number).columns
                         if c != "yield_q_ha"]

    # ── PHASE 2 — Prediction Agent ───────────────────────────────────────────

    def run_phase2(self):
        print("\n" + "═"*65)
        print("  PHASE 2 — PREDICTION AGENT")
        print("═"*65)
        self.agent2_pred = PredictionAgent(self.feat_names, self.seed)
        self._timed("train", self.agent2_pred.train,
                    self.X_train, self.y_train, self.X_val, self.y_val)
        results = self.agent2_pred.evaluate_all(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.X_test, self.y_test
        )
        self._timed("plot_prediction",
                    self.agent2_pred.plot_results,
                    self.X_test, self.y_test,
                    save_path=f"{OUTPUT_DIR}/prediction_results.png")
        return results

    # ── PHASE 3 — Causal Agent ───────────────────────────────────────────────

    def run_phase3(self):
        print("\n" + "═"*65)
        print("  PHASE 3 — CAUSAL AGENT")
        print("═"*65)
        self.agent3_causal = CausalAgent()
        self._timed("build_dag",  self.agent3_causal.build_dag)
        self._timed("estimate_ate", self.agent3_causal.estimate_ate, self.clean_df)
        self._timed("fit_cf_model", self.agent3_causal.fit_counterfactual_model,
                    self.clean_df, self.num_cols, self.scaler)
        self._timed("plot_dag", self.agent3_causal.plot_dag,
                    f"{OUTPUT_DIR}/causal_dag.png")
        self._timed("plot_ate", self.agent3_causal.plot_ate,
                    f"{OUTPUT_DIR}/causal_ate.png")

    # ── PHASE 4 — Explanation Agent ──────────────────────────────────────────

    def run_phase4(self):
        print("\n" + "═"*65)
        print("  PHASE 4 — EXPLANATION AGENT")
        print("═"*65)
        self.agent4_explain = ExplanationAgent(
            self.agent2_pred, self.agent3_causal,
            self.X_train, self.feat_names,
            self.scaler, self.clean_df
        )
        self.agent4_explain.global_explanation(10)
        self.agent4_explain.causal_explanation()
        self._timed("plot_explanations", self.agent4_explain.plot_all_explanations,
                    self.X_test[10], self.X_test[5], self.X_test[20],
                    f"{OUTPUT_DIR}/explanations.png")

        # Explanation card for one sample
        sample_dict = self.clean_df[self.num_cols].iloc[10].to_dict()
        card = self.agent4_explain.generate_explanation_card(
            X_instance     = self.X_test[10],
            instance_label = "Warangal · Kharif Rice · 2022",
            instance_dict  = sample_dict,
            intervention   = {"irrigation_coverage_pct": 80.0},
        )
        print(card)

    # ── PHASE 5 — Advisory Agent ─────────────────────────────────────────────

    def run_phase5(self):
        print("\n" + "═"*65)
        print("  PHASE 5 — ADVISORY AGENT")
        print("═"*65)
        self.agent5_advisory = AdvisoryAgent(
            self.agent2_pred, self.agent3_causal, self.agent4_explain
        )

        # Demo farm (stress scenario — rich advisory output)
        stress_farm = {
            "irrigation_coverage_pct": 18.0,
            "rainfall_annual":          420.0,
            "soil_moisture":            22.0,
            "soil_ph":                   5.2,
            "organic_carbon":            0.22,
            "npk_dosage_kg_ha":         55.0,
            "temp_mean":                39.5,
            "variety_improved":          0,
            "sowing_week":               8,
            "prev_year_yield":          18.0,
            **{f: float(self.clean_df[f].median())
               for f in self.num_cols
               if f not in ["irrigation_coverage_pct","rainfall_annual","soil_moisture",
                              "soil_ph","organic_carbon","npk_dosage_kg_ha","temp_mean",
                              "variety_improved","sowing_week","prev_year_yield"]}
        }

        advisory = self.agent5_advisory.generate_advisory(
            self.X_test[0], stress_farm, "Nalgonda · Kharif Bajra · 2023"
        )
        self.agent5_advisory.print_advisory(advisory)
        chain = self.agent5_advisory.traceability_chain(advisory, stress_farm)
        print(chain)
        self._timed("plot_dashboard", self.agent5_advisory.plot_advisory_dashboard,
                    advisory, f"{OUTPUT_DIR}/advisory_dashboard.png")

    # ── SUMMARY TIMING REPORT ────────────────────────────────────────────────

    def timing_report(self):
        print("\n" + "═"*50)
        print("  PIPELINE TIMING SUMMARY")
        print("═"*50)
        total = sum(self.timings.values())
        for step, secs in self.timings.items():
            bar = "█" * int(secs / total * 30)
            print(f"  {step:<25} {secs:5.2f}s  {bar}")
        print(f"  {'TOTAL':<25} {total:.2f}s")
        print("═"*50)

    # ── MASTER COMPOSITE FIGURE ──────────────────────────────────────────────

    def create_master_figure(self):
        """One-page summary figure showing all key results."""
        import matplotlib.image as mpimg

        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        panels = [
            (f"{OUTPUT_DIR}/prediction_results.png", "Agent 2 — Prediction Results"),
            (f"{OUTPUT_DIR}/causal_dag.png",          "Agent 3 — Causal DAG"),
            (f"{OUTPUT_DIR}/causal_ate.png",           "Agent 3 — Average Treatment Effects"),
            (f"{OUTPUT_DIR}/explanations.png",         "Agent 4 — All Explanation Types"),
            (f"{OUTPUT_DIR}/advisory_dashboard.png",   "Agent 5 — Advisory Dashboard"),
        ]

        for i, (path, title) in enumerate(panels):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            try:
                img = mpimg.imread(path)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, f"[{title}]", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.axis("off")

        # Last panel: timing summary text
        ax_last = fig.add_subplot(gs[1, 2])
        ax_last.axis("off")
        timing_text = "Pipeline Timing\n" + "─"*22 + "\n"
        total = sum(self.timings.values())
        for k, v in self.timings.items():
            timing_text += f"{k[:20]:<22} {v:.1f}s\n"
        timing_text += "─"*22 + f"\nTotal: {total:.1f}s"
        ax_last.text(0.05, 0.95, timing_text, transform=ax_last.transAxes,
                     fontsize=8.5, verticalalignment="top",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

        plt.suptitle("MA-XAI Framework — Full Pipeline Results",
                     fontsize=16, fontweight="bold", y=0.995)
        path = f"{OUTPUT_DIR}/MA_XAI_Master_Figure.png"
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n[Orchestrator] Master figure saved → {path}")
        return path

    # ── RUN ALL ──────────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "█"*65)
        print("  MA-XAI MULTI-AGENT EXPLAINABLE AI FRAMEWORK")
        print("  Crop Yield Prediction & Agricultural Decision Support")
        print("█"*65)

        self.run_phase1()
        self.run_phase2()
        self.run_phase3()
        self.run_phase4()
        self.run_phase5()
        self.timing_report()
        master_path = self.create_master_figure()

        print("\n" + "█"*65)
        print("  FRAMEWORK COMPLETE — ALL 5 AGENTS OPERATIONAL")
        print(f"  Outputs in: {OUTPUT_DIR}/")
        print("█"*65 + "\n")
        return master_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    orchestrator = MAXAIOrchestrator(n_samples=5000)
    orchestrator.run()
