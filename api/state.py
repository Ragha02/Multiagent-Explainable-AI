"""
MA-XAI API — Global pipeline state singleton.
Holds trained agent instances and cached results so predictions
are fast (< 100ms) once the pipeline has been initialized.
"""
import asyncio
import threading
import numpy as np
import sys, os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PipelineState:
    """Thread-safe singleton holding all trained MA-XAI agents."""

    def __init__(self):
        self.status: str = "idle"       # idle | running | ready | error
        self.progress: int = 0
        self.current_step: str = "Not started"
        self.error: str | None = None

        # Trained agents
        self.agent2_pred = None
        self.agent3_causal = None
        self.agent4_explain = None
        self.agent5_advisory = None
        self.agent6_recommend = None

        # Data artefacts
        self.feat_names: list = []
        self.scaler = None
        self.clean_df = None
        self.num_cols: list = []

        # Cached results
        self.model_metrics: dict = {}
        self.ate_table: list = []
        self.global_shap: list = []

        self._lock = threading.Lock()

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def is_ready(self) -> bool:
        return self.status == "ready"


# Module-level singleton
pipeline = PipelineState()


async def run_pipeline_async():
    """Run the full 5-agent pipeline in a thread and update state."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_pipeline_sync)


def _run_pipeline_sync():
    """Blocking pipeline execution — runs in thread pool."""
    import numpy as np

    try:
        pipeline.update(status="running", progress=5, current_step="Initialising agents…")

        # ── Agent 1 ───────────────────────────────────────────────────────
        pipeline.update(progress=10, current_step="Agent 1 — Generating dataset…")
        from agents.agent1_data import (
            generate_synthetic_dataset, inject_missing_values,
            clean_data, engineer_features, encode_and_split,
            _REAL_DATA_FILES,
        )
        raw_df = generate_synthetic_dataset(50_000)  # auto-detects real Kaggle data

        # Skip synthetic noise injection when using real Kaggle data
        if all(os.path.exists(f) for f in _REAL_DATA_FILES):
            pipeline.update(progress=20, current_step="Agent 1 — Real data loaded, cleaning…")
            dirty_df = raw_df
        else:
            dirty_df = inject_missing_values(raw_df)

        clean_df, _ = clean_data(dirty_df)
        eng_df = engineer_features(clean_df)
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         feat_names, scaler, enc) = encode_and_split(eng_df)

        num_cols = [c for c in clean_df.select_dtypes(include=np.number).columns
                    if c != "yield_q_ha"]

        pipeline.update(
            progress=25, current_step="Agent 1 — Complete ✓",
            feat_names=feat_names, scaler=scaler,
            clean_df=clean_df, num_cols=num_cols,
        )

        # ── Agent 6 (fast — trains on small CSV, runs after Agent 1) ─────────
        pipeline.update(progress=27, current_step="Agent 6 — Training crop recommender…")
        try:
            from agents.agent6_recommend import CropRecommendAgent
            agent6 = CropRecommendAgent()
            agent6.train("data/Crop_recommendation.csv", clean_df=clean_df)
            pipeline.update(agent6_recommend=agent6)
        except Exception as e:
            print(f"[Agent6] Skipped (CSV not found or error): {e}")

        # ── Agent 2 ───────────────────────────────────────────────────────
        pipeline.update(progress=30, current_step="Agent 2 — Training prediction models…")
        from agents.agent2_prediction import PredictionAgent
        agent2 = PredictionAgent(feat_names)
        agent2.train(X_train, y_train, X_val, y_val)
        results_df = agent2.evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test)

        # Save prediction plot
        os.makedirs("ma_xai_outputs", exist_ok=True)
        agent2.plot_results(X_test, y_test, save_path="ma_xai_outputs/prediction_results.png")

        metrics = results_df.set_index("model").to_dict(orient="index")
        fi = agent2.global_feature_importance(15)
        global_shap = fi.to_dict(orient="records")

        pipeline.update(
            progress=50, current_step="Agent 2 — Complete ✓",
            agent2_pred=agent2, model_metrics=metrics, global_shap=global_shap,
        )

        # ── Agent 3 ───────────────────────────────────────────────────────
        pipeline.update(progress=55, current_step="Agent 3 — Building causal DAG…")
        from agents.agent3_causal import CausalAgent
        agent3 = CausalAgent()
        agent3.build_dag()
        agent3.estimate_ate(clean_df)
        agent3.fit_counterfactual_model(clean_df, num_cols, scaler)
        agent3.plot_dag(save_path="ma_xai_outputs/causal_dag.png")
        agent3.plot_ate(save_path="ma_xai_outputs/causal_ate.png")

        ate_rows = []
        for _, row in agent3.ate_table.iterrows():
            ate_rows.append({
                "treatment": row["treatment"],
                "t_low": row["t_low"],
                "t_high": row["t_high"],
                "ate_qha": round(float(row["ATE (q/ha)"]), 3),
                "ate_pct": round(float(row["ATE (%)"]), 2),
                "ci_low": round(float(row["CI_low"]), 3),
                "ci_high": round(float(row["CI_high"]), 3),
            })

        pipeline.update(
            progress=70, current_step="Agent 3 — Complete ✓",
            agent3_causal=agent3, ate_table=ate_rows,
        )

        # ── Agent 4 ───────────────────────────────────────────────────────
        pipeline.update(progress=75, current_step="Agent 4 — Computing explanations…")
        from agents.agent4_explanation import ExplanationAgent
        agent4 = ExplanationAgent(agent2, agent3, X_train, feat_names, scaler, clean_df)
        agent4.plot_all_explanations(
            X_test[10], X_test[5], X_test[20],
            save_path="ma_xai_outputs/explanations.png",
        )

        pipeline.update(
            progress=88, current_step="Agent 4 — Complete ✓",
            agent4_explain=agent4,
        )

        # ── Agent 5 ───────────────────────────────────────────────────────
        pipeline.update(progress=92, current_step="Agent 5 — Building advisory engine…")
        from agents.agent5_advisory import AdvisoryAgent
        agent5 = AdvisoryAgent(agent2, agent3, agent4)

        pipeline.update(
            progress=100, current_step="All 5 agents operational ✓",
            status="ready", agent5_advisory=agent5,
        )

    except Exception as exc:
        import traceback
        pipeline.update(status="error", error=str(exc) + "\n" + traceback.format_exc())
        raise
