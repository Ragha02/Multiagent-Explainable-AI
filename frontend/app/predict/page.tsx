"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { API_BASE, FarmInput, PredictionResult, ShapFeature, defaultFarm, parseApiError } from "@/lib/api";
import FarmForm from "@/components/FarmForm";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend,
} from "recharts";

const PAGE = { maxWidth: 1100, margin: "0 auto", padding: "100px 28px 80px" };

// Pretty names for engineered features that come from the real model
const FEAT_LABELS: Record<string, string> = {
  prev_year_yield:         "Prev. Year Yield",
  rainfall_adequacy:       "Rainfall Adequacy",
  water_supply_index:      "Water Supply Index",
  soil_quality_index:      "Soil Quality Index",
  nutrient_balance:        "Nutrient Balance",
  temp_stress_index:       "Temp. Stress Index",
  rainfall_annual:         "Annual Rainfall",
  temp_mean:               "Mean Temperature",
  irrigation_coverage_pct: "Irrigation Coverage",
  npk_dosage_kg_ha:        "NPK Dosage",
  soil_moisture:           "Soil Moisture",
  soil_ph:                 "Soil pH",
  organic_carbon:          "Organic Carbon",
  variety_improved:        "Variety (HYV)",
  sowing_week:             "Sowing Week",
  season:                  "Season",
  crop:                    "Crop Type",
  state:                   "State / Region",
};

function featLabel(key: string) {
  return FEAT_LABELS[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function PredictPage() {
  const [farm, setFarm]     = useState<FarmInput>(defaultFarm);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);

  const submit = async (f?: FarmInput) => {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(f ?? farm),
      });
      if (!r.ok) throw new Error(await parseApiError(r, "Prediction failed"));
      setResult(await r.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  // Auto-submit with latest farm state when location is filled
  const handleLocationFilled = () => submit(farm);

  const shap    = result?.top_shap_features ?? [];
  const maxShap = Math.max(...shap.map((s) => Math.abs(s.shap_value)), 1);
  const models  = result?.model_metrics;

  return (
    <div style={PAGE}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        style={{ marginBottom: 40 }}
      >
        <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
          <span className="badge badge-green">Agent 2 — Prediction</span>
          <span className="badge badge-teal">Real India Data</span>
        </div>
        <h1 className="font-display" style={{ fontSize: 60, color: "var(--text)", marginBottom: 10 }}>
          Yield Prediction
        </h1>
        <p style={{ fontSize: 14, color: "#6b7280", maxWidth: 540, lineHeight: 1.65 }}>
          Ensemble (RF + XGBoost, <strong style={{ color: "#22c55e" }}>R²=0.9644</strong>) trained on
          50k real India records. Select a <strong style={{ color: "#94a3b8" }}>state → district → crop</strong> to
          auto-predict instantly, or tune the sliders manually.
        </p>
      </motion.div>

      <div style={{ display: "grid", gridTemplateColumns: "370px 1fr", gap: 22, alignItems: "start" }}>

        {/* ── FORM ──────────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
          className="card"
          style={{ padding: 26, position: "sticky", top: 80 }}
        >
          <div className="section-label" style={{ marginBottom: 18 }}>Farm Parameters</div>
          <FarmForm
            value={farm}
            onChange={setFarm}
            loading={loading}
            onSubmit={() => submit()}
            onLocationFilled={handleLocationFilled}
          />
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                style={{
                  marginTop: 12, padding: "10px 14px", borderRadius: 9, fontSize: 13,
                  background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#fca5a5",
                }}
              >⚠ {error}</motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* ── RESULTS ───────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15, duration: 0.5 }}
          style={{ display: "flex", flexDirection: "column", gap: 18 }}
        >
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div key="loading" className="card"
                style={{ padding: "56px 28px", textAlign: "center" }}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              >
                <div style={{ fontSize: 36, marginBottom: 14 }} className="anim-float">🌾</div>
                <p style={{ fontSize: 15, fontWeight: 700, color: "#22c55e", marginBottom: 6 }}>Analysing farm…</p>
                <p style={{ color: "#4b5563", fontSize: 12 }}>Running ensemble prediction</p>
                <div className="progress-track" style={{ margin: "18px auto 0", maxWidth: 200 }}>
                  <div className="shimmer" style={{ height: 3, borderRadius: 99 }} />
                </div>
              </motion.div>
            ) : result ? (
              <motion.div key="result" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4 }} className="card"
                style={{
                  padding: "38px 28px", textAlign: "center",
                  borderColor: "var(--green)",
                  background: "var(--surface)",
                }}
              >
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", color: "var(--text-dim)", marginBottom: 12 }}>
                  MA-XAI ENSEMBLE PREDICTION
                </div>
                <motion.div className="stat-num"
                  initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.1, type: "spring", stiffness: 180 }}
                  style={{ fontSize: 90, color: "var(--green)", lineHeight: 1, marginBottom: 6 }}
                >
                  {result.predicted_yield.toFixed(1)}
                </motion.div>
                <div style={{ fontSize: 15, color: "#6b7280", marginBottom: 16 }}>quintals per hectare</div>
                <div style={{
                  display: "inline-flex", gap: 8, alignItems: "center",
                  padding: "5px 14px", borderRadius: 99,
                  background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                  fontSize: 12, color: "#94a3b8",
                }}>
                  {result.farm_label}
                </div>
              </motion.div>
            ) : (
              <motion.div key="empty" className="card"
                style={{ padding: "56px 28px", textAlign: "center" }}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              >
                <div style={{ fontSize: 44, marginBottom: 16, opacity: 0.12 }}>🌾</div>
                <p style={{ fontSize: 15, fontWeight: 600, color: "#eef2ff", marginBottom: 8 }}>Select a Location</p>
                <p style={{ color: "#6b7280", fontSize: 13, lineHeight: 1.6 }}>
                  Pick a <strong style={{ color: "#22c55e" }}>State → District → Crop</strong> to auto-predict,
                  <br />or adjust sliders and click <strong style={{ color: "#22c55e" }}>Run Prediction</strong>
                </p>
              </motion.div>
            )}
          </AnimatePresence>


          {/* SHAP bars */}
          <AnimatePresence>
            {shap.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.08 }}
                className="card" style={{ padding: 26 }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                  <div className="section-label">SHAP Feature Attribution</div>
                  <span style={{ fontSize: 11, color: "#4b5563" }}>XGBoost explainer · real data</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
                  {shap.slice(0, 8).map((s: ShapFeature, i) => {
                    const pos = s.shap_value >= 0;
                    const w   = (Math.abs(s.shap_value) / maxShap) * 100;
                    return (
                      <motion.div
                        key={s.feature}
                        initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                            <span style={{
                              width: 16, fontSize: 9, fontWeight: 700, textAlign: "center",
                              color: i < 2 ? "#22c55e" : "#4b5563",
                            }}>#{i + 1}</span>
                            <span style={{ fontSize: 12, color: "#94a3b8" }}>{featLabel(s.feature)}</span>
                          </div>
                          <span className="stat-num" style={{ fontSize: 12, fontWeight: 800, color: pos ? "#22c55e" : "#ef4444" }}>
                            {pos ? "+" : ""}{s.shap_value.toFixed(3)}
                          </span>
                        </div>
                        <div style={{ height: 4, borderRadius: 99, background: "rgba(255,255,255,0.04)", overflow: "hidden" }}>
                          <motion.div
                            initial={{ width: 0 }} animate={{ width: `${w}%` }}
                            transition={{ delay: 0.15 + i * 0.04, duration: 0.6, ease: "easeOut" }}
                            style={{
                              height: "100%",
                            background: pos ? "var(--text)" : "var(--red)",
                          }}
                        />
                      </div>
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Model comparison chart */}
          <AnimatePresence>
            {models && (() => {
              const MODEL_COLORS: Record<string, string> = {
                "Random Forest": "var(--blue)",
                "XGBoost": "var(--teal)",
                "Linear Regression": "var(--border-bright)",
                "MA-XAI Ensemble": "var(--green)",
              };
              const chartData = Object.entries(models).map(([name, m]) => ({
                name: name.replace("MA-XAI ", "").replace("Linear Regression", "Linear Reg."),
                fullName: name,
                R2: parseFloat((m as any).R2?.toFixed(4) ?? "0"),
                RMSE: parseFloat((m as any).RMSE?.toFixed(2) ?? "0"),
              }));
              return (
                <motion.div
                  initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.16 }}
                  className="card" style={{ padding: 26 }}
                >
                  <div className="section-label" style={{ marginBottom: 6 }}>Model Comparison — R² Score</div>
                  <p style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 20 }}>Higher is better · Real India test set</p>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={chartData} barSize={28} margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis dataKey="name" tick={{ fill: "var(--text-dim)", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
                      <YAxis domain={[0.85, 1.0]} tick={{ fill: "var(--text-dim)", fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(2)} />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const d = payload[0].payload;
                          return (
                            <div style={{ background: "var(--surface)", border: "1px solid var(--border-bright)", padding: "8px 12px", fontSize: 11 }}>
                              <p style={{ color: "var(--text)", fontWeight: 700, marginBottom: 4 }}>{d.fullName}</p>
                              <p style={{ color: "var(--green)" }}>R²: {d.R2}</p>
                              <p style={{ color: "var(--text-dim)" }}>RMSE: {d.RMSE} q/ha</p>
                            </div>
                          );
                        }}
                        cursor={{ fill: "rgba(255,255,255,0.03)" }}
                      />
                      <Bar dataKey="R2" isAnimationActive animationDuration={800} radius={[2,2,0,0]}>
                        {chartData.map((entry, i) => (
                          <Cell key={i} fill={MODEL_COLORS[entry.fullName] ?? "var(--border-bright)"} fillOpacity={entry.fullName.includes("Ensemble") ? 1 : 0.65} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </motion.div>
              );
            })()}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}
