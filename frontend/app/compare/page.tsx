"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  API_BASE, FarmInput, CompareResult, DeltaShapEntry, DiceAction,
  defaultFarm, stressFarm, highYieldFarm, runComparison,
} from "@/lib/api";
import FarmForm from "@/components/FarmForm";

const PAGE  = { maxWidth: 1280, margin: "0 auto", padding: "100px 28px 100px" };
const GREEN = "var(--green)";
const BLUE  = "var(--blue)";
const RED   = "var(--red)";

// Pretty names for model features
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

const DICE_ICONS: Record<string, string> = {
  irrigation_coverage_pct: "💧",
  npk_dosage_kg_ha:        "🌿",
  soil_moisture:           "🪣",
  organic_carbon:          "🍂",
  variety_improved:        "🌾",
  sowing_week:             "📅",
  prev_year_yield:         "📈",
  rainfall_annual:         "🌧",
  temp_mean:               "🌡",
  soil_ph:                 "⚗",
};

function featLabel(key: string) {
  return FEAT_LABELS[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Animated number
function AnimNum({ val, decimals = 1, color = "inherit" }: { val: number; decimals?: number; color?: string }) {
  return (
    <motion.span
      key={val}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="stat-num"
      style={{ color }}
    >
      {val.toFixed(decimals)}
    </motion.span>
  );
}

// Small horizontal SHAP bar
function ShapBar({ val, max, color }: { val: number; max: number; color: string }) {
  const w = max > 0 ? (Math.abs(val) / max) * 100 : 0;
  return (
    <div style={{ height: 3, background: "rgba(255,255,255,0.05)", overflow: "hidden", borderRadius: 99, flex: 1 }}>
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${w}%` }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        style={{ height: "100%", background: color, borderRadius: 99 }}
      />
    </div>
  );
}

export default function ComparePage() {
  const [farmA, setFarmA] = useState<FarmInput>(defaultFarm);
  const [farmB, setFarmB] = useState<FarmInput>(stressFarm);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]    = useState<string | null>(null);

  const compare = async () => {
    setLoading(true); setError(null);
    try {
      const r = await runComparison(farmA, farmB);
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Comparison failed");
    } finally {
      setLoading(false);
    }
  };

  // Derived from result
  const maxDelta = result
    ? Math.max(...result.delta_shap.map((d) => Math.abs(d.delta)), 1)
    : 1;
  const flips = result?.delta_shap.filter((d) => d.direction_flip) ?? [];
  const winner = result ? (result.delta_yield >= 0 ? "A" : "B") : null;
  const deltaAbs = result ? Math.abs(result.delta_yield) : 0;

  return (
    <div style={PAGE}>

      {/* ── HEADER ──────────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        style={{ marginBottom: 44 }}
      >
        <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
          <span className="badge badge-purple">Agent 4 — Explanation</span>
          <span className="badge badge-teal">SHAP · LIME · ΔSHAP</span>
          <span className="badge badge-orange">DiCE Counterfactuals</span>
        </div>
        <h1 className="font-display" style={{ fontSize: "clamp(44px, 6vw, 72px)", color: "var(--text)", marginBottom: 10 }}>
          XAI Farm Comparison
        </h1>
        <p style={{ fontSize: 14, color: "var(--text-dim)", maxWidth: 600, lineHeight: 1.7 }}>
          Configure two farms and run the full explanation stack side-by-side.
          SHAP attribution divergence, LIME local contrast, and{" "}
          <strong style={{ color: "var(--orange)" }}>DiCE minimal-change actions</strong>{" "}
          reveal exactly why the model treats them differently — and what to change.
        </p>
      </motion.div>

      {/* ── DUAL FARM PANELS ──────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, marginBottom: 24 }}>

        {/* Farm A */}
        <motion.div
          initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.08 }}
          className="card"
          style={{ padding: 26, borderColor: `${GREEN}30` }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{
                width: 28, height: 28, background: GREEN, display: "flex",
                alignItems: "center", justifyContent: "center",
                fontSize: 13, fontWeight: 900, color: "var(--bg)",
              }}>A</div>
              <div className="section-label" style={{ margin: 0, padding: 0, border: "none", color: GREEN }}>Farm A</div>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {[
                { label: "Healthy", farm: defaultFarm },
                { label: "High-Yield", farm: highYieldFarm },
              ].map(({ label, farm }) => (
                <button key={label} onClick={() => setFarmA({ ...farm })}
                  style={{
                    padding: "4px 10px", fontSize: 10, fontWeight: 700, cursor: "pointer",
                    background: "rgba(137,184,132,0.08)", border: "1px solid rgba(137,184,132,0.2)",
                    color: GREEN, borderRadius: 4,
                  }}
                >{label}</button>
              ))}
            </div>
          </div>
          <FarmForm value={farmA} onChange={setFarmA} loading={loading} onSubmit={compare} submitLabel="SET FARM A" />
        </motion.div>

        {/* Farm B */}
        <motion.div
          initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.12 }}
          className="card"
          style={{ padding: 26, borderColor: `${BLUE}30` }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{
                width: 28, height: 28, background: BLUE, display: "flex",
                alignItems: "center", justifyContent: "center",
                fontSize: 13, fontWeight: 900, color: "var(--bg)",
              }}>B</div>
              <div className="section-label" style={{ margin: 0, padding: 0, border: "none", color: BLUE }}>Farm B</div>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {[
                { label: "Stressed", farm: stressFarm },
                { label: "Healthy",  farm: defaultFarm },
              ].map(({ label, farm }) => (
                <button key={label} onClick={() => setFarmB({ ...farm })}
                  style={{
                    padding: "4px 10px", fontSize: 10, fontWeight: 700, cursor: "pointer",
                    background: "rgba(122,148,191,0.08)", border: "1px solid rgba(122,148,191,0.2)",
                    color: BLUE, borderRadius: 4,
                  }}
                >{label}</button>
              ))}
            </div>
          </div>
          <FarmForm value={farmB} onChange={setFarmB} loading={loading} onSubmit={compare} submitLabel="SET FARM B" />
        </motion.div>
      </div>

      {/* ── RUN CTA ───────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 40 }}>
        <button
          onClick={compare}
          disabled={loading}
          className="btn-primary"
          style={{ fontSize: 22, padding: "14px 48px", opacity: loading ? 0.5 : 1 }}
        >
          {loading ? (
            <><span className="anim-spin" style={{ display: "inline-block" }}>⟳</span> Analysing…</>
          ) : "⟶  RUN COMPARISON"}
        </button>
      </div>

      {/* ── ERROR ─────────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{
              marginBottom: 32, padding: "12px 18px",
              background: "rgba(194,112,112,0.08)", border: "1px solid rgba(194,112,112,0.2)",
              color: "var(--red)", fontSize: 13,
            }}
          >⚠ {error}</motion.div>
        )}
      </AnimatePresence>

      {/* ── RESULTS ───────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {result && (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            style={{ display: "flex", flexDirection: "column", gap: 20 }}
          >

            {/* ── 1. YIELD DELTA HERO ──────────────────────────────────────── */}
            <div className="card" style={{
              padding: "36px 40px",
              background: "var(--surface)",
              borderColor: winner === "A" ? `${GREEN}40` : `${BLUE}40`,
            }}>
              <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.15em", color: "var(--text-dim)", marginBottom: 24 }}>
                YIELD COMPARISON — MA-XAI ENSEMBLE
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
                {/* Farm A */}
                <div style={{ flex: 1, textAlign: "center" }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: GREEN, marginBottom: 8, letterSpacing: "0.08em" }}>
                    FARM A — {result.farm_a.label}
                  </div>
                  <div style={{ fontSize: 72, lineHeight: 1 }}>
                    <AnimNum val={result.farm_a.predicted_yield} color={GREEN} />
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 6 }}>quintals / hectare</div>
                </div>

                {/* Delta */}
                <div style={{ padding: "0 32px", textAlign: "center", flexShrink: 0 }}>
                  <div style={{
                    fontSize: 10, fontWeight: 800, color: "var(--text-dim)",
                    letterSpacing: "0.1em", marginBottom: 8,
                  }}>ΔYIELD</div>
                  <div style={{
                    fontSize: 48, fontWeight: 900,
                    color: result.delta_yield >= 0 ? GREEN : BLUE,
                    fontFamily: "'Bebas Neue', sans-serif",
                  }}>
                    {result.delta_yield >= 0 ? "+" : ""}{result.delta_yield.toFixed(1)}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-dim)" }}>q/ha</div>
                  <div style={{
                    marginTop: 10, padding: "4px 12px", display: "inline-block",
                    fontSize: 10, fontWeight: 700,
                    background: winner === "A" ? `${GREEN}15` : `${BLUE}15`,
                    border: `1px solid ${winner === "A" ? GREEN : BLUE}40`,
                    color: winner === "A" ? GREEN : BLUE,
                  }}>
                    Farm {winner} leads by {deltaAbs.toFixed(1)} q/ha
                  </div>
                </div>

                {/* Farm B */}
                <div style={{ flex: 1, textAlign: "center" }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: BLUE, marginBottom: 8, letterSpacing: "0.08em" }}>
                    FARM B — {result.farm_b.label}
                  </div>
                  <div style={{ fontSize: 72, lineHeight: 1 }}>
                    <AnimNum val={result.farm_b.predicted_yield} color={BLUE} />
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 6 }}>quintals / hectare</div>
                </div>
              </div>
            </div>

            {/* ── 2. ΔSHAP DIVERGENCE CHART ────────────────────────────────── */}
            <div className="card" style={{ padding: 28 }}>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 20 }}>
                <div>
                  <div className="section-label" style={{ marginBottom: 4 }}>ΔSHAP Attribution Divergence</div>
                  <p style={{ fontSize: 11, color: "var(--text-dim)" }}>
                    Per-feature SHAP gap (Farm A − Farm B). Green = Farm A advantage · Blue = Farm B advantage.
                  </p>
                </div>
                <div style={{ display: "flex", gap: 12, alignItems: "center", flexShrink: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <div style={{ width: 10, height: 10, background: GREEN }} />
                    <span style={{ fontSize: 10, color: "var(--text-dim)" }}>Farm A</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <div style={{ width: 10, height: 10, background: BLUE }} />
                    <span style={{ fontSize: 10, color: "var(--text-dim)" }}>Farm B</span>
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {result.delta_shap.map((d: DeltaShapEntry, i) => {
                  const wA = maxDelta > 0 ? (Math.abs(d.shap_a) / maxDelta) * 100 : 0;
                  const wB = maxDelta > 0 ? (Math.abs(d.shap_b) / maxDelta) * 100 : 0;
                  return (
                    <motion.div
                      key={d.feature}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.03 }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
                          <span style={{ fontSize: 9, fontWeight: 700, color: "var(--muted)", width: 16, textAlign: "right" }}>
                            #{i + 1}
                          </span>
                          <span style={{ fontSize: 12, color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {featLabel(d.feature)}
                          </span>
                          {d.direction_flip && (
                            <span style={{
                              fontSize: 8, fontWeight: 800, color: "var(--orange)",
                              background: "rgba(199,160,124,0.12)", border: "1px solid rgba(199,160,124,0.25)",
                              padding: "1px 5px", flexShrink: 0,
                            }}>⇄ FLIP</span>
                          )}
                        </div>
                        <div style={{ display: "flex", gap: 10, flexShrink: 0, marginLeft: 10 }}>
                          <span className="stat-num" style={{ fontSize: 11, color: d.shap_a >= 0 ? GREEN : RED, width: 52, textAlign: "right" }}>
                            {d.shap_a >= 0 ? "+" : ""}{d.shap_a.toFixed(3)}
                          </span>
                          <span className="stat-num" style={{ fontSize: 11, color: d.shap_b >= 0 ? BLUE : RED, width: 52, textAlign: "right" }}>
                            {d.shap_b >= 0 ? "+" : ""}{d.shap_b.toFixed(3)}
                          </span>
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                        <ShapBar val={d.shap_a} max={maxDelta} color={d.shap_a >= 0 ? GREEN : RED} />
                        <ShapBar val={d.shap_b} max={maxDelta} color={d.shap_b >= 0 ? BLUE : RED} />
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>

            {/* ── 3. DIRECTION FLIP TABLE ──────────────────────────────────── */}
            {flips.length > 0 && (
              <div className="card" style={{ padding: 28 }}>
                <div className="section-label" style={{ marginBottom: 6 }}>Direction Flip Features</div>
                <p style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 20 }}>
                  These features have <strong style={{ color: "var(--orange)" }}>opposite SHAP signs</strong> between Farm A and Farm B —
                  meaning the model's reasoning about them completely reversed. The most revealing divergence insight.
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {flips.map((d, i) => (
                    <motion.div
                      key={d.feature}
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05 }}
                      style={{
                        display: "grid", gridTemplateColumns: "1fr auto auto auto",
                        gap: 16, alignItems: "center",
                        padding: "10px 14px",
                        background: "rgba(199,160,124,0.05)",
                        border: "1px solid rgba(199,160,124,0.18)",
                      }}
                    >
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", marginBottom: 2 }}>
                          {featLabel(d.feature)}
                        </div>
                        <div style={{ fontSize: 9, color: "var(--muted)", fontFamily: "monospace" }}>{d.feature}</div>
                      </div>
                      <div style={{ textAlign: "center" }}>
                        <div style={{ fontSize: 9, color: "var(--text-dim)", marginBottom: 3 }}>FARM A</div>
                        <span className="stat-num" style={{ fontSize: 14, color: d.shap_a >= 0 ? GREEN : RED }}>
                          {d.shap_a >= 0 ? "▲" : "▼"} {Math.abs(d.shap_a).toFixed(3)}
                        </span>
                      </div>
                      <div style={{ textAlign: "center" }}>
                        <div style={{ fontSize: 9, color: "var(--text-dim)", marginBottom: 3 }}>FARM B</div>
                        <span className="stat-num" style={{ fontSize: 14, color: d.shap_b >= 0 ? BLUE : RED }}>
                          {d.shap_b >= 0 ? "▲" : "▼"} {Math.abs(d.shap_b).toFixed(3)}
                        </span>
                      </div>
                      <div style={{
                        padding: "3px 8px", fontSize: 9, fontWeight: 700,
                        color: "var(--orange)", border: "1px solid rgba(199,160,124,0.3)",
                        background: "rgba(199,160,124,0.1)",
                      }}>⇄ REVERSED</div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>

              {/* ── 4. DiCE ACTIONS ────────────────────────────────────────── */}
              <div className="card" style={{
                padding: 28,
                borderColor: "rgba(199,160,124,0.25)",
                background: "rgba(199,160,124,0.03)",
              }}>
                <div className="section-label" style={{ marginBottom: 4 }}>DiCE Counterfactual Actions</div>
                <p style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 22, lineHeight: 1.6 }}>
                  Minimal parameter changes to close the yield gap.
                  Generated by <strong style={{ color: "var(--orange)" }}>Diverse Counterfactual Explanations (DiCE)</strong> — changes
                  are optimised to be small and realistic.
                </p>

                {result.dice_actions.length === 0 ? (
                  <div style={{
                    padding: "22px 18px", textAlign: "center",
                    background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)",
                  }}>
                    <div style={{ fontSize: 24, marginBottom: 10, opacity: 0.4 }}>✓</div>
                    <p style={{ fontSize: 13, color: "var(--text-dim)" }}>
                      Farms are very similar — no significant counterfactual actions needed.
                    </p>
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {result.dice_actions.map((action: DiceAction, i) => {
                      const up = action.to_val > action.from_val;
                      const gainColor = action.estimated_gain >= 0 ? GREEN : RED;
                      return (
                        <motion.div
                          key={action.feature}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.07 }}
                          style={{
                            display: "flex", alignItems: "flex-start", gap: 14,
                            padding: "14px 16px",
                            background: "rgba(255,255,255,0.025)",
                            border: "1px solid rgba(199,160,124,0.15)",
                          }}
                        >
                          <div style={{
                            width: 32, height: 32, flexShrink: 0,
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 18, background: "rgba(199,160,124,0.08)",
                            border: "1px solid rgba(199,160,124,0.2)",
                          }}>
                            {DICE_ICONS[action.feature] ?? "⚙"}
                          </div>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", marginBottom: 5 }}>
                              {featLabel(action.feature)}
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                              <span className="stat-num" style={{ fontSize: 13, color: "var(--text-dim)" }}>
                                {action.from_val.toFixed(action.unit === "" ? 1 : 0)}{action.unit}
                              </span>
                              <span style={{ color: "var(--orange)", fontSize: 14 }}>{up ? "→" : "↓"}</span>
                              <span className="stat-num" style={{ fontSize: 13, color: "var(--orange)", fontWeight: 800 }}>
                                {action.to_val.toFixed(action.unit === "" ? 1 : 0)}{action.unit}
                              </span>
                            </div>
                          </div>
                          <div style={{ textAlign: "right", flexShrink: 0 }}>
                            <div style={{ fontSize: 9, color: "var(--text-dim)", marginBottom: 3 }}>EST. GAIN</div>
                            <span className="stat-num" style={{ fontSize: 16, color: gainColor, fontWeight: 900 }}>
                              {action.estimated_gain >= 0 ? "+" : ""}{action.estimated_gain.toFixed(1)}
                            </span>
                            <span style={{ fontSize: 9, color: "var(--text-dim)", marginLeft: 2 }}>q/ha</span>
                          </div>
                        </motion.div>
                      );
                    })}
                    {/* Total gain */}
                    <div style={{
                      display: "flex", justifyContent: "space-between", alignItems: "center",
                      padding: "10px 16px",
                      background: "rgba(199,160,124,0.08)", border: "1px solid rgba(199,160,124,0.25)",
                    }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: "var(--orange)", letterSpacing: "0.06em" }}>
                        TOTAL ESTIMATED GAIN
                      </span>
                      <span className="stat-num" style={{ fontSize: 20, color: "var(--orange)", fontWeight: 900 }}>
                        +{result.dice_actions.reduce((s, a) => s + Math.max(a.estimated_gain, 0), 0).toFixed(1)} q/ha
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* ── 5. LIME CONTRAST ─────────────────────────────────────────── */}
              <div className="card" style={{ padding: 28 }}>
                <div className="section-label" style={{ marginBottom: 4 }}>LIME Local Contrast</div>
                <p style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 20, lineHeight: 1.6 }}>
                  Local linear approximation per farm — how each feature condition contributes to the
                  model's local decision at that specific input point.
                </p>

                {result.lime_contrast.length === 0 ? (
                  <div style={{ padding: "20px", textAlign: "center", color: "var(--text-dim)", fontSize: 13 }}>
                    LIME contrast unavailable
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {result.lime_contrast.slice(0, 7).map((entry, i) => {
                      const maxLime = Math.max(
                        ...result.lime_contrast.map((e) =>
                          Math.max(Math.abs(e.contribution_a), Math.abs(e.contribution_b))
                        ), 0.001
                      );
                      const wA = (Math.abs(entry.contribution_a) / maxLime) * 100;
                      const wB = (Math.abs(entry.contribution_b) / maxLime) * 100;
                      return (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.04 }}
                        >
                          <div style={{ fontSize: 10, color: "var(--text-dim)", marginBottom: 5, lineHeight: 1.4 }}>
                            {entry.feature_condition.length > 42
                              ? entry.feature_condition.slice(0, 42) + "…"
                              : entry.feature_condition}
                          </div>
                          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                            {/* Farm A bar */}
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              <span style={{ fontSize: 8, fontWeight: 700, color: GREEN, width: 12, flexShrink: 0 }}>A</span>
                              <div style={{ flex: 1, height: 3, background: "rgba(255,255,255,0.04)", overflow: "hidden", borderRadius: 99 }}>
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${wA}%` }}
                                  transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 + i * 0.04 }}
                                  style={{ height: "100%", background: entry.contribution_a >= 0 ? GREEN : RED }}
                                />
                              </div>
                              <span className="stat-num" style={{ fontSize: 9, color: entry.contribution_a >= 0 ? GREEN : RED, width: 42, textAlign: "right" }}>
                                {entry.contribution_a >= 0 ? "+" : ""}{entry.contribution_a.toFixed(3)}
                              </span>
                            </div>
                            {/* Farm B bar */}
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              <span style={{ fontSize: 8, fontWeight: 700, color: BLUE, width: 12, flexShrink: 0 }}>B</span>
                              <div style={{ flex: 1, height: 3, background: "rgba(255,255,255,0.04)", overflow: "hidden", borderRadius: 99 }}>
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${wB}%` }}
                                  transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 + i * 0.04 }}
                                  style={{ height: "100%", background: entry.contribution_b >= 0 ? BLUE : RED }}
                                />
                              </div>
                              <span className="stat-num" style={{ fontSize: 9, color: entry.contribution_b >= 0 ? BLUE : RED, width: 42, textAlign: "right" }}>
                                {entry.contribution_b >= 0 ? "+" : ""}{entry.contribution_b.toFixed(3)}
                              </span>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* ── 6. INDIVIDUAL SHAP BREAKDOWNS ────────────────────────────── */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
              {[
                { farm: result.farm_a, label: "Farm A", color: GREEN },
                { farm: result.farm_b, label: "Farm B", color: BLUE },
              ].map(({ farm, label, color }) => {
                const maxS = Math.max(...farm.shap.map((s) => Math.abs(s.shap_value)), 1);
                return (
                  <div key={label} className="card" style={{ padding: 24, borderColor: `${color}25` }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18 }}>
                      <div style={{
                        width: 20, height: 20, background: color,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 10, fontWeight: 900, color: "var(--bg)",
                      }}>{label.slice(-1)}</div>
                      <div className="section-label" style={{ margin: 0, padding: 0, border: "none", color }}>
                        {label} — SHAP Attribution
                      </div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
                      {farm.shap.slice(0, 8).map((s, i) => {
                        const pos = s.shap_value >= 0;
                        const w = (Math.abs(s.shap_value) / maxS) * 100;
                        return (
                          <div key={s.feature}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                              <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
                                <span style={{ fontSize: 9, color: "var(--muted)", marginRight: 5 }}>#{i + 1}</span>
                                {featLabel(s.feature)}
                              </span>
                              <span className="stat-num" style={{ fontSize: 11, color: pos ? color : RED }}>
                                {pos ? "+" : ""}{s.shap_value.toFixed(3)}
                              </span>
                            </div>
                            <div style={{ height: 3, background: "rgba(255,255,255,0.04)", overflow: "hidden", borderRadius: 99 }}>
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${w}%` }}
                                transition={{ delay: i * 0.04, duration: 0.5 }}
                                style={{ height: "100%", background: pos ? color : RED }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>

          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
