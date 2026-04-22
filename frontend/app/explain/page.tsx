"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import type { Transition } from "framer-motion";
import { API_BASE } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const PAGE = { maxWidth: 1100, margin: "0 auto", padding: "100px 28px 80px" };
const EASE: Transition = { duration: 0.55, ease: "easeOut" };

const XAI_TYPES = [
  { type: "Type 1", title: "Global SHAP",       color: "var(--green)",  icon: "🌍",
    desc: "Mean |SHAP values| across 50k real records. prev_year_yield ranks #1." },
  { type: "Type 2", title: "Local LIME",          color: "var(--teal)",   icon: "🔬",
    desc: "Per-prediction tabular LIME explanation on the test set." },
  { type: "Type 3", title: "Contrastive ΔSHAP",  color: "var(--blue)",   icon: "⚖",
    desc: "Feature attribution delta between two real farm instances." },
  { type: "Type 4", title: "Causal ATE",          color: "var(--purple)", icon: "📊",
    desc: "Causal effect of irrigation / NPK interventions from DoWhy." },
];

const label = (key: string) =>
  key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace("Npk", "NPK")
    .replace("Ate", "ATE");

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div style={{
        background: "var(--surface)", border: "1px solid var(--border-bright)",
        padding: "10px 14px", fontSize: 12,
      }}>
        <p style={{ fontWeight: 700, color: "var(--text)", marginBottom: 4 }}>{label(d.feature)}</p>
        <p style={{ color: "var(--green)" }}>Mean |SHAP|: {d.mean_abs_shap.toFixed(3)}</p>
      </div>
    );
  }
  return null;
};

export default function ExplainPage() {
  const [ready, setReady]  = useState(false);
  const [shap, setShap]    = useState<{ feature: string; mean_abs_shap: number }[]>([]);

  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/pipeline/results`);
        const d = await r.json();
        if (d.status === "ready") { setShap(d.global_shap ?? []); setReady(true); clearInterval(id); }
      } catch {}
    }, 1500);
    return () => clearInterval(id);
  }, []);

  const top12 = shap.slice(0, 12);
  const maxShap = Math.max(...top12.map((s) => s.mean_abs_shap), 1);

  return (
    <div style={PAGE}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
        transition={{ ...EASE }}
        style={{ marginBottom: 44 }}
      >
        <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
          <span className="badge badge-purple">Agent 4 — Explanation</span>
          <span className="badge badge-green">50k Real Records</span>
        </div>
        <h1 className="font-display" style={{ fontSize: 60, color: "var(--text)", marginBottom: 10 }}>
          All 4 XAI Types
        </h1>
        <p style={{ fontSize: 14, color: "var(--text-dim)", maxWidth: 540, lineHeight: 1.65 }}>
          XGBoost SHAP and tabular LIME computed from models trained on real India district-level data.
          Global rankings reflect authentic agronomic insight, not synthetic correlations.
        </p>
      </motion.div>

      {/* XAI type cards */}
      <motion.div
        initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
        transition={{ ...EASE, delay: 0.06 }}
        style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 32 }}
      >
        {XAI_TYPES.map((x) => (
          <div key={x.type} style={{
            padding: "16px 16px", background: "var(--surface)", border: "1px solid var(--border)",
            borderBottom: `3px solid ${x.color}`,
          }}>
            <div style={{ fontSize: 20, marginBottom: 8 }}>{x.icon}</div>
            <span style={{ fontSize: 9, fontWeight: 700, color: x.color, letterSpacing: "0.1em", textTransform: "uppercase" }}>{x.type}</span>
            <p style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", marginTop: 4, marginBottom: 5 }}>{x.title}</p>
            <p style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.5 }}>{x.desc}</p>
          </div>
        ))}
      </motion.div>

      {!ready ? (
        <motion.div
          initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
          transition={{ ...EASE, delay: 0.1 }}
          className="card" style={{ padding: 80, textAlign: "center" }}
        >
          <div style={{ fontSize: 40, marginBottom: 16, color: "var(--text-dim)" }}>✦</div>
          <p style={{ color: "var(--text-dim)" }}>Waiting for pipeline to complete…</p>
        </motion.div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>

          {/* ── Type 1: Global SHAP — Recharts Horizontal Bar ── */}
          <motion.div
            initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
            transition={{ ...EASE, delay: 0.1 }}
            className="card" style={{ padding: 28 }}
          >
            <div className="section-label" style={{ marginBottom: 8 }}>Type 1 — Global SHAP Ranking</div>
            <p style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 22, lineHeight: 1.6 }}>
              Mean |SHAP| values — learned from 246k India records → 50k sample. Reflects true agronomic drivers.
            </p>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart
                data={top12}
                layout="vertical"
                margin={{ top: 0, right: 60, bottom: 0, left: 140 }}
                barSize={14}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fill: "var(--text-dim)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={135}
                  tick={{ fill: "var(--text-dim)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={label}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                <Bar dataKey="mean_abs_shap" isAnimationActive animationDuration={900} radius={[0,2,2,0]}>
                  {top12.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={i === 0 ? "var(--text)" : i < 3 ? "var(--border-bright)" : "var(--border)"}
                      fillOpacity={i === 0 ? 1 : 0.8}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            {/* Insight callout */}
            <div style={{
              marginTop: 20, padding: "12px 14px",
              background: "rgba(0,0,0,0.2)", border: "1px solid var(--border-bright)",
              borderLeft: "3px solid var(--green)",
            }}>
              <p style={{ fontSize: 11, color: "var(--green)", fontWeight: 700, marginBottom: 4 }}>📌 Key Real-Data Insight</p>
              <p style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.55 }}>
                <strong style={{ color: "var(--text)" }}>prev_year_yield</strong> dominates — historical performance
                is the strongest predictor of current yield, reflecting persistent soil health &amp; management quality across districts.
              </p>
            </div>
          </motion.div>

          {/* ── Types 2, 3, 4: Composite explanation PNG ── */}
          <motion.div
            initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
            transition={{ ...EASE, delay: 0.14 }}
            className="card" style={{ padding: 28 }}
          >
            <div className="section-label" style={{ marginBottom: 8 }}>Types 2, 3 & 4 — Composite Explanation Panel</div>
            <p style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 22, lineHeight: 1.6 }}>
              Agent 4 renders Local LIME (Type 2), Contrastive ΔSHAP (Type 3), and Causal ATE (Type 4)
              in a single figure using the XGBoost SHAP explainer &amp; tabular LIME on 50k records.
            </p>
            <div style={{ overflow: "hidden", background: "rgba(0,0,0,0.2)", border: "1px solid var(--border)" }}>
              <img
                src={`${API_BASE}/outputs/explanations.png`}
                alt="Types 2,3,4 — LIME, ΔSHAP and ATE composite"
                style={{ width: "100%", display: "block", objectFit: "contain", maxHeight: 560 }}
              />
            </div>
          </motion.div>

        </div>
      )}
    </div>
  );
}
