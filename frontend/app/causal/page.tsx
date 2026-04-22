"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import type { Transition } from "framer-motion";
import { API_BASE, ATERow } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ErrorBar,
  ResponsiveContainer,
  Cell,
} from "recharts";

const PAGE = { maxWidth: 1100, margin: "0 auto", padding: "100px 28px 80px" };
const EASE: Transition = { duration: 0.55, ease: "easeOut" };

// ── DAG Definition (mirrors agent3_causal.py exactly) ────────────────────────
const DAG_NODES = [
  // Layer 1 — Exogenous
  { id: "rainfall",              label: "rainfall",              layer: 1, color: "#4e9af1", x: 60,  y: 120 },
  { id: "temperature",           label: "temperature",           layer: 1, color: "#4e9af1", x: 60,  y: 260 },
  { id: "soil_type",             label: "soil_type",             layer: 1, color: "#4e9af1", x: 60,  y: 400 },
  // Layer 2 — Environmental State
  { id: "soil_moisture",         label: "soil_moisture",         layer: 2, color: "#5cb85c", x: 260, y: 180 },
  { id: "nutrient_availability", label: "nutrient_avail.",       layer: 2, color: "#5cb85c", x: 260, y: 340 },
  // Layer 3 — Controllable
  { id: "irrigation_applied",    label: "irrigation",            layer: 3, color: "#f0ad4e", x: 480, y: 100 },
  { id: "fertilizer_applied",    label: "fertilizer",            layer: 3, color: "#f0ad4e", x: 480, y: 260 },
  { id: "sowing_date",           label: "sowing_date",           layer: 3, color: "#f0ad4e", x: 480, y: 400 },
  // Layer 4 — Intermediate / Outcome
  { id: "crop_growth",           label: "crop_growth",           layer: 4, color: "#9b59b6", x: 700, y: 200 },
  { id: "yield",                 label: "yield",                 layer: 4, color: "#e74c3c", x: 900, y: 260 },
];

const DAG_EDGES = [
  ["rainfall",          "soil_moisture"],
  ["temperature",       "soil_moisture"],
  ["rainfall",          "nutrient_availability"],
  ["soil_type",         "nutrient_availability"],
  ["soil_type",         "soil_moisture"],
  ["soil_moisture",     "irrigation_applied"],
  ["nutrient_availability", "fertilizer_applied"],
  ["rainfall",          "sowing_date"],
  ["temperature",       "sowing_date"],
  ["irrigation_applied","crop_growth"],
  ["fertilizer_applied","crop_growth"],
  ["sowing_date",       "crop_growth"],
  ["soil_moisture",     "crop_growth"],
  ["nutrient_availability","crop_growth"],
  ["temperature",       "crop_growth"],
  ["crop_growth",       "yield"],
  ["irrigation_applied","yield"],
  ["fertilizer_applied","yield"],
  ["soil_moisture",     "yield"],
];

const NODE_R = 34;

function nodeById(id: string) {
  return DAG_NODES.find((n) => n.id === id)!;
}

function CausalDAG() {
  const W = 980, H = 500;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="rgba(255,255,255,0.35)" />
        </marker>
      </defs>

      {/* Edges */}
      {DAG_EDGES.map(([src, tgt], i) => {
        const s = nodeById(src);
        const t = nodeById(tgt);
        if (!s || !t) return null;
        const dx = t.x - s.x, dy = t.y - s.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const ux = dx / len, uy = dy / len;
        const x1 = s.x + ux * NODE_R;
        const y1 = s.y + uy * NODE_R;
        const x2 = t.x - ux * (NODE_R + 6);
        const y2 = t.y - uy * (NODE_R + 6);
        // slight curve
        const mx = (x1 + x2) / 2 - (uy * 18);
        const my = (y1 + y2) / 2 + (ux * 18);
        return (
          <path
            key={i}
            d={`M${x1},${y1} Q${mx},${my} ${x2},${y2}`}
            fill="none"
            stroke="rgba(255,255,255,0.2)"
            strokeWidth={1.5}
            markerEnd="url(#arrow)"
          />
        );
      })}

      {/* Nodes */}
      {DAG_NODES.map((n) => (
        <g key={n.id}>
          <circle
            cx={n.x} cy={n.y} r={NODE_R}
            fill={n.color}
            fillOpacity={0.85}
            stroke={n.color}
            strokeWidth={2}
          />
          <text
            x={n.x} y={n.y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={9}
            fontWeight="700"
            fill="white"
            style={{ fontFamily: "Inter, sans-serif" }}
          >
            {n.label.length > 10 ? n.label.slice(0, 10) + "…" : n.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ── ATE Chart ────────────────────────────────────────────────────────────────
const ATE_COLORS = ["#7a94bf", "#89b884", "#c7a07c"];

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div style={{
        background: "var(--surface)", border: "1px solid var(--border-bright)",
        padding: "10px 14px", fontSize: 12,
      }}>
        <p style={{ fontWeight: 700, color: "var(--text)", marginBottom: 4 }}>{d.treatment}</p>
        <p style={{ color: "var(--text-dim)" }}>{d.t_low} → {d.t_high}</p>
        <p style={{ color: d.ate_qha >= 0 ? "var(--green)" : "var(--red)", fontWeight: 700, marginTop: 4 }}>
          ATE: {d.ate_qha > 0 ? "+" : ""}{d.ate_qha.toFixed(2)} q/ha
        </p>
        <p style={{ color: "var(--text-dim)", fontSize: 11 }}>
          95% CI: [{d.ci_low.toFixed(2)}, {d.ci_high.toFixed(2)}]
        </p>
      </div>
    );
  }
  return null;
};

const LAYER_LEGEND = [
  { color: "#4e9af1", label: "L1 — Exogenous (Climate/Soil)" },
  { color: "#5cb85c", label: "L2 — Environmental State" },
  { color: "#f0ad4e", label: "L3 — Controllable Actions" },
  { color: "#9b59b6", label: "L4 — Intermediate" },
  { color: "#e74c3c", label: "L4 — Yield (Outcome)" },
];

export default function CausalPage() {
  const [ate, setAte]     = useState<ATERow[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/pipeline/results`);
        const d = await r.json();
        if (d.status === "ready") { setAte(d.ate_table ?? []); setReady(true); clearInterval(id); }
      } catch {}
    }, 1500);
    return () => clearInterval(id);
  }, []);

  // Prepare chart data with error bars
  const chartData = ate.map((row) => ({
    ...row,
    errorY: [row.ate_qha - row.ci_low, row.ci_high - row.ate_qha] as [number, number],
  }));

  return (
    <div style={PAGE}>
      <motion.div
        initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
        transition={{ ...EASE }}
        style={{ marginBottom: 44 }}
      >
        <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
          <span className="badge badge-teal">Agent 3 — Causal</span>
          <span className="badge badge-blue">Real India Crop Data</span>
        </div>
        <h1 className="font-display" style={{ fontSize: 60, color: "var(--text)", marginBottom: 10 }}>
          Causal Analysis
        </h1>
        <p style={{ fontSize: 14, color: "var(--text-dim)", maxWidth: 540, lineHeight: 1.65 }}>
          A 5-layer directed acyclic graph built from real district-level India crop data.
          DoWhy estimates counterfactual ATEs for the 3 controllable intervention variables.
        </p>
      </motion.div>

      {!ready ? (
        <motion.div
          initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
          transition={{ ...EASE, delay: 0.1 }}
          className="card" style={{ padding: 80, textAlign: "center" }}
        >
          <div style={{ fontSize: 40, marginBottom: 16, color: "var(--text-dim)" }}>⟶</div>
          <p style={{ color: "var(--text-dim)" }}>Waiting for pipeline to complete…</p>
        </motion.div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

          {/* ── DAG Card ── */}
          <motion.div
            initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
            transition={{ ...EASE, delay: 0.1 }}
            className="card" style={{ padding: 30 }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 22, flexWrap: "wrap", gap: 16 }}>
              <div>
                <div className="section-label" style={{ marginBottom: 8 }}>
                  Causal DAG — 4-Layer Structure ({DAG_NODES.length} nodes, {DAG_EDGES.length} edges)
                </div>
                <p style={{ fontSize: 12, color: "var(--text-dim)", maxWidth: 500, lineHeight: 1.6 }}>
                  Only Layer 3 (orange) nodes are farmer-controllable. DoWhy uses back-door adjustment
                  to estimate causal effects from observational data.
                </p>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                {LAYER_LEGEND.map((x) => (
                  <div key={x.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-dim)" }}>
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: x.color, flexShrink: 0 }} />
                    <span>{x.label}</span>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: "rgba(0,0,0,0.25)", border: "1px solid var(--border)", padding: 16 }}>
              <CausalDAG />
            </div>
          </motion.div>

          {/* ── ATE Two-Col ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>

            {/* ATE Table */}
            <motion.div
              initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
              transition={{ ...EASE, delay: 0.18 }}
              className="card" style={{ padding: 26 }}
            >
              <div className="section-label" style={{ marginBottom: 20 }}>Average Treatment Effects</div>
              <p style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 22, lineHeight: 1.5 }}>
                DoWhy back-door adjustment · 95% bootstrap CI · real India district data
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
                {ate.map((row, i) => {
                  const pos = row.ate_qha >= 0;
                  const maxAte = Math.max(...ate.map((r) => Math.abs(r.ate_qha)), 1);
                  const w = (Math.abs(row.ate_qha) / maxAte) * 100;
                  return (
                    <motion.div
                      key={row.treatment}
                      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 + i * 0.1 }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, gap: 10 }}>
                        <div style={{ flex: 1 }}>
                          <p style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", marginBottom: 2 }}>{row.treatment}</p>
                          <p style={{ fontSize: 10, color: "var(--text-dim)" }}>{row.t_low} → {row.t_high}</p>
                        </div>
                        <div style={{ textAlign: "right", flexShrink: 0 }}>
                          <span className="stat-num" style={{ fontSize: 22, color: pos ? "var(--green)" : "var(--red)" }}>
                            {pos ? "+" : ""}{row.ate_qha.toFixed(2)}
                          </span>
                          <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: 3 }}>q/ha</span>
                          <p style={{ fontSize: 10, color: pos ? "var(--green)" : "var(--red)", marginTop: 1 }}>
                            {pos ? "+" : ""}{row.ate_pct.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <div style={{ height: 4, background: "var(--border)", overflow: "hidden", marginBottom: 4 }}>
                        <motion.div
                          initial={{ width: 0 }} animate={{ width: `${w}%` }}
                          transition={{ delay: 0.4 + i * 0.1, duration: 0.7, ease: "easeOut" }}
                          style={{ height: "100%", background: pos ? "var(--text)" : "var(--red)" }}
                        />
                      </div>
                      <p style={{ fontSize: 10, color: "var(--border-bright)" }}>
                        95% CI: [{row.ci_low.toFixed(3)}, {row.ci_high.toFixed(3)}]
                      </p>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>

            {/* ATE Bar Chart via Recharts */}
            <motion.div
              initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
              transition={{ ...EASE, delay: 0.22 }}
              className="card" style={{ padding: 26 }}
            >
              <div className="section-label" style={{ marginBottom: 20 }}>ATE Chart — with 95% CI Error Bars</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={chartData}
                  margin={{ top: 20, right: 20, bottom: 60, left: 20 }}
                  barSize={40}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis
                    dataKey="treatment"
                    tick={{ fill: "var(--text-dim)", fontSize: 10 }}
                    tickLine={false}
                    axisLine={{ stroke: "var(--border)" }}
                    angle={-20}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis
                    tick={{ fill: "var(--text-dim)", fontSize: 10 }}
                    tickLine={false}
                    axisLine={{ stroke: "var(--border)" }}
                    tickFormatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}`}
                    label={{ value: "q/ha", angle: -90, position: "insideLeft", fill: "var(--text-dim)", fontSize: 10 }}
                  />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                  <ReferenceLine y={0} stroke="var(--border-bright)" strokeDasharray="4 4" />
                  <Bar dataKey="ate_qha" isAnimationActive animationDuration={800} radius={[2,2,0,0]}>
                    {chartData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={ATE_COLORS[i % ATE_COLORS.length]}
                        fillOpacity={0.85}
                      />
                    ))}
                    <ErrorBar dataKey="errorY" width={4} strokeWidth={2} stroke="var(--text)" direction="y" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ marginTop: 14, padding: "10px 14px", background: "rgba(0,0,0,0.2)", border: "1px solid var(--border)" }}>
                <p style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.6 }}>
                  <strong style={{ color: "var(--text)" }}>Note:</strong> ATEs estimated from real observational India crop data.
                  Error bars show 95% bootstrap confidence intervals.
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      )}
    </div>
  );
}
