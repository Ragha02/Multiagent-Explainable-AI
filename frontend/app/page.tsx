"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { API_BASE, PipelineStatus } from "@/lib/api";

// ── Real Kaggle data stats ──────────────────────────────────────────────────
const STATS = [
  { label: "Source Records",     value: "246k",  sub: "crop_production.csv",          color: "#22c55e", icon: "🗃" },
  { label: "Training Samples",   value: "50k",   sub: "stratified by state × crop",   color: "#14b8a6", icon: "📦" },
  { label: "Feature Dimensions", value: "41",    sub: "after encoding & engineering",  color: "#3b82f6", icon: "📐" },
  { label: "XAI Methods",        value: "4",     sub: "SHAP · LIME · ΔSHAP · ATE",    color: "#a855f7", icon: "✦"  },
];

const PAGES = [
  {
    href: "/predict", icon: "🌾", color: "#22c55e", tag: "Agent 2",
    title: "Yield Prediction",
    desc: "Set farm parameters — irrigation, rainfall, soil, NPK — and get instant ensemble predictions with SHAP attribution.",
  },
  {
    href: "/causal", icon: "⟶", color: "#14b8a6", tag: "Agent 3",
    title: "Causal Analysis",
    desc: "4-layer causal DAG over real India crop data with average treatment effects for controllable interventions.",
  },
  {
    href: "/explain", icon: "✦", color: "#a855f7", tag: "Agent 4",
    title: "Explanations",
    desc: "Global SHAP rankings, local LIME waterfall, contrastive ΔSHAP and causal ATEs — all from real data.",
  },
  {
    href: "/advisory", icon: "▲", color: "#f97316", tag: "Agent 5",
    title: "Advisory Report",
    desc: "IF-THEN agronomic rules + ML counterfactuals, priority-ranked across pre, in & post-season phases.",
  },
];

const AGENTS = [
  { n: "A1", name: "Data",       thresh: 10, desc: "246k rows → 50k sample" },
  { n: "A2", name: "Prediction", thresh: 30, desc: "RF + XGBoost + MLR"     },
  { n: "A3", name: "Causal",     thresh: 55, desc: "DAG + ATE"              },
  { n: "A4", name: "Explain",    thresh: 75, desc: "SHAP + LIME"            },
  { n: "A5", name: "Advisory",   thresh: 92, desc: "Rules + Counterfactual" },
];

const SOURCES = [
  { name: "crop_production.csv",      label: "Gov. crop production",  color: "#22c55e", rows: "246k" },
  { name: "yield_df.csv",             label: "FAO climate (India)",   color: "#14b8a6", rows: "4k"   },
  { name: "Crop_recommendation.csv",  label: "Soil / NPK medians",   color: "#a855f7", rows: "2.2k" },
];

function agentState(progress: number, thresh: number) {
  if (progress >= thresh + 15) return "done";
  if (progress >= thresh)       return "active";
  return "idle";
}

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show:   { opacity: 1, y:  0 },
};
const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.07, delayChildren: 0 } } };

export default function Dashboard() {
  const [status, setStatus] = useState<PipelineStatus>({
    status: "idle", progress: 0, current_step: "Connecting…",
  });

  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/pipeline/status`);
        const d: PipelineStatus = await r.json();
        setStatus(d);
        if (d.status === "ready" || d.status === "error") clearInterval(id);
      } catch {}
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const ensemble  = status.metrics?.["MA-XAI Ensemble"];
  const rf        = status.metrics?.["Random Forest"];
  const isReady   = status.status === "ready";
  const isRunning = status.status === "running";

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "100px 28px 100px" }}>

      {/* ── HERO ──────────────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.4,0,0.2,1] }}
        style={{ textAlign: "center", marginBottom: 64, position: "relative" }}
      >
        <div style={{
          position: "absolute", top: -80, left: "50%", transform: "translateX(-50%)",
          width: "100%", height: 300,
          background: "url('data:image/svg+xml,%3Csvg viewBox=\"0 0 200 200\" xmlns=\"http://www.w3.org/2000/svg\"%3E%3Cfilter id=\"noiseFilter\"%3E%3CfeTurbulence type=\"fractalNoise\" baseFrequency=\"0.85\" numOctaves=\"3\" stitchTiles=\"stitch\"/%3E%3C/filter%3E%3Crect width=\"100%25\" height=\"100%25\" filter=\"url(%23noiseFilter)\" opacity=\"0.1\"/%3E%3C/svg%3E')",
          maskImage: "linear-gradient(to bottom, black, transparent)",
          WebkitMaskImage: "linear-gradient(to bottom, black, transparent)",
          pointerEvents: "none", zIndex: -1
        }} />

        {/* Real data badge */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          style={{ marginBottom: 20, display: "flex", alignItems: "center", justifyContent: "center", gap: 16 }}
        >
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 6,
            padding: "5px 14px", border: "1px solid var(--border-bright)", background: "var(--surface)",
            fontSize: 12, fontWeight: 700, color: "var(--green)", textTransform: "uppercase", letterSpacing: "0.05em",
          }}>
            <span style={{ width: 6, height: 6, background: "var(--green)" }} />
            Real Kaggle Data Active
          </span>
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 6,
            padding: "5px 14px", border: "1px solid var(--border)", background: "var(--surface)",
            fontSize: 12, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.05em"
          }}>🇮🇳 India Agricultural Census</span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="font-display"
          style={{
            fontSize: "clamp(44px, 7vw, 80px)", lineHeight: 1, marginBottom: 24, color: "var(--text)",
          }}
        >
          Multi-Agent <br/>
          <span className="grad-text">Explainable AI</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ delay: 0.35 }}
          style={{ fontSize: 16, color: "#6b7280", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}
        >
          A 5-agent pipeline trained on <strong style={{ color: "#94a3b8" }}>246,000 real India crop records</strong> —
          delivering yield prediction, causal inference, SHAP/LIME explanations &amp; actionable advisories.
        </motion.p>
      </motion.div>

      {/* ── STATS ─────────────────────────────────────────────────────────── */}
      <motion.div
        variants={stagger} initial="hidden" animate="show"
        style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 32 }}
      >
        {STATS.map((s) => (
          <motion.div key={s.label} variants={fadeUp}>
            <div className="card" style={{ padding: "22px 18px" }}>
              <div style={{ fontSize: 20, marginBottom: 10 }}>{s.icon}</div>
              <div className="stat-num" style={{ fontSize: 32, fontWeight: 900, color: s.color, lineHeight: 1, marginBottom: 5 }}>
                {s.value}
              </div>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#eef2ff", marginBottom: 3 }}>{s.label}</div>
              <div style={{ fontSize: 10, color: "#4b5563", fontStyle: "italic" }}>{s.sub}</div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* ── DATA SOURCES ──────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
        className="card" style={{ padding: "20px 24px", marginBottom: 32 }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 20, flexWrap: "wrap" }}>
          <span className="section-label">Kaggle Datasets</span>
          {SOURCES.map((s) => (
            <div key={s.name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: s.color, flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: "#94a3b8", fontWeight: 500 }}>{s.label}</span>
              <span className="stat-num" style={{ fontSize: 11, color: s.color, fontWeight: 700 }}>{s.rows} rows</span>
            </div>
          ))}
          <div style={{ marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" }}>
            <span style={{ fontSize: 11, color: "#4b5563" }}>Merged →</span>
            <span className="stat-num" style={{ fontSize: 12, fontWeight: 800, color: "#22c55e" }}>50k stratified sample</span>
          </div>
        </div>
      </motion.div>

      {/* ── PIPELINE STATUS ───────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card" style={{ padding: 28, marginBottom: 32 }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
          <div>
            <div className="section-label" style={{ marginBottom: 6 }}>5-Agent ML Pipeline</div>
            <p style={{ fontSize: 14, color: "#eef2ff", fontWeight: 600 }}>{status.current_step}</p>
          </div>
          <div style={{
            padding: "5px 14px", borderRadius: 99, fontSize: 12, fontWeight: 700,
            background: isReady ? "rgba(34,197,94,0.1)" : isRunning ? "rgba(234,179,8,0.1)" : "rgba(100,116,139,0.1)",
            border: `1px solid ${isReady ? "rgba(34,197,94,0.3)" : isRunning ? "rgba(234,179,8,0.3)" : "rgba(100,116,139,0.2)"}`,
            color: isReady ? "#22c55e" : isRunning ? "#eab308" : "#6b7280",
          }}>
            {isReady ? "✓ Ready" : isRunning ? `${status.progress}%` : "Idle"}
          </div>
        </div>

        <div className="progress-track" style={{ marginBottom: 24 }}>
          <div className="progress-fill" style={{ width: `${status.progress}%` }} />
        </div>

        {/* Agent nodes */}
        <div style={{ display: "flex", alignItems: "stretch", gap: 0 }}>
          {AGENTS.map((a, i) => {
            const state = agentState(status.progress, a.thresh);
            const C = {
              done:   { bg: "rgba(34,197,94,0.08)",   border: "rgba(34,197,94,0.3)",   text: "#22c55e",  label: "#86efac"  },
              active: { bg: "rgba(234,179,8,0.1)",    border: "rgba(234,179,8,0.45)",   text: "#eab308",  label: "#fde047"  },
              idle:   { bg: "rgba(255,255,255,0.02)", border: "rgba(255,255,255,0.06)", text: "#374151",  label: "#374151"  },
            }[state];
            return (
              <div key={a.n} style={{ display: "flex", alignItems: "center", flex: 1 }}>
                <div style={{
                  flex: 1, padding: "12px 10px", borderRadius: 10,
                  background: C.bg, border: `1px solid ${C.border}`,
                  transition: "all 0.5s ease",
                  boxShadow: state === "active" ? "0 0 16px rgba(234,179,8,0.1)" : "none",
                }}>
                  <div style={{ fontSize: 10, fontWeight: 800, color: C.text, marginBottom: 3 }}>
                    {state === "done" ? "✓ " : state === "active" ? "⟳ " : ""}{a.n}
                  </div>
                  <div style={{ fontSize: 12, color: state === "idle" ? "#374151" : "#eef2ff", fontWeight: 600, marginBottom: 2 }}>
                    {a.name}
                  </div>
                  <div style={{ fontSize: 9, color: C.label, opacity: 0.8 }}>{a.desc}</div>
                </div>
                {i < AGENTS.length - 1 && (
                  <div style={{
                    width: 22, textAlign: "center", fontSize: 12,
                    color: state === "done" ? "#22c55e" : "#1f2937",
                    transition: "color 0.5s ease", flexShrink: 0,
                  }}>→</div>
                )}
              </div>
            );
          })}
        </div>
      </motion.div>

      {/* ── METRICS (when ready) ──────────────────────────────────────────── */}
      {isReady && ensemble && (
        <motion.div
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="card"
          style={{ padding: 28, marginBottom: 32, borderColor: "rgba(34,197,94,0.2)", background: "rgba(34,197,94,0.03)" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
            <div className="section-label">✓ Ensemble Performance — Real India Data</div>
            {rf && (
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Random Forest R² <span className="stat-num" style={{ color: "#22c55e", fontWeight: 700 }}>{rf.R2?.toFixed(3)}</span>
              </div>
            )}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
            {[
              { k: "RMSE", v: ensemble.RMSE?.toFixed(2), u: "q/ha", note: "lower is better"  },
              { k: "MAE",  v: ensemble.MAE?.toFixed(2),  u: "q/ha", note: "lower is better"  },
              { k: "R²",   v: ensemble.R2?.toFixed(4),   u: "",     note: "closer to 1.0"    },
              { k: "MAPE", v: ensemble.MAPE?.toFixed(1), u: "%",    note: "% error"           },
            ].map((m, i) => (
              <motion.div
                key={m.k}
                initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.07 }}
                style={{
                  padding: "18px 14px", borderRadius: 12, textAlign: "center",
                  background: "rgba(34,197,94,0.05)", border: "1px solid rgba(34,197,94,0.12)",
                }}
              >
                <div className="stat-num" style={{ fontSize: 26, fontWeight: 900, color: "#22c55e", marginBottom: 3 }}>
                  {m.v}<span style={{ fontSize: 12, color: "#6b7280", marginLeft: 2 }}>{m.u}</span>
                </div>
                <div style={{ fontSize: 11, fontWeight: 700, color: "#eef2ff", marginBottom: 2 }}>{m.k}</div>
                <div style={{ fontSize: 10, color: "#4b5563" }}>{m.note}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* ── PAGES ─────────────────────────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }}>
        <div className="section-label" style={{ marginBottom: 18 }}>Explore the Framework</div>
      </motion.div>

      <motion.div
        variants={stagger} initial="hidden" animate="show"
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}
      >
        {PAGES.map((p) => (
          <motion.div key={p.href} variants={fadeUp}>
            <Link
              href={p.href}
              style={{
                display: "flex", alignItems: "center", gap: 16,
                padding: "22px 20px", borderRadius: 16,
                background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.07)",
                textDecoration: "none", transition: "all 0.25s ease",
              }}
              onMouseEnter={(e) => {
                const el = e.currentTarget as HTMLElement;
                el.style.background = "rgba(255,255,255,0.05)";
                el.style.borderColor = `${p.color}28`;
                el.style.transform = "translateY(-2px)";
              }}
              onMouseLeave={(e) => {
                const el = e.currentTarget as HTMLElement;
                el.style.background = "rgba(255,255,255,0.025)";
                el.style.borderColor = "rgba(255,255,255,0.07)";
                el.style.transform = "translateY(0)";
              }}
            >
              <div style={{
                width: 46, height: 46, borderRadius: 12, flexShrink: 0,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 20, background: `${p.color}12`, border: `1px solid ${p.color}25`,
              }}>{p.icon}</div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: p.color, marginBottom: 3, letterSpacing: "0.07em" }}>{p.tag}</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#eef2ff", marginBottom: 4 }}>{p.title}</div>
                <div style={{ fontSize: 12, color: "#6b7280", lineHeight: 1.55 }}>{p.desc}</div>
              </div>
              <div style={{ color: p.color, fontSize: 16, flexShrink: 0, opacity: 0.6 }}>→</div>
            </Link>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
