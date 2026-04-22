"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { API_BASE, FarmInput, AdvisoryResult, Recommendation, defaultFarm, parseApiError } from "@/lib/api";
import FarmForm from "@/components/FarmForm";
import { AlertCircle, AlertTriangle, Info, CheckCircle2, Download } from "lucide-react";
import { downloadAdvisoryPDF } from "@/components/AdvisoryPDF";

const PAGE = { maxWidth: 1100, margin: "0 auto", padding: "100px 28px 80px" };

const PRI = {
  CRITICAL: { icon: <AlertCircle size={18} color="var(--red)" />,    color: "var(--red)",    bg: "var(--surface)", border: "var(--red)" },
  HIGH:     { icon: <AlertTriangle size={18} color="var(--orange)" />, color: "var(--orange)", bg: "var(--surface)", border: "var(--orange)" },
  MEDIUM:   { icon: <Info size={18} color="var(--yellow)" />,          color: "var(--yellow)", bg: "var(--surface)", border: "var(--yellow)" },
  LOW:      { icon: <CheckCircle2 size={18} color="var(--green)" />,   color: "var(--green)",  bg: "var(--surface)", border: "var(--green)" },
};
const PHASES = ["PRE-SEASON", "IN-SEASON", "POST-SEASON"] as const;
const PHASE_COLORS: Record<string, string> = {
  "PRE-SEASON": "#3b82f6", "IN-SEASON": "#22c55e", "POST-SEASON": "#f97316",
};

export default function AdvisoryPage() {
  const [farm, setFarm]         = useState<FarmInput>(defaultFarm);
  const [advisory, setAdvisory] = useState<AdvisoryResult | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [showTrace, setShowTrace] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);

  const exportPDF = async () => {
    if (!advisory) return;
    setPdfLoading(true);
    try { await downloadAdvisoryPDF(advisory, farm as unknown as Record<string, unknown>); }
    finally { setPdfLoading(false); }
  };

  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/advisory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(farm),
      });
      if (!r.ok) throw new Error(await parseApiError(r, "Advisory failed"));
      setAdvisory(await r.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error");
    } finally {
      setLoading(false);
    }
  };

  const byPhase: Record<string, Recommendation[]> = {};
  advisory?.recommendations.forEach((r) => {
    byPhase[r.phase] = [...(byPhase[r.phase] ?? []), r];
  });

  return (
    <div style={PAGE}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        style={{ marginBottom: 44 }}
      >
        <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
          <span className="badge badge-orange">Agent 5 — Advisory</span>
          <span className="badge badge-green">IF-THEN + Counterfactual</span>
        </div>
        <h1 className="font-display" style={{ fontSize: 60, color: "var(--text)", marginBottom: 10 }}>
          Advisory Report
        </h1>
        <p style={{ fontSize: 14, color: "#6b7280", maxWidth: 540, lineHeight: 1.65 }}>
          Agronomic rules calibrated to real India crop data, combined with ML counterfactual
          uplift estimates. Recommendations are ranked CRITICAL → LOW and grouped by season.
        </p>
      </motion.div>

      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: 22, alignItems: "start" }}>

        {/* ── FORM ────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
          className="card" style={{ padding: 26, position: "sticky", top: 80 }}
        >
          <div className="section-label" style={{ marginBottom: 16 }}>Farm Configuration</div>
          <FarmForm
            value={farm} onChange={setFarm} loading={loading}
            onSubmit={submit} submitLabel="▲ Generate Advisory"
          />
          <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ marginTop: 12, padding: "10px 14px", borderRadius: 9, fontSize: 13,
                  background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)", color: "#fca5a5" }}>
                ⚠ {error}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* ── RESULTS ─────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <AnimatePresence mode="wait">
            {advisory ? (
              <motion.div key="advisory" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                style={{ display: "flex", flexDirection: "column", gap: 18 }}>

                {/* Summary strip */}
                <div className="card" style={{
                  padding: "22px 26px",
                  borderColor: "rgba(249,115,22,0.2)", background: "rgba(249,115,22,0.03)",
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  flexWrap: "wrap", gap: 16,
                }}>
                  <div>
                    <p style={{ fontSize: 16, fontWeight: 700, color: "#eef2ff", marginBottom: 6 }}>
                      {advisory.farm_label}
                    </p>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {advisory.n_critical > 0 && (
                        <span style={{ fontSize: 11, padding: "2px 10px", borderRadius: 99,
                          background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.25)", color: "#fca5a5" }}>
                          🔴 {advisory.n_critical} Critical
                        </span>
                      )}
                      {advisory.n_high > 0 && (
                        <span style={{ fontSize: 11, padding: "2px 10px", borderRadius: 99,
                          background: "rgba(249,115,22,0.1)", border: "1px solid rgba(249,115,22,0.25)", color: "#fdba74" }}>
                          🟠 {advisory.n_high} High
                        </span>
                      )}
                      <span style={{ fontSize: 11, color: "#6b7280", padding: "2px 4px" }}>
                        {advisory.recommendations.length} total recommendations
                      </span>
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
                    <div className="stat-num" style={{ fontSize: 44, color: "var(--green)", lineHeight: 1 }}>
                      {advisory.predicted_yield.toFixed(1)}
                    </div>
                    <div style={{ fontSize: 11, color: "#6b7280", marginTop: 3 }}>q/ha predicted</div>
                    <button onClick={exportPDF} disabled={pdfLoading}
                      style={{ display: "flex", alignItems: "center", gap: 6, padding: "7px 14px",
                        background: "transparent", border: "1px solid var(--border-bright)",
                        color: "var(--text-dim)", fontSize: 11, cursor: pdfLoading ? "not-allowed" : "pointer",
                        fontWeight: 600, letterSpacing: "0.05em" }}>
                      <Download size={12} />
                      {pdfLoading ? "Generating…" : "Export PDF"}
                    </button>
                  </div>
                </div>

                {/* Phase sections */}
                {PHASES.filter((p) => byPhase[p]?.length).map((phase, pi) => (
                  <motion.div key={phase} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: pi * 0.08 }}>
                    {/* Phase header */}
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                      <div style={{
                        padding: "3px 11px", borderRadius: 99, fontSize: 10, fontWeight: 800,
                        background: `${PHASE_COLORS[phase]}12`,
                        border: `1px solid ${PHASE_COLORS[phase]}28`,
                        color: PHASE_COLORS[phase],
                      }}>{phase}</div>
                      <div style={{ height: 1, flex: 1, background: "rgba(255,255,255,0.05)" }} />
                      <span style={{ fontSize: 10, color: "#4b5563" }}>{byPhase[phase].length} rec{byPhase[phase].length > 1 ? "s" : ""}</span>
                    </div>

                    {/* Cards */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
                      {byPhase[phase].map((rec, i) => {
                        const cfg = PRI[rec.priority] ?? PRI.LOW;
                        return (
                          <motion.div key={i}
                            initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.04 * i }}
                            style={{ padding: "16px 18px", borderRadius: 12, background: cfg.bg, border: `1px solid ${cfg.border}` }}>
                            <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                              <span style={{ fontSize: 16, lineHeight: 1, flexShrink: 0 }}>{cfg.icon}</span>
                              <div style={{ flex: 1 }}>
                                <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6, flexWrap: "wrap" }}>
                                  <span style={{
                                    fontSize: 9, fontWeight: 900, letterSpacing: "0.07em",
                                    padding: "2px 7px", borderRadius: 3,
                                    background: `${cfg.color}18`, color: cfg.color,
                                  }}>{rec.priority}</span>
                                  <span style={{ fontSize: 10, color: "#6b7280" }}>{rec.confidence}</span>
                                  {rec.delta_yield !== undefined && rec.delta_yield > 0 && (
                                    <span style={{ fontSize: 10, fontWeight: 700, color: "#22c55e" }}>
                                      +{rec.delta_yield.toFixed(1)} q/ha gain
                                    </span>
                                  )}
                                </div>
                                <p style={{ fontSize: 13, fontWeight: 600, color: "#eef2ff", marginBottom: 5, lineHeight: 1.5 }}>
                                  {rec.recommendation}
                                </p>
                                <p style={{ fontSize: 11, color: "#6b7280", lineHeight: 1.5 }}>📌 {rec.basis}</p>
                              </div>
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  </motion.div>
                ))}

                {/* Traceability */}
                {advisory.traceability && (
                  <div className="card" style={{ overflow: "hidden" }}>
                    <button
                      onClick={() => setShowTrace((v) => !v)}
                      style={{
                        width: "100%", padding: "16px 22px", display: "flex",
                        alignItems: "center", justifyContent: "space-between",
                        background: "none", border: "none", cursor: "pointer",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{ fontSize: 14 }}>🔗</span>
                        <span style={{ fontSize: 13, fontWeight: 600, color: "#eef2ff" }}>End-to-End Traceability Chain</span>
                        <span style={{ fontSize: 10, color: "#4b5563" }}>Data → Model → Advisory</span>
                      </div>
                      <span style={{ color: "#6b7280", fontSize: 12 }}>{showTrace ? "▲" : "▼"}</span>
                    </button>
                    <AnimatePresence>
                      {showTrace && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.3 }}
                          style={{ overflow: "hidden" }}
                        >
                          <div style={{ padding: "0 22px 22px" }}>
                            <pre style={{
                              fontSize: 11, lineHeight: 1.7, color: "#94a3b8",
                              background: "rgba(0,0,0,0.3)", borderRadius: 10, padding: 16,
                              overflowX: "auto",
                              fontFamily: "JetBrains Mono, 'Courier New', monospace",
                              border: "1px solid rgba(255,255,255,0.05)",
                            }}>{advisory.traceability}</pre>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}

              </motion.div>
            ) : (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="card" style={{ padding: 80, textAlign: "center" }}>
                <div style={{ fontSize: 44, marginBottom: 16, opacity: 0.12 }}>▲</div>
                <p style={{ fontSize: 15, fontWeight: 700, color: "#eef2ff", marginBottom: 8 }}>
                  Configure Your Farm
                </p>
                <p style={{ color: "#6b7280", fontSize: 13, lineHeight: 1.6 }}>
                  Set parameters on the left and click{" "}
                  <strong style={{ color: "#f97316" }}>Generate Advisory</strong> to receive
                  priority-ranked recommendations calibrated to real India crop data.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
