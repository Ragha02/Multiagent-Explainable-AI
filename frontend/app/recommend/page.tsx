"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { API_BASE, parseApiError } from "@/lib/api";
import { Sprout, Droplets, Thermometer, Wind, FlaskConical, Zap } from "lucide-react";

const PAGE = { maxWidth: 1100, margin: "0 auto", padding: "100px 28px 80px" };

const CROP_COLORS: Record<string, string> = {
  Rice: "#14b8a6", Wheat: "#f59e0b", Maize: "#eab308", Soybean: "#84cc16",
  Cotton: "#8b5cf6", Sugarcane: "#22c55e", Groundnut: "#f97316",
  Bajra: "#fb923c", Jowar: "#a78bfa", Mustard: "#facc15",
};

const CROP_EMOJI: Record<string, string> = {
  Rice: "🌾", Wheat: "🌽", Maize: "🌽", Soybean: "🫘", Cotton: "🌿",
  Sugarcane: "🎋", Groundnut: "🥜", Bajra: "🌾", Jowar: "🌾", Mustard: "🌻",
};

interface Recommendation {
  rank: number;
  crop: string;
  confidence: number;
  yield_p25: number;
  yield_median: number;
  yield_p75: number;
  sample_size: number;
  note: string;
}

const FIELDS = [
  { key: "N",          label: "Nitrogen",       icon: <FlaskConical size={14}/>, unit: "kg/ha",  min: 0,   max: 300, step: 5,   default: 80  },
  { key: "P",          label: "Phosphorus",      icon: <FlaskConical size={14}/>, unit: "kg/ha",  min: 0,   max: 145, step: 2,   default: 40  },
  { key: "K",          label: "Potassium",       icon: <FlaskConical size={14}/>, unit: "kg/ha",  min: 0,   max: 205, step: 2,   default: 40  },
  { key: "ph",         label: "Soil pH",         icon: <Zap size={14}/>,          unit: "",       min: 3.5, max: 9.5, step: 0.1, default: 6.5 },
  { key: "humidity",   label: "Humidity",        icon: <Droplets size={14}/>,     unit: "%",      min: 10,  max: 100, step: 1,   default: 65  },
  { key: "rainfall",   label: "Rainfall",        icon: <Wind size={14}/>,         unit: "mm/yr",  min: 50,  max: 3000,step: 50,  default: 700 },
  { key: "temperature",label: "Temperature",     icon: <Thermometer size={14}/>,  unit: "°C",     min: 5,   max: 50,  step: 0.5, default: 27  },
];

export default function RecommendPage() {
  const [params, setParams] = useState<Record<string, number>>(
    Object.fromEntries(FIELDS.map((f) => [f.key, f.default]))
  );
  const [results, setResults] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [ran, setRan]         = useState(false);

  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...params, top_k: 3 }),
      });
      if (!r.ok) throw new Error(await parseApiError(r, "Recommendation failed"));
      const d = await r.json();
      setResults(d.recommendations ?? []);
      setRan(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={PAGE}>
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: 44 }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
          <span className="badge badge-green">Agent 6 — Recommend</span>
          <span className="badge badge-teal">Crop_recommendation.csv</span>
        </div>
        <h1 className="font-display" style={{ fontSize: 60, color: "var(--text)", marginBottom: 10 }}>
          Crop Recommendation
        </h1>
        <p style={{ fontSize: 14, color: "var(--text-dim)", maxWidth: 540, lineHeight: 1.65 }}>
          Enter your soil and climate parameters. Our RF + Gradient Boosting ensemble trained on
          2,200 real ICAR soil records recommends the top 3 most suitable crops.
        </p>
      </motion.div>

      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: 22, alignItems: "start" }}>

        {/* Form */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}
          className="card" style={{ padding: 26, position: "sticky", top: 80 }}>
          <div className="section-label" style={{ marginBottom: 20 }}>Soil & Climate Parameters</div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {FIELDS.map((f) => (
              <div key={f.key}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
                    <span style={{ color: "var(--green)" }}>{f.icon}</span>
                    {f.label}
                  </label>
                  <span className="stat-num" style={{ fontSize: 13, color: "var(--text)" }}>
                    {params[f.key]}{f.unit && ` ${f.unit}`}
                  </span>
                </div>
                <input
                  type="range" min={f.min} max={f.max} step={f.step}
                  value={params[f.key]}
                  onChange={(e) => setParams((p) => ({ ...p, [f.key]: parseFloat(e.target.value) }))}
                  style={{ width: "100%", accentColor: "var(--green)", cursor: "pointer" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--border-bright)", marginTop: 2 }}>
                  <span>{f.min}{f.unit}</span>
                  <span>{f.max}{f.unit}</span>
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={submit} disabled={loading}
            className="btn"
            style={{ width: "100%", marginTop: 20, background: "var(--green)", color: "var(--bg)", border: "none",
              padding: "12px", fontSize: 13, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
              letterSpacing: "0.05em", textTransform: "uppercase", opacity: loading ? 0.6 : 1 }}
          >
            {loading ? "Analysing…" : "🌾 Get Crop Recommendations"}
          </button>

          {error && (
            <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(239,68,68,0.08)",
              border: "1px solid var(--red)", color: "var(--red)", fontSize: 12 }}>
              ⚠ {error}
            </div>
          )}
        </motion.div>

        {/* Results */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <AnimatePresence mode="wait">
            {!ran ? (
              <motion.div key="empty" className="card" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                style={{ padding: "72px 28px", textAlign: "center" }}>
                <Sprout size={48} color="var(--border-bright)" style={{ margin: "0 auto 20px" }} />
                <p style={{ fontSize: 15, fontWeight: 600, color: "var(--text)", marginBottom: 8 }}>Configure Your Soil</p>
                <p style={{ color: "var(--text-dim)", fontSize: 13, lineHeight: 1.6 }}>
                  Adjust the sliders to match your field's soil and climate profile,<br />
                  then click <strong style={{ color: "var(--green)" }}>Get Crop Recommendations</strong>.
                </p>
              </motion.div>
            ) : (
              <motion.div key="results" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
                {/* Top result highlight */}
                {results.length > 0 && (
                  <motion.div className="card" initial={{ scale: 0.96 }} animate={{ scale: 1 }}
                    style={{ padding: 28, marginBottom: 16, borderColor: CROP_COLORS[results[0].crop] ?? "var(--green)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 20, flexWrap: "wrap" }}>
                      <div>
                        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", color: "var(--text-dim)", marginBottom: 10 }}>
                          TOP RECOMMENDATION
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                          <span style={{ fontSize: 48 }}>{CROP_EMOJI[results[0].crop] ?? "🌱"}</span>
                          <div>
                            <h2 className="font-display" style={{ fontSize: 42, color: CROP_COLORS[results[0].crop] ?? "var(--green)", lineHeight: 1 }}>
                              {results[0].crop}
                            </h2>
                            <p style={{ fontSize: 13, color: "var(--text-dim)", marginTop: 4 }}>{results[0].note}</p>
                          </div>
                        </div>
                      </div>
                      <div style={{ textAlign: "right", flexShrink: 0 }}>
                        <div className="stat-num" style={{ fontSize: 52, color: CROP_COLORS[results[0].crop] ?? "var(--green)" }}>
                          {results[0].confidence.toFixed(0)}%
                        </div>
                        <div style={{ fontSize: 11, color: "var(--text-dim)" }}>model confidence</div>
                      </div>
                    </div>

                    {/* Yield range */}
                    {results[0].yield_median > 0 && (
                      <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
                        <div style={{ fontSize: 10, color: "var(--text-dim)", marginBottom: 10, letterSpacing: "0.08em" }}>
                          EXPECTED YIELD RANGE — from {results[0].sample_size.toLocaleString()} real India records
                        </div>
                        <div style={{ display: "flex", gap: 24 }}>
                          {[
                            { label: "25th pct", v: results[0].yield_p25 },
                            { label: "Median",   v: results[0].yield_median },
                            { label: "75th pct", v: results[0].yield_p75 },
                          ].map((d) => (
                            <div key={d.label}>
                              <div className="stat-num" style={{ fontSize: 22, color: "var(--text)" }}>{d.v}</div>
                              <div style={{ fontSize: 10, color: "var(--text-dim)" }}>{d.label} q/ha</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* Runner-up cards */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                  {results.slice(1).map((rec, i) => (
                    <motion.div key={rec.rank} className="card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 + i * 0.07 }}
                      style={{ padding: 22, borderColor: CROP_COLORS[rec.crop] ?? "var(--border)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                          <span style={{ fontSize: 24 }}>{CROP_EMOJI[rec.crop] ?? "🌱"}</span>
                          <div>
                            <p style={{ fontSize: 10, color: "var(--text-dim)", fontWeight: 700, letterSpacing: "0.08em" }}>RANK #{rec.rank}</p>
                            <p className="font-display" style={{ fontSize: 22, color: CROP_COLORS[rec.crop] ?? "var(--text)" }}>{rec.crop}</p>
                          </div>
                        </div>
                        <div className="stat-num" style={{ fontSize: 28, color: CROP_COLORS[rec.crop] ?? "var(--text-dim)" }}>
                          {rec.confidence.toFixed(0)}%
                        </div>
                      </div>
                      <p style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.55, marginBottom: 12 }}>{rec.note}</p>
                      {rec.yield_median > 0 && (
                        <div style={{ display: "flex", gap: 12, fontSize: 11, color: "var(--text-dim)", paddingTop: 10, borderTop: "1px solid var(--border)" }}>
                          <span>p25: <strong style={{ color: "var(--text)" }}>{rec.yield_p25} q/ha</strong></span>
                          <span>Median: <strong style={{ color: "var(--text)" }}>{rec.yield_median} q/ha</strong></span>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
