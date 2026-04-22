"use client";
import { useEffect, useState } from "react";
import {
  FarmInput, defaultFarm, stressFarm, highYieldFarm,
  fetchLocations, fetchLocationDefaults, LocationData,
} from "@/lib/api";

// ── Slider config ─────────────────────────────────────────────────────────────
const SLIDERS = [
  { key: "prev_year_yield"         as const, label: "Prev. Year Yield",     unit: "q/ha",  min: 2,    max: 100,  step: 0.5, color: "#22c55e",
    hint: "#1 SHAP predictor — historical yield" },
  { key: "rainfall_annual"         as const, label: "Annual Rainfall",      unit: "mm",    min: 50,   max: 2500, step: 10,  color: "#3b82f6",
    hint: "District avg rainfall" },
  { key: "temp_mean"               as const, label: "Mean Temperature",     unit: "°C",    min: 10,   max: 48,   step: 0.5, color: "#ef4444",
    hint: "Seasonal average" },
  { key: "irrigation_coverage_pct" as const, label: "Irrigation Coverage",  unit: "%",     min: 0,    max: 100,  step: 1,   color: "#14b8a6",
    hint: "% area irrigated" },
  { key: "npk_dosage_kg_ha"        as const, label: "NPK Dosage",           unit: "kg/ha", min: 10,   max: 400,  step: 5,   color: "#eab308",
    hint: "Total fertiliser applied" },
  { key: "soil_moisture"           as const, label: "Soil Moisture",        unit: "%",     min: 5,    max: 90,   step: 1,   color: "#a855f7",
    hint: "Surface soil moisture" },
  { key: "soil_ph"                 as const, label: "Soil pH",              unit: "",      min: 4.0,  max: 9.5,  step: 0.1, color: "#f97316",
    hint: "Optimal 6.0–7.5" },
  { key: "organic_carbon"          as const, label: "Organic Carbon",       unit: "%",     min: 0.05, max: 3.5,  step: 0.05,color: "#22c55e",
    hint: "Soil health indicator" },
  { key: "sowing_week"             as const, label: "Sowing Week",          unit: "wk",    min: 1,    max: 52,   step: 1,   color: "#14b8a6",
    hint: "Week of year (1–52)" },
];

const PRESETS = [
  { label: "✦ Healthy",    farm: defaultFarm,   color: "#22c55e", bg: "rgba(34,197,94,0.08)",  border: "rgba(34,197,94,0.25)"  },
  { label: "⬆ High-Yield", farm: highYieldFarm, color: "#3b82f6", bg: "rgba(59,130,246,0.08)", border: "rgba(59,130,246,0.25)" },
  { label: "⚠ Stress",     farm: stressFarm,    color: "#ef4444", bg: "rgba(239,68,68,0.08)",  border: "rgba(239,68,68,0.25)"  },
];

type Props = {
  value: FarmInput;
  onChange: (f: FarmInput) => void;
  loading?: boolean;
  onSubmit: () => void;
  onLocationFilled?: () => void;   // auto-submit hook
  submitLabel?: string;
};

const SELECT_STYLE: React.CSSProperties = {
  width: "100%", padding: "8px 30px 8px 10px", borderRadius: 9, fontSize: 12,
  background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.09)",
  color: "#eef2ff", outline: "none", cursor: "pointer", boxSizing: "border-box",
  appearance: "none" as "none", WebkitAppearance: "none" as "none",
};

export default function FarmForm({ value, onChange, loading, onSubmit, onLocationFilled, submitLabel = "🌾 Run Prediction" }: Props) {
  const [locs, setLocs]               = useState<LocationData | null>(null);
  const [locsLoading, setLocsLoading] = useState(false);
  const [locsError, setLocsError]     = useState<string | null>(null);
  const [selState, setSelState]       = useState<string>("");
  const [selDistrict, setSelDistrict] = useState<string>("");
  const [selCrop, setSelCrop]         = useState<string>("");
  const [fillLoading, setFillLoading] = useState(false);
  const [fillInfo, setFillInfo]       = useState<string | null>(null);

  const set = (key: keyof FarmInput, v: number | string) => onChange({ ...value, [key]: v });

  // Load location list once pipeline is ready
  useEffect(() => {
    setLocsLoading(true);
    fetchLocations()
      .then(setLocs)
      .catch(() => setLocsError("Locations unavailable — pipeline still loading?"))
      .finally(() => setLocsLoading(false));
  }, []);

  // Apply defaults fetched from API
  const applyDefaults = async (state: string, district: string, crop?: string) => {
    setFillLoading(true); setFillInfo(null);
    try {
      const res = await fetchLocationDefaults(state, district, crop);
      const d   = res.defaults;
      onChange({
        ...value,
        ...(d.prev_year_yield         !== undefined && { prev_year_yield: d.prev_year_yield }),
        ...(d.rainfall_annual         !== undefined && { rainfall_annual: d.rainfall_annual }),
        ...(d.temp_mean               !== undefined && { temp_mean: d.temp_mean }),
        ...(d.soil_ph                 !== undefined && { soil_ph: d.soil_ph }),
        ...(d.irrigation_coverage_pct !== undefined && { irrigation_coverage_pct: d.irrigation_coverage_pct }),
        ...(d.soil_moisture           !== undefined && { soil_moisture: d.soil_moisture }),
        ...(d.organic_carbon          !== undefined && { organic_carbon: d.organic_carbon }),
        ...(d.npk_dosage_kg_ha        !== undefined && { npk_dosage_kg_ha: d.npk_dosage_kg_ha }),
        ...(d.sowing_week             !== undefined && { sowing_week: d.sowing_week }),
        ...(d.variety_improved        !== undefined && { variety_improved: d.variety_improved }),
        farm_label: crop
          ? `${crop} — ${district}, ${state}`
          : `${district}, ${state}`,
      });
      const src = crop ? `${d.n_records} real ${crop} records in ${district}` : `${d.n_records} real records in ${district}`;
      setFillInfo(`✓ Auto-filled from ${src}`);
      // Notify parent so it can auto-submit
      onLocationFilled?.();
    } catch {
      setFillInfo("Could not load defaults — sliders unchanged");
    } finally {
      setFillLoading(false);
    }
  };

  // When district selected — reset crop, load district defaults
  const handleDistrictSelect = (district: string) => {
    setSelDistrict(district); setSelCrop("");
    if (district && selState) applyDefaults(selState, district);
  };

  // When crop selected — reload defaults filtered to that crop
  const handleCropSelect = (crop: string) => {
    setSelCrop(crop);
    if (crop && selState && selDistrict) applyDefaults(selState, selDistrict, crop);
  };

  const districts = selState && locs ? (locs.districts[selState] ?? []) : [];
  const crops = selState && selDistrict && locs
    ? (locs.crops_by_district?.[`${selState}||${selDistrict}`] ?? [])
    : [];

  const clearLocation = () => {
    setSelState(""); setSelDistrict(""); setSelCrop(""); setFillInfo(null);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* ── LOCATION SELECTOR ──────────────────────────────────────────── */}
      <div style={{
        padding: "16px 14px", borderRadius: 12,
        background: "rgba(34,197,94,0.05)", border: "1px solid rgba(34,197,94,0.15)",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
          <span style={{ fontSize: 10, fontWeight: 800, color: "#22c55e", letterSpacing: "0.08em" }}>
            📍 SELECT LOCATION
          </span>
          {selState && (
            <button
              onClick={clearLocation}
              style={{
                fontSize: 10, color: "#4b5563", background: "none", border: "none",
                cursor: "pointer", padding: "1px 6px",
              }}
            >✕ Clear</button>
          )}
        </div>

        {locsLoading ? (
          <p style={{ fontSize: 11, color: "#4b5563" }}>Loading locations…</p>
        ) : locsError ? (
          <p style={{ fontSize: 11, color: "#ef4444" }}>{locsError}</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>

            {/* State select */}
            <div style={{ position: "relative" }}>
              <label style={{ fontSize: 9, fontWeight: 700, color: "#6b7280", display: "block", marginBottom: 4, letterSpacing: "0.06em" }}>
                STATE
              </label>
              <select
                value={selState}
                onChange={(e) => { setSelState(e.target.value); setSelDistrict(""); setSelCrop(""); setFillInfo(null); }}
                style={SELECT_STYLE}
              >
                <option value="">— Select a state —</option>
                {locs?.states.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
              <span style={{ position: "absolute", right: 10, top: 27, color: "#6b7280", fontSize: 10, pointerEvents: "none" }}>▼</span>
            </div>

            {/* District select — only when state chosen */}
            {selState && (
              <div style={{ position: "relative" }}>
                <label style={{ fontSize: 9, fontWeight: 700, color: "#6b7280", display: "block", marginBottom: 4, letterSpacing: "0.06em" }}>
                  DISTRICT  <span style={{ color: "#374151", fontSize: 9 }}>({districts.length} available)</span>
                </label>
                <select
                  value={selDistrict}
                  onChange={(e) => handleDistrictSelect(e.target.value)}
                  style={SELECT_STYLE}
                >
                  <option value="">— Select a district —</option>
                  {districts.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
                <span style={{ position: "absolute", right: 10, top: 27, color: "#6b7280", fontSize: 10, pointerEvents: "none" }}>▼</span>
              </div>
            )}

            {/* Crop select — only when district chosen */}
            {selDistrict && (
              <div style={{ position: "relative" }}>
                <label style={{ fontSize: 9, fontWeight: 700, color: "#6b7280", display: "block", marginBottom: 4, letterSpacing: "0.06em" }}>
                  CROP  <span style={{ color: "#374151", fontSize: 9 }}>({crops.length} grown here)</span>
                </label>
                <select
                  value={selCrop}
                  onChange={(e) => handleCropSelect(e.target.value)}
                  style={SELECT_STYLE}
                >
                  <option value="">— All crops (district avg) —</option>
                  {crops.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
                <span style={{ position: "absolute", right: 10, top: 27, color: "#6b7280", fontSize: 10, pointerEvents: "none" }}>▼</span>
              </div>
            )}

            {/* Fill status */}
            {fillLoading && (
              <p style={{ fontSize: 10, color: "#eab308" }}>⟳ Loading district data…</p>
            )}
            {fillInfo && !fillLoading && (
              <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#22c55e", flexShrink: 0 }} />
                <p style={{ fontSize: 10, color: "#86efac" }}>{fillInfo}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── QUICK PRESETS ──────────────────────────────────────────────── */}
      <div>
        <div style={{ fontSize: 9, fontWeight: 700, color: "#4b5563", marginBottom: 6, letterSpacing: "0.06em" }}>
          OR LOAD PRESET
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 7 }}>
          {PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => { clearLocation(); onChange({ ...p.farm, farm_label: p.farm.farm_label }); }}
              style={{
                padding: "7px 5px", borderRadius: 9, fontSize: 10, fontWeight: 700,
                cursor: "pointer", textAlign: "center",
                background: p.bg, border: `1px solid ${p.border}`, color: p.color,
                transition: "all 0.2s",
              }}
            >{p.label}</button>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 14 }}>
        <div style={{ fontSize: 9, fontWeight: 700, color: "#4b5563", marginBottom: 12, letterSpacing: "0.06em" }}>
          FINE-TUNE PARAMETERS
        </div>

        {/* Farm label */}
        <div style={{ marginBottom: 14 }}>
          <label style={{ fontSize: 9, fontWeight: 700, color: "#6b7280", display: "block", marginBottom: 5, letterSpacing: "0.06em" }}>
            FARM LABEL
          </label>
          <input
            type="text" value={value.farm_label}
            onChange={(e) => set("farm_label", e.target.value)}
            placeholder="e.g. Kharif Rice — GUNTUR 2024"
            style={{
              width: "100%", padding: "8px 10px", borderRadius: 9, fontSize: 12,
              background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.09)",
              color: "#eef2ff", outline: "none", boxSizing: "border-box",
            }}
          />
        </div>

        {/* Variety toggle */}
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "10px 12px", borderRadius: 10, marginBottom: 14,
          background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
        }}>
          <div>
            <p style={{ fontSize: 11, fontWeight: 700, color: "#eef2ff", marginBottom: 1 }}>Crop Variety</p>
            <p style={{ fontSize: 9, color: "#4b5563" }}>HYV adds ~25–40% avg yield</p>
          </div>
          <button
            onClick={() => set("variety_improved", value.variety_improved === 1 ? 0 : 1)}
            style={{
              padding: "5px 12px", borderRadius: 7, fontSize: 11, fontWeight: 800, cursor: "pointer",
              background: value.variety_improved ? "rgba(34,197,94,0.12)" : "rgba(239,68,68,0.08)",
              border: `1px solid ${value.variety_improved ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.25)"}`,
              color: value.variety_improved ? "#22c55e" : "#fca5a5",
              transition: "all 0.25s",
            }}
          >{value.variety_improved ? "✓ Improved" : "✗ Traditional"}</button>
        </div>

        {/* Sliders */}
        <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
          {SLIDERS.map((s) => {
            const val  = value[s.key] as number;
            const pct  = ((val - s.min) / (s.max - s.min)) * 100;
            const decs = s.step < 0.1 ? 2 : s.step < 1 ? 1 : 0;
            return (
              <div key={s.key}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                  <div>
                    <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 500 }}>{s.label}</span>
                    <span style={{ fontSize: 9, color: "#374151", marginLeft: 5 }}>{s.hint}</span>
                  </div>
                  <span className="stat-num" style={{ fontSize: 12, fontWeight: 800, color: s.color }}>
                    {val.toFixed(decs)}{s.unit && <span style={{ fontSize: 9, fontWeight: 500, marginLeft: 2, color: "#6b7280" }}>{s.unit}</span>}
                  </span>
                </div>
                <input
                  type="range" min={s.min} max={s.max} step={s.step} value={val}
                  onChange={(e) => set(s.key, parseFloat(e.target.value))}
                  style={{
                    background: `linear-gradient(to right, ${s.color} ${pct}%, rgba(255,255,255,0.07) ${pct}%)`,
                  }}
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={onSubmit} disabled={!!loading}
        className="btn-primary"
        style={{ marginTop: 4, opacity: loading ? 0.5 : 1 }}
      >
        {loading
          ? <><span className="anim-spin" style={{ display: "inline-block" }}>⟳</span> Running…</>
          : submitLabel}
      </button>

    </div>
  );
}
