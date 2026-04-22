// API base URL — change to your backend if deployed
export const API_BASE = "http://localhost:8000";

/**
 * Safely parse a non-OK fetch Response into a human-readable error string.
 * Handles FastAPI's 422 array-detail format, plain string details, and
 * non-JSON bodies — so the UI never shows "[object Object]".
 */
export async function parseApiError(response: Response, fallback = "Request failed"): Promise<string> {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (!detail) return `${response.status} ${response.statusText || fallback}`;
    if (typeof detail === "string") return detail;
    // FastAPI 422 returns detail as an array of {loc, msg, type}
    if (Array.isArray(detail)) {
      return detail.map((d: { msg?: string; loc?: string[] }) =>
        [d.loc?.slice(1).join(" → "), d.msg].filter(Boolean).join(": ")
      ).join(" · ");
    }
    return JSON.stringify(detail);
  } catch {
    return `${response.status} ${response.statusText || fallback}`;
  }
}

export type PipelineStatus = {
  status: "idle" | "running" | "ready" | "error";
  progress: number;
  current_step: string;
  metrics?: Record<string, Record<string, number>> | null;
  ate_table?: ATERow[] | null;
  error?: string | null;
};

export type ATERow = {
  treatment: string;
  t_low: string;
  t_high: string;
  ate_qha: number;
  ate_pct: number;
  ci_low: number;
  ci_high: number;
};

export type FarmInput = {
  irrigation_coverage_pct: number;
  rainfall_annual: number;
  soil_moisture: number;
  soil_ph: number;
  organic_carbon: number;
  npk_dosage_kg_ha: number;
  temp_mean: number;
  variety_improved: number;
  sowing_week: number;
  prev_year_yield: number;
  farm_label: string;
};

export type ShapFeature = { feature: string; shap_value: number };

export type PredictionResult = {
  farm_label: string;
  predicted_yield: number;
  model_metrics?: Record<string, Record<string, number>>;
  top_shap_features?: ShapFeature[];
};

export type Recommendation = {
  phase: string;
  priority: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  recommendation: string;
  basis: string;
  confidence: string;
  delta_yield?: number;
};

export type AdvisoryResult = {
  farm_label: string;
  predicted_yield: number;
  n_critical: number;
  n_high: number;
  recommendations: Recommendation[];
  traceability?: string;
};

// ── Comparison types ──────────────────────────────────────────────────────────

export type DeltaShapEntry = {
  feature: string;
  shap_a: number;
  shap_b: number;
  delta: number;
  direction_flip: boolean;
};

export type LimeContrastEntry = {
  feature_condition: string;
  contribution_a: number;
  contribution_b: number;
};

export type DiceAction = {
  feature: string;
  from_val: number;
  to_val: number;
  unit: string;
  estimated_gain: number;
};

export type FarmSummary = {
  label: string;
  predicted_yield: number;
  shap: ShapFeature[];
};

export type CompareResult = {
  farm_a: FarmSummary;
  farm_b: FarmSummary;
  delta_yield: number;
  delta_shap: DeltaShapEntry[];
  lime_contrast: LimeContrastEntry[];
  dice_actions: DiceAction[];
};

export async function runComparison(
  farm_a: FarmInput,
  farm_b: FarmInput,
): Promise<CompareResult> {
  const r = await fetch(`${API_BASE}/api/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ farm_a, farm_b }),
  });
  if (!r.ok) throw new Error(await parseApiError(r, "Comparison failed"));
  return r.json();
}

// ── Location types ────────────────────────────────────────────────────────────

export type LocationData = {
  states: string[];
  districts: Record<string, string[]>;
  crops_by_district: Record<string, string[]>;  // key: "State||DISTRICT"
};

export type LocationDefaults = {
  state: string;
  district: string;
  crop: string | null;
  defaults: Partial<FarmInput> & {
    n_records: number;
    nitrogen_kg_ha?: number;
    available_crops?: string[];
    filtered_by_crop?: string | null;
  };
};

// ── Location fetch helpers ─────────────────────────────────────────────────────

export async function fetchLocations(): Promise<LocationData> {
  const r = await fetch(`${API_BASE}/api/locations`);
  if (!r.ok) throw new Error("Failed to fetch locations");
  return r.json();
}

export async function fetchLocationDefaults(
  state: string,
  district: string,
  crop?: string,
): Promise<LocationDefaults> {
  const s = encodeURIComponent(state);
  const d = encodeURIComponent(district);
  const q = crop ? `?crop=${encodeURIComponent(crop)}` : "";
  const r = await fetch(`${API_BASE}/api/locations/defaults/${s}/${d}${q}`);
  if (!r.ok) throw new Error("Failed to fetch location defaults");
  return r.json();
}

export const defaultFarm: FarmInput = {
  irrigation_coverage_pct: 65,
  rainfall_annual: 950,
  soil_moisture:   50,
  soil_ph:         6.8,
  organic_carbon:  0.60,
  npk_dosage_kg_ha: 140,
  temp_mean:       26,
  variety_improved: 1,
  sowing_week:     21,
  prev_year_yield: 30,
  farm_label:      "Kharif Rice — Telangana",
};

export const stressFarm: FarmInput = {
  irrigation_coverage_pct: 18,
  rainfall_annual: 380,
  soil_moisture:   20,
  soil_ph:         5.1,
  organic_carbon:  0.22,
  npk_dosage_kg_ha: 45,
  temp_mean:       38,
  variety_improved: 0,
  sowing_week:     30,
  prev_year_yield: 10,
  farm_label:      "Stressed Dryland Farm — Rajasthan",
};

export const highYieldFarm: FarmInput = {
  irrigation_coverage_pct: 92,
  rainfall_annual: 1400,
  soil_moisture:   70,
  soil_ph:         6.5,
  organic_carbon:  1.20,
  npk_dosage_kg_ha: 220,
  temp_mean:       24,
  variety_improved: 1,
  sowing_week:     18,
  prev_year_yield: 55,
  farm_label:      "High-Yield Paddy — Punjab",
};
