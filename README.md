<div align="center">

# MA-XAI
### Multi-Agent Explainable AI for Agricultural Decision Support

<br/>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=flat-square&logo=next.js&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6600?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=flat-square)
![DoWhy](https://img.shields.io/badge/DoWhy-Causal_Inference-E91E63?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

<br/>

> *Crop yield models give you a number. MA-XAI gives you the reason, the cause, and a ranked action plan.*

</div>

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [The 6-Agent Pipeline](#4-the-6-agent-pipeline)
   - [Agent 1 — Data](#agent-1--data)
   - [Agent 2 — Prediction](#agent-2--prediction)
   - [Agent 3 — Causal Inference](#agent-3--causal-inference)
   - [Agent 4 — Explanation](#agent-4--explanation)
   - [Agent 5 — Advisory](#agent-5--advisory)
   - [Agent 6 — Crop Recommendation](#agent-6--crop-recommendation)
5. [Data Sources](#5-data-sources)
6. [REST API Reference](#6-rest-api-reference)
7. [Frontend Dashboard](#7-frontend-dashboard)
8. [Performance Results](#8-performance-results)
9. [Quick Start](#9-quick-start)
10. [Dependencies](#10-dependencies)
11. [Contributing & Collaboration](#11-contributing--collaboration)

---

## 1. What This Project Does

Most crop yield ML models are one-dimensional: they take inputs and return a predicted number. An agricultural officer cannot make decisions from `22.4 q/ha` alone.

**MA-XAI solves this by answering four questions simultaneously:**

| Question | Solved by |
|---|---|
| *What will the yield be?* | Agent 2 — weighted ensemble (RF + XGBoost) |
| *Why did the model predict that?* | Agent 4 — SHAP, LIME, contrastive attribution |
| *What would actually move yield if changed?* | Agent 3 — causal DAG + DoWhy ATE estimation |
| *What should the farmer do, in what order?* | Agent 5 — rule engine + counterfactual uplift |

A sixth agent adds **crop suitability recommendation** — given soil and climate parameters, it recommends the top-3 crops with confidence scores and expected yield ranges.

The entire system runs end-to-end on **246,000 real Indian government agricultural records** and exposes its results through a FastAPI backend and a live Next.js 15 dashboard.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     MA-XAI 6-Agent Pipeline                        │
│                                                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐  │
│  │ Agent 1 │─▶│ Agent 2 │─▶│ Agent 3 │─▶│ Agent 4 │─▶│Agent 5 │  │
│  │  Data   │  │ Predict │  │ Causal  │  │ Explain │  │Advisory│  │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └────────┘  │
│       │                                                    ▲       │
│  ┌─────────┐                                               │       │
│  │ Agent 6 │────────────────── Crop Recommend ────────────┘       │
│  └─────────┘                                                       │
│                                                                    │
│  FastAPI Backend (/api/*) ──▶  Next.js 15 Dashboard               │
│  Thread-safe PipelineState singleton                               │
│  Background async pipeline runner                                  │
└────────────────────────────────────────────────────────────────────┘
```

**Key design principle:** Each agent is a self-contained Python module — independently importable, testable, and replaceable. They share state through a thread-safe singleton in production (API mode) and communicate through clean function interfaces in CLI mode (orchestrator).

---

## 3. Project Structure

```
Multiagent-Explainable-AI/
│
├── agents/                        ← The 6-agent ML pipeline (pure Python)
│   ├── agent1_data.py             ← Data ingestion, merging, cleaning, feature engineering
│   ├── agent2_prediction.py       ← RF + XGBoost ensemble + SHAP explainer
│   ├── agent3_causal.py           ← Causal DAG + DoWhy ATE estimation + counterfactuals
│   ├── agent4_explanation.py      ← 4 XAI methods: SHAP, LIME, contrastive, causal
│   ├── agent5_advisory.py         ← 32-rule IF-THEN engine + ML-informed advisory
│   └── agent6_recommend.py        ← Crop suitability classifier (RF + GB ensemble)
│
├── api/                           ← FastAPI backend
│   ├── main.py                    ← App factory, CORS, lifespan pipeline kickoff
│   ├── state.py                   ← Thread-safe PipelineState singleton + async runner
│   ├── models.py                  ← Pydantic request/response schemas
│   └── routes/
│       ├── pipeline.py            ← POST /api/pipeline, GET /api/pipeline/status
│       ├── predict.py             ← POST /api/predict
│       ├── advisory.py            ← POST /api/advisory
│       ├── recommend.py           ← POST /api/recommend
│       ├── compare.py             ← POST /api/compare (farm comparison + SHAP divergence)
│       └── locations.py           ← GET /api/locations (district/state data for map)
│
├── frontend/                      ← Next.js 15 App Router dashboard
│   ├── app/
│   │   ├── page.tsx               ← Dashboard — live pipeline progress + metrics
│   │   ├── predict/page.tsx       ← Farm sliders → SHAP waterfall + model comparison table
│   │   ├── causal/page.tsx        ← DAG image viewer + ATE bars with 95% CI
│   │   ├── explain/page.tsx       ← 4 XAI type cards + global SHAP leaderboard
│   │   ├── advisory/page.tsx      ← Priority recommendation cards (Pre/In/Post season)
│   │   ├── recommend/page.tsx     ← Soil/climate → top-3 crop recommendations
│   │   └── compare/page.tsx       ← Side-by-side farm comparison + SHAP divergence
│   ├── components/
│   │   ├── Navbar.tsx             ← Responsive nav with live pipeline status indicator
│   │   ├── FarmForm.tsx           ← Parametric farm slider panel with 3 presets
│   │   ├── AgentPipeline.tsx      ← Animated pipeline step tracker
│   │   └── AdvisoryPDF.tsx        ← PDF report export component
│   └── lib/
│       └── api.ts                 ← API client, type definitions, farm presets
│
├── scripts/
│   └── download_datasets.py       ← One-shot Kaggle downloader (run once)
│
├── data/                          ← Downloaded Kaggle CSVs (git-ignored — see §5)
│   ├── crop_production.csv        ← 246k India district-level crop records
│   ├── yield_df.csv               ← FAO climate data (rainfall + temperature)
│   ├── Crop_recommendation.csv    ← ICAR soil + NPK medians by crop
│   └── geo/
│       └── india_states.geojson   ← State boundary polygons for map visualisation
│
├── ma_xai_outputs/                ← Generated plots (git-ignored — regenerate via pipeline)
├── orchestrator.py                ← CLI runner — executes all agents end-to-end
└── requirements.txt               ← Python dependencies
```

> **`data/` and `ma_xai_outputs/` are gitignored.** Run `scripts/download_datasets.py` to fetch data and `orchestrator.py` to regenerate plots.

---

## 4. The 6-Agent Pipeline

---

### Agent 1 — Data

**File:** `agents/agent1_data.py`

**Responsibilities:** Ingest, merge, validate, impute, and engineer features from three real datasets.

**How it works:**

1. **Real data detection** — Checks at startup whether Kaggle CSVs exist in `data/`. If yes, loads ~246k real records. If no, falls back to a 5,000-row synthetic dataset so the entire pipeline still runs without internet or credentials.

2. **Three-way merge:**
   ```
   crop_production.csv  ──[crop, year]──▶  yield_df.csv (India FAO rows)
         └──────────────[crop]──────────▶  Crop_recommendation.csv (soil medians)
   ```

3. **Gap filling** — 18 columns absent from Kaggle data (e.g. `irrigation_coverage_pct`, `solar_radiation`) are filled with agronomically-validated random distributions. The downstream feature schema never changes.

4. **Stratified sampling** — The full cross-joined set approaches 1M+ rows. A stratified 50k sample (proportional by `state × crop`) keeps training time under ~60 seconds while preserving India's regional diversity.

5. **Data Quality Report** — Grades every column A/B/C by missingness, runs temporal + spatial interpolation, then global median fallback.

6. **Feature engineering** — Derives six composite indices:

| Feature | Formula |
|---|---|
| `rainfall_adequacy` | `rainfall / crop_water_need` |
| `temp_stress_index` | `\|temp − 25\|² / 100` |
| `nutrient_balance` | `(N/400 + P/60 + K/400) / 3` |
| `water_supply_index` | `irrigation × reliability + rainfall_adequacy` |
| `soil_quality_index` | weighted OC + pH proximity + nutrients |
| `rainfall_mid_season` | mid-season rainfall window proxy |

**Output:** `X_train, X_val, X_test, y_train, y_val, y_test, feat_names, scaler, encoders`

---

### Agent 2 — Prediction

**File:** `agents/agent2_prediction.py`

**Responsibilities:** Train a multi-model ensemble, compute SHAP explanations, evaluate against baselines.

**Models trained:**

| Model | Specification |
|---|---|
| MLR | Multivariate Linear Regression — baseline |
| Random Forest | 500 estimators, `min_samples_leaf=5`, all features |
| XGBoost | 300 estimators, `lr=0.05`, `max_depth=6`, early stopping |
| **MA-XAI Ensemble** | Optimised weighted blend of RF + XGBoost |

**Key design — learned ensemble weights:**

Rather than fixing a 50/50 split, `scipy.optimize.minimize_scalar` searches for the RF weight `w` that minimises validation-set RMSE. In practice this converges near `w_RF ≈ 0.40`, `w_XGB ≈ 0.60`.

```python
ŷ = w_RF × ŷ_RF + w_XGB × ŷ_XGB
```

**SHAP:** XGBoost's `TreeExplainer` computes both global feature importance (`mean |SHAP|` across training set) and per-instance local attributions. These flow downstream to Agent 4 and the `/predict` API endpoint.

---

### Agent 3 — Causal Inference

**File:** `agents/agent3_causal.py`

**Responsibilities:** Build a causal DAG over the farm domain, estimate treatment effects, serve counterfactual queries.

**DAG — 5 layers, 10 nodes, 19 directed edges:**

```
L1 — Exogenous     rainfall_annual, temp_mean
L2 — Soil state    soil_ph, nitrogen, soil_moisture, organic_carbon
L3 — Controllable  irrigation_coverage_pct, npk_dosage_kg_ha, sowing_week
L4 — Intermediate  rainfall_adequacy, water_supply_index
L5 — Outcome       yield_q_ha
```

**ATE estimation:** Uses [DoWhy](https://github.com/py-why/dowhy) with back-door criterion adjustment. For each controllable variable, compares low vs. high intervention while conditioning on all confounders. Results include bootstrapped 95% confidence intervals.

**Counterfactual queries:** A separate Random Forest trains on the numerical columns to answer "if I change `feature X` to value `v`, what is the predicted yield delta?" — used by Agent 5 to attach uplift estimates to every recommendation.

**Why this matters:** SHAP answers *"which features drove this prediction?"*. Causal analysis answers *"if a farmer actually changes irrigation from 20% to 80%, what is the causal effect on yield?"* The distinction between correlation and intervention is the core value proposition of Agent 3.

---

### Agent 4 — Explanation

**File:** `agents/agent4_explanation.py`

**Responsibilities:** Produce all four XAI explanation types for any farm instance.

| Type | Method | Question it answers |
|---|---|---|
| **Global** | Mean \|SHAP\| over training set | Which features matter universally, across all farms? |
| **Local** | Tabular LIME on a single instance | Why did *this specific farm* get *this specific prediction*? |
| **Contrastive** | ΔSHAP between two instances | Why does farm A outperform farm B, feature by feature? |
| **Causal** | DAG path analysis + ATEs from Agent 3 | What can the farmer *actually control* to move yield? |

All four types are rendered in a single composite matplotlib figure and also returned as structured JSON through the API.

**Explanation card** — Agent 4 generates a human-readable card for any instance combining the local LIME explanation with the top causal intervention, formatted for the frontend advisory panel.

---

### Agent 5 — Advisory

**File:** `agents/agent5_advisory.py`

**Responsibilities:** Translate model outputs into ranked, actionable, farmer-readable recommendations with full traceability.

**Two-layer system:**

**Layer 1 — IF-THEN Rule Engine (32 rules)**

Domain-validated agronomic rules based on ICAR / FAO guidelines, organised into three temporal phases:

- **PRE-SEASON (11 rules):** irrigation setup, soil pH correction, organic carbon, NPK basal dose, variety selection, sowing window
- **IN-SEASON (7 rules):** soil moisture monitoring, NPK top-dress, heat/cold stress response, pest/disease risk, weed management
- **POST-SEASON (7 rules):** yield trend analysis, OC rebuilding, crop rotation planning, nutrient replacement

**Layer 2 — ML Counterfactual Uplift**

For each recommendation, Agent 3's counterfactual model estimates the actual yield gain if the advice is followed. This converts qualitative rules into quantified decisions:

```json
{
  "phase": "IN-SEASON",
  "priority": "CRITICAL",
  "recommendation": "Apply supplemental irrigation of 40–50mm immediately",
  "basis": "Soil moisture 22% — below permanent wilting point",
  "confidence": "High",
  "delta_yield": 9.0
}
```

Recommendations are sorted: `PRE-SEASON → IN-SEASON → POST-SEASON`, and within each phase: `CRITICAL → HIGH → MEDIUM → LOW`.

**Traceability chain** — Every advisory includes a full audit trail tracing the recommendation back through all 5 agents, from raw input values through model predictions, SHAP attributions, causal estimates, and the triggering rule.

---

### Agent 6 — Crop Recommendation

**File:** `agents/agent6_recommend.py`

**Responsibilities:** Given soil and climate parameters, recommend the top-k crops with confidence scores and historical yield ranges.

**Model:**

- Random Forest (300 trees) + Gradient Boosting (200 estimators) ensemble
- Trained on `Crop_recommendation.csv` (ICAR soil samples, 22 crops)
- 5-fold cross-validation accuracy ≈ 99%
- Equal-weight probability averaging across both classifiers

**Input features:** N, P, K (kg/ha), soil pH, humidity (%), annual rainfall (mm), temperature (°C)

**Output per crop recommendation:**

```json
{
  "rank": 1,
  "crop": "Rice",
  "confidence": 87.4,
  "yield_p25": 14.2,
  "yield_median": 22.1,
  "yield_p75": 38.7,
  "sample_size": 4821,
  "note": "High water demand (>1200mm). Best in Kharif. Heavy clay soils preferred."
}
```

Yield percentiles are pulled from the real production data (Agent 1's `clean_df`) so the ranges reflect actual India district outcomes, not assumptions.

---

## 5. Data Sources

| Dataset | Source | Rows | Key Columns |
|---|---|---|---|
| `crop_production.csv` | India Agri Census via Kaggle | 246,091 | State, District, Crop, Year, Area, Production |
| `yield_df.csv` | FAO / World Bank via Kaggle | 28,242 (4,048 India rows) | avg_rainfall, avg_temp, year |
| `Crop_recommendation.csv` | ICAR soil samples via Kaggle | 2,200 | N, P, K, pH, humidity, rainfall, temperature, label |

### Downloading Data

```bash
# Set up Kaggle credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download all three datasets (~17 MB total)
python3 scripts/download_datasets.py
```

> **No Kaggle key?** Skip this step. The pipeline auto-detects missing CSVs and falls back to a 5,000-row synthetic dataset — the full system still runs.

---

## 6. REST API Reference

**Base URL:** `http://localhost:8000`

All agents are loaded once into a thread-safe `PipelineState` singleton on startup. Subsequent requests reuse in-memory models — no re-loading.

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | GET | Liveness check — pipeline status + progress % |
| `POST /api/pipeline` | POST | Trigger the full 6-agent pipeline in background |
| `GET /api/pipeline/status` | GET | Poll progress (0–100%), current step, metrics |
| `GET /api/pipeline/results` | GET | Full results: global SHAP, ATE table, model metrics |
| `POST /api/predict` | POST | Predict yield for a farm + return SHAP values |
| `POST /api/advisory` | POST | Generate priority advisory report for a farm |
| `POST /api/recommend` | POST | Top-k crop recommendations for given soil/climate |
| `POST /api/compare` | POST | Side-by-side farm comparison with SHAP divergence |
| `GET /api/locations` | GET | District/state data for geospatial map |
| `GET /outputs/{filename}` | GET | Serve generated plot PNGs |

### Example: Predict Yield

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "irrigation_coverage_pct": 75,
    "rainfall_annual": 1100,
    "soil_moisture": 60,
    "soil_ph": 6.8,
    "organic_carbon": 0.7,
    "npk_dosage_kg_ha": 160,
    "temp_mean": 26,
    "variety_improved": 1,
    "sowing_week": 21,
    "prev_year_yield": 28,
    "farm_label": "Kharif Rice — Telangana"
  }'
```

**Response:**
```json
{
  "predicted_yield": 58.81,
  "top_shap_features": [
    { "feature": "prev_year_yield",   "shap_value": 17.19 },
    { "feature": "rainfall_adequacy", "shap_value":  3.34 }
  ]
}
```

### Example: Crop Recommendation

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90, "P": 42, "K": 43,
    "ph": 6.5,
    "humidity": 82,
    "rainfall": 1200,
    "temperature": 27
  }'
```

---

## 7. Frontend Dashboard

**URL:** `http://localhost:3000`

Built with **Next.js 15 App Router**, **TypeScript**, **Tailwind CSS**, and **Framer Motion**.

| Page | Route | What you see |
|---|---|---|
| **Dashboard** | `/` | Live pipeline progress tracker, data source cards, ensemble performance metrics |
| **Predict** | `/predict` | Farm parameter sliders (3 presets), animated SHAP waterfall, model comparison table |
| **Causal** | `/causal` | 5-layer DAG image, ATE bars with 95% CIs, DoWhy methodology panel |
| **Explain** | `/explain` | 4 XAI type cards, composite explanation figure, global SHAP leaderboard |
| **Advisory** | `/advisory` | Phase-grouped recommendation cards, priority badges, collapsible traceability chain |
| **Recommend** | `/recommend` | Soil/climate form → top-3 crop cards with yield percentile bars |
| **Compare** | `/compare` | Two-farm side-by-side with SHAP divergence bars and counterfactual gap analysis |

**Design system:** Dark glassmorphism theme · `Inter` (UI) + `JetBrains Mono` (data) · Framer Motion entrance animations · responsive grid layouts

---

## 8. Performance Results

> Evaluated on real India crop data — 7,500-row held-out temporal test set (post-2010 records).

### Model Comparison

| Model | RMSE ↓ | MAE ↓ | R² ↑ | MAPE ↓ |
|---|---|---|---|---|
| MLR (baseline) | 16.27 q/ha | 10.42 q/ha | 0.720 | 58.2% |
| Random Forest | **7.99 q/ha** | **2.51 q/ha** | **0.933** | **15.4%** |
| XGBoost | 8.63 q/ha | 3.34 q/ha | 0.921 | 20.3% |
| **MA-XAI Ensemble** | 8.04 q/ha | 2.64 q/ha | 0.932 | 16.2% |

### Top SHAP Features (real India data, global)

| Rank | Feature | Mean \|SHAP\| | Agronomic Interpretation |
|---|---|---|---|
| #1 | `prev_year_yield` | **17.17** | Soil health + management quality compound across seasons |
| #2 | `season` | 1.79 | Kharif vs. Rabi fundamentally shifts yield expectations |
| #3 | `crop` | 1.33 | Crop type embeds its own water and soil requirements |
| #4 | `rainfall_adequacy` | 0.63 | Ratio of actual rainfall to crop water need |
| #5 | `rainfall_mid_season` | 0.60 | Critical growth-stage rainfall window |

> The dominance of `prev_year_yield` (17.17 vs. 1.79 for the next feature) reflects a real agronomic truth — microclimate persistence, soil organic matter buildup, and cumulative management quality dwarf any single-season intervention.

### Causal ATEs (DoWhy back-door, real India data)

| Intervention | Range | ATE | 95% CI |
|---|---|---|---|
| NPK Fertiliser | 60 → 180 kg/ha | **+1.71 q/ha (+5.9%)** | [−0.94, +3.65] |
| Sowing Week | Week 10 → 22 | −0.30 q/ha (−1.0%) | [−2.17, +1.68] |
| Irrigation Coverage | 20% → 80% | −0.09 q/ha (−0.3%) | [−1.73, +1.92] |

> Wide confidence intervals reflect genuine heterogeneity in observational field data — different crops, states, and soils respond differently. The back-door adjustment controls for confounders but cannot eliminate individual farm variation.

---

## 9. Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+
- A [Kaggle API key](https://www.kaggle.com/settings/account) *(optional — synthetic fallback available)*

### Step 1 — Clone and install

```bash
git clone https://github.com/Ragha02/Multiagent-Explainable-AI.git
cd Multiagent-Explainable-AI
pip install -r requirements.txt
```

### Step 2 — Download data (optional)

```bash
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
python3 scripts/download_datasets.py
```

### Step 3 — Start the backend

```bash
python3 -m uvicorn api.main:app --port 8000 --reload
```

The pipeline starts automatically in the background on API startup.

### Step 4 — Start the frontend

```bash
cd frontend
npm install
npm run dev
```

### Step 5 — Open the dashboard

Navigate to **http://localhost:3000**. Click **Start Pipeline** on the dashboard. All 6 agents execute in the background (~60 seconds on 50k rows) and the UI updates in real time.

### CLI mode — skip the web UI entirely

```bash
python3 orchestrator.py
```

Runs all 5 core agents sequentially and saves five publication-quality figures to `ma_xai_outputs/` plus a master composite figure.

---

## 10. Dependencies

### Python (`requirements.txt`)

| Package | Used by |
|---|---|
| `pandas`, `numpy` | All agents — data manipulation |
| `scikit-learn` | Agents 2, 3, 6 — RF, MLR, StandardScaler, LabelEncoder |
| `xgboost` | Agent 2 — gradient-boosted ensemble member |
| `shap` | Agents 2, 4 — TreeExplainer, global + local importance |
| `lime` | Agent 4 — tabular local explainer |
| `dowhy` | Agent 3 — causal graph, back-door ATE estimation |
| `dice-ml` | Agent 3 — diverse counterfactual generation |
| `networkx` | Agent 3 — DAG construction and rendering |
| `matplotlib`, `seaborn` | Agents 2–5 — plot generation |
| `scipy` | Agent 2 — ensemble weight optimisation |
| `fastapi`, `uvicorn` | API layer |
| `kaggle` | `scripts/download_datasets.py` |

### JavaScript (`frontend/package.json`)

| Package | Used by |
|---|---|
| `next` 15 | App Router, SSR, static assets |
| `react`, `react-dom` | UI components |
| `framer-motion` | Entrance animations, progress bars |
| `tailwindcss` | Design system + responsive layouts |
| `typescript` | Type safety across all components |

---

## 11. Contributing & Collaboration

**MA-XAI is open to features, research extensions, and collaborations — all via GitHub.**

Whether you want to add a new agent, improve an XAI method, plug in a different crop dataset, or extend the system to a new geography — contributions are welcome.

### Ways to contribute

| Type | How |
|---|---|
| 🐛 **Bug fix** | Open an issue describing the problem, then submit a PR |
| ✨ **New feature** | Open a feature request issue first — discuss before building |
| 🤖 **New agent** | Follow the existing agent pattern — standalone module, clean interface |
| 📊 **New dataset** | Extend `agent1_data.py` with a new data source or region |
| 🧪 **Tests** | Any test coverage is welcome — `agents/` modules are pure Python |
| 📝 **Research collab** | Open an issue tagged `collaboration` and describe your research angle |

### Workflow

```bash
# Fork → clone your fork
git clone https://github.com/YOUR_USERNAME/Multiagent-Explainable-AI.git

# Create a feature branch
git checkout -b feat/your-feature-name

# Make changes, then push and open a Pull Request
git push origin feat/your-feature-name
```

> Open PRs against the `main` branch. Include a short description of what it does and why. If it modifies an agent's public interface, update the relevant docstring.

---

<div align="center">

*MA-XAI — because a number without a reason is just noise.*

</div>
