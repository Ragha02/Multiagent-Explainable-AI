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

<br/>

> *Most crop yield models give you a number. MA-XAI gives you a reason, a cause, and a plan.*

</div>

---

## The Problem

A district-level agricultural officer is looking at a model output: `predicted yield = 22.4 q/ha`. That number is useless on its own.

**What they actually need to know:**

- *Why* is the model predicting that specific number?
- Which inputs are truly *causing* low yield — not just correlated with it?
- What interventions, if actioned right now, would move the needle?
- How confident should they be in any of this?

No existing system answers all four questions together. MA-XAI does.

---

## What It Is

MA-XAI is a **five-agent AI pipeline** that chains prediction, causal inference, and multi-method explanation into a single decision support system — built on 246,000 real Indian agricultural records.

The output isn't just a prediction. It's a **ranked, traceable advisory report**: every recommendation links back through the causal graph, through the model's SHAP values, all the way to the raw input that triggered it.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MA-XAI Pipeline                                  │
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐  │
│  │ Agent 1  │───▶│ Agent 2  │───▶│ Agent 3  │───▶│ Agent 4  │───▶│  Agent 5   │  │
│  │  Data    │    │ Predict  │    │  Causal  │    │ Explain  │    │  Advisory  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └────────────┘  │
│       │               │               │               │                │          │
│  Ingest &        RF + XGBoost     DoWhy DAG       4 XAI types    IF-THEN rules   │
│  Engineer        ensemble +       back-door        SHAP · LIME   + counterfactual │
│  features        SHAP values      ATE estimation  contrastive     uplift estimates│
│                                                    causal                         │
└─────────────────────────────────────────────────────────────────────────┘
```

Each agent is independently importable and testable. They share state through a thread-safe singleton — no re-loading models between API requests.

---

## The Five Agents

### Agent 1 — Data
Ingests three Kaggle datasets (India Agri Census, FAO climate data, ICAR soil samples), merges them on `[crop, year]`, fills schema gaps with agronomically-validated distributions, and engineers six composite indices — including a `water_supply_index` and `soil_quality_index` that capture multi-factor interactions a raw column never could.

### Agent 2 — Prediction
Trains an MLR baseline, a 500-tree Random Forest, and an XGBoost model in parallel, then finds the optimal ensemble weight between RF and XGB using `scipy.optimize` on the validation set. SHAP's `TreeExplainer` runs on the XGBoost member — making every prediction explainable at zero additional compute cost.

```
ŷ = w_RF × ŷ_RF + w_XGB × ŷ_XGB     (weights optimised on val-set RMSE)
```

### Agent 3 — Causal
Constructs a five-layer directed acyclic graph (DAG) encoding domain knowledge about how rainfall, soil, irrigation, and farming practice interact. Uses **DoWhy's back-door criterion** to estimate average treatment effects (ATEs) for each controllable variable — answering *"if a farmer actually changes irrigation coverage from 20% to 80%, what is the causal effect on yield?"* with bootstrapped 95% confidence intervals.

### Agent 4 — Explanation
Produces four complementary explanation types in one pass:

| Type | Method | Question answered |
|---|---|---|
| **Global** | Mean \|SHAP\| over training set | Which features matter most, universally? |
| **Local** | LIME on single instance | Why did *this* farm get *this* prediction? |
| **Contrastive** | ΔSHAP between two farms | Why does farm A outperform farm B? |
| **Causal** | DAG + ATE from Agent 3 | What can the farmer actually *change*? |

### Agent 5 — Advisory
Converts all of the above into a tiered action plan. A 30+ rule IF-THEN engine fires season-aware recommendations (`CRITICAL → HIGH → MEDIUM → LOW`), and Agent 3's counterfactual model attaches a concrete yield-delta estimate to each one. Every recommendation carries a full **traceability chain** — input value → rule → model output → advisory — so nothing is a black box.

---

## Results

> Evaluated on a held-out temporal test set (post-2010 records, ~7,500 rows).

| Model | RMSE ↓ | R² ↑ | MAPE ↓ |
|---|---|---|---|
| MLR (baseline) | 16.27 q/ha | 0.720 | 58.2% |
| Random Forest | 7.99 q/ha | 0.933 | 15.4% |
| XGBoost | 8.63 q/ha | 0.921 | 20.3% |
| **MA-XAI Ensemble** | **8.04 q/ha** | **0.932** | **16.2%** |

The most revealing result isn't the ensemble score — it's what SHAP surfaces on real data. `prev_year_yield` carries a mean absolute SHAP value of **17.17**, roughly 27× stronger than any single-season intervention variable. That's not a model quirk. It's agronomics: soil health, management quality, and microclimate persistence compound across seasons. The causal layer captures this explicitly; the advisory layer uses it to re-rank interventions accordingly.

---

## Project Structure

```
ma-xai/
├── agents/                 ← The five ML agents (pure Python, no web deps)
│   ├── agent1_data.py      ← Data ingestion, merging, cleaning, feature engineering
│   ├── agent2_prediction.py← RF + XGBoost ensemble, SHAP explainer
│   ├── agent3_causal.py    ← DAG construction, DoWhy ATE estimation
│   ├── agent4_explanation.py← Four XAI methods in one composable module
│   ├── agent5_advisory.py  ← Rule engine + counterfactual uplift + traceability
│   └── agent6_recommend.py ← Crop recommendation sub-module
│
├── api/                    ← FastAPI backend
│   ├── main.py             ← App factory, CORS, lifespan pipeline kickoff
│   ├── state.py            ← Thread-safe pipeline singleton + async runner
│   ├── models.py           ← Pydantic schemas
│   └── routes/             ← predict · advisory · compare · recommend · locations
│
├── frontend/               ← Next.js 15 (App Router) dashboard
│   └── app/                ← Dashboard · Predict · Causal · Explain · Advisory
│
├── scripts/
│   └── download_datasets.py← One-shot Kaggle downloader
│
├── orchestrator.py         ← CLI — runs all five agents end-to-end
└── requirements.txt
```

---

## Quick Start

**Prerequisites:** Python 3.12+, Node.js 20+, a [Kaggle API key](https://www.kaggle.com/settings/account)

```bash
# 1 — Clone and install
git clone https://github.com/Ragha02/Multiagent-Explainable-AI.git
cd Multiagent-Explainable-AI
pip install -r requirements.txt

# 2 — Set up Kaggle credentials and pull the data
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
python3 scripts/download_datasets.py

# 3 — Start the backend
python3 -m uvicorn api.main:app --port 8000 --reload

# 4 — Start the frontend (new terminal)
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000** → click **Start Pipeline** → watch all five agents run live.

> **No Kaggle key?** The pipeline falls back to a 5,000-row synthetic dataset automatically — the full system still runs, explains, and advises.

---

## CLI Mode

Skip the web UI entirely and run the full pipeline from the terminal:

```bash
python3 orchestrator.py
```

Outputs five publication-quality figures to `ma_xai_outputs/` — prediction results, causal DAG, ATE bars, XAI composite panel, and an advisory dashboard — plus a master figure stitching all of them together.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Models** | scikit-learn (RF, MLR), XGBoost |
| **Explainability** | SHAP (TreeExplainer), LIME (tabular) |
| **Causal Inference** | DoWhy (back-door criterion), NetworkX (DAG) |
| **Counterfactuals** | DiCE-ML |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Next.js 15, TypeScript, Tailwind CSS, Framer Motion |
| **Data** | pandas, NumPy, SciPy |

---

## Contributing & Collaboration

**MA-XAI is open to features, research extensions, and collaborations — all via GitHub.**

Whether you want to propose a new agent, improve an existing XAI method, plug in a new dataset, or just fix a bug — contributions are welcome.

### Ways to contribute

| Type | How |
|---|---|
| 🐛 **Bug fix** | Open an issue describing the problem, then submit a PR |
| ✨ **New feature** | Open a feature request issue first — discuss before building |
| 🤖 **New agent** | Follow the existing agent pattern (standalone module, clean interface) |
| 📊 **New dataset** | Extend `agent1_data.py` with a new data source or region |
| 🧪 **Tests** | Any coverage is welcome — the `agents/` modules are pure Python |
| 📝 **Research collab** | Open an issue tagged `collaboration` and describe your angle |

### Workflow

```bash
# Fork → clone your fork
git clone https://github.com/YOUR_USERNAME/Multiagent-Explainable-AI.git

# Create a feature branch
git checkout -b feat/your-feature-name

# Make changes, then push and open a Pull Request
git push origin feat/your-feature-name
```

> Open your PR against the `main` branch. Include a short description of what it does and why. If it touches an agent's interface, update the relevant docstring.

---

<div align="center">

*MA-XAI — because a number without a reason is just noise.*

</div>
