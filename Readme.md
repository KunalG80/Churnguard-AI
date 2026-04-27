# ChurnGuard AI 📉

> **CFO-grade customer retention intelligence** — not just "who will churn?" but "where should the budget go?"

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-22%20passing-green)](#testing)

---

## Model Performance

| Metric | Value |
|---|---|
| **CV ROC-AUC** | **0.8478 ± 0.010** (5-fold stratified) |
| **Test ROC-AUC** | **0.8461** |
| CV ↔ Test gap | 0.0017 — no overfitting |
| Best F1 threshold | 0.55 (Prec=53% / Rec=78%) |
| Best Net-ROI threshold | **0.70** (Prec=65% / Net=₹16,826 / 1,409 customers) |
| Tuning | Optuna — 50 trials, TPE sampler |

### Model Comparison (XGBoost vs Logistic Regression vs Random Forest)

| Model | CV AUC | Test AUC | F1 | Precision |
|---|---|---|---|---|
| Logistic Regression | ~0.800 | ~0.798 | ~0.59 | ~0.50 |
| Random Forest | ~0.820 | ~0.818 | ~0.61 | ~0.52 |
| **XGBoost (tuned)** | **0.848** | **0.846** | **0.633** | **0.65\*** |

\* At threshold=0.70

---

## Project Structure

```
├── main.py                    # Training pipeline
├── app.py                     # Streamlit dashboard
├── src/
│   ├── config.py              # All paths, constants, XGB params
│   ├── data/
│   │   ├── load_data.py       # CSV loader with encoding fallback
│   │   └── preprocess.py      # Cleaning, tenure grouping
│   ├── features/
│   │   └── build_features.py  # Feature engineering (4 new features)
│   ├── models/
│   │   ├── train_xgboost.py   # XGBoost training
│   │   ├── evaluate.py        # Metrics, ROC, PR curves, metadata
│   │   ├── shap_explain.py    # TreeExplainer, SHAP summary plot
│   │   ├── tune_xgboost.py    # Optuna 50-trial tuning
│   │   ├── threshold_tuner.py # Precision/recall/ROI table
│   │   └── model_comparison.py # LR vs RF vs XGBoost comparison
│   └── utils/
│       ├── segment_roi.py     # ROI analysis + budget-constrained targeting
│       ├── executive_summary.py
│       ├── pdf_export.py      # ReportLab CFO memo
│       ├── ppt_export.py      # 5-slide boardroom deck
│       └── live_budget_chart.py
├── tests/                     # 22 unit tests
├── Dockerfile
├── docker-compose.yml
├── runtime.txt                # Streamlit Cloud
└── packages.txt               # Streamlit Cloud apt deps
```

---

## Quickstart

### Local

```bash
git clone https://github.com/yourname/churnguard-ai
cd churnguard-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place dataset
cp /path/to/Churn.csv data/raw/

# Train
python main.py

# Dashboard
streamlit run app.py
```

### Docker

```bash
# Train + launch dashboard
docker compose up

# With Optuna tuning
TUNE=1 docker compose up train

# Model comparison
docker compose run compare
```

### Optuna tuning

```bash
python src/models/tune_xgboost.py        # standalone
# or
TUNE=1 python main.py                    # baked into pipeline
```

### Threshold selection

```bash
python -m src.models.threshold_tuner
```

### Model comparison

```bash
python -m src.models.model_comparison
# saves reports/figures/model_comparison.png
```

### Tests

```bash
python -m pytest tests/ -v
```

---

## Deploy to Streamlit Cloud

1. Push repo to GitHub (exclude `data/`, `models/`, `reports/` via `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo → `app.py` → Deploy
4. Upload `Churn.csv` via the dashboard uploader at runtime

> The app loads a pre-trained model if `models/churn_model.pkl` exists in the repo.
> For Cloud deployment, commit the pkl file or add a training step to your CI.

---

## Key Features

| Feature | Detail |
|---|---|
| **Budget-constrained targeting** | Ranks churners by net ROI, greedy budget allocation |
| **SHAP explanations** | TreeExplainer, top-20 feature importance |
| **Threshold tuning** | PR curve + ROI table across all thresholds |
| **CFO reports** | PDF memo + 5-slide PPTX deck auto-generated |
| **Schema alignment** | Inference-time feature drift handled automatically |
| **Model versioning** | `models/metadata.json` saved on every run |

---

## Engineered Features

Beyond the raw Telco dataset columns, four high-signal features are added:

| Feature | Formula | Why |
|---|---|---|
| `service_count` | sum of active add-on services | More services → higher switching cost |
| `charge_per_service` | MonthlyCharges / (service_count+1) | Detects overpriced bundles |
| `avg_monthly_spend` | TotalCharges / (tenure+1) | Normalises spend trajectory |
| `is_new_customer` | tenure ≤ 3 months | Highest churn-risk window |

---

## Tech Stack

`XGBoost` · `scikit-learn` · `Streamlit` · `Plotly` · `SHAP` · `Optuna` · `ReportLab` · `python-pptx` · `Docker`
