# Healthcare Mortality Risk Analytics

An end-to-end machine learning system that predicts patient mortality risk from a 20,000-record Indian healthcare dataset — covering data cleaning, exploratory analysis, feature engineering, model training, hyperparameter tuning, explainability, a FastAPI prediction backend, and a React analytics dashboard.

> **This is a statistical/educational project, not a clinical tool.** See [Healthcare disclaimer](#healthcare-disclaimer).

---

## Table of contents

- [Project overview](#project-overview)
- [Features](#features)
- [ML problem statement](#ml-problem-statement)
- [Dataset](#dataset)
- [Model A vs Model B](#model-a-vs-model-b)
- [Final model choice](#final-model-choice)
- [Evaluation metrics](#evaluation-metrics)
- [Technologies used](#technologies-used)
- [Project structure](#project-structure)
- [Installation and setup](#installation-and-setup)
- [Running the backend](#running-the-backend)
- [Running the frontend](#running-the-frontend)
- [API reference](#api-reference)
- [Using the prediction system](#using-the-prediction-system)
- [Reproducing the ML pipeline](#reproducing-the-ml-pipeline)
- [Healthcare disclaimer](#healthcare-disclaimer)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

---

## Project overview

This project takes a raw healthcare dataset through a complete, documented ML workflow — cleaning, EDA, feature engineering, preprocessing, baseline modeling, hyperparameter tuning, explainability, and deployment — and exposes the result as a web application with a prediction interface and analytics dashboard.

Every phase's decisions, findings, and outputs are preserved in `reports/` so the reasoning is auditable, not just the final numbers.

## Features

- Full EDA with 10+ charts covering distributions, mortality patterns, correlations, and symptom analysis
- Two parallel models (**Model A** and **Model B** — see below) trained on 4 algorithms each, tuned, and compared
- SHAP-based explainability for tree models and coefficient-based explainability for the final Logistic Regression models
- A FastAPI backend exposing dataset stats, EDA results, model performance, and live predictions
- A React + Tailwind dashboard with 6 pages: Dashboard, Data Analysis, Model Performance, Explainability, Prediction, and Dataset
- A reusable, validated prediction interface with per-patient contributing-factor breakdowns

## ML problem statement

**Predict `death_flag` (0 = survived/ongoing care, 1 = deceased)** from patient demographic, clinical, socioeconomic, and treatment-context features, using binary classification on an imbalanced target (~9.1% positive class).

The target was chosen over alternatives (`severity`, `outcome`, `treatment_cost_inr`) after an explicit comparative analysis — see `reports/EDA_report.md` for the full reasoning, including why each alternative was rejected.

## Dataset

- **20,000 patient records, 32 original columns** — Indian healthcare data covering demographics, disease, severity, treatment, cost, and outcome
- Delivered as an Excel workbook mislabeled `.csv` (handled transparently in `ml/preprocessing/01_clean_data.py`)
- Cleaned of a real Excel auto-date-corruption bug (`age_group` "10-19" silently became the date 2019-10-01 on 2,062 rows) and structurally-missing categorical values
- No duplicate rows, no rows removed as "outliers" — every extreme value (BMI, cost, hospital stay) was judged clinically plausible and kept (see `reports/EDA_report.md`)

## Model A vs Model B

Two parallel models are trained on the same data, differing in what information they're allowed to use:

| | **Model A — Full clinical-course** | **Model B — Early-risk** |
|---|---|---|
| Includes | Everything, including `treatment_type`, `days_hospitalized`, `treatment_cost` | Only information available **at or near diagnosis/admission** |
| Excludes | — | `treatment_type`, `days_hospitalized`, `treatment_cost_log`, `length_of_stay_category` — all excluded after an explicit temporal-availability audit (Phase 5) confirmed they're only known *after* treatment has progressed |
| Use case | Best achievable accuracy, using the full patient record | A genuine admission-time risk-screening tool |

Both models also exclude identified leakage columns entirely: `patient_id`, `outcome`, `cause_of_death`, `recovery_days`, and `follow_up_required` (the last one was found during EDA to be a near-perfect leakage proxy — 0 of 1,822 deceased patients had `follow_up_required = Yes`).

## Final model choice

**Logistic Regression, threshold = 0.40, for both Model A and Model B.**

XGBoost was tuned as a competing candidate (`RandomizedSearchCV`, 5-fold CV, PR-AUC scoring) and won narrowly on cross-validated PR-AUC. But on the untouched test set, tuned Logistic Regression **matched or beat** tuned XGBoost on nearly every metric while remaining fully interpretable (coefficients directly explain each prediction — no SHAP layer required):

| Metric | Model A — Logistic Regression | Model A — XGBoost |
|---|---|---|
| Accuracy | 83.7% | 80.5% |
| Precision | 32.5% | 28.9% |
| Recall | 73.9% | **78.6%** |
| F1 | **45.2%** | 42.3% |
| ROC-AUC | **88.6%** | 88.1% |
| PR-AUC | **61.4%** | 61.3% |

XGBoost's only real advantage was higher recall on Model A (78.6% vs 73.9%) — Logistic Regression won recall on Model B outright (72.3% vs 70.1%). Given the near-tie and the value of interpretability in a healthcare setting, Logistic Regression was selected for both models. Full reasoning in `reports/EXPLAINABILITY_report.md`.

## Evaluation metrics

Final test-set performance (test set touched exactly once, after all tuning was finalized on training/CV data):

| Metric | Model A | Model B |
|---|---|---|
| Accuracy | 83.7% | 82.7% |
| Precision | 32.5% | 30.8% |
| Recall (sensitivity) | 73.9% | 72.3% |
| Specificity | 84.7% | 83.8% |
| F1 score | 45.2% | 43.2% |
| ROC-AUC | 88.6% | 87.3% |
| PR-AUC | 61.4% | 53.2% |
| False negatives | 95 of 364 | 101 of 364 |
| False positives | 558 | 590 |

The dataset is ~9.1% positive (deceased), so **accuracy alone is misleading** — a model predicting "survived" for everyone would score ~91% accuracy while catching zero at-risk patients. Recall, F1, ROC-AUC, and PR-AUC were prioritized throughout for this reason.

## Technologies used

**ML pipeline:** Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, joblib
**Backend:** FastAPI, Pydantic, Uvicorn, python-dotenv
**Frontend:** React, Vite, Tailwind CSS v4, React Router, Recharts, Axios

## Project structure

```
healthcare-ml-project/
├── data/
│   ├── raw/                    # original uploaded dataset
│   └── processed/              # cleaned, featured, train/test splits
├── ml/
│   ├── preprocessing/          # cleaning, feature engineering, pipelines
│   ├── analysis/                # EDA and explainability scripts
│   ├── training/                 # baseline training and hyperparameter tuning
│   └── prediction/                # the reusable prediction class (used by the backend)
├── models/                      # trained model artifacts (see note below)
├── reports/
│   ├── figures/                  # all generated charts (30+ PNGs)
│   ├── model_results/            # machine-readable results (JSON)
│   ├── EDA_report.md
│   └── EXPLAINABILITY_report.md
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entrypoint
│   │   ├── config.py             # centralized file paths
│   │   ├── schemas/               # Pydantic request/response models
│   │   ├── services/               # dataset/model/predictor service layer
│   │   └── routes/                  # /api/predict, /api/dataset, /api/models, /api/eda
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/client.js           # single API service layer
│   │   ├── hooks/useApi.js          # shared loading/error/data hook
│   │   ├── components/               # Layout, StatCard, MetricsTable, FigureCard, etc.
│   │   └── pages/                     # Dashboard, DataAnalysis, ModelPerformance, Explainability, Prediction, Dataset
│   ├── package.json
│   └── .env.example
└── README.md
```

**Note on `models/`:** only `final_model_a.joblib` and `final_model_b.joblib` (Logistic Regression, ~16KB each — the models actually used by the backend) are essential. The smaller baseline/tuned artifacts for Logistic Regression, XGBoost, and HistGradientBoosting are also included for reference (all under ~1MB each). The large Random Forest artifacts (~215MB combined across baseline/tuned × Model A/B) were **excluded from this package** — Random Forest was evaluated and rejected early (Phase 6-7) in favor of Logistic Regression and XGBoost, its full performance record is already preserved in `reports/model_results/*.json`, and it can be regenerated in minutes by re-running `ml/training/05_train_baseline_models.py` and `ml/training/06_tune_and_select.py` if needed.

## Installation and setup

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- Git (optional, for cloning)

### 1. Get the project

Unzip the project archive, or clone the repository, then open a terminal in the project root (`healthcare-ml-project/`).

### 2. Set up the backend

```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate


pip install -r requirements.txt
copy .env.example .env        # Windows
pip install -r requirements.txt
```

### 3. Set up the frontend

```bash
cd frontend
npm install
copy .env.example .env        # Windows

```

## Running the backend

From the `backend/` directory, with the virtual environment activated:

```bash
uvicorn app.main:app --reload
```

- API root: `http://127.0.0.1:8000/`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

The backend loads both prediction models into memory at startup, and serves generated charts at `/static/figures/`.

## Running the frontend

In a **separate terminal**, from the `frontend/` directory:

```bash
npm run dev
```

Then open `http://localhost:5173/` in a browser. The frontend reads the backend URL from `VITE_API_BASE_URL` in `frontend/.env` (defaults to `http://127.0.0.1:8000`).

**Run the backend first** — the frontend's pages will show a "Could not reach the backend" error until it's up.

## API reference

| Endpoint | Purpose |
|---|---|
| `GET /api/dataset/overview` | Dataset-level summary statistics |
| `GET /api/dataset/columns` | Per-column type/range/uniqueness info |
| `GET /api/dataset/preview` | Paginated record preview (leakage/PII columns excluded) |
| `GET /api/dataset/search` | Search records by column + value |
| `GET /api/dataset/missing-values` | Missing-value summary |
| `GET /api/dataset/valid-categories` | Valid values for every categorical field (used to build the prediction form) |
| `GET /api/dataset/tracked-symptoms` | The exact symptom list the models were trained on |
| `GET /api/models/summary` | Final Model A / Model B headline metrics |
| `GET /api/models/baseline` | Phase 6 baseline results, all algorithms |
| `GET /api/models/decision-matrix` | Logistic Regression vs XGBoost final comparison |
| `GET /api/models/explainability` | SHAP/coefficient findings |
| `GET /api/models/feature-config` | Model A / Model B feature-set definitions |
| `GET /api/eda/report` | Full EDA markdown report |
| `GET /api/eda/figures` | List of available chart filenames |
| `POST /api/predict/model-a` | Prediction using the full clinical-course model |
| `POST /api/predict/model-b` | Prediction using the early-risk model |

Full request/response schemas are in the Swagger docs at `/docs` once the backend is running.

## Using the prediction system

1. Open the **Prediction** page in the frontend.
2. Choose **Model A** (if you have treatment-course information) or **Model B** (admission-time only).
3. Fill in the form — all dropdown options are pulled live from the backend, not hardcoded.
4. Submit to get: predicted probability of death, risk category (Low/Moderate/High/Critical), the decision threshold used, and the top contributing factors for that specific patient.

Every prediction response includes an explicit disclaimer field — see below.

## Reproducing the ML pipeline

Each phase has its own script, runnable independently, in order:

```bash
python ml/preprocessing/01_clean_data.py          # Phase 2: cleaning
python ml/analysis/02_eda.py                       # Phase 3: EDA
python ml/preprocessing/03_feature_engineering.py    # Phase 4: feature engineering
python ml/preprocessing/04_preprocessing.py           # Phase 5: train/test split + pipelines
python ml/training/05_train_baseline_models.py          # Phase 6: baseline models
python ml/training/06_tune_and_select.py                 # Phase 7: hyperparameter tuning
python ml/analysis/07_explainability.py                    # Phase 8: SHAP + coefficients
python ml/prediction/predictor.py                            # Phase 9: prediction system demo
```

Run these from the project root with the backend's virtual environment activated (they share the same dependencies).

## Healthcare disclaimer

**This system produces statistical risk estimates, not medical diagnoses.** It is a machine learning demonstration built on a historical dataset. It must not be used to make, support, or influence real clinical decisions, and it is not a substitute for professional medical judgment. Always consult a qualified healthcare provider for diagnosis and treatment decisions. This disclaimer is also returned in every prediction API response.

## Limitations

- Trained on a single historical dataset; performance on other populations or data-collection practices is unknown.
- Precision is modest (~30-33%) at the chosen threshold — the models are tuned to prioritize catching at-risk patients (recall) over minimizing false alarms, which is a deliberate trade-off, not an oversight, but it does mean roughly 2 in 3 flagged patients did not die.
- Model A's post-admission features (treatment cost, length of stay) make it more accurate but less useful as an *early* warning tool — use Model B when only admission-time information is available.
- One coefficient in the Logistic Regression model (`state_Mizoram`) reflects a small-sample artifact (34 patients, 0 recorded deaths) rather than a real geographic effect — documented in `reports/EXPLAINABILITY_report.md`.
- No external validation against real-world clinical outcomes has been performed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized (Windows) | Reinstall Python from python.org and check "Add Python to PATH" during install |
| `pip install` fails / permission errors | Make sure the virtual environment is activated (prompt should show `(venv)`) before running pip |
| Virtual environment won't activate (Windows) | If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry |
| `ModuleNotFoundError: No module named 'ml'` | Make sure you're running backend commands from inside `backend/`, and that the project folder structure hasn't been rearranged — the backend adds the project root to `sys.path` automatically |
| `npm` / Node errors | Confirm `node --version` is 18+; delete `frontend/node_modules` and re-run `npm install` if a previous install seems corrupted |
| Frontend shows "Could not reach the backend" | Confirm the backend is running (`http://127.0.0.1:8000/health` should return `{"status":"ok"}`) and that `VITE_API_BASE_URL` in `frontend/.env` matches |
| CORS errors in the browser console | Check `CORS_ALLOWED_ORIGINS` in `backend/.env` includes your frontend's exact URL (e.g. `http://localhost:5173`) |
| Port already in use | Change the port: `uvicorn app.main:app --reload --port 8001` (backend) or `npm run dev -- --port 5174` (frontend), and update `.env` accordingly |
| Model file not found | Confirm `models/final_model_a.joblib` and `models/final_model_b.joblib` are present — these are required by the backend and are included in this package |
| Dataset path errors when re-running ML scripts | Run all `ml/` scripts from the project root, not from inside `ml/` |
