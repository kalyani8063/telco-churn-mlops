# Customer Churn Risk Prediction — MLOps Pipeline

An end-to-end, deployed system that predicts which telecom customers are likely to churn,
explains *why* per customer via SHAP, and was built around a fully open-source MLOps stack:
MLflow (experiment tracking + model registry), SHAP (explainability), scikit-learn/XGBoost/
LightGBM (modeling), and Streamlit (serving). Live demo below.

**[Live app →](https://YOUR-APP-URL.streamlit.app)** *(replace with your actual deployed URL)*

## Problem Statement

Retention teams can't reach out to every customer — budget and time are limited. This project
scores each customer's churn risk and surfaces the specific factors driving that risk, so a
retention team could realistically prioritize outreach instead of guessing.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM sample dataset) — 7,043 customers, 20 raw features (demographics, account info,
subscribed services), binary churn label. ~26.5% churn rate.

**Known data quality issue handled explicitly:** 11 customers have a blank `TotalCharges`
value. Rather than blindly imputing, `src/preprocess.py` first *asserts* that every blank-value
row has `tenure == 0` (i.e. they're brand-new customers who genuinely haven't been billed yet)
before filling with 0 — so the code fails loudly instead of silently mis-imputing if that
assumption is ever wrong on a different data pull.

## Pipeline
data/raw/telco_churn.csv (original IBM CSV, real string categories)
│
▼
src/preprocess.py

drops customerID (no predictive value)
imputes blank TotalCharges (asserted as new-customer edge case)
label-encodes binary Yes/No columns
one-hot encodes multi-category columns (Contract, PaymentMethod, InternetService, etc.)
→ 21 raw columns become 40 model-ready features
│
▼
src/train.py
trains 5 models: Logistic Regression (baseline), Random Forest,
XGBoost (class-weighted), XGBoost (SMOTE-resampled), LightGBM (class-weighted)
every run logged to MLflow (params, metrics, model artifact)
│
▼
MLflow Model Registry — best model (see below) promoted via alias "production"
│
▼
src/explain.py — SHAP global + per-prediction feature importance
│
▼
app/streamlit_app.py — live churn risk scoring + SHAP explanation UI
│
▼
Deployed on Streamlit Community Cloud



## Results

| Model | Accuracy | F1 (churn class) | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.806 | 0.604 | 0.657 | 0.559 | **0.843** |
| Random Forest | 0.803 | 0.583 | 0.664 | 0.519 | 0.842 |
| XGBoost (class-weighted) | 0.752 | 0.619 | 0.522 | 0.759 | 0.834 |
| XGBoost (SMOTE) | 0.784 | 0.584 | 0.596 | 0.572 | 0.834 |
| **LightGBM (class-weighted)** | 0.757 | **0.627** | 0.528 | **0.770** | 0.836 |

**Final model: LightGBM (class-weighted).**

ROC-AUC is essentially tied across all 5 models (0.83–0.84), meaning they all separate churners
from non-churners similarly well overall. The real decision is the precision/recall tradeoff:
missing an actual churner (false negative) costs the business a lost customer with zero chance
to intervene, while a false alarm (false positive) just costs one unnecessary retention email —
cheap by comparison. Given that asymmetry, I optimized for recall over raw accuracy or precision.
LightGBM (class-weighted) catches **77% of actual churners**, meaningfully more than the 56%
caught by the logistic regression baseline, at the acceptable cost of more false positives.

*(Logistic regression's higher precision/ROC-AUC would be the better pick if the retention
outreach itself were expensive or intrusive enough that false alarms had a real cost — that
tradeoff is business-context-dependent, not a universal "better" model.)*

## Explainability — does the model's reasoning make sense?

Rather than trust the metrics alone, I ran the deployed model against 3 hand-constructed
customer profiles to check whether its behavior matches real-world churn intuition:

| Profile | Predicted churn risk | Sanity check |
|---|---|---|
| Long tenure (60mo), two-year contract, has support | **11.8%** ✅ | Correctly low — classic "sticky" customer |
| New customer (1mo), month-to-month, fiber, no support | **91.7%** ✅ | Correctly high — classic churn-risk profile |
| Mid-tenure, one-year contract, fiber, no support, no dependents | **10.1%** | Lower than expected — contract commitment outweighed the risk factors present |

**A genuine finding, not something I hand-picked:** `PaymentMethod = Electronic check` showed
up as a risk-*increasing* SHAP factor in every test case. This matches a well-documented pattern
in telecom churn analysis — customers on manual electronic-check payment (vs. autopay) tend to
churn more, likely correlating with lower account engagement/commitment. The model independently
surfaced this without being told to look for it.

## Architecture note: a real deployment lesson

Initial serving loaded the model directly from MLflow's local file-based registry
(`models:/churn-model@production`). This worked locally but **broke on deployment** — MLflow's
local store records absolute filesystem paths at run-creation time, which don't resolve on a
different machine (Streamlit Cloud clones the repo into a different path than my local Windows
machine). Fix: exported the promoted model to a standalone portable file
(`models/churn_model.pkl`) rather than depending on live registry resolution at serving time.

This is a known real-world MLOps gotcha — local file-based tracking/registries aren't portable
across environments. A production setup at scale would use a remote MLflow tracking server
(hosted, backed by a real DB + S3/GCS artifact store) instead. Documenting and working around
this limitation, rather than hiding it, felt more honest than pretending the registry pattern
worked end-to-end without a hitch.

## What's containerized / what's not (honest status)

- A `Dockerfile` is included and describes the intended container build (Python 3.11-slim,
  installs `requirements.txt`, serves via Streamlit on port 8501).
- Local Docker Desktop/WSL2 environment issues prevented a fully verified local `docker build`
  during development — noting this rather than claiming a container image was tested end-to-end
  I didn't confirm.

## Running locally

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
mlflow ui                          # inspect experiments at localhost:5000
streamlit run app/streamlit_app.py
```

## Data drift monitoring

```bash
python src/monitor.py   # generates evidently_report.html
```

*(Compares the train/test split as reference vs. current — with a real production system this
would compare training data against live incoming traffic instead.)*

## Tech stack

Python, pandas, scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE), MLflow, SHAP,
Streamlit, Docker, GitHub Actions (CI), Evidently AI.

## License

MIT