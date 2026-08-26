# Customer Churn Prediction — End-to-End MLOps Pipeline

Predicts which telecom customers are likely to churn, ranks them by risk, and explains *why*
(via SHAP) so a retention team could realistically act on the output. Built with a fully
open-source MLOps stack: DVC (data versioning), MLflow (experiment tracking + model registry),
SHAP (explainability), Docker (containerization), GitHub Actions (CI), Evidently AI (drift
monitoring), and Streamlit (serving UI).

## Problem Statement

Telecom companies lose revenue when customers churn, and retention teams have limited budget —
they can't reach out to everyone. This project builds a model that scores each customer's churn
risk and surfaces the top factors driving that risk per customer, so a retention team could
prioritize outreach instead of guessing.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/easonlai/sample-telco-customer-churn-dataset)
— 7,011 customers, 20 features (demographics, account info, services subscribed), binary churn
label. Source CSV mirrored at `data/raw/telco_churn.csv`.

## Pipeline

```
data/raw (DVC-tracked)
      │
      ▼
src/preprocess.py  ──▶ data/processed (train/test splits, encoded features)
      │
      ▼
src/train.py  ──▶ trains LogisticRegression / RandomForest / XGBoost / LightGBM
      │            with class-weight and SMOTE variants, logs every run to MLflow
      ▼
MLflow Model Registry (best model promoted to "Production")
      │
      ▼
src/explain.py  ──▶ SHAP values for global + per-prediction explanations
      │
      ▼
app/streamlit_app.py  ──▶ loads registered model, serves predictions + SHAP chart
      │
      ▼
Docker container ──▶ deployed on Streamlit Community Cloud / HF Spaces
      │
      ▼
src/monitor.py  ──▶ Evidently AI drift report (train vs. new-batch data)
```

## Repo structure

```
churn-mlops/
├── data/
│   ├── raw/            # DVC-tracked raw data
│   └── processed/       # train/test splits
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── explain.py
│   └── monitor.py
├── app/
│   └── streamlit_app.py
├── models/               # local MLflow artifact store (or point to remote)
├── tests/
│   └── test_preprocess.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── requirements.txt
├── dvc.yaml
└── README.md
```

## Results

*(fill in after training — this is the section reviewers read first)*

| Model | Accuracy | F1 (churn class) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | | | |
| Random Forest | | | |
| XGBoost | | | |
| XGBoost + SMOTE | | | |

**Final model:** _(name)_, chosen because _(precision/recall tradeoff reasoning tied to
business cost of false negatives vs false positives)_.

## Error analysis

*(fill in — e.g. "model most often confuses month-to-month customers with short tenure;
recommend routing predictions with confidence <0.6 to manual review")*

## Running locally

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
mlflow ui          # inspect experiments at localhost:5000
streamlit run app/streamlit_app.py
```

## Data drift monitoring

```bash
python src/monitor.py   # generates evidently_report.html
```

## License

MIT
