"""
Train and compare multiple churn models, logging every run to MLflow.

Run:
    mlflow ui  &            # optional, to watch runs live at localhost:5000
    python src/train.py
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

PROCESSED_DIR = Path("data/processed")
EXPERIMENT_NAME = "churn-prediction"

mlflow.set_experiment(EXPERIMENT_NAME)


def load_splits():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]
    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def log_run(run_name: str, model, X_train, y_train, X_test, y_test, params: dict):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{run_name:30s} | " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        return metrics


def main():
    X_train, y_train, X_test, y_test = load_splits()

    results = {}

    # --- Baseline: Logistic Regression ---
    results["logreg_baseline"] = log_run(
        "logreg_baseline",
        LogisticRegression(max_iter=1000),
        X_train, y_train, X_test, y_test,
        params={"model": "LogisticRegression", "imbalance_strategy": "none"},
    )

    # --- Random Forest ---
    results["random_forest"] = log_run(
        "random_forest",
        RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
        X_train, y_train, X_test, y_test,
        params={"model": "RandomForest", "n_estimators": 300, "max_depth": 8, "imbalance_strategy": "none"},
    )

    # --- XGBoost, class-weighted ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    results["xgboost_class_weight"] = log_run(
        "xgboost_class_weight",
        xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
        ),
        X_train, y_train, X_test, y_test,
        params={"model": "XGBoost", "imbalance_strategy": "class_weight"},
    )

    # --- XGBoost, SMOTE-resampled ---
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    results["xgboost_smote"] = log_run(
        "xgboost_smote",
        xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        ),
        X_train_sm, y_train_sm, X_test, y_test,
        params={"model": "XGBoost", "imbalance_strategy": "SMOTE"},
    )

    # --- LightGBM, class-weighted ---
    results["lightgbm_class_weight"] = log_run(
        "lightgbm_class_weight",
        lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1,
        ),
        X_train, y_train, X_test, y_test,
        params={"model": "LightGBM", "imbalance_strategy": "class_weight"},
    )

    print("\n=== Summary (sorted by F1) ===")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True):
        print(f"{name:30s} F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}  Recall={m['recall']:.3f}")

    print(
        "\nNext step: open the MLflow UI (`mlflow ui`), inspect all runs, "
        "pick the best tradeoff for your business case (not just highest F1 — "
        "consider recall if missing a churner is costlier than a false alarm), "
        "and promote that run's model to the MLflow Model Registry as 'Production'."
    )


if __name__ == "__main__":
    main()
