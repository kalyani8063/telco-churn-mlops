"""
Generate SHAP explanations for a trained model.

This assumes you've already trained an XGBoost/LightGBM/RandomForest model
in train.py. Point MODEL_URI at the MLflow run/model you want to explain,
or load a locally pickled model directly.

Run:
    python src/explain.py
"""
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set this to the run_id of your chosen best model from MLflow, e.g.
# MODEL_URI = "runs:/<run_id>/model"
MODEL_URI = "models:/churn-model/Production"  # or point at a specific runs:/ URI


def load_test_data():
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = test.drop(columns=["Churn"])
    return X_test


def explain(model, X_test: pd.DataFrame, n_samples: int = 200):
    sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)

    explainer = shap.Explainer(model.predict, sample)
    shap_values = explainer(sample)

    # Global feature importance
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved global SHAP summary to {OUTPUT_DIR / 'shap_summary.png'}")

    return explainer, shap_values


def explain_single_customer(explainer, X_row: pd.DataFrame):
    """Used by the Streamlit app to show per-customer 'why' reasons."""
    shap_values = explainer(X_row)
    contributions = pd.Series(shap_values.values[0], index=X_row.columns)
    top_reasons = contributions.abs().sort_values(ascending=False).head(3)
    return contributions[top_reasons.index]


if __name__ == "__main__":
    X_test = load_test_data()
    model = mlflow.sklearn.load_model(MODEL_URI)
    explainer, shap_values = explain(model, X_test)
    print("Done. Use explain_single_customer() inside the Streamlit app for per-prediction reasons.")
