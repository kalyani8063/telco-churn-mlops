"""
Streamlit demo: customer churn risk scoring with SHAP-based explanations.

Run locally:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import mlflow
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import clean, encode

st.set_page_config(page_title="Churn Risk Predictor", layout="centered")

MODEL_URI = os.environ.get("MODEL_URI", "models:/churn-model@production")
PROCESSED_TRAIN_PATH = Path("data/processed/train.csv")


@st.cache_resource
def load_model():
    import joblib
    return joblib.load("models/churn_model.pkl")

@st.cache_resource
def load_explainer(_model, background_df):
    return shap.Explainer(_model.predict, background_df)


@st.cache_resource
def load_training_columns():
    """The exact one-hot-encoded feature columns (and order) the model was trained on."""
    try:
        return [c for c in pd.read_csv(PROCESSED_TRAIN_PATH, nrows=0).columns if c != "Churn"]
    except FileNotFoundError:
        st.error(
            "Couldn't find `data/processed/train.csv`. Run `python src/preprocess.py` "
            "first so the app knows the model's expected feature columns."
        )
        st.stop()


def get_background_sample():
    df = pd.read_csv(PROCESSED_TRAIN_PATH).drop(columns=["Churn"])
    return df.sample(min(100, len(df)), random_state=42)


def build_model_row(
    model_columns, *, tenure, monthly_charges, total_charges, contract, internet,
    senior, partner, dependents, paperless, tech_support,
):
    """Build a single-customer row through the same clean()/encode() pipeline used
    at training time, then align it to the model's actual training columns — so a
    category not chosen in the form correctly ends up as 0 rather than missing."""
    raw_row = pd.DataFrame([{
        "customerID": "DEMO-0000",
        "gender": "Female",
        "SeniorCitizen": int(senior),
        "Partner": "Yes" if partner else "No",
        "Dependents": "Yes" if dependents else "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes" if paperless else "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": "No",
    }])
    encoded = encode(clean(raw_row)).drop(columns=["Churn"])
    return encoded.reindex(columns=model_columns, fill_value=0)


def main():
    st.title("📉 Customer Churn Risk Predictor")
    st.caption(
        "Enter a customer's profile to get a churn probability and the top factors "
        "driving that prediction — built to help a retention team prioritize outreach."
    )

    model = load_model()
    model_columns = load_training_columns()
    background = get_background_sample()
    explainer = load_explainer(model, background)

    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with col2:
            senior = st.checkbox("Senior Citizen")
            partner = st.checkbox("Has Partner")
            dependents = st.checkbox("Has Dependents")
            paperless = st.checkbox("Paperless Billing", value=True)
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("Predict churn risk")

    if submitted:
        row = build_model_row(
            model_columns,
            tenure=tenure, monthly_charges=monthly_charges, total_charges=total_charges,
            contract=contract, internet=internet, senior=senior, partner=partner,
            dependents=dependents, paperless=paperless, tech_support=tech_support,
        )

        proba = model.predict_proba(row)[0, 1]
        st.metric("Predicted churn probability", f"{proba:.1%}")

        if proba > 0.5:
            st.warning("⚠️ High risk — recommend prioritizing for retention outreach.")
        else:
            st.success("✅ Low risk.")

        st.subheader("Top factors driving this prediction")
        shap_values = explainer(row)
        contributions = pd.Series(shap_values.values[0], index=row.columns)
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(5)

        fig, ax = plt.subplots()
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top.values]
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        ax.set_xlabel("Impact on churn probability (SHAP value)")
        st.pyplot(fig)
        st.caption("Red = pushes prediction toward churn. Green = pushes toward staying.")


if __name__ == "__main__":
    main()
