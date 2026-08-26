"""
Streamlit demo: customer churn risk scoring with SHAP-based explanations.

Run locally:
    streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import mlflow
import os

st.set_page_config(page_title="Churn Risk Predictor", layout="centered")

MODEL_URI = os.environ.get("MODEL_URI", "models:/churn-model/Production")

FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "Tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


@st.cache_resource
def load_model():
    try:
        return mlflow.sklearn.load_model(MODEL_URI)
    except Exception as e:
        st.error(
            f"Couldn't load model from `{MODEL_URI}`. "
            f"Train a model first with `python src/train.py`, then register it "
            f"in MLflow as 'churn-model' / stage 'Production'. Error: {e}"
        )
        st.stop()


@st.cache_resource
def load_explainer(_model, background_df):
    return shap.Explainer(_model.predict, background_df)


def get_background_sample():
    """Small reference sample used by SHAP; falls back to zeros if no data available."""
    try:
        df = pd.read_csv("data/processed/train.csv").drop(columns=["Churn"])
        return df.sample(min(100, len(df)), random_state=42)
    except FileNotFoundError:
        return pd.DataFrame([np.zeros(len(FEATURE_COLS))], columns=FEATURE_COLS)


def main():
    st.title("📉 Customer Churn Risk Predictor")
    st.caption(
        "Enter a customer's profile to get a churn probability and the top factors "
        "driving that prediction — built to help a retention team prioritize outreach."
    )

    model = load_model()
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
        # NOTE: this mapping must match whatever encoding preprocess.py used at
        # training time. Wire this up to the real LabelEncoder mappings you saved
        # during preprocessing rather than hand-coding indices like this demo does.
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
        tech_map = {"Yes": 1, "No": 0, "No internet service": 2}

        row = pd.DataFrame([{
            "gender": 0, "SeniorCitizen": int(senior), "Partner": int(partner),
            "Dependents": int(dependents), "Tenure": tenure, "PhoneService": 1,
            "MultipleLines": 0, "InternetService": internet_map[internet],
            "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
            "TechSupport": tech_map[tech_support], "StreamingTV": 0, "StreamingMovies": 0,
            "Contract": contract_map[contract], "PaperlessBilling": int(paperless),
            "PaymentMethod": 2, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])[FEATURE_COLS]

        proba = model.predict_proba(row)[0, 1]
        st.metric("Predicted churn probability", f"{proba:.1%}")

        if proba > 0.5:
            st.warning("⚠️ High risk — recommend prioritizing for retention outreach.")
        else:
            st.success("✅ Low risk.")

        st.subheader("Top factors driving this prediction")
        shap_values = explainer(row)
        contributions = pd.Series(shap_values.values[0], index=FEATURE_COLS)
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(5)

        fig, ax = plt.subplots()
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top.values]
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        ax.set_xlabel("Impact on churn probability (SHAP value)")
        st.pyplot(fig)
        st.caption("Red = pushes prediction toward churn. Green = pushes toward staying.")


if __name__ == "__main__":
    main()
