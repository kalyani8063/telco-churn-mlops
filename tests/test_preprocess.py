import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import clean, encode


def make_raw_row(**overrides):
    row = {
        "customerID": "0001-AAAAA",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_clean_handles_blank_total_charges_for_new_customers():
    df = pd.DataFrame([
        make_raw_row(customerID="0001", tenure=0, TotalCharges=" "),
        make_raw_row(customerID="0002", tenure=34, TotalCharges="1889.5"),
    ])
    cleaned = clean(df)
    assert cleaned["TotalCharges"].iloc[0] == 0
    assert cleaned["TotalCharges"].iloc[1] == 1889.5


def test_clean_raises_if_blank_total_charges_has_nonzero_tenure():
    df = pd.DataFrame([
        make_raw_row(customerID="0001", tenure=5, TotalCharges=" "),
    ])
    with pytest.raises(AssertionError):
        clean(df)


def test_clean_drops_customer_id():
    df = pd.DataFrame([make_raw_row()])
    cleaned = clean(df)
    assert "customerID" not in cleaned.columns


def test_clean_encodes_gender_and_churn():
    df = pd.DataFrame([
        make_raw_row(customerID="0001", gender="Female", Churn="No"),
        make_raw_row(customerID="0002", gender="Male", Churn="Yes"),
    ])
    cleaned = clean(df)
    assert cleaned["gender"].tolist() == [0, 1]
    assert cleaned["Churn"].tolist() == [0, 1]


def test_clean_encodes_binary_yes_no_columns():
    df = pd.DataFrame([
        make_raw_row(Partner="Yes", Dependents="No", PhoneService="Yes", PaperlessBilling="No"),
    ])
    cleaned = clean(df)
    assert cleaned["Partner"].iloc[0] == 1
    assert cleaned["Dependents"].iloc[0] == 0
    assert cleaned["PhoneService"].iloc[0] == 1
    assert cleaned["PaperlessBilling"].iloc[0] == 0


def test_encode_one_hot_encodes_multi_category_columns():
    df = pd.DataFrame([
        make_raw_row(Contract="Month-to-month"),
        make_raw_row(customerID="0002", Contract="Two year"),
    ])
    cleaned = clean(df)
    encoded = encode(cleaned)
    assert "Contract" not in encoded.columns
    assert "Contract_Month-to-month" in encoded.columns
    assert "Contract_Two year" in encoded.columns
    assert encoded["Contract_Month-to-month"].tolist() == [1, 0]


def test_encode_keeps_three_valued_categories_distinct():
    df = pd.DataFrame([
        make_raw_row(OnlineSecurity="Yes"),
        make_raw_row(customerID="0002", OnlineSecurity="No"),
        make_raw_row(customerID="0003", OnlineSecurity="No internet service"),
    ])
    cleaned = clean(df)
    encoded = encode(cleaned)
    assert "OnlineSecurity_Yes" in encoded.columns
    assert "OnlineSecurity_No" in encoded.columns
    assert "OnlineSecurity_No internet service" in encoded.columns


def test_encode_output_has_no_object_columns():
    df = pd.DataFrame([make_raw_row(), make_raw_row(customerID="0002")])
    cleaned = clean(df)
    encoded = encode(cleaned)
    assert encoded.select_dtypes(include="object").columns.empty
