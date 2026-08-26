import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import clean, encode


def test_clean_handles_blank_total_charges():
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "TotalCharges": [" ", "100.5"],
        "Churn": [0, 1],
    })
    cleaned = clean(df)
    assert cleaned["TotalCharges"].iloc[0] == 0
    assert cleaned["TotalCharges"].iloc[1] == 100.5
    assert "customerID" not in cleaned.columns


def test_clean_drops_customer_id():
    df = pd.DataFrame({"customerID": ["001"], "TotalCharges": ["10"], "Churn": [0]})
    cleaned = clean(df)
    assert "customerID" not in cleaned.columns


def test_encode_handles_no_object_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    encoded = encode(df)
    pd.testing.assert_frame_equal(df, encoded)


def test_encode_label_encodes_object_columns():
    df = pd.DataFrame({"category": ["yes", "no", "yes"]})
    encoded = encode(df)
    assert encoded["category"].dtype != object
