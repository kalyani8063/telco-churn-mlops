"""
Preprocess the RAW Telco churn CSV (original IBM/Kaggle format — real string
categories, real data quality issues) into clean, encoded train/test splits.

Key real-world issues this handles:

1. `TotalCharges` is stored as a string and is BLANK for 11 customers — all of
   whom have `tenure == 0` (brand-new customers who haven't been billed yet).
   Decision: impute these as 0, since that's factually correct (they've paid
   $0 in total so far). This is verified with an assertion, not assumed.

2. Several categorical columns use 3 values instead of 2 — e.g. OnlineSecurity
   is "Yes" / "No" / "No internet service". These are kept as-is via one-hot
   encoding rather than collapsed, so the model can use the distinction if
   it's predictive.

3. Binary Yes/No columns are label-encoded (0/1). Multi-category columns are
   one-hot encoded, since they have no ordinal relationship.

4. `customerID` is dropped — unique identifier, zero predictive signal.

Run:
    python src/preprocess.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BINARY_YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

MULTI_CATEGORY_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    blank_mask = df["TotalCharges"].isna()
    if blank_mask.any():
        assert (df.loc[blank_mask, "tenure"] == 0).all(), (
            "Found blank TotalCharges rows with nonzero tenure — the "
            "'new customer' imputation assumption doesn't hold, investigate "
            "before filling with 0."
        )
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    for col in BINARY_YES_NO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=MULTI_CATEGORY_COLS, drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def split_and_save(df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train.assign(**{target_col: y_train}).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(**{target_col: y_test}).to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Train shape: {X_train.shape} | churn rate: {y_train.mean():.3f}")
    print(f"Test shape:  {X_test.shape} | churn rate: {y_test.mean():.3f}")
    print(f"Final feature count after one-hot encoding: {X_train.shape[1]}")


if __name__ == "__main__":
    df = load_raw()
    print(f"Raw shape: {df.shape}")

    n_blank_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    print(f"Blank TotalCharges rows found: {n_blank_charges}")

    df = clean(df)
    df = encode(df)
    split_and_save(df)
    print(f"Saved processed splits to {PROCESSED_DIR}/")
