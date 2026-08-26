"""
Preprocess the raw Telco churn CSV into clean, encoded train/test splits.

Run:
    python src/preprocess.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

RAW_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges sometimes arrives as blank strings for new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop identifier column, it's not predictive
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: columns in this specific mirror of the dataset are already
    integer-encoded (0/1/2...) rather than raw strings. This function is
    defensive — if you swap in the original string-valued Kaggle CSV,
    it will label-encode any remaining object columns automatically.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
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


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    df = encode(df)
    split_and_save(df)
    print(f"Saved processed splits to {PROCESSED_DIR}/")
