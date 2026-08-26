"""
Generate a data-drift report comparing training data ("reference") against
a simulated "new" batch of incoming data.

Since we don't have live production traffic, we simulate drift by holding
out a later slice of the data as the "current" batch — a legitimate and
common way to demo drift monitoring without a live system.

Run:
    python src/monitor.py
"""
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path("evidently_report.html")


def main():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=train, current_data=test, column_mapping=None)
    report.save_html(str(OUTPUT_PATH))

    print(f"Drift report saved to {OUTPUT_PATH} — open it in a browser to inspect.")
    print(
        "Note: train.csv and test.csv are a random split, so drift here should be "
        "minimal — that's expected and worth stating explicitly in your README. "
        "To demonstrate *real* drift for the resume story, try splitting by a "
        "column like tenure or Contract type instead, and note the difference."
    )


if __name__ == "__main__":
    main()
