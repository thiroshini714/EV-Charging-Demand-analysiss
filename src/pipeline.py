"""End-to-end pipeline: raw data -> clean -> EDA -> features -> models -> explainability.

Usage:
    python -m src.pipeline                # full run
    python -m src.pipeline --fast         # smaller/faster models (smoke test)
    python -m src.pipeline --rows 20000   # synthetic row count (only when no raw CSV exists)
"""
import argparse
import json

import pandas as pd

from src import optional_deps as deps
from src.classification import train_classification
from src.config import CLEAN_FILE, FEATURE_FILE, RAW_FILE, REPORTS_DIR
from src.data_cleaning import clean_data, describe_data
from src.data_generation import generate_synthetic_data
from src.eda import run_eda
from src.explainability import run_explainability
from src.feature_engineering import build_features, split_chronological
from src.forecasting import run_forecasting
from src.regression import train_regression


def load_or_generate_raw(rows: int) -> tuple:
    """Use data/raw/charging_data.csv if present, otherwise create synthetic data."""
    if RAW_FILE.exists():
        print(f"[data] loading real/raw file {RAW_FILE}")
        return pd.read_csv(RAW_FILE), False
    print(f"[data] {RAW_FILE.name} not found -> generating SYNTHETIC data ({rows:,} rows)")
    raw = generate_synthetic_data(n_rows=rows)
    raw.to_csv(RAW_FILE, index=False)
    return raw, True


def main(rows: int = 72000, fast: bool = False) -> dict:
    summary = {"fast_mode": fast}
    raw, synthetic = load_or_generate_raw(rows)
    summary["synthetic_data"] = synthetic
    summary["raw_profile"] = describe_data(raw)

    print("[clean] cleaning ...")
    clean, report = clean_data(raw)
    summary["cleaning_report"] = report
    clean.to_csv(CLEAN_FILE, index=False)

    print("[features] engineering ...")
    feats = build_features(clean)
    feats.to_csv(FEATURE_FILE, index=False)

    print("[eda] figures ...")
    summary["eda_findings"] = run_eda(feats)["findings"]

    train, test = split_chronological(feats)
    summary["split"] = {"train_rows": len(train), "test_rows": len(test),
                        "train_end": str(train["StartDatetime"].max()),
                        "test_start": str(test["StartDatetime"].min())}

    print("[regression] training ...")
    reg = train_regression(train, test, fast)
    summary["regression_best"] = reg["best_name"]

    print("[classification] training ...")
    clf = train_classification(train, test, fast)
    summary["classification_best"] = clf["best_name"]
    summary["class_balance"] = clf["imbalance"]

    print("[explainability] SHAP / LIME ...")
    summary["explainability"] = run_explainability(reg["best_pipeline"], clf["best_pipeline"], test, train)

    print("[forecast] time series ...")
    fc = run_forecasting(feats, test_days=30 if fast else 60, horizon=30)
    summary["forecast_best"] = fc["best"]

    summary["libraries_available"] = {"xgboost": deps.HAS_XGB, "lightgbm": deps.HAS_LGBM,
                                      "imbalanced-learn": deps.HAS_IMBLEARN, "shap": deps.HAS_SHAP,
                                      "lime": deps.HAS_LIME, "statsmodels": deps.HAS_STATSMODELS}
    summary["skipped"] = list(dict.fromkeys(reg["skipped"] + clf["skipped"]))
    (REPORTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[done] artifacts written to models/ and reports/")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=72000)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    main(a.rows, a.fast)
