"""Unit tests for cleaning, features, leakage safeguards and modelling helpers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification import _oversample, imbalance_report  # noqa: E402
from src.config import FEATURES  # noqa: E402
from src.data_cleaning import clean_data  # noqa: E402
from src.data_generation import generate_synthetic_data  # noqa: E402
from src.feature_engineering import (build_features, demand_threshold,  # noqa: E402
                                     make_class_target, make_inference_frame,
                                     split_chronological, time_of_day)
from src.regression import regression_metrics  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return generate_synthetic_data(n_rows=4000, seed=1)


@pytest.fixture(scope="module")
def clean(raw):
    return clean_data(raw)[0]


@pytest.fixture(scope="module")
def feats(clean):
    return build_features(clean)


def test_raw_schema_matches_pdf(raw):
    expected = ["UserID", "ChargerID", "ChargerCompany", "Location", "ChargerType", "StartDay",
                "StartTime", "EndDay", "EndTime", "StartDatetime", "EndDatetime", "Duration", "Demand"]
    assert list(raw.columns) == expected


def test_cleaning_removes_problems(raw):
    clean, report = clean_data(raw)
    assert report["duplicates_removed"] > 0
    assert clean.duplicated().sum() == 0
    assert clean["Demand"].notna().all() and (clean["Demand"] > 0).all()
    assert clean[["ChargerCompany", "Location", "ChargerType", "Duration"]].notna().all().all()
    assert pd.api.types.is_datetime64_any_dtype(clean["StartDatetime"])
    assert clean["StartDatetime"].is_monotonic_increasing


def test_cleaning_rejects_missing_columns():
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame({"UserID": [1]}))


def test_time_features(feats):
    assert feats["hour"].between(0, 23).all()
    assert set(feats["is_weekend"].unique()) <= {0, 1}
    assert set(feats["time_of_day"].unique()) <= {"Night", "Morning", "Afternoon", "Evening"}
    assert (feats["is_weekend"] == (feats["day_of_week"] >= 5).astype(int)).all()


def test_time_of_day_bins():
    hours = pd.Series([0, 5, 6, 11, 12, 16, 17, 21, 22, 23])
    assert time_of_day(hours).tolist() == ["Night", "Night", "Morning", "Morning", "Afternoon",
                                           "Afternoon", "Evening", "Evening", "Night", "Night"]


def test_history_features_do_not_leak(feats):
    """First session of any charger has no history; later averages exclude the current row."""
    first = feats.groupby("ChargerID").head(1)
    assert (first["charger_prev_sessions"] == 0).all()
    assert first["charger_prev_avg_demand"].isna().all()
    ch = feats[feats["ChargerID"] == feats["ChargerID"].iloc[0]].reset_index(drop=True)
    expected = ch["Demand"].shift(1).expanding().mean()
    pd.testing.assert_series_equal(ch["charger_prev_avg_demand"], expected,
                                   check_names=False, check_exact=False, rtol=1e-9)


def test_sessions_prev_24h(feats):
    ch = feats[feats["ChargerID"] == feats["ChargerID"].iloc[0]].reset_index(drop=True)
    t = ch["StartDatetime"]
    for i in (0, 5, len(ch) // 2, len(ch) - 1):
        manual = ((t[:i] >= t[i] - pd.Timedelta(hours=24))).sum()
        assert ch.loc[i, "charger_sessions_prev_24h"] == manual


def test_chronological_split_has_no_overlap(feats):
    train, test = split_chronological(feats, 0.2)
    assert train["StartDatetime"].max() <= test["StartDatetime"].min()
    assert len(train) + len(test) == len(feats)


def test_threshold_from_train_and_target_definition(feats):
    train, test = split_chronological(feats)
    thr = demand_threshold(train["Demand"])
    assert thr == pytest.approx(train["Demand"].median())
    y = make_class_target(train["Demand"], thr)
    assert ((train["Demand"] >= thr).astype(int) == y).all()
    assert abs(y.mean() - 0.5) < 0.05           # median split -> roughly balanced


def test_imbalance_report():
    rep = imbalance_report(pd.Series([0] * 90 + [1] * 10))
    assert rep["imbalanced"] and rep["minority_to_majority_ratio"] == pytest.approx(10 / 90, abs=1e-3)
    assert not imbalance_report(pd.Series([0] * 50 + [1] * 50))["imbalanced"]


def test_manual_oversampling_balances_classes():
    X = np.random.default_rng(0).normal(size=(100, 3))
    y = np.array([0] * 90 + [1] * 10)
    Xr, yr = _oversample(X, y, "manual")
    assert (yr == 0).sum() == (yr == 1).sum() == 90 and len(Xr) == len(yr)


def test_regression_metrics():
    m = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert m["MAE"] == pytest.approx(1 / 3)
    assert m["MSE"] == pytest.approx(1 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(1 / 3))


def test_inference_frame_columns(feats):
    row = make_inference_frame({"ChargerCompany": "VoltGrid", "Location": "Downtown",
                                "ChargerType": "DC Fast", "StartDatetime": "2024-06-01 18:30"}, feats)
    assert list(row.columns) == FEATURES and len(row) == 1
    assert row.loc[0, "is_weekend"] == 1          # 2024-06-01 is a Saturday
    assert not np.isnan(row.loc[0, "location_hour_prev_avg_demand"])
