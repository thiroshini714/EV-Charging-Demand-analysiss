"""Feature engineering: calendar features and leakage-free historical features.

Historical features for a session use ONLY sessions that started strictly
before it (cumulative sums minus the current row), so no target leakage.
"""
import numpy as np
import pandas as pd

from src.config import FEATURES, HIGH_DEMAND_QUANTILE, TARGET, TEST_SIZE


def time_of_day(hour: pd.Series) -> pd.Series:
    """Map hour (0-23) to Night / Morning / Afternoon / Evening."""
    bins = [-1, 5, 11, 16, 21, 23]
    labels = ["Night", "Morning", "Afternoon", "Evening", "Night "]
    out = pd.cut(hour, bins=bins, labels=labels, ordered=False).astype(str)
    return out.str.strip()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["StartDatetime"]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["time_of_day"] = time_of_day(df["hour"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def _prev_stats(df: pd.DataFrame, keys, prefix: str) -> pd.DataFrame:
    """Count and mean Demand of earlier sessions in the same group (df time-sorted)."""
    g = df.groupby(keys, sort=False)
    prev_count = g.cumcount()
    prev_sum = g[TARGET].cumsum() - df[TARGET]
    avg = prev_sum / prev_count.replace(0, np.nan)
    return prev_count, avg


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add usage-history features. `df` must be sorted by StartDatetime."""
    df = df.sort_values("StartDatetime").reset_index(drop=True).copy()
    df["charger_prev_sessions"], df["charger_prev_avg_demand"] = _prev_stats(df, "ChargerID", "charger")
    df["user_prev_sessions"], df["user_prev_avg_demand"] = _prev_stats(df, "UserID", "user")
    _, df["location_hour_prev_avg_demand"] = _prev_stats(df, ["Location", "hour"], "loc_hour")

    # sessions on the same charger during the 24 h before this session starts
    counts = np.zeros(len(df), dtype=int)
    times = df["StartDatetime"].values.astype("datetime64[ns]").astype("int64")
    window = np.int64(24 * 3600 * 10**9)
    for _, idx in df.groupby("ChargerID", sort=False).indices.items():
        t = times[idx]                       # already ascending (df sorted)
        counts[idx] = np.arange(len(t)) - np.searchsorted(t, t - window, side="left")
    df["charger_sessions_prev_24h"] = counts
    return df


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Full feature table (sorted by time)."""
    return add_history_features(add_time_features(clean.sort_values("StartDatetime")))


def split_chronological(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Train on earlier sessions, test on the most recent ones (no look-ahead)."""
    df = df.sort_values("StartDatetime").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def make_class_target(demand: pd.Series, threshold: float) -> pd.Series:
    """1 = High Demand (Demand >= threshold), 0 = Normal Demand."""
    return (demand >= threshold).astype(int)


def demand_threshold(train_demand: pd.Series, q: float = HIGH_DEMAND_QUANTILE) -> float:
    """Threshold derived from TRAINING data only (PDF: median demand)."""
    return float(train_demand.quantile(q))


def make_inference_frame(inputs: dict, history: pd.DataFrame) -> pd.DataFrame:
    """Build one model-ready row from user inputs (used by the dashboard).

    `inputs` needs: ChargerCompany, Location, ChargerType, StartDatetime and may
    contain history values; missing history values stay NaN and are imputed by
    the model pipeline. `history` is the feature table (for defaults).
    """
    row = {"StartDatetime": pd.Timestamp(inputs["StartDatetime"])}
    frame = add_time_features(pd.DataFrame([{**row, **{k: inputs[k] for k in
                              ("ChargerCompany", "Location", "ChargerType")}}]))
    for col in ("charger_prev_sessions", "charger_prev_avg_demand", "charger_sessions_prev_24h",
                "user_prev_sessions", "user_prev_avg_demand", "location_hour_prev_avg_demand"):
        frame[col] = inputs.get(col, np.nan)
    if np.isnan(frame.loc[0, "location_hour_prev_avg_demand"]) and history is not None:
        m = history[(history["Location"] == inputs["Location"]) & (history["hour"] == frame.loc[0, "hour"])]
        if len(m):
            frame.loc[0, "location_hour_prev_avg_demand"] = m[TARGET].mean()
    return frame[FEATURES]
