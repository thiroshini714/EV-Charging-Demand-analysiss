"""Time-series forecasting of total daily charging demand (kWh).

Models: seasonal-naive baseline, Holt-Winters (if statsmodels is installed) and a
lag-feature gradient-boosting model with recursive multi-step forecasting.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import optional_deps as deps
from src.config import RANDOM_STATE, REPORTS_DIR, TARGET

LAGS = (1, 2, 7, 14, 28)


def daily_series(df: pd.DataFrame) -> pd.Series:
    """Total daily demand (kWh), continuous daily index (missing days = 0)."""
    s = df.set_index("StartDatetime")[TARGET].resample("D").sum()
    return s.asfreq("D", fill_value=0.0)


def _calendar(index: pd.DatetimeIndex, t0: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "dow": index.dayofweek, "month": index.month,
        "doy_sin": np.sin(2 * np.pi * index.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * index.dayofyear / 365.25),
        "trend": (index - t0).days}, index=index)


def _lag_row(history: pd.Series, day: pd.Timestamp, t0: pd.Timestamp) -> pd.DataFrame:
    feats = _calendar(pd.DatetimeIndex([day]), t0)
    for lag in LAGS:
        feats[f"lag_{lag}"] = history.get(day - pd.Timedelta(days=lag), np.nan)
    window = history[history.index < day].tail(7)
    feats["roll_7"] = window.mean()
    return feats


def _train_matrix(series: pd.Series):
    t0 = series.index[0]
    X = _calendar(series.index, t0)
    for lag in LAGS:
        X[f"lag_{lag}"] = series.shift(lag)
    X["roll_7"] = series.shift(1).rolling(7).mean()
    mask = X.notna().all(axis=1)
    return X[mask], series[mask], t0


def _recursive_forecast(model, history: pd.Series, horizon: int, t0) -> pd.Series:
    hist = history.copy()
    preds = {}
    for i in range(1, horizon + 1):
        day = history.index[-1] + pd.Timedelta(days=i)
        p = float(model.predict(_lag_row(hist, day, t0))[0])
        preds[day] = max(p, 0.0)
        hist.loc[day] = preds[day]
    return pd.Series(preds)


def metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred),
            "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)}


def run_forecasting(df: pd.DataFrame, test_days: int = 60, horizon: int = 30) -> dict:
    """Backtest on the last `test_days`, then forecast `horizon` days beyond the data."""
    series = daily_series(df)
    train_s, test_s = series.iloc[:-test_days], series.iloc[-test_days:]
    rows, back = [], pd.DataFrame({"Actual": test_s})

    # 1. seasonal naive (same weekday last week)
    # repeat the last observed training week (no peeking into the test window)
    naive = pd.Series([train_s.iloc[-7 + (i % 7)] for i in range(len(test_s))], index=test_s.index)
    rows.append({"Model": "Seasonal Naive (7d)", **metrics(test_s.values, naive.values)})
    back["Seasonal Naive (7d)"] = naive

    # 2. Holt-Winters (optional)
    if deps.HAS_STATSMODELS:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw = ExponentialSmoothing(train_s, trend="add", seasonal="add", seasonal_periods=7).fit()
        hw_pred = pd.Series(np.clip(hw.forecast(test_days).values, 0, None), index=test_s.index)
        rows.append({"Model": "Holt-Winters", **metrics(test_s.values, hw_pred.values)})
        back["Holt-Winters"] = hw_pred

    # 3. lag-feature gradient boosting, recursive forecast over the test window
    X, y, t0 = _train_matrix(train_s)
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE).fit(X, y)
    gbm_pred = _recursive_forecast(gbm, train_s, test_days, t0)
    rows.append({"Model": "Gradient Boosting (lags)", **metrics(test_s.values, gbm_pred.values)})
    back["Gradient Boosting (lags)"] = gbm_pred

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    # future forecast with the best model refit on ALL data
    best = results.loc[0, "Model"]
    if best == "Holt-Winters":
        full = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7).fit()
        idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        future = pd.Series(np.clip(full.forecast(horizon).values, 0, None), index=idx)
    elif best == "Gradient Boosting (lags)":
        Xf, yf, t0f = _train_matrix(series)
        gbm_full = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                 random_state=RANDOM_STATE).fit(Xf, yf)
        future = _recursive_forecast(gbm_full, series, horizon, t0f)
    else:
        last_week = series.iloc[-7:].values
        idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        future = pd.Series([last_week[i % 7] for i in range(horizon)], index=idx)

    results.to_csv(REPORTS_DIR / "forecast_metrics.csv", index=False)
    back.rename_axis("Date").to_csv(REPORTS_DIR / "forecast_backtest.csv")
    future.rename("Forecast_kWh").rename_axis("Date").to_csv(REPORTS_DIR / "forecast_future.csv")
    return {"results": results, "best": best, "backtest": back, "future": future}
