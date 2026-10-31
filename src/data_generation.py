"""Synthetic stand-in for the "Charging Record" dataset.

The real CSV was not available when this project was built, so this module
generates ~72k sessions with the SAME schema as the PDF:
UserID, ChargerID, ChargerCompany, Location, ChargerType, StartDay, StartTime,
EndDay, EndTime, StartDatetime, EndDatetime, Duration, Demand.

Deliberate data-quality problems (missing values, duplicates, invalid and
extreme values) are injected so the cleaning step has real work to do.
All results produced from this data are synthetic and NOT real-world findings.
"""
import numpy as np
import pandas as pd

COMPANIES = ["VoltGrid", "EcoCharge", "PowerPlug", "GreenWave", "UrbanVolt"]
LOCATIONS = ["Downtown", "Airport", "Shopping Mall", "University Campus",
             "Residential Area", "Highway Rest Stop", "Business Park", "Hospital"]
CHARGER_TYPES = ["AC Level 1", "AC Level 2", "DC Fast"]
POWER_KW = {"AC Level 1": 3.3, "AC Level 2": 7.2, "DC Fast": 50.0}
BASE_DURATION_MIN = {"AC Level 1": 240, "AC Level 2": 120, "DC Fast": 35}
LOCATION_DURATION = {"Downtown": 1.0, "Airport": 1.6, "Shopping Mall": 0.7,
                     "University Campus": 1.2, "Residential Area": 1.8,
                     "Highway Rest Stop": 0.6, "Business Park": 1.1, "Hospital": 1.0}
LOCATION_POPULARITY = {"Downtown": 1.4, "Airport": 1.0, "Shopping Mall": 1.3,
                       "University Campus": 0.9, "Residential Area": 0.8,
                       "Highway Rest Stop": 0.9, "Business Park": 1.1, "Hospital": 0.6}
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"


def _hour_weights(location: str) -> np.ndarray:
    """Hourly arrival profile depending on the kind of location."""
    h = np.arange(24)
    if location in ("Business Park", "Downtown", "Hospital"):
        w = 0.2 + np.exp(-0.5 * ((h - 8.5) / 1.6) ** 2) + 0.8 * np.exp(-0.5 * ((h - 17.5) / 1.8) ** 2)
    elif location in ("Shopping Mall", "Airport", "Highway Rest Stop"):
        w = 0.15 + np.exp(-0.5 * ((h - 13.5) / 3.5) ** 2) + 0.6 * np.exp(-0.5 * ((h - 18.5) / 2.0) ** 2)
    else:  # residential / campus: evening plug-in peak
        w = 0.2 + 0.4 * np.exp(-0.5 * ((h - 9) / 2) ** 2) + np.exp(-0.5 * ((h - 19) / 2.2) ** 2)
    return w / w.sum()


def generate_synthetic_data(n_rows: int = 72000, seed: int = 42) -> pd.DataFrame:
    """Return a raw (dirty) synthetic charging-session DataFrame."""
    rng = np.random.default_rng(seed)

    # Charger fleet -------------------------------------------------------
    n_chargers = 120
    ch_location = rng.choice(LOCATIONS, n_chargers)
    ch_type = np.array([
        rng.choice(CHARGER_TYPES, p=[0.05, 0.35, 0.60] if loc == "Highway Rest Stop"
                   else [0.15, 0.60, 0.25])
        for loc in ch_location])
    ch_company = rng.choice(COMPANIES, n_chargers, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    chargers = pd.DataFrame({
        "ChargerID": [f"CH{i:03d}" for i in range(n_chargers)],
        "ChargerCompany": ch_company, "Location": ch_location, "ChargerType": ch_type})
    popularity = np.array([LOCATION_POPULARITY[l] for l in ch_location]) * rng.lognormal(0, 0.5, n_chargers)
    popularity = popularity / popularity.sum()

    # Users: each has a typical battery size (makes user history informative)
    n_users = 8000
    user_battery = rng.uniform(30, 100, n_users)
    user_w = rng.pareto(2.0, n_users) + 1
    user_w = user_w / user_w.sum()

    # Sessions ------------------------------------------------------------
    sess = chargers.iloc[rng.choice(n_chargers, n_rows, p=popularity)].reset_index(drop=True)
    user_idx = rng.choice(n_users, n_rows, p=user_w)
    sess["UserID"] = [f"U{u:05d}" for u in user_idx]

    days = pd.date_range(START_DATE, END_DATE, freq="D")
    t = np.arange(len(days))
    dow = days.dayofweek.values
    day_w = (1 + 0.5 * t / len(days)) * np.where(dow >= 5, 0.8, 1.0) \
        * (1 + 0.15 * np.sin(2 * np.pi * (days.dayofyear.values - 80) / 365))
    day_idx = rng.choice(len(days), n_rows, p=day_w / day_w.sum())

    hours = np.zeros(n_rows, dtype=int)
    for loc in LOCATIONS:
        m = (sess["Location"] == loc).values
        hours[m] = rng.choice(24, m.sum(), p=_hour_weights(loc))
    minutes = rng.integers(0, 60, n_rows)
    seconds = rng.integers(0, 60, n_rows)
    start = (days[day_idx] + pd.to_timedelta(hours, "h")
             + pd.to_timedelta(minutes, "m") + pd.to_timedelta(seconds, "s"))

    # Duration (minutes) and energy demand (kWh) ----------------------------
    base = sess["ChargerType"].map(BASE_DURATION_MIN).values
    loc_mult = sess["Location"].map(LOCATION_DURATION).values
    night_mult = np.where((hours >= 21) | (hours <= 5), 1.5, 1.0)
    duration = np.clip(base * loc_mult * night_mult * rng.lognormal(0, 0.45, n_rows), 5, 720)
    power = sess["ChargerType"].map(POWER_KW).values
    efficiency = np.where(sess["ChargerType"].values == "DC Fast", 0.6, 0.9)
    battery = user_battery[user_idx] * rng.uniform(0.4, 0.95, n_rows)
    demand = np.minimum(power * efficiency * duration / 60.0, battery)
    demand = np.round(np.clip(demand * rng.normal(1.0, 0.05, n_rows), 0.1, None), 2)
    end = start + pd.to_timedelta(duration, "m")

    df = sess.assign(
        StartDay=start.strftime("%Y-%m-%d"), StartTime=start.strftime("%H:%M:%S"),
        EndDay=end.strftime("%Y-%m-%d"), EndTime=end.strftime("%H:%M:%S"),
        StartDatetime=start.strftime("%Y-%m-%d %H:%M:%S"),
        EndDatetime=end.strftime("%Y-%m-%d %H:%M:%S"),
        Duration=np.round(duration, 2), Demand=demand)
    df = df[["UserID", "ChargerID", "ChargerCompany", "Location", "ChargerType",
             "StartDay", "StartTime", "EndDay", "EndTime", "StartDatetime",
             "EndDatetime", "Duration", "Demand"]]

    return _inject_dirt(df, rng)


def _inject_dirt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject missing values, invalid values, outliers and duplicates."""
    df = df.copy()
    n = len(df)

    def pick(frac):
        return rng.choice(n, int(n * frac), replace=False)

    df.loc[pick(0.010), "Location"] = np.nan
    df.loc[pick(0.010), "ChargerCompany"] = np.nan
    df.loc[pick(0.005), "Demand"] = np.nan
    df.loc[pick(0.005), "Duration"] = np.nan
    df.loc[pick(0.003), "StartDatetime"] = np.nan
    df.loc[pick(0.001), "Demand"] = -1.0                      # invalid
    out_idx = pick(0.003)
    df.loc[out_idx, "Demand"] = df.loc[out_idx, "Demand"] * 10  # extreme outliers
    dup = df.sample(frac=0.004, random_state=int(rng.integers(0, 10**6)))
    return pd.concat([df, dup], ignore_index=True).sample(
        frac=1.0, random_state=int(rng.integers(0, 10**6))).reset_index(drop=True)


if __name__ == "__main__":
    from src.config import RAW_FILE
    data = generate_synthetic_data()
    data.to_csv(RAW_FILE, index=False)
    print(f"Wrote {len(data):,} rows to {RAW_FILE}")
