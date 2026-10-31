"""Data understanding and cleaning."""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["UserID", "ChargerID", "ChargerCompany", "Location",
                    "ChargerType", "StartDatetime", "Duration", "Demand"]


def describe_data(df: pd.DataFrame) -> dict:
    """Basic data-understanding summary (shape, dtypes, missing, duplicates)."""
    return {
        "rows": int(len(df)), "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_data(raw: pd.DataFrame, outlier_iqr_factor: float = 3.0):
    """Clean the raw charging records.

    Steps: validate schema -> drop duplicates -> parse datetimes -> repair
    Duration -> impute charger attributes -> drop invalid/missing target ->
    remove extreme per-charger-type Demand outliers -> sort by time.
    Returns (clean_df, report_dict).
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    report = {"rows_raw": int(len(raw))}
    df = raw.copy()

    # 1. duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)

    # 2. datetimes (rebuild from Day + Time columns when the combined one is missing)
    df["StartDatetime"] = pd.to_datetime(df["StartDatetime"], errors="coerce")
    if {"StartDay", "StartTime"}.issubset(df.columns):
        rebuilt = pd.to_datetime(df["StartDay"].astype(str) + " " + df["StartTime"].astype(str),
                                 errors="coerce")
        report["start_datetime_rebuilt"] = int(df["StartDatetime"].isna().sum() - rebuilt.isna().sum()
                                               if rebuilt.notna().sum() else 0)
        df["StartDatetime"] = df["StartDatetime"].fillna(rebuilt)
    if "EndDatetime" in df.columns:
        df["EndDatetime"] = pd.to_datetime(df["EndDatetime"], errors="coerce")
        if {"EndDay", "EndTime"}.issubset(df.columns):
            df["EndDatetime"] = df["EndDatetime"].fillna(pd.to_datetime(
                df["EndDay"].astype(str) + " " + df["EndTime"].astype(str), errors="coerce"))
    before = len(df)
    df = df.dropna(subset=["StartDatetime"])
    report["dropped_bad_start_datetime"] = before - len(df)

    # 3. numeric types + Duration repair (minutes)
    for col in ("Duration", "Demand"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "EndDatetime" in df.columns:
        calc = (df["EndDatetime"] - df["StartDatetime"]).dt.total_seconds() / 60.0
        fix = df["Duration"].isna() & calc.notna() & (calc > 0)
        df.loc[fix, "Duration"] = calc[fix]
        report["duration_repaired"] = int(fix.sum())
        bad_time = df["EndDatetime"].notna() & (df["EndDatetime"] < df["StartDatetime"])
        report["dropped_end_before_start"] = int(bad_time.sum())
        df = df[~bad_time]
    df["Duration"] = df["Duration"].where(df["Duration"] > 0)
    df["Duration"] = df["Duration"].fillna(df["Duration"].median())

    # 4. impute charger attributes from the charger's most common value
    for col in ("ChargerCompany", "Location", "ChargerType"):
        mode = df.groupby("ChargerID")[col].agg(
            lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        report[f"{col}_imputed"] = int(df[col].isna().sum())
        df[col] = df[col].fillna(df["ChargerID"].map(mode)).fillna("Unknown")

    # 5. target must be present and physically valid
    before = len(df)
    df = df[df["Demand"].notna() & (df["Demand"] > 0)]
    report["dropped_invalid_demand"] = before - len(df)

    # 6. extreme outliers per charger type (IQR fences)
    q1 = df.groupby("ChargerType")["Demand"].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby("ChargerType")["Demand"].transform(lambda s: s.quantile(0.75))
    upper = q3 + outlier_iqr_factor * (q3 - q1)
    out = df["Demand"] > upper
    report["outliers_removed"] = int(out.sum())
    df = df[~out]

    df = df.sort_values("StartDatetime").reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    return df, report
