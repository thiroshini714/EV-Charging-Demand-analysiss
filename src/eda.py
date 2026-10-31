"""Exploratory data analysis: figures saved to reports/figures and summary tables."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import FIGURES_DIR, TARGET  # noqa: E402

sns.set_theme(style="whitegrid")
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, name: str) -> str:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def summary_tables(df: pd.DataFrame) -> dict:
    """Aggregate tables used by both the EDA report and the dashboard."""
    agg = {"sessions": ("Demand", "size"), "total_kwh": ("Demand", "sum"),
           "avg_kwh": ("Demand", "mean")}
    return {
        "by_location": df.groupby("Location").agg(**agg).sort_values("avg_kwh", ascending=False),
        "by_charger_type": df.groupby("ChargerType").agg(**agg),
        "by_company": df.groupby("ChargerCompany").agg(**agg).sort_values("avg_kwh", ascending=False),
        "by_hour": df.groupby("hour").agg(**agg),
        "by_day_of_week": df.groupby("day_of_week").agg(**agg),
        "by_month": df.groupby("month").agg(**agg),
        "by_charger": df.groupby("ChargerID").agg(**agg).sort_values("sessions"),
    }


def run_eda(df: pd.DataFrame) -> dict:
    """Create the EDA figures and return {name: path} plus key findings."""
    figs = {}
    t = summary_tables(df)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], bins=50, kde=True, ax=ax[0])
    ax[0].set(title="Demand distribution (kWh)", xlabel="Demand (kWh)")
    sns.boxplot(data=df, x="ChargerType", y=TARGET, ax=ax[1])
    ax[1].set(title="Demand by charger type (outliers shown)")
    figs["demand_distribution"] = _save(fig, "01_demand_distribution.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    t["by_location"]["avg_kwh"].plot.barh(ax=ax[0], title="Avg demand by location")
    t["by_charger_type"]["avg_kwh"].plot.bar(ax=ax[1], title="Avg demand by charger type", rot=0)
    t["by_company"]["avg_kwh"].plot.bar(ax=ax[2], title="Avg demand by company", rot=30)
    figs["demand_by_category"] = _save(fig, "02_demand_by_category.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    t["by_hour"]["sessions"].plot(marker="o", ax=ax[0], title="Sessions by hour")
    t["by_day_of_week"]["sessions"].set_axis(DAY_NAMES).plot.bar(ax=ax[1], title="Sessions by day of week", rot=0)
    t["by_month"]["sessions"].plot.bar(ax=ax[2], title="Sessions by month", rot=0)
    figs["time_patterns"] = _save(fig, "03_time_patterns.png")

    heat = df.pivot_table(index="day_of_week", columns="hour", values=TARGET, aggfunc="count")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(heat.set_axis(DAY_NAMES, axis=0), cmap="YlGnBu", ax=ax)
    ax.set(title="Session count: day of week x hour", ylabel="")
    figs["heatmap_dow_hour"] = _save(fig, "04_heatmap_dow_hour.png")

    daily = df.set_index("StartDatetime")[TARGET].resample("D").sum()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    daily.plot(ax=ax, alpha=0.5, label="daily kWh")
    daily.rolling(30, min_periods=7).mean().plot(ax=ax, label="30-day mean")
    ax.set(title="Total daily charging demand (kWh)", xlabel="")
    ax.legend()
    figs["daily_trend"] = _save(fig, "05_daily_trend.png")

    num = df[["Demand", "Duration", "hour", "day_of_week", "is_weekend",
              "charger_prev_avg_demand", "user_prev_avg_demand",
              "charger_sessions_prev_24h", "location_hour_prev_avg_demand"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Correlation matrix")
    figs["correlation"] = _save(fig, "06_correlation.png")

    peak_hour = int(t["by_hour"]["sessions"].idxmax())
    findings = {
        "peak_hour": peak_hour,
        "busiest_location": t["by_location"]["sessions"].idxmax(),
        "highest_avg_demand_location": t["by_location"]["avg_kwh"].idxmax(),
        "highest_avg_demand_charger_type": t["by_charger_type"]["avg_kwh"].idxmax(),
        "least_used_chargers": t["by_charger"].head(5).index.tolist(),
        "weekend_vs_weekday_avg_kwh": {("Weekend" if k else "Weekday"): v for k, v in
                                       df.groupby("is_weekend")[TARGET].mean().round(2).items()},
    }
    return {"figures": figs, "findings": findings}
