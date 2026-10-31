"""Central configuration: paths, random seed and feature definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Put the real "Charging Record" CSV here to use it instead of synthetic data.
RAW_FILE = RAW_DIR / "charging_data.csv"
CLEAN_FILE = PROCESSED_DIR / "charging_clean.csv"
FEATURE_FILE = PROCESSED_DIR / "charging_features.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2                 # last 20 % of sessions (chronological split)
HIGH_DEMAND_QUANTILE = 0.5      # PDF: High Demand = Demand >= median Demand
IMBALANCE_RATIO_THRESHOLD = 0.8  # minority/majority below this => imbalanced

# Duration, EndDatetime etc. are only known AFTER a session ends, so they are
# excluded from model inputs to avoid target leakage (Demand ~ power x time).
CATEGORICAL_FEATURES = ["ChargerCompany", "Location", "ChargerType", "time_of_day"]
NUMERIC_FEATURES = [
    "hour", "day", "month", "day_of_week", "week_of_year", "is_weekend",
    "hour_sin", "hour_cos",
    "charger_prev_sessions", "charger_prev_avg_demand", "charger_sessions_prev_24h",
    "user_prev_sessions", "user_prev_avg_demand",
    "location_hour_prev_avg_demand",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Demand"
CLASS_TARGET = "HighDemand"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
