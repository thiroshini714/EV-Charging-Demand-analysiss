# EV Charging Demand Prediction and Usage Pattern Analysis

End-to-end machine-learning project (GUVI x HCL capstone brief): clean EV charging
session data, explore usage patterns, engineer time/history features, predict charging
**Demand** (regression), classify **High vs Normal Demand** sessions (classification,
class-imbalance handling), explain models (feature importance, SHAP, LIME), forecast
daily demand, and present everything in an interactive **Streamlit** dashboard.

> **Data note.** The brief links a public "Charging Record" CSV (~72k sessions), but that
> file was not available when this project was built. The repo therefore generates a
> **synthetic dataset with the exact same schema**
> (`UserID, ChargerID, ChargerCompany, Location, ChargerType, StartDay, StartTime, EndDay,
> EndTime, StartDatetime, EndDatetime, Duration, Demand`) including deliberate dirt
> (missing values, duplicates, invalid values, outliers). **All numbers and "insights" from
> synthetic data are illustrative, not real-world findings.** To use the real data, save it
> as `data/raw/charging_data.csv` and re-run the pipeline - no code changes needed
> (if its column formats differ, adjust `src/data_cleaning.py`).

## Project structure

```
data/raw/          raw CSV (real file, or generated synthetic file)
data/processed/    cleaned data + engineered feature table
notebooks/         01_end_to_end_walkthrough.ipynb
src/               config, data_generation, data_cleaning, feature_engineering,
                   preprocessing, eda, regression, classification, explainability,
                   forecasting, pipeline (orchestrator)
models/            best_regressor.joblib, best_classifier.joblib, demand_threshold.json
reports/           metric tables (CSV/JSON), EDA + SHAP figures
dashboard/app.py   Streamlit app
tests/             pytest unit tests
docs/              project documentation
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No credentials, API keys or database are used (nothing sensitive is hard-coded).

## Run

```bash
python -m src.pipeline            # full run (~1-3 min): data -> clean -> EDA -> features -> models -> SHAP/LIME -> forecast
python -m src.pipeline --fast     # quicker smoke run with smaller models
python -m pytest -q tests         # unit tests
streamlit run dashboard/app.py    # open http://localhost:8501
```

## Method summary

| Step | What is done |
|------|--------------|
| Cleaning | drop duplicates, rebuild datetimes from Day+Time, repair Duration, impute Location/Company from the charger's mode, drop invalid/missing Demand, remove extreme per-charger-type outliers (3xIQR) |
| Features | hour, day, month, day of week, week of year, weekend flag, time of day, cyclical hour; **history features** (charger/user previous session count and mean demand, location-hour mean demand, charger sessions in prior 24 h) computed from *earlier sessions only* |
| Leakage control | chronological 80/20 split (train = earlier sessions); preprocessing fitted inside sklearn `Pipeline` on train only; class threshold (median Demand) taken from **train** only; `Duration`/`End*` excluded from model inputs because they are only known after the session ends |
| Regression | Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM - MAE, MSE, RMSE, R2 |
| Classification | High = Demand >= median, Normal = below. Logistic Regression, Decision Tree, Random Forest, KNN, SVM (RBF, trained on an 8k subsample for speed, probability-calibrated), XGBoost, LightGBM - Accuracy, Precision, Recall, F1, ROC-AUC |
| Imbalance | class ratio reported; Random Oversampling / SMOTE / ADASYN compared on training data only; applied to the final models **only if** minority/majority < 0.8. A median split is ~50/50, so resampling is normally *not* required (see `reports/imbalance_comparison.csv`) |
| Explainability | permutation feature importance (always), SHAP summary plots, LIME for single predictions |
| Forecasting | daily total kWh: seasonal-naive baseline, Holt-Winters (statsmodels), lag-feature gradient boosting; 60-day backtest + 30-day forecast |

## Dashboard tabs

Overview KPIs - Demand analysis (location / charger type / company / utilisation) -
Time trends - Predict demand (regression value + High-demand probability) - Model performance
(+ imbalance check) - Feature importance - Explainability (SHAP, LIME) - Forecast.
Sidebar filters: location, charger type, company, date range.

## Known limitations

- Synthetic data (see note above); results will differ on the real CSV.
- Because `Duration` is excluded (leakage), regression R2 is moderate by design.
- RBF-SVM uses an 8,000-row stratified subsample of the training set.
- If XGBoost / LightGBM / imbalanced-learn / SHAP / LIME / statsmodels are not installed the
  pipeline still runs, skips those parts, and records what was skipped in
  `reports/run_summary.json` and the dashboard.
- Version control: initialise Git yourself (`git init`, commit regularly) - this repo has no history.
