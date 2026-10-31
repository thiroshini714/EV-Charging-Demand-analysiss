# Project Documentation

**All results below come from SYNTHETIC data (71,176 cleaned sessions) generated with the
PDF's schema. Re-run `python -m src.pipeline` on the real CSV for real findings.**
Run environment: sklearn models only - XGBoost, LightGBM, SMOTE/ADASYN, SHAP, LIME,
Holt-Winters and Streamlit could not be installed in the build sandbox (see README).

## 1. Data preprocessing
Raw 72,288 rows -> 288 duplicates removed; 216 StartDatetimes rebuilt from StartDay+StartTime;
360 Durations repaired from End-Start; 720 Location and 720 ChargerCompany values imputed from
the charger's most common value; 432 rows dropped (missing/invalid Demand); 392 extreme Demand
outliers removed (3xIQR per charger type) -> 71,176 rows.

## 2. EDA (figures in `reports/figures/`)
Demand distribution and boxplots, demand by location / charger type / company, sessions by
hour / weekday / month, weekday-hour heatmap, daily trend, correlation matrix.
Synthetic-data patterns: peak hour 18:00, busiest location Business Park, highest average
demand at Residential Area and Airport (long stays), DC Fast highest by type, no weekday/weekend
difference in average kWh (16.80 vs 16.73).

## 3. Feature engineering
See README. History features use only earlier sessions (unit-tested for leakage).

## 4. Regression (test = latest 20 % of sessions)
| Model | MAE | MSE | RMSE | R2 |
|---|---|---|---|---|
| Linear Regression | 5.538 | 53.088 | 7.286 | 0.373 |
| Random Forest | 5.595 | 53.720 | 7.329 | 0.365 |
| Gradient Boosting | 5.560 | 53.723 | 7.330 | 0.365 |
| Decision Tree | 5.611 | 55.219 | 7.431 | 0.347 |

XGBoost and LightGBM: **not run** (packages unavailable in the build environment).
Top drivers (permutation importance): location-hour historical demand, Location, ChargerType.
Models are close because the synthetic signal is mostly linear in the encoded features.

## 5. Classification (High = Demand >= 14.77 kWh, the training median)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.720 | 0.754 | 0.662 | 0.705 | 0.799 |
| Logistic Regression | 0.720 | 0.756 | 0.658 | 0.703 | 0.799 |
| Decision Tree | 0.711 | 0.737 | 0.666 | 0.700 | 0.790 |
| KNN | 0.708 | 0.734 | 0.661 | 0.696 | 0.781 |
| SVM (8k subsample) | 0.716 | 0.747 | 0.661 | 0.702 | 0.770 |

Class balance: 28,465 normal vs 28,475 high in train (ratio 0.9996) -> balanced, so resampling
is not required. Random Oversampling made no meaningful difference (F1 0.7035 -> 0.7020).
SMOTE and ADASYN: **not run** (imbalanced-learn unavailable).

## 6. Explainability
Permutation importance: run. SHAP and LIME: code written, **not executed** here.

## 7. Time series
Daily total kWh, 60-day backtest: lag-feature gradient boosting RMSE 229 kWh (R2 0.01),
seasonal naive RMSE 350. Forecast skill is weak on this synthetic series (strong trend plus
Poisson-like noise); treat the forecast as a demonstration of the method.
