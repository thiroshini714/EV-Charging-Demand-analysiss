"""Regression: predict charging Demand (kWh) per session."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import optional_deps as deps
from src.config import FEATURES, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from src.preprocessing import build_preprocessor


def get_regression_models(fast: bool = False):
    """Return ({name: estimator}, [skipped model notes])."""
    n = 40 if fast else 120
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=20,
                                               random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=n, max_depth=14, min_samples_leaf=5,
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=n, max_depth=4, subsample=0.8,
                                                       random_state=RANDOM_STATE),
    }
    skipped = []
    if deps.HAS_XGB:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=n * 3, max_depth=6, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                         random_state=RANDOM_STATE)
    else:
        skipped.append("XGBoost (package not installed)")
    if deps.HAS_LGBM:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(n_estimators=n * 3, learning_rate=0.05, num_leaves=63,
                                           subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                           random_state=RANDOM_STATE, verbose=-1)
    else:
        skipped.append("LightGBM (package not installed)")
    return models, skipped


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def train_regression(train: pd.DataFrame, test: pd.DataFrame, fast: bool = False) -> dict:
    """Train/compare all regressors; persist best pipeline and reports."""
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]
    models, skipped = get_regression_models(fast)

    rows, fitted = [], {}
    for name, est in models.items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        m = regression_metrics(y_te, pred)
        m.update(Model=name, TrainSeconds=round(time.time() - t0, 1))
        rows.append(m)
        fitted[name] = pipe
        print(f"  [regression] {name:18s} RMSE={m['RMSE']:.3f} R2={m['R2']:.3f}")

    results = pd.DataFrame(rows)[["Model", "MAE", "MSE", "RMSE", "R2", "TrainSeconds"]]
    results = results.sort_values("RMSE").reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best = fitted[best_name]

    # Permutation importance on ORIGINAL features (model-agnostic)
    sample = test.sample(min(3000, len(test)), random_state=RANDOM_STATE)
    perm = permutation_importance(best, sample[FEATURES], sample[TARGET], n_repeats=5,
                                  random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error")
    importance = pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean,
                               "std": perm.importances_std}).sort_values("importance", ascending=False)

    preds = test[["StartDatetime", "ChargerID", "Location", "ChargerType", TARGET]].copy()
    preds["Predicted"] = best.predict(X_te)

    results.to_csv(REPORTS_DIR / "regression_metrics.csv", index=False)
    importance.to_csv(REPORTS_DIR / "regression_feature_importance.csv", index=False)
    preds.to_csv(REPORTS_DIR / "regression_predictions.csv", index=False)
    joblib.dump(best, MODELS_DIR / "best_regressor.joblib")
    (REPORTS_DIR / "regression_best.json").write_text(json.dumps(
        {"best_model": best_name, "skipped": skipped}, indent=2))
    return {"results": results, "best_name": best_name, "best_pipeline": best,
            "importance": importance, "skipped": skipped}
