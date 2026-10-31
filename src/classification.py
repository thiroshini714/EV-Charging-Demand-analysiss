"""Classification: High-Demand (Demand >= median) vs Normal-Demand sessions."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from src import optional_deps as deps
from src.config import (FEATURES, IMBALANCE_RATIO_THRESHOLD, MODELS_DIR,
                        RANDOM_STATE, REPORTS_DIR, TARGET)
from src.feature_engineering import demand_threshold, make_class_target
from src.preprocessing import build_preprocessor

SVM_MAX_TRAIN_ROWS = 8000  # RBF-SVM scales ~O(n^2); trained on a stratified subsample


def get_classification_models(fast: bool = False):
    n = 40 if fast else 120
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20,
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=n, max_depth=14, min_samples_leaf=5,
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        "SVM": CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE),
                                      ensemble=False, cv=3),
    }
    skipped = []
    if deps.HAS_XGB:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=n * 3, max_depth=6, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                          eval_metric="logloss", random_state=RANDOM_STATE)
    else:
        skipped.append("XGBoost (package not installed)")
    if deps.HAS_LGBM:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=n * 3, learning_rate=0.05, num_leaves=63,
                                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                            random_state=RANDOM_STATE, verbose=-1)
    else:
        skipped.append("LightGBM (package not installed)")
    return models, skipped


def classification_metrics(y_true, y_pred, y_score) -> dict:
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, y_score)}


def imbalance_report(y: pd.Series) -> dict:
    """Class balance check (PDF: 'Check for class imbalance')."""
    counts = y.value_counts().sort_index()
    ratio = float(counts.min() / counts.max())
    return {"normal_count": int(counts.get(0, 0)), "high_count": int(counts.get(1, 0)),
            "minority_to_majority_ratio": round(ratio, 4),
            "imbalanced": bool(ratio < IMBALANCE_RATIO_THRESHOLD)}


def get_resamplers():
    """Return ({name: resampler-or-None}, [skipped notes]). ROS has a sklearn fallback."""
    skipped, out = [], {"None": None}
    if deps.HAS_IMBLEARN:
        from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
        out["Random Oversampling"] = RandomOverSampler(random_state=RANDOM_STATE)
        out["SMOTE"] = SMOTE(random_state=RANDOM_STATE)
        out["ADASYN"] = ADASYN(random_state=RANDOM_STATE)
    else:
        out["Random Oversampling"] = "manual"
        skipped += ["SMOTE (imbalanced-learn not installed)", "ADASYN (imbalanced-learn not installed)"]
    return out, skipped


def _oversample(X, y, sampler):
    """Resample TRAINING data only."""
    if sampler is None:
        return X, y

    if sampler == "manual":
        y = np.asarray(y)
        minority = int(pd.Series(y).value_counts().idxmin())
        X_min, X_maj = X[y == minority], X[y != minority]
        X_up = resample(
            X_min, replace=True, n_samples=len(X_maj),
            random_state=RANDOM_STATE
        )
        return (
            np.vstack([X_maj, X_up]),
            np.concatenate([
                np.full(len(X_maj), 1 - minority),
                np.full(len(X_up), minority)
            ])
        )

    try:
        return sampler.fit_resample(X, y)
    except ValueError as e:
        if "No samples will be generated" in str(e):
            return X, y
        raise

def compare_imbalance_methods(train, test, y_tr, y_te) -> pd.DataFrame:
    """Compare no-resampling vs ROS / SMOTE / ADASYN with one reference model."""
    prep = build_preprocessor().fit(train[FEATURES])
    Xt, Xe = prep.transform(train[FEATURES]), prep.transform(test[FEATURES])
    samplers, _ = get_resamplers()
    rows = []
    for name, sampler in samplers.items():
        Xr, yr = _oversample(Xt, y_tr.values, sampler)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(Xr, yr)
        m = classification_metrics(y_te, clf.predict(Xe), clf.predict_proba(Xe)[:, 1])
        m.update(Strategy=name, TrainRows=len(yr))
        rows.append(m)
    return pd.DataFrame(rows)[["Strategy", "TrainRows", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]


def train_classification(train: pd.DataFrame, test: pd.DataFrame, fast: bool = False) -> dict:
    """Train/compare classifiers; persist best pipeline, threshold and reports."""
    threshold = demand_threshold(train[TARGET])        # from TRAIN only -> no leakage
    y_tr = make_class_target(train[TARGET], threshold)
    y_te = make_class_target(test[TARGET], threshold)
    imb = imbalance_report(y_tr)
    imb["threshold_kwh"] = round(threshold, 3)
    print(f"  [classification] median threshold={threshold:.2f} kWh, balance={imb}")

    imb_table = compare_imbalance_methods(train, test, y_tr, y_te)
    _, imb_skipped = get_resamplers()

    # Use resampling only when the data is actually imbalanced (PDF: "when required")
    strategy = "None"
    if imb["imbalanced"]:
        strategy = imb_table.sort_values("F1", ascending=False).iloc[0]["Strategy"]
    imb["strategy_used"] = strategy

    models, skipped = get_classification_models(fast)
    rows, fitted, cms = [], {}, {}
    for name, est in models.items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        X_fit, y_fit = train[FEATURES], y_tr
        if name == "SVM" and len(train) > SVM_MAX_TRAIN_ROWS:
            sub = train.sample(SVM_MAX_TRAIN_ROWS, random_state=RANDOM_STATE)
            X_fit, y_fit = sub[FEATURES], y_tr.loc[sub.index]
        if strategy == "None":
            pipe.fit(X_fit, y_fit)
        else:                                            # resample after preprocessing, train only
            prep = pipe.named_steps["prep"].fit(X_fit)
            Xr, yr = _oversample(prep.transform(X_fit), y_fit.values, get_resamplers()[0][strategy])
            est.fit(Xr, yr)
        proba = pipe.predict_proba(test[FEATURES])[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = classification_metrics(y_te, pred, proba)
        m.update(Model=name, TrainSeconds=round(time.time() - t0, 1))
        rows.append(m)
        fitted[name] = pipe
        cms[name] = confusion_matrix(y_te, pred).tolist()
        print(f"  [classification] {name:20s} AUC={m['ROC_AUC']:.3f} F1={m['F1']:.3f}")

    results = pd.DataFrame(rows)[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "TrainSeconds"]]
    results = results.sort_values(["ROC_AUC", "F1"], ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best = fitted[best_name]

    sample = test.sample(min(3000, len(test)), random_state=RANDOM_STATE)
    perm = permutation_importance(best, sample[FEATURES], y_te.loc[sample.index], n_repeats=5,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    importance = pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean,
                               "std": perm.importances_std}).sort_values("importance", ascending=False)

    results.to_csv(REPORTS_DIR / "classification_metrics.csv", index=False)
    imb_table.to_csv(REPORTS_DIR / "imbalance_comparison.csv", index=False)
    importance.to_csv(REPORTS_DIR / "classification_feature_importance.csv", index=False)
    joblib.dump(best, MODELS_DIR / "best_classifier.joblib")
    (MODELS_DIR / "demand_threshold.json").write_text(json.dumps({"threshold_kwh": threshold}))
    (REPORTS_DIR / "classification_best.json").write_text(json.dumps(
        {"best_model": best_name, "imbalance": imb, "confusion_matrices": cms,
         "skipped": skipped + imb_skipped}, indent=2))
    return {"results": results, "best_name": best_name, "best_pipeline": best,
            "importance": importance, "imbalance": imb, "imbalance_table": imb_table,
            "threshold": threshold, "y_test": y_te, "skipped": skipped + imb_skipped}
