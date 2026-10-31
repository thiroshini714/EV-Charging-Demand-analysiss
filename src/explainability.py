"""Model explainability: feature importance, SHAP and LIME.

SHAP and LIME are optional imports; functions return None / raise a clear
RuntimeError when the library is missing so callers can show a friendly note.
"""
import numpy as np
import pandas as pd

from src import optional_deps as deps
from src.config import FEATURES, FIGURES_DIR, RANDOM_STATE, REPORTS_DIR


def _split_pipeline(pipeline):
    return pipeline.named_steps["prep"], pipeline.named_steps["model"]


def transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessor and return a DataFrame with readable column names."""
    prep, _ = _split_pipeline(pipeline)
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    return pd.DataFrame(prep.transform(X), columns=names, index=X.index)


def _positive_class_values(values):
    """SHAP returns list / 3-D arrays for classifiers depending on version -> 2-D (positive class)."""
    if isinstance(values, list):
        return np.asarray(values[-1])
    values = np.asarray(values)
    return values[:, :, -1] if values.ndim == 3 else values


def compute_shap(pipeline, X: pd.DataFrame, max_rows: int = 500, task: str = "regression"):
    """Return (shap_values 2-D array, transformed sample DataFrame) for the model."""
    if not deps.HAS_SHAP:
        raise RuntimeError("SHAP is not installed (pip install shap).")
    import shap
    _, model = _split_pipeline(pipeline)
    sample = X.sample(min(max_rows, len(X)), random_state=RANDOM_STATE)
    Xt = transformed_frame(pipeline, sample)
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, Xt)
        values = explainer.shap_values(Xt)
    elif hasattr(model, "feature_importances_") or model.__class__.__name__.startswith(("XGB", "LGBM")):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xt)
    else:
        background = shap.sample(Xt, 50, random_state=RANDOM_STATE)
        fn = model.predict_proba if task == "classification" else model.predict
        values = shap.KernelExplainer(fn, background).shap_values(Xt.iloc[:100])
        Xt = Xt.iloc[:100]
    return _positive_class_values(values), Xt


def shap_importance_table(shap_values: np.ndarray, Xt: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per transformed feature, descending."""
    return (pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def save_shap_summary_plot(shap_values, Xt: pd.DataFrame, name: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    shap.summary_plot(shap_values, Xt, show=False, max_display=15)
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return str(path)


def lime_explain(pipeline, X_train: pd.DataFrame, instance: pd.DataFrame,
                 task: str = "regression", num_features: int = 10):
    """LIME explanation for one row -> list of (feature condition, weight)."""
    if not deps.HAS_LIME:
        raise RuntimeError("LIME is not installed (pip install lime).")
    from lime.lime_tabular import LimeTabularExplainer
    _, model = _split_pipeline(pipeline)
    bg = transformed_frame(pipeline, X_train.sample(min(2000, len(X_train)), random_state=RANDOM_STATE))
    inst = transformed_frame(pipeline, instance)
    explainer = LimeTabularExplainer(
        bg.values, feature_names=list(bg.columns), mode=task,
        class_names=["Normal Demand", "High Demand"] if task == "classification" else None,
        random_state=RANDOM_STATE)
    predict_fn = model.predict_proba if task == "classification" else model.predict
    exp = explainer.explain_instance(inst.values[0], predict_fn, num_features=num_features)
    return exp.as_list()


def run_explainability(reg_pipeline, clf_pipeline, test: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Generate SHAP plots/tables and one LIME example per task; return status notes."""
    status = {"shap": "skipped (not installed)", "lime": "skipped (not installed)"}
    if deps.HAS_SHAP:
        for task, pipe, tag in (("regression", reg_pipeline, "regression"),
                                ("classification", clf_pipeline, "classification")):
            try:
                vals, Xt = compute_shap(pipe, test[FEATURES], task=task)
                shap_importance_table(vals, Xt).to_csv(REPORTS_DIR / f"shap_{tag}_importance.csv", index=False)
                save_shap_summary_plot(vals, Xt, f"shap_{tag}_summary.png")
            except Exception as exc:  # keep pipeline running; surface the reason
                status["shap"] = f"failed for {tag}: {exc}"
                break
        else:
            status["shap"] = "ok"
    if deps.HAS_LIME:
        try:
            row = test[FEATURES].iloc[[0]]
            for task, pipe in (("regression", reg_pipeline), ("classification", clf_pipeline)):
                pd.DataFrame(lime_explain(pipe, train[FEATURES], row, task),
                             columns=["condition", "weight"]).to_csv(
                    REPORTS_DIR / f"lime_{task}_example.csv", index=False)
            status["lime"] = "ok"
        except Exception as exc:
            status["lime"] = f"failed: {exc}"
    return status
