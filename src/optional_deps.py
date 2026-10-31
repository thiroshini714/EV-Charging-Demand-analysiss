"""Graceful handling of optional libraries (XGBoost, LightGBM, imbalanced-learn, SHAP, LIME)."""
import importlib


def available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


HAS_XGB = available("xgboost")
HAS_LGBM = available("lightgbm")
HAS_IMBLEARN = available("imblearn")
HAS_SHAP = available("shap")
HAS_LIME = available("lime")
HAS_STATSMODELS = available("statsmodels")
