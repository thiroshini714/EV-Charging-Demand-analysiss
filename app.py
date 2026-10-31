"""Streamlit dashboard: EV Charging Demand Prediction and Usage Pattern Analysis.

Run:  streamlit run dashboard/app.py   (after:  python -m src.pipeline)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import optional_deps as deps  # noqa: E402
from src.config import (FEATURE_FILE, FEATURES, MODELS_DIR, REPORTS_DIR,  # noqa: E402
                        TARGET)
from src.feature_engineering import make_inference_frame  # noqa: E402

st.set_page_config(page_title="EV Charging Demand", page_icon="⚡", layout="wide")
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ----------------------------------------------------------------- loaders
@st.cache_data(show_spinner="Loading data ...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_FILE, parse_dates=["StartDatetime"])
    return df


@st.cache_resource(show_spinner="Loading models ...")
def load_model(name: str):
    path = MODELS_DIR / name
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_report(name: str):
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ------------------------------------------------------------------- setup
if not FEATURE_FILE.exists():
    st.error("Processed data not found. Run `python -m src.pipeline` first, then reload this page.")
    st.stop()

data = load_data()
reg_model = load_model("best_regressor.joblib")
clf_model = load_model("best_classifier.joblib")
threshold = load_json(MODELS_DIR / "demand_threshold.json").get("threshold_kwh")
run_summary = load_json(REPORTS_DIR / "run_summary.json")

st.title("⚡ EV Charging Demand Prediction & Usage Pattern Analysis")
if run_summary.get("synthetic_data"):
    st.warning("The data in this dashboard is SYNTHETIC (stand-in for the 'Charging Record' CSV). "
               "Insights are illustrative only. Put the real CSV in data/raw/charging_data.csv and "
               "re-run the pipeline to use real data.")

# ----------------------------------------------------------------- filters
st.sidebar.header("Filters")
locations = st.sidebar.multiselect("Location", sorted(data["Location"].unique()))
types = st.sidebar.multiselect("Charger type", sorted(data["ChargerType"].unique()))
companies = st.sidebar.multiselect("Charger company", sorted(data["ChargerCompany"].unique()))
d_min, d_max = data["StartDatetime"].dt.date.min(), data["StartDatetime"].dt.date.max()
date_range = st.sidebar.date_input("Date range", (d_min, d_max), min_value=d_min, max_value=d_max)

view = data
if locations:
    view = view[view["Location"].isin(locations)]
if types:
    view = view[view["ChargerType"].isin(types)]
if companies:
    view = view[view["ChargerCompany"].isin(companies)]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    view = view[(view["StartDatetime"].dt.date >= date_range[0])
                & (view["StartDatetime"].dt.date <= date_range[1])]
st.sidebar.caption(f"{len(view):,} of {len(data):,} sessions selected")

tabs = st.tabs(["Overview", "Demand analysis", "Time trends", "Predict demand",
                "Model performance", "Feature importance", "Explainability", "Forecast"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    if view.empty:
        st.info("No sessions match the current filters. Widen the filters in the sidebar.")
    else:
        c = st.columns(5)
        c[0].metric("Sessions", f"{len(view):,}")
        c[1].metric("Total energy (kWh)", f"{view[TARGET].sum():,.0f}")
        c[2].metric("Avg demand / session (kWh)", f"{view[TARGET].mean():.1f}")
        c[3].metric("Avg duration (min)", f"{view['Duration'].mean():.0f}")
        c[4].metric("Active chargers", f"{view['ChargerID'].nunique()}")
        c = st.columns(3)
        busiest = view["Location"].value_counts()
        c[0].metric("Busiest location", busiest.index[0], f"{busiest.iloc[0]:,} sessions")
        c[1].metric("Peak hour", f"{int(view['hour'].value_counts().idxmax()):02d}:00")
        c[2].metric("Unique users", f"{view['UserID'].nunique():,}")
        st.plotly_chart(px.histogram(view, x=TARGET, nbins=50, title="Demand distribution (kWh)"),
                        use_container_width=True)

# ---------------------------------------------------------- demand analysis
with tabs[1]:
    if view.empty:
        st.info("No data for the selected filters.")
    else:
        left, right = st.columns(2)
        by_loc = view.groupby("Location").agg(sessions=(TARGET, "size"), avg_kwh=(TARGET, "mean"),
                                              total_kwh=(TARGET, "sum")).reset_index()
        left.plotly_chart(px.bar(by_loc.sort_values("avg_kwh"), x="avg_kwh", y="Location",
                                 orientation="h", title="Average demand by location (kWh)"),
                          use_container_width=True)
        by_type = view.groupby("ChargerType").agg(sessions=(TARGET, "size"),
                                                  avg_kwh=(TARGET, "mean")).reset_index()
        right.plotly_chart(px.bar(by_type, x="ChargerType", y="avg_kwh",
                                  title="Average demand by charger type (kWh)"),
                           use_container_width=True)
        by_co = view.groupby("ChargerCompany").agg(sessions=(TARGET, "size"),
                                                   avg_kwh=(TARGET, "mean")).reset_index()
        left.plotly_chart(px.bar(by_co, x="ChargerCompany", y="sessions",
                                 title="Sessions by charger company"), use_container_width=True)
        grp = view.groupby(["Location", "ChargerType"])[TARGET].mean().reset_index()
        right.plotly_chart(px.density_heatmap(grp, x="ChargerType", y="Location", z=TARGET,
                                              histfunc="avg", title="Avg demand: location x charger type"),
                           use_container_width=True)
        st.subheader("Charger utilisation (least-used chargers)")
        util = view.groupby(["ChargerID", "Location", "ChargerType", "ChargerCompany"]).agg(
            sessions=(TARGET, "size"), total_kwh=(TARGET, "sum")).reset_index().sort_values("sessions")
        st.dataframe(util.head(15), use_container_width=True, hide_index=True)

# ------------------------------------------------------------- time trends
with tabs[2]:
    if view.empty:
        st.info("No data for the selected filters.")
    else:
        daily = view.set_index("StartDatetime")[TARGET].resample("D").sum().reset_index()
        st.plotly_chart(px.line(daily, x="StartDatetime", y=TARGET, title="Total daily demand (kWh)"),
                        use_container_width=True)
        left, right = st.columns(2)
        hourly = view.groupby("hour").agg(sessions=(TARGET, "size"), avg_kwh=(TARGET, "mean")).reset_index()
        left.plotly_chart(px.bar(hourly, x="hour", y="sessions", title="Sessions by hour of day"),
                          use_container_width=True)
        heat = view.pivot_table(index="day_of_week", columns="hour", values=TARGET, aggfunc="count")
        heat.index = [DAY_NAMES[i] for i in heat.index]
        right.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlGnBu",
                                     title="Sessions: day of week x hour"), use_container_width=True)
        monthly = view.groupby("month").agg(sessions=(TARGET, "size")).reset_index()
        st.plotly_chart(px.bar(monthly, x="month", y="sessions", title="Sessions by month"),
                        use_container_width=True)

# ------------------------------------------------------------- predictions
with tabs[3]:
    st.subheader("Predict charging demand for a new session")
    if reg_model is None or clf_model is None:
        st.error("Trained models not found. Run `python -m src.pipeline` first.")
    else:
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            p_loc = c1.selectbox("Location", sorted(data["Location"].unique()))
            p_type = c2.selectbox("Charger type", sorted(data["ChargerType"].unique()))
            p_co = c3.selectbox("Charger company", sorted(data["ChargerCompany"].unique()))
            c1, c2 = st.columns(2)
            p_date = c1.date_input("Start date", d_max)
            p_time = c2.time_input("Start time", pd.Timestamp("18:00").time())
            st.caption("Optional usage history (leave at 0 for an unknown / new user or charger).")
            c1, c2, c3 = st.columns(3)
            h_cs = c1.number_input("Charger: previous sessions", 0, 100000, 0)
            h_ca = c2.number_input("Charger: avg previous demand (kWh)", 0.0, 200.0, 0.0)
            h_24 = c3.number_input("Charger: sessions in last 24 h", 0, 200, 0)
            c1, c2 = st.columns(2)
            h_us = c1.number_input("User: previous sessions", 0, 100000, 0)
            h_ua = c2.number_input("User: avg previous demand (kWh)", 0.0, 200.0, 0.0)
            submitted = st.form_submit_button("Predict")
        if submitted:
            nan = float("nan")
            row = make_inference_frame({
                "ChargerCompany": p_co, "Location": p_loc, "ChargerType": p_type,
                "StartDatetime": pd.Timestamp.combine(p_date, p_time),
                "charger_prev_sessions": h_cs, "charger_prev_avg_demand": h_ca if h_cs else nan,
                "charger_sessions_prev_24h": h_24,
                "user_prev_sessions": h_us, "user_prev_avg_demand": h_ua if h_us else nan}, data)
            kwh = float(reg_model.predict(row)[0])
            p_high = float(clf_model.predict_proba(row)[0, 1])
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted demand", f"{max(kwh, 0):.1f} kWh")
            c2.metric("High-demand probability", f"{p_high:.0%}")
            c3.metric("Category", "High Demand" if p_high >= 0.5 else "Normal Demand")
            if threshold is not None:
                st.caption(f"High Demand means Demand >= {threshold:.1f} kWh (median of training sessions).")
            st.session_state["last_row"] = row

# --------------------------------------------------------- model comparison
with tabs[4]:
    reg_m, clf_m = load_report("regression_metrics.csv"), load_report("classification_metrics.csv")
    st.subheader("Regression models (lower RMSE is better)")
    if reg_m is not None:
        st.dataframe(reg_m.style.format({c: "{:.3f}" for c in ["MAE", "MSE", "RMSE", "R2"]}),
                     use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(reg_m, x="Model", y="RMSE", title="RMSE by model"),
                        use_container_width=True)
    st.subheader("Classification models (High vs Normal demand)")
    if clf_m is not None:
        st.dataframe(clf_m.style.format({c: "{:.3f}" for c in
                                         ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]}),
                     use_container_width=True, hide_index=True)
        long = clf_m.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
                          var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                               title="Classification metrics"), use_container_width=True)
    imb = load_json(REPORTS_DIR / "classification_best.json").get("imbalance")
    imb_t = load_report("imbalance_comparison.csv")
    st.subheader("Class imbalance check")
    if imb:
        st.write(f"Normal: **{imb['normal_count']:,}** | High: **{imb['high_count']:,}** | "
                 f"minority/majority ratio: **{imb['minority_to_majority_ratio']}** -> "
                 f"{'imbalanced' if imb['imbalanced'] else 'balanced (median threshold)'}; "
                 f"resampling used: **{imb['strategy_used']}**")
    if imb_t is not None:
        st.dataframe(imb_t, use_container_width=True, hide_index=True)
    skipped = run_summary.get("skipped", [])
    if skipped:
        st.info("Not run in the last pipeline execution: " + "; ".join(skipped))

# ------------------------------------------------------ feature importance
with tabs[5]:
    for title, file in (("Regression (permutation importance, RMSE increase)", "regression_feature_importance.csv"),
                        ("Classification (permutation importance, AUC drop)", "classification_feature_importance.csv")):
        imp = load_report(file)
        st.subheader(title)
        if imp is None:
            st.info("Run the pipeline to generate feature importance.")
        else:
            st.plotly_chart(px.bar(imp.head(15).sort_values("importance"), x="importance", y="feature",
                                   orientation="h"), use_container_width=True)

# ------------------------------------------------------------ explainability
with tabs[6]:
    st.subheader("SHAP (global) and LIME (single prediction)")
    if not deps.HAS_SHAP:
        st.info("SHAP is not installed in this environment (`pip install shap`).")
    else:
        for tag in ("regression", "classification"):
            img = REPORTS_DIR / "figures" / f"shap_{tag}_summary.png"
            if img.exists():
                st.markdown(f"**SHAP summary - {tag}**")
                st.image(str(img))
            else:
                st.info(f"SHAP {tag} plot not found; re-run the pipeline.")
    st.markdown("**LIME explanation for the last prediction made on the 'Predict demand' tab**")
    if not deps.HAS_LIME:
        st.info("LIME is not installed in this environment (`pip install lime`).")
    elif "last_row" not in st.session_state:
        st.info("Make a prediction first, then come back here.")
    else:
        from src.explainability import lime_explain
        task = st.radio("Explain", ["regression", "classification"], horizontal=True)
        model = reg_model if task == "regression" else clf_model
        with st.spinner("Computing LIME explanation ..."):
            expl = pd.DataFrame(lime_explain(model, data[FEATURES], st.session_state["last_row"], task),
                                columns=["condition", "weight"])
        st.plotly_chart(px.bar(expl.sort_values("weight"), x="weight", y="condition", orientation="h",
                               color="weight", color_continuous_scale="RdBu"), use_container_width=True)

# ---------------------------------------------------------------- forecast
with tabs[7]:
    st.subheader("Daily demand forecast (kWh)")
    fm, back, fut = (load_report("forecast_metrics.csv"), load_report("forecast_backtest.csv"),
                     load_report("forecast_future.csv"))
    if fm is None:
        st.info("Run the pipeline to generate forecasts.")
    else:
        st.dataframe(fm, use_container_width=True, hide_index=True)
        history = data.set_index("StartDatetime")[TARGET].resample("D").sum().reset_index()
        history.columns = ["Date", "Actual"]
        fig = px.line(history.tail(120), x="Date", y="Actual", title="Recent actual demand and 30-day forecast")
        if fut is not None:
            fig.add_scatter(x=fut["Date"], y=fut["Forecast_kWh"], mode="lines", name="Forecast")
        st.plotly_chart(fig, use_container_width=True)
        if back is not None:
            st.plotly_chart(px.line(back, x="Date", y=[c for c in back.columns if c != "Date"],
                                    title="Backtest: actual vs models"), use_container_width=True)
