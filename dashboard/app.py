"""PulsePredict Streamlit dashboard.   Run:  streamlit run dashboard/app.py
(Start the API first:  uvicorn api.main:app --reload)"""
import json
import os
import sys
from pathlib import Path

# Make `src` importable when Streamlit runs this file from the dashboard/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src import eda  # noqa: E402
from src.config import FIGURES_DIR, METRICS_DIR, PROCESSED_DATA_PATH  # noqa: E402

API_URL = os.environ.get("PULSEPREDICT_API_URL", "http://localhost:8000")
DISCLAIMER = ("**Educational portfolio project – not a medical diagnostic system.** "
              "Outputs are model estimates learned from a training dataset and must not be used "
              "for health decisions. Consult a qualified clinician for medical advice.")

st.set_page_config(page_title="PulsePredict", page_icon="🫀", layout="wide")


# ---------- cached loaders ----------
@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((METRICS_DIR / name).read_text())


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_resource
def load_eda_figures():
    return eda.make_all(load_clean_data())


def artefacts_ready() -> bool:
    return (METRICS_DIR / "selected_model.json").exists() and PROCESSED_DATA_PATH.exists()


# ---------- pages ----------
def page_overview():
    st.title("🫀 PulsePredict")
    st.subheader("Cardiovascular Risk Analytics Platform")
    st.write(
        "PulsePredict is an end-to-end project that takes cardiovascular risk-factor data from "
        "raw CSV to a working application: data cleaning and exploration, model comparison and "
        "evaluation, a validated prediction API, SQL logging of predictions, automated tests and CI."
    )
    summary, selected = load_json("dataset_summary.json"), load_json("selected_model.json")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records (after cleaning)", f"{summary['observations_clean']:,}")
    c2.metric("Input features", f"{summary['input_features']} (+ BMI)")
    c3.metric("Positive class", f"{summary['positive_class_rate']*100:.1f}%")
    c4.metric(f"Best ROC-AUC ({selected['model_name']})", f"{selected['metrics']['roc_auc']:.3f}")

    st.markdown("#### Pipeline")
    st.code("Cardiovascular data → preprocessing → ML model → FastAPI → prediction", language=None)
    st.markdown(
        "1. **Data/ML** – cleaning with documented rules, EDA, three compared models, saved sklearn pipeline.\n"
        "2. **Backend** – FastAPI validates input with Pydantic, runs the saved pipeline, logs to SQLite.\n"
        "3. **Application** – this Streamlit dashboard talks to the API over HTTP."
    )
    st.info(DISCLAIMER)


def page_explorer():
    st.title("Data Explorer")
    st.caption("Cleaned dataset used for training. Eight charts chosen to answer specific questions.")
    figs = load_eda_figures()
    names = list(figs)
    for i in range(0, 6, 2):
        left, right = st.columns(2)
        left.pyplot(figs[names[i]])
        right.pyplot(figs[names[i + 1]])
    st.pyplot(figs[names[6]])
    st.pyplot(figs[names[7]])

    st.markdown("### Dataset summary")
    s, report = load_json("dataset_summary.json"), load_json("cleaning_report.json")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw observations", f"{s['observations_raw']:,}")
    c2.metric("After cleaning", f"{s['observations_clean']:,}")
    c3.metric("Missing values (raw)", s["missing_values_raw"])
    c4.metric("Class balance (CVD=1)", f"{s['positive_class_rate']*100:.1f}%")
    st.write(
        f"{report['rows_removed_total']:,} rows ({report['percent_removed']}%) were removed. "
        "Each rule is based on physiological plausibility rather than arbitrary percentile trimming. "
        "Age was converted from days to years and BMI engineered from height and weight."
    )
    steps = pd.DataFrame(report["steps"]).rename(columns={
        "rule": "Cleaning rule", "rows_removed": "Rows removed", "rationale": "Why"})
    st.dataframe(steps[["Cleaning rule", "Rows removed", "Why"]], hide_index=True)


def page_performance():
    st.title("Model Performance")
    comp = pd.read_csv(METRICS_DIR / "model_comparison.csv")
    sel = load_json("selected_model.json")
    table = pd.DataFrame({
        "Model": comp["model"],
        "Accuracy": comp["accuracy"].round(3),
        "Precision": comp["precision"].round(3),
        "Recall": comp["recall"].round(3),
        "F1": comp["f1"].round(3),
        "ROC-AUC": comp["roc_auc"].round(3),
        "CV ROC-AUC (train)": comp["cv_roc_auc_mean"].round(3),
    })
    st.markdown("### Model comparison (held-out test set)")
    st.dataframe(table, hide_index=True)
    st.success(f"Selected model: **{sel['model_name']}**")

    c1, c2 = st.columns(2)
    c1.image(str(FIGURES_DIR / "confusion_matrix.png"))
    c2.image(str(FIGURES_DIR / "roc_curve.png"))
    st.image(str(FIGURES_DIR / "feature_importance.png"))
    st.caption("Height and weight are shuffled independently here, so BMI's contribution is "
               "shared between them.")

    st.markdown("### Why this model?")
    st.write(sel["selection_rationale"])
    cm = sel["confusion_matrix"]
    st.markdown(
        f"**False negatives vs false positives.** In a health-screening context a false negative "
        f"(a higher-risk profile labelled lower-risk – {cm['fn']:,} here) is usually costlier than a "
        f"false positive ({cm['fp']:,} here), which typically leads to a further check. That is why "
        f"recall and ROC-AUC are weighed alongside accuracy, and why the decision threshold could be "
        f"lowered in a real screening tool to trade precision for recall."
    )


def _call_api(path: str, payload: dict | None = None):
    if payload is None:
        return requests.get(f"{API_URL}{path}", timeout=5)
    return requests.post(f"{API_URL}{path}", json=payload, timeout=10)


def page_prediction():
    st.title("Risk Prediction")
    st.warning(DISCLAIMER)
    try:
        ok = _call_api("/health").json().get("model_loaded", False)
        st.caption(f"API status: {'🟢 connected, model loaded' if ok else '🟠 connected, model NOT loaded'}  ·  {API_URL}")
    except requests.RequestException:
        st.caption(f"API status: 🔴 not reachable at {API_URL} – start it with `uvicorn api.main:app --reload`")

    levels = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", 18, 100, 45)
        sex = c1.selectbox("Sex", [1, 2], format_func=lambda v: "Female" if v == 1 else "Male", index=1)
        height = c1.number_input("Height (cm)", 100, 250, 170)
        weight = c2.number_input("Weight (kg)", 30, 300, 78)
        sbp = c2.number_input("Systolic BP (mmHg)", 70, 250, 135)
        dbp = c2.number_input("Diastolic BP (mmHg)", 40, 150, 85)
        chol = c3.selectbox("Cholesterol", [1, 2, 3], format_func=levels.get)
        gluc = c3.selectbox("Glucose", [1, 2, 3], format_func=levels.get)
        smoking = c3.checkbox("Smoker")
        alcohol = c3.checkbox("Regular alcohol intake")
        active = c3.checkbox("Physically active", value=True)
        submitted = st.form_submit_button("Analyse Risk Profile")

    if submitted:
        payload = {
            "age": int(age), "sex": int(sex), "height": float(height), "weight": float(weight),
            "systolic_bp": int(sbp), "diastolic_bp": int(dbp), "cholesterol": int(chol),
            "glucose": int(gluc), "smoking": int(smoking), "alcohol": int(alcohol), "active": int(active),
        }
        try:
            r = _call_api("/predict", payload)
        except requests.RequestException:
            st.error("Could not reach the API. Is it running?")
        else:
            if r.status_code == 200:
                res = r.json()
                a, b, c = st.columns(3)
                a.metric("Classification", res["risk_category"])
                b.metric("Model probability", f"{res['probability']:.2f}")
                c.metric("Model version", res["model_version"])
                st.progress(min(max(res["probability"], 0.0), 1.0))
                st.caption("This is the model's output for the training dataset's patterns – "
                           "not a personal chance of having cardiovascular disease.")
            elif r.status_code == 422:
                st.error("The API rejected these values:")
                for err in r.json().get("detail", []):
                    loc = ".".join(str(x) for x in err.get("loc", [])[1:])
                    st.write(f"- **{loc or 'input'}**: {err.get('msg')}")
            else:
                st.error(f"API error {r.status_code}: {r.text}")

    st.markdown("### Internal analytics (SQLite prediction log)")
    try:
        stats = _call_api("/stats").json()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predictions made", stats["predictions_made"])
        m2.metric("Average predicted probability", f"{stats['average_probability']:.2f}")
        m3.metric("Higher-risk classifications", stats["higher_risk_classifications"])
        if stats["recent"]:
            st.dataframe(pd.DataFrame(stats["recent"]), hide_index=True)
    except requests.RequestException:
        st.caption("Analytics unavailable while the API is offline.")


# ---------- navigation ----------
PAGES = {
    "Overview": page_overview,
    "Data Explorer": page_explorer,
    "Model Performance": page_performance,
    "Risk Prediction": page_prediction,
}

st.sidebar.title("PulsePredict")
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption("Educational portfolio project – not a medical device.")

if choice != "Risk Prediction" and not artefacts_ready():
    st.error("Training artefacts not found. Run `python -m src.train` first.")
else:
    PAGES[choice]()
