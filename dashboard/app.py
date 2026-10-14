"""AahaarTwin dashboard (Phases 5+6) — interactive front end for the
personalized meal-response digital twin.

Research prototype only — not a medical device; not for diagnosis or
treatment. Every metric shown is loaded from a real project artifact
(models/*.json); anything hypothetical is labeled a "demonstration example".

Import-safe: importing this module performs no Streamlit UI calls. Run with
``streamlit run dashboard/app.py`` (entry point: ``main()``).
"""

import json
import math
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Path bootstrap (no Streamlit calls here — import must stay UI-free).
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if os.path.join(PROJ, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJ, "src"))

from features import STATIC_FEATURES, load_bio  # noqa: E402
from ranking import CARBS_P01, CARBS_P99, rank_meals  # noqa: E402
from twin_state import (  # noqa: E402
    DRIFT_WINDOW,
    EXCURSION_PEAK_THRESHOLD,
    MODEL_VERSION,
    compute_drift_bound,
    get_profile,
    predict_meal,
    seed_from_meals,
)

DB_PATH = os.path.join(PROJ, "twin_state.db")
MODELS_DIR = os.path.join(PROJ, "models")
RESULTS_PHASE1 = os.path.join(PROJ, "results", "phase1", "metrics.json")
MEALS_CSV = os.path.join(PROJ, "data", "processed", "meals.csv")
RAW_DIR = os.path.join(PROJ, "..", "feas-data", "cgmacros_raw", "CGMacros")
BIO_CSV = os.path.join(RAW_DIR, "bio.csv")

DOI = "10.13026/3z8q-x658"
DISCLAIMER = ("Research prototype \u2014 not a medical device;"
              " not for diagnosis or treatment.")
DEMO_TAG = "demonstration example"

ARTIFACT_FILES = ["metrics", "metrics_ci", "calibration", "personalization",
                  "ranking", "feature_importance", "conformal"]

BOOSTER_CARDS = [
    ("lgbm_iauc.txt", "2h glucose load (iAUC)", "point prediction"),
    ("lgbm_peak.txt", "2h glucose peak", "point prediction"),
    ("lgbm_excursion.txt", "excursion probability (peak \u2265 180 mg/dL)",
     "binary classifier"),
    ("lgbm_iauc_q05.txt", "iAUC 5th percentile", "quantile model"),
    ("lgbm_iauc_q95.txt", "iAUC 95th percentile", "quantile model"),
]

MACRO_LIMITS = {
    "carbs": (0.0, 500.0),
    "protein": (0.0, 300.0),
    "fat": (0.0, 300.0),
    "fiber": (0.0, 150.0),
    "calories": (0.0, 5000.0),
}
PRE_KEYS = ["g0", "slope_60min", "hr_mean_2h", "act_kcal_2h"]
PRE_LABELS = {"g0": "Pre-meal glucose g0 (mg/dL)",
              "slope_60min": "Glucose slope, prior 60 min (mg/dL/min)",
              "hr_mean_2h": "Mean heart rate, prior 2 h (bpm)",
              "act_kcal_2h": "Activity calories, prior 2 h (kcal)"}


# --------------------------------------------------------------------------- #
# Pure helpers (UI-free; unit-testable).
# --------------------------------------------------------------------------- #
def validate_macros(macros):
    """Validate a macro dict. Returns a list of error strings (empty = OK).

    Rules: every key present, numeric, non-negative, within documented bounds.
    """
    errors = []
    macros = macros or {}
    for key, (lo, hi) in MACRO_LIMITS.items():
        v = macros.get(key)
        if v is None:
            errors.append(f"{key}: missing value.")
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            errors.append(f"{key}: not a number ({v!r}).")
            continue
        if math.isnan(f):
            errors.append(f"{key}: not a number (NaN).")
        elif f < lo:
            errors.append(f"{key}: {f} is negative \u2014 values below 0 are rejected.")
        elif f > hi:
            errors.append(f"{key}: {f} exceeds the {key} input range 0\u2013{hi:g}.")
    return errors


def validate_premeal(premeal):
    """Light validation for pre-meal context overrides (None = auto/impute)."""
    errors = []
    premeal = premeal or {}
    for key in PRE_KEYS:
        v = premeal.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            errors.append(f"{key}: not a number ({v!r}).")
            continue
        if math.isnan(f):
            errors.append(f"{key}: not a number (NaN).")
    return errors


def risk_level(prob_excursion, thr_f1):
    """Excursion-risk badge from the excursion probability.

    HIGH  : prob >= 0.5 (majority of model mass on an excursion)
    MODERATE: 0.347 <= prob < 0.5 (above the OOF max-F1 operating point)
    LOW   : prob < 0.347
    """
    if prob_excursion >= 0.5:
        return "HIGH"
    if prob_excursion >= thr_f1:
        return "MODERATE"
    return "LOW"


def clip_interval_lo(lo):
    """Display clipping: interval lower bound clipped at 0 (documented)."""
    return max(0.0, float(lo))


def _safe_num(x, default=0.0):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


# --------------------------------------------------------------------------- #
# Cached data / artifact loading.
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    """Load every metrics artifact; fail loudly if any is missing/corrupt."""
    out = {}
    missing = []
    for name in ARTIFACT_FILES:
        path = os.path.join(MODELS_DIR, f"{name}.json")
        try:
            with open(path) as fh:
                out[name] = json.load(fh)
        except FileNotFoundError:
            missing.append(f"models/{name}.json")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Artifact models/{name}.json is not valid JSON: {exc}") from exc
    if missing:
        raise RuntimeError(
            "Missing required metric artifact(s): " + ", ".join(missing) +
            ". Metrics are never invented; the dashboard refuses to start "
            "without them.")
    # Baseline metrics (Phase-1 time-ordered evaluation, real artifact).
    try:
        with open(RESULTS_PHASE1) as fh:
            out["baselines"] = json.load(fh)["baselines"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Missing/corrupt baseline artifact: results/phase1/metrics.json "
            f"({exc}).") from exc
    return out


@st.cache_resource
def warm_models():
    """Load the 5 LightGBM boosters into memory (cached across reruns)."""
    from twin_state import _boosters
    return _boosters()


@st.cache_data
def load_meals():
    return pd.read_csv(MEALS_CSV, parse_dates=["ts"])


@st.cache_data
def load_bio_df():
    return load_bio(BIO_CSV)


def patient_pids():
    meals = load_meals()
    return sorted(int(p) for p in meals["pid"].unique())


def latest_premeal_context(pid):
    """Auto-fill pre-meal context from the patient's latest usable meal row."""
    meals = load_meals()
    g = meals[(meals["pid"] == pid) & (meals["usable"].astype(bool))].sort_values("ts")
    if g.empty:
        return {k: None for k in PRE_KEYS}, None
    row = g.iloc[-1]
    ctx = {k: (None if pd.isna(row[k]) else float(row[k])) for k in PRE_KEYS}
    return ctx, str(row["ts"])


def query_df(sql, params=()):
    con = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


@st.cache_data
def load_cgm_day(pid, day_iso):
    """Raw 1-min CGM/HR for one patient-day, downsampled to 15 min.

    Returns (glucose_series, hr_series) or None when the raw directory/file is
    unavailable or has no rows for that day.
    """
    path = os.path.join(RAW_DIR, f"CGMacros-{pid:03d}", f"CGMacros-{pid:03d}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    if "Timestamp" not in df.columns or "Dexcom GL" not in df.columns:
        return None
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    day = df[df["Timestamp"].dt.strftime("%Y-%m-%d") == day_iso]
    if day.empty:
        return None
    day = day.set_index("Timestamp").sort_index()
    glu = pd.to_numeric(day["Dexcom GL"], errors="coerce").resample("15min").mean()
    glu = glu.dropna()
    hr = None
    if "HR" in day.columns:
        hr = pd.to_numeric(day["HR"], errors="coerce").resample("15min").mean().dropna()
    if glu.empty:
        return None
    return glu, hr


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
def page_overview(art):
    st.header("Overview")
    st.markdown(
        "**AahaarTwin** is a research prototype of a *personalized meal-response "
        "digital twin*: given a patient's pre-meal glucose/wearable context and a "
        "meal's macronutrients, it predicts the 2-hour post-meal glucose response "
        "(iAUC load, peak, excursion probability with a 90% interval), keeps a "
        "stateful per-patient response profile (SQLite), and ranks candidate "
        "meals by predicted response. " + DISCLAIMER)

    m = art["metrics"]
    ci = art["metrics_ci"]
    st.subheader("Dataset status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Usable meals", f"{int(m['n_usable_meals']):,}")
    c2.metric("Patients", int(m["n_patients"]))
    c3.metric("Source dataset", "CGMacros")
    st.caption(f"CGMacros dataset \u2014 DOI {DOI}. Meal table: data/processed/meals.csv.")

    st.subheader("Model status")
    try:
        warm_models()
        st.success(f"5 LightGBM boosters loaded \u2014 model version `{MODEL_VERSION}`.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model files could not be loaded: {exc}")
    with st.expander("The 5 boosters", expanded=False):
        for fname, target, kind in BOOSTER_CARDS:
            st.markdown(f"- `{fname}` \u2014 {target} ({kind})")

    st.subheader("Out-of-fold performance (loaded from artifacts)")
    reg = m["regression"]
    cls = m["classification"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Regression**")
        st.write(f"iAUC MAE **{reg['iauc']['mae']:.2f}** mg/dL\u00b7h "
                 f"(95% CI {ci['mae_iauc'][0]:.2f}\u2013{ci['mae_iauc'][1]:.2f})")
        st.write(f"Peak MAE **{reg['peak']['mae']:.2f}** mg/dL "
                 f"(95% CI {ci['mae_peak'][0]:.2f}\u2013{ci['mae_peak'][1]:.2f})")
        st.write(f"Interval: nominal 90% q05\u2013q95; empirical OOF coverage "
                 f"{m['uncertainty']['empirical_coverage_oof']:.1%}, "
                 f"mean width {m['uncertainty']['mean_width']:.1f}")
    with col2:
        st.markdown("**Excursion classification** (peak \u2265 180 mg/dL)")
        st.write(f"PR-AUC **{cls['pr_auc']:.3f}** "
                 f"(95% CI {ci['pr_auc'][0]:.2f}\u2013{ci['pr_auc'][1]:.2f}); "
                 f"ROC-AUC **{cls['roc_auc']:.3f}** "
                 f"(95% CI {ci['roc_auc'][0]:.2f}\u2013{ci['roc_auc'][1]:.2f})")
        st.write(f"At OOF max-F1 threshold {cls['threshold_max_f1_oof']:.3f}: "
                 f"P {cls['at_max_f1']['precision']:.3f}, "
                 f"R {cls['at_max_f1']['recall']:.3f}, "
                 f"F1 {cls['at_max_f1']['f1']:.3f}")
        st.write(f"At 0.5: P {cls['at_0_5']['precision']:.3f}, "
                 f"R {cls['at_0_5']['recall']:.3f}, F1 {cls['at_0_5']['f1']:.3f}; "
                 f"ECE (raw) {cls['ece_raw']:.3f}")

    st.warning(
        "**Limitations.** Research prototype: predictions are associational, "
        "not causal, and not medical advice. The population model has an OOF "
        "iAUC MAE of ~32 mg/dL\u00b7h with wide uncertainty intervals; raw "
        "calibration is imperfect (ECE 0.122). Excursion labels are derived "
        "from the peak \u2265 180 mg/dL rule, not measured events. Per-patient "
        "fine-tuning (E1) is evaluated but not wired into the prediction path. "
        "See the Evaluation and Safety pages for the full list.")

    st.subheader("Demo patients")
    st.caption("Illustrative selection from the data \u2014 not a clinical grouping.")
    bio = load_bio_df()
    demo = {
        35: f"pid 35 \u2014 A1c {bio.loc[35, 'hba1c']:.1f}% (in the diabetes range, \u22656.5%; illustrative)",
        31: f"pid 31 \u2014 A1c {bio.loc[31, 'hba1c']:.1f}% (healthy range; illustrative)",
    }
    sel = st.selectbox("Demo patient", options=list(demo.keys()),
                       format_func=lambda p: demo[p])
    prof = get_profile(DB_PATH, int(sel))
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed meals", prof["n_meals_observed"])
    c2.metric("Carb sensitivity (iAUC/g)",
              "n/a" if prof["carb_sensitivity"] is None
              else f"{prof['carb_sensitivity']:.2f}")
    c3.metric("Drift flag", "YES" if prof["drift_flag"] else "no")


# --------------------------------------------------------------------------- #
# Page: Patient explorer
# --------------------------------------------------------------------------- #
def page_patient(art):
    st.header("Patient explorer")
    pid = int(st.selectbox("Patient", options=patient_pids(), index=0))
    bio = load_bio_df()
    st.subheader(f"Profile \u2014 pid {pid}")
    if pid in bio.index:
        row = bio.loc[pid]
        info = {
            "Age (y)": row.get("age"), "Gender": "M" if row.get("gender_m") == 1 else "F",
            "BMI": row.get("bmi"), "HbA1c (%)": row.get("hba1c"),
            "Fasting glucose (mg/dL)": row.get("fasting_glu"),
            "Fasting insulin (mcU/mL)": row.get("fasting_insulin"),
            "Triglycerides (mg/dL)": row.get("trig"),
            "Cholesterol (mg/dL)": row.get("chol"),
            "HDL (mg/dL)": row.get("hdl"), "LDL (mg/dL)": row.get("ldl"),
        }
        cols = st.columns(5)
        for i, (k, v) in enumerate(info.items()):
            txt = "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else (
                v if isinstance(v, str) else f"{v:.1f}")
            cols[i % 5].metric(k, txt)
        st.caption("Static variables from bio.csv (one row per patient).")
    else:
        st.info("No bio row for this patient.")

    prof = get_profile(DB_PATH, pid)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observed meals", prof["n_meals_observed"])
    c2.metric("Carb sensitivity",
              "n/a (<3 meals)" if prof["carb_sensitivity"] is None
              else f"{prof['carb_sensitivity']:.2f} iAUC/g")
    c3.metric("EWMA residual",
              "n/a" if prof["ewma_residual"] is None else f"{prof['ewma_residual']:.1f}")
    c4.metric("Drift", "FLAGGED" if prof["drift_flag"] else "none")

    st.subheader("CGM & wearable trend (raw sensor data)")
    meals = load_meals()
    g = meals[(meals["pid"] == pid)].sort_values("ts")
    default_day = g["ts"].dt.date.iloc[0] if not g.empty else pd.Timestamp.now().date()
    day = st.date_input("Day", value=default_day)
    day_iso = pd.Timestamp(day).strftime("%Y-%m-%d")
    cgm = load_cgm_day(pid, day_iso)
    if cgm is None:
        st.info(
            f"No raw CGM rows for pid {pid} on {day_iso}. The raw sensor directory "
            "is optional for the dashboard; predictions and meal history below do "
            "not depend on it.")
    else:
        glu, hr = cgm
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=glu.index, y=glu.values, mode="lines",
                                 name="Dexcom GL (mg/dL, 15-min mean)"))
        fig.update_layout(title=f"Glucose \u2014 pid {pid}, {day_iso}",
                          xaxis_title="Time", yaxis_title="mg/dL",
                          margin=dict(l=40, r=20, t=40, b=40), height=320)
        st.plotly_chart(fig, use_container_width=True)
        if hr is not None and not hr.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hr.index, y=hr.values, mode="lines",
                                      name="HR (bpm, 15-min mean)"))
            fig2.update_layout(title=f"Heart rate \u2014 pid {pid}, {day_iso}",
                               xaxis_title="Time", yaxis_title="bpm",
                               margin=dict(l=40, r=20, t=40, b=40), height=280)
            st.plotly_chart(fig2, use_container_width=True)
        st.caption("Downsampled from the 1-minute raw CGM grid to 15-minute means.")

    st.subheader("Meal history \u2014 predicted vs observed")
    obs = query_df(
        "SELECT ts, meal_type, carbs, iauc_obs, peak_obs, source FROM observations"
        " WHERE pid = ? ORDER BY ts", (pid,))
    pred = query_df(
        "SELECT ts, pred_iauc, pred_peak, prob_excursion, iauc_lo, iauc_hi, note"
        " FROM predictions WHERE pid = ?", (pid,))
    if obs.empty:
        st.info("No meals recorded for this patient.")
    else:
        hist = obs.merge(pred, on="ts", how="left", suffixes=("", "_p"))
        hist["residual"] = hist["iauc_obs"] - hist["pred_iauc"]
        show = hist[["ts", "meal_type", "carbs", "pred_iauc", "iauc_obs",
                     "residual", "pred_peak", "peak_obs", "source"]].copy()
        show.columns = ["time", "meal", "carbs (g)", "pred iAUC", "obs iAUC",
                        "residual", "pred peak", "obs peak", "source"]
        st.dataframe(show.round(1), use_container_width=True)
        st.caption("`source` is **observed** for real logged meals; **simulated** "
                   "rows are what-if scenarios that never update the profile.")

    st.subheader("Evidence quality")
    if pred.empty:
        st.info("No predictions logged for this patient yet.")
    else:
        labels = pred["note"].str.extract(r"evidence:(\w+)")[0].fillna("unknown")
        counts = labels.value_counts()
        fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.values.tolist()))
        fig.update_layout(title="Predictions by evidence label",
                          xaxis_title="evidence", yaxis_title="count",
                          margin=dict(l=40, r=20, t=40, b=40), height=280)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Evidence = fraction of the 4 pre-meal context features present: "
                   "high = 4/4, medium = 3/4, low = \u22642/4.")


# --------------------------------------------------------------------------- #
# Page: Prediction
# --------------------------------------------------------------------------- #
def page_prediction(art):
    st.header("Prediction")
    st.markdown("Predict one meal's 2-hour response for a patient. "
                + DISCLAIMER)
    m = art["metrics"]
    thr_f1 = float(m["classification"]["threshold_max_f1_oof"])

    pid = int(st.selectbox("Patient", options=patient_pids(), key="pred_pid"))
    c1, c2, c3 = st.columns(3)
    with c1:
        d = st.date_input("Meal date", value=pd.Timestamp.now().date(), key="pred_d")
    with c2:
        t = st.time_input("Meal time", value=pd.Timestamp.now().time().replace(
            second=0, microsecond=0), key="pred_t")
    with c3:
        meal_type = st.selectbox("Meal type",
                                 ["breakfast", "lunch", "dinner", "snack"],
                                 key="pred_mt")
    ts_iso = pd.Timestamp.combine(d, t).strftime("%Y-%m-%d %H:%M:%S")

    st.subheader("Meal macros")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    carbs = mc1.number_input("Carbs (g)", min_value=0.0, max_value=500.0,
                             value=60.0, step=5.0)
    protein = mc2.number_input("Protein (g)", min_value=0.0, value=20.0, step=5.0)
    fat = mc3.number_input("Fat (g)", min_value=0.0, value=12.0, step=2.0)
    fiber = mc4.number_input("Fiber (g)", min_value=0.0, value=5.0, step=1.0)
    calories = mc5.number_input("Calories (kcal)", min_value=0.0, value=450.0,
                                step=25.0)
    macros = {"carbs": carbs, "protein": protein, "fat": fat, "fiber": fiber,
              "calories": calories}

    st.subheader("Pre-meal context")
    ctx, ctx_ts = latest_premeal_context(pid)
    if ctx_ts:
        st.caption(f"Auto-filled from the patient's latest usable meal row "
                   f"({ctx_ts}); override any value. Blank = median-imputed.")
    else:
        st.caption("No usable meal row for this patient; all context blank "
                   "(median-imputed).")
    pc = {}
    cols = st.columns(4)
    for i, k in enumerate(PRE_KEYS):
        default = ctx.get(k)
        pc[k] = cols[i].number_input(
            PRE_LABELS[k], value=None if default is None else float(default),
            placeholder="auto/impute", key=f"pre_{k}")

    if st.button("Predict", type="primary"):
        errs = validate_macros(macros) + validate_premeal(pc)
        if errs:
            for e in errs:
                st.error(e)
        else:
            try:
                with st.spinner("Running the 5 boosters\u2026"):
                    pred = predict_meal(DB_PATH, pid, ts_iso, meal_type, macros, pc)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}. Check the model files and "
                         f"inputs, then retry.")
            else:
                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted iAUC (mg/dL\u00b7h)", f"{pred['pred_iauc']:.1f}")
                r2.metric("Predicted peak (mg/dL)", f"{pred['pred_peak']:.1f}")
                r3.metric("Excursion probability", f"{pred['prob_excursion']:.1%}")
                lo = clip_interval_lo(pred["iauc_lo"])
                hi = float(pred["iauc_hi"])
                st.metric("90% iAUC interval (display-clipped)",
                          f"[{lo:.1f}, {hi:.1f}]")
                st.caption("Interval = q05\u2013q95 \u00b1 conformal adjustment "
                           f"({art['conformal']['adjustment']:.2f}); the lower "
                           "bound is clipped at 0 for display only \u2014 the "
                           "underlying interval is symmetric/additive and may "
                           "extend slightly negative.")
                lvl = risk_level(pred["prob_excursion"], thr_f1)
                badge = {"HIGH": "\U0001F534 HIGH", "MODERATE": "\U0001F7E1 MODERATE",
                         "LOW": "\U0001F7E2 LOW"}[lvl]
                st.markdown(f"**Excursion-risk badge: {badge}**")
                st.caption(f"HIGH if prob \u2265 0.5; MODERATE if \u2265 {thr_f1:.3f} "
                           "(OOF max-F1 operating point); else LOW.")
                st.info(f"Evidence: **{pred['evidence_label']}** "
                        f"(score {pred['evidence_score']:.2f}) \u2014 fraction of "
                        "the 4 pre-meal context features present.")
                st.caption(f"Model `{pred['model_version']}`; profile at prediction "
                           f"time: {pred['n_meals_observed']} observed meals. This "
                           "prediction was logged to the twin database "
                           "(predictions table) for provenance.")


# --------------------------------------------------------------------------- #
# Page: Meal comparison
# --------------------------------------------------------------------------- #
def _candidate_form(i, art):
    defaults = [
        {"name": "Demo: veg thali (moderate carbs)", "meal_type": "lunch",
         "carbs": 70.0, "protein": 18.0, "fat": 15.0, "fiber": 10.0,
         "calories": 500.0},
        {"name": "Demo: high-protein bowl (lower carbs)", "meal_type": "lunch",
         "carbs": 30.0, "protein": 35.0, "fat": 12.0, "fiber": 8.0,
         "calories": 420.0},
        {"name": "Demo: sweet dessert + chai (high carbs)", "meal_type": "snack",
         "carbs": 110.0, "protein": 8.0, "fat": 14.0, "fiber": 2.0,
         "calories": 620.0},
    ]
    d = defaults[i]
    with st.expander(f"Candidate {i + 1}", expanded=(i == 0)):
        st.caption(f"Prefilled values are a **{DEMO_TAG}** \u2014 not real "
                   "nutritional facts. Edit freely.")
        name = st.text_input("Name", value=d["name"], key=f"cand{i}_name")
        mt = st.selectbox("Meal type",
                          ["breakfast", "lunch", "dinner", "snack"],
                          index=["breakfast", "lunch", "dinner", "snack"].index(
                              d["meal_type"]), key=f"cand{i}_mt")
        c1, c2, c3, c4, c5 = st.columns(5)
        carbs = c1.number_input("Carbs (g)", 0.0, 500.0, d["carbs"], 5.0,
                                key=f"cand{i}_carbs")
        protein = c2.number_input("Protein (g)", 0.0, None, d["protein"], 5.0,
                                  key=f"cand{i}_prot")
        fat = c3.number_input("Fat (g)", 0.0, None, d["fat"], 2.0, key=f"cand{i}_fat")
        fiber = c4.number_input("Fiber (g)", 0.0, None, d["fiber"], 1.0,
                                key=f"cand{i}_fib")
        calories = c5.number_input("Calories (kcal)", 0.0, None, d["calories"], 25.0,
                                   key=f"cand{i}_cal")
    return {"name": name, "meal_type": mt, "carbs": carbs, "protein": protein,
            "fat": fat, "fiber": fiber, "calories": calories}


def page_comparison(art):
    st.header("Meal comparison")
    st.markdown("Rank up to 3 candidate meals by predicted 2h glucose load "
                "(ascending). Candidate macros are **your inputs** \u2014 the "
                "dashboard invents no foods. " + DISCLAIMER)
    m = art["metrics"]
    thr_f1 = float(m["classification"]["threshold_max_f1_oof"])

    pid = int(st.selectbox("Patient", options=patient_pids(), key="cmp_pid"))
    ctx, ctx_ts = latest_premeal_context(pid)
    if ctx_ts:
        st.caption(f"Pre-meal context auto-filled from the patient's latest usable "
                   f"meal row ({ctx_ts}).")

    candidates = [_candidate_form(i, art) for i in range(3)]
    candidates = [c for c in candidates if c["name"].strip()]
    if len(candidates) < 2:
        st.info("Enter at least 2 named candidates to compare.")
        return

    errs = []
    for c in candidates:
        errs += [f"{c['name']}: {e}" for e in validate_macros(c)]
    if errs:
        for e in errs:
            st.error(e)
        return

    if st.button("Rank meals", type="primary"):
        premeal = {k: ctx.get(k) for k in PRE_KEYS}
        premeal["ts_iso"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        def _predict_fn(ts_iso, meal_type, macros, premeal_ctx):
            return predict_meal(DB_PATH, pid, ts_iso, meal_type, macros,
                                {k: premeal_ctx.get(k) for k in PRE_KEYS})

        try:
            with st.spinner("Scoring candidates\u2026"):
                ranked = rank_meals(_predict_fn, premeal, candidates)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ranking failed: {exc}. Check the model files and inputs.")
            return

        st.subheader("Ranking")
        table = pd.DataFrame([{
            "rank": i + 1, "meal": r["name"],
            "pred iAUC": round(r["pred_iauc"], 1),
            "pred peak": round(r["pred_peak"], 1),
            "excursion prob": f"{r['prob_excursion']:.1%}",
            "90% interval": f"[{clip_interval_lo(r['iauc_lo']):.0f}, {r['iauc_hi']:.0f}]",
            "evidence": r["evidence_label"],
            "macros": "missing — exploratory" if r["macros_missing"] else "provided",
            "OOD": "exploratory" if r["ood_flag"] else "in-range",
        } for i, r in enumerate(ranked)])
        st.dataframe(table, use_container_width=True, hide_index=True)

        names = [r["name"] for r in ranked]
        iaucs = [r["pred_iauc"] for r in ranked]
        lo = [r["pred_iauc"] - clip_interval_lo(r["iauc_lo"]) for r in ranked]
        hi = [r["iauc_hi"] - r["pred_iauc"] for r in ranked]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names, y=iaucs, error_y=dict(type="data", symmetric=False,
                                           array=hi, arrayminus=lo),
            marker_color=["#2ca02c" if i == 0 else "#7f7f7f"
                          for i in range(len(ranked))],
            name="pred iAUC \u00b1 90% interval"))
        fig.update_layout(title="Predicted 2h glucose load by candidate",
                          xaxis_title="candidate", yaxis_title="iAUC (mg/dL\u00b7h)",
                          margin=dict(l=40, r=20, t=40, b=40), height=360)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=names,
                              y=[100.0 * r["prob_excursion"] for r in ranked],
                              name="excursion prob (%)"))
        fig2.add_hline(y=100.0 * thr_f1, line_dash="dash",
                       annotation_text=f"OOF max-F1 threshold {thr_f1:.3f}")
        fig2.add_hline(y=50.0, line_dash="dot",
                       annotation_text="HIGH-risk boundary 0.5")
        fig2.update_layout(title="Excursion probability by candidate",
                           xaxis_title="candidate", yaxis_title="%",
                           margin=dict(l=40, r=20, t=40, b=40), height=320)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Explanations")
        for i, r in enumerate(ranked):
            if r["ood_flag"]:
                st.warning(
                    f"**{r['name']}** is exploratory: its carbs fall outside the "
                    f"training 1st\u201399th percentile range "
                    f"({CARBS_P01:.1f}\u2013{CARBS_P99:.1f} g). Treat this "
                    "projection with extra skepticism.")
            if r["macros_missing"]:
                st.warning(
                    f"**{r['name']}** is exploratory: no nutritional information "
                    "was provided, so its prediction is fully median-imputed. "
                    "Do not compare it as if it were equally reliable.")
            st.info(f"**#{i + 1} {r['name']}** \u2014 {r['explanation']}")
        st.caption("Each ranking call logs one prediction per candidate to the "
                   "twin database (predictions table) for provenance.")


# --------------------------------------------------------------------------- #
# Page: Twin history
# --------------------------------------------------------------------------- #
def page_history(art):
    st.header("Twin history")
    st.markdown("How this patient's twin has behaved over time: predictions vs "
                "what was actually observed, the update log, and drift status.")
    pid = int(st.selectbox("Patient", options=patient_pids(), key="hist_pid"))
    prof = get_profile(DB_PATH, pid)

    obs = query_df(
        "SELECT ts, iauc_obs, peak_obs, excursion_obs, source FROM observations"
        " WHERE pid = ? ORDER BY ts", (pid,))
    pred = query_df(
        "SELECT ts, pred_iauc, pred_peak, prob_excursion FROM predictions"
        " WHERE pid = ? ORDER BY ts", (pid,))
    paired = pred.merge(obs, on="ts", how="inner")

    st.subheader("Predicted vs observed iAUC")
    if paired.empty:
        st.info("No paired prediction/observation rows for this patient yet.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=paired["pred_iauc"], y=paired["iauc_obs"],
                                 mode="markers",
                                 text=paired["ts"],
                                 name="meals",
                                 marker=dict(color=paired["source"].map(
                                     {"observed": "#1f77b4",
                                      "simulated": "#ff7f0e"}))))
        lo = float(min(paired["pred_iauc"].min(), paired["iauc_obs"].min()))
        hi = float(max(paired["pred_iauc"].max(), paired["iauc_obs"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 name="perfect prediction",
                                 line=dict(dash="dash", color="gray")))
        fig.update_layout(title=f"Predicted vs observed iAUC \u2014 pid {pid}",
                          xaxis_title="predicted iAUC (mg/dL\u00b7h)",
                          yaxis_title="observed iAUC (mg/dL\u00b7h)",
                          margin=dict(l=40, r=20, t=40, b=40), height=380)
        st.plotly_chart(fig, use_container_width=True)

        ts = pd.to_datetime(paired["ts"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts, y=paired["pred_iauc"], mode="lines+markers",
                                  name="predicted iAUC"))
        fig2.add_trace(go.Scatter(x=ts, y=paired["iauc_obs"], mode="lines+markers",
                                  name="observed iAUC"))
        fig2.update_layout(title="Prediction and observation over time",
                           xaxis_title="meal time", yaxis_title="iAUC (mg/dL\u00b7h)",
                           margin=dict(l=40, r=20, t=40, b=40), height=340)
        st.plotly_chart(fig2, use_container_width=True)
        mae = float((paired["pred_iauc"] - paired["iauc_obs"]).abs().mean())
        st.caption(f"Paired meals: {len(paired)} \u2014 in-twin MAE "
                   f"{mae:.1f} mg/dL\u00b7h. (Demonstration replay of logged "
                   "meals, not an out-of-sample evaluation.) "
                   "Blue = **observed** meals (update the profile); orange = "
                   "**simulated** what-ifs (logged for provenance only).")

    st.subheader("Update log")
    log = query_df(
        "SELECT ts, residual, ewma_before, ewma_after, drift_flag FROM updates_log"
        " WHERE pid = ? ORDER BY ts", (pid,))
    if log.empty:
        st.info("No profile updates logged for this patient yet.")
    else:
        st.dataframe(log.round(2), use_container_width=True, hide_index=True)
        st.caption("residual = observed \u2212 predicted iAUC; EWMA \u03b1=0.3.")

    st.subheader("Drift status")
    bound = float(prof["drift_bound"])
    if prof["drift_flag"]:
        st.error(f"\U0001F6A9 Drift FLAGGED for pid {pid}.")
    else:
        st.success(f"No drift flagged for pid {pid}.")
    st.markdown(
        f"**Drift bound: {bound:.2f}** mg/dL\u00b7h (= 1.5 \u00d7 overall OOF "
        "iAUC MAE). In plain language: the twin watches the last "
        f"{DRIFT_WINDOW} meals' prediction errors. If their average size stays "
        f"above {bound:.2f} for {DRIFT_WINDOW} meals in a row, something "
        "systematic may have changed \u2014 the patient's physiology, their "
        "logging habits, or the sensor \u2014 and the twin flags it for human "
        "review. A single bad meal can never trigger it.")


# --------------------------------------------------------------------------- #
# Page: Evaluation
# --------------------------------------------------------------------------- #
def page_evaluation(art):
    st.header("Evaluation")
    st.markdown("Every number below comes from a project artifact "
                "(models/*.json, results/phase1/metrics.json) \u2014 nothing is "
                "invented. " + DISCLAIMER)
    m = art["metrics"]
    ci = art["metrics_ci"]
    b = art["baselines"]
    cls = m["classification"]

    st.subheader("Baseline comparison (iAUC MAE, time-ordered within-patient "
                 "evaluation)")
    base_rows = [
        ("A. Patient-mean", b["mae_patient_mean"], None,
         "predict test iAUC with the patient's own train mean"),
        ("B. Carb-only linear (per patient)", b["mae_carb_only_linear"], None,
         f"beats patient-mean in {b['frac_carb_beats_patient_mean']:.0%} of patients"),
        ("C. Population linear (LOPO, 6 features)", b["mae_population_linear"], None,
         f"beats patient-mean in {b['frac_pop_beats_patient_mean']:.0%} of patients"),
        ("D. Personal linear (per patient, 5 features)", b["mae_personal_linear"], None,
         f"beats population linear in {b['frac_personal_beats_population']:.0%} of patients"),
        ("LightGBM (population, OOF)", m["regression"]["iauc"]["mae"],
         (ci["mae_iauc"][0], ci["mae_iauc"][1]),
         "95% CI by patient bootstrap (2,000 resamples)"),
    ]
    df = pd.DataFrame([{
        "model": name,
        "iAUC MAE": f"{mae:.2f}",
        "95% CI": "not computed" if civ is None else f"{civ[0]:.2f}\u2013{civ[1]:.2f}",
        "note": note,
    } for name, mae, civ, note in base_rows])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Baseline MAEs are from the Phase-1 time-ordered evaluation "
               "(results/phase1/metrics.json); the LightGBM row is the OOF "
               "evaluation (models/metrics.json). Different evaluation schemes \u2014 "
               "compare with care.")

    st.subheader("Excursion classification (peak \u2265 180 mg/dL)")
    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC", f"{cls['pr_auc']:.3f}",
              f"CI {ci['pr_auc'][0]:.2f}\u2013{ci['pr_auc'][1]:.2f}")
    c2.metric("ROC-AUC", f"{cls['roc_auc']:.3f}",
              f"CI {ci['roc_auc'][0]:.2f}\u2013{ci['roc_auc'][1]:.2f}")
    c3.metric("Event rate", f"{cls['event_rate']:.1%}")
    st.write(f"At OOF max-F1 threshold **{cls['threshold_max_f1_oof']:.3f}**: "
             f"precision {cls['at_max_f1']['precision']:.3f}, recall "
             f"{cls['at_max_f1']['recall']:.3f}, F1 {cls['at_max_f1']['f1']:.3f}.")
    st.write(f"At 0.5: precision {cls['at_0_5']['precision']:.3f}, recall "
             f"{cls['at_0_5']['recall']:.3f}, F1 {cls['at_0_5']['f1']:.3f}. "
             f"Raw ECE {cls['ece_raw']:.3f} (calibrated \u2248 0 by construction).")

    st.subheader("Calibration curve")
    cal = art["calibration"]
    fig = go.Figure()
    for key, name in [("bins_raw", "raw"), ("bins_calibrated", "calibrated")]:
        bins = cal[key]
        x = [(bb["bin"][0] + bb["bin"][1]) / 2 for bb in bins]
        y = [bb["accuracy"] for bb in bins]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="perfect", line=dict(dash="dash", color="gray")))
    fig.update_layout(title="Reliability diagram (excursion classifier)",
                      xaxis_title="predicted probability",
                      yaxis_title="observed excursion rate",
                      margin=dict(l=40, r=20, t=40, b=40), height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("From models/calibration.json. Raw probabilities are "
               "miscalibrated (ECE 0.122); the calibrated curve is a fitted "
               "mapping, not new evidence.")

    st.subheader("Experiment E1 \u2014 personalization")
    e1 = art["personalization"]["summary"]
    st.info(f"**Verdict: {e1['verdict']}**\n\n"
            f"Mean \u0394MAE (personal \u2212 population): "
            f"{e1['mean_dMAE_personal_minus_pop']:.2f} mg/dL\u00b7h, 95% CI "
            f"[{e1['bootstrap_95ci_mean_dMAE'][0]:.2f}, "
            f"{e1['bootstrap_95ci_mean_dMAE'][1]:.2f}], win rate "
            f"{e1['win_rate']:.0%} over {e1['n_patients']} patients. "
            f"Decision rule: {e1['decision_rule']}.")
    st.caption("Caveat: E1 fine-tuning is evaluated but NOT wired into the "
               "dashboard prediction path, which uses the population models.")

    st.subheader("Experiment E2 \u2014 ranking")
    e2 = art["ranking"]["summary"]
    st.info(f"**Mean Spearman \u03c1 (predicted vs observed iAUC): "
            f"{e2['mean_spearman_pred']:.3f}** vs carb-only "
            f"{e2['mean_spearman_carb']:.3f}; the model ranking beats the "
            f"carb-only ranking in {e2['frac_pred_beats_carb']:.1%} of "
            f"{e2['n_patients']} patients.")

    st.subheader("Feature importance (split gain, iAUC model)")
    fi = art["feature_importance"]["lgbm_iauc.txt"]
    items = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:12]
    fig2 = go.Figure(go.Bar(x=[v for _, v in items][::-1],
                            y=[k for k, _ in items][::-1],
                            orientation="h"))
    fig2.update_layout(title="Top features by LightGBM split gain (iAUC)",
                       xaxis_title="total gain",
                       margin=dict(l=140, r=20, t=40, b=40), height=420)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Gain = total loss reduction attributed to splits on the feature. "
               "It is associational, not causal: a high-gain feature is "
               "predictive, not necessarily a lever that changes outcomes.")

    st.subheader("Limitations")
    st.markdown("""
- **Research prototype, not a medical device.** Nothing here is medical advice.
- Population-model OOF iAUC MAE \u224832 mg/dL\u00b7h \u2014 large relative to
  typical meal responses; intervals are wide (mean width \u224871).
- Excursion labels derive from the peak \u2265180 mg/dL rule, not adjudicated events.
- Interval lower bounds can go slightly negative; the app clips them at 0 for
  display only (the underlying interval is symmetric/additive).
- The seed replay is a demonstration of the state machine, not an evaluation
  (final boosters saw all usable meals).
- Personal fine-tuning (E1) is not in the prediction path.
- Demo candidate meals are illustrative inputs, not nutritional facts.
- Raw-data trend plots depend on the optional raw CGM directory.
""")


# --------------------------------------------------------------------------- #
# Page: Safety
# --------------------------------------------------------------------------- #
def page_safety(art):
    st.header("Safety & explainability")
    st.error("**" + DISCLAIMER + "** This dashboard is a research prototype for "
             "a health-data challenge. It must not be used for diagnosis, "
             "treatment, or any medical decision.")
    m = art["metrics"]

    st.subheader("What the model uses")
    st.markdown("""
The prediction is built from **25 features in 4 groups**, all available at or
before meal onset (no post-meal leakage):

- **Meal** \u2014 carbs, protein, fat, fiber, calories (from the meal log).
- **Pre-meal context** \u2014 pre-meal glucose (g0), 60-min glucose slope, mean
  heart rate and activity calories over the prior 2 h (CGM + wearable).
- **Time** \u2014 circadian encoding (hour sin/cos) and meal-type one-hots.
- **Static** \u2014 age, sex, BMI, HbA1c, fasting glucose/insulin, lipids
  (bio.csv, one row per patient).

Missing values are filled with medians computed on the **training** data only.
""")

    st.subheader("What feature importances mean (and don't)")
    st.markdown("""
The Evaluation page shows **LightGBM split gain**: how much each feature
reduced the training loss across all tree splits. This is **not SHAP** and it
is **not causal**. A high-gain feature (e.g. carbs) is *predictive* of the
response in this dataset; it does not prove that changing it changes any
individual's outcome, and gain can be inflated by correlated features.
Treat importances as descriptive, never prescriptive.
""")

    st.subheader("Associational, not causal")
    st.markdown("""
Every ranking explanation carries this sentence: *"Model projection
(associational, not causal) \u2014 research prototype, not medical advice."*
The models learn statistical associations between logged meals and glucose
traces. They do not model physiology, cannot account for unlogged factors
(stress, illness, medication, sleep), and must not be read as dietary guidance.
""")

    st.subheader("Uncertainty, honestly")
    st.markdown(f"""
- Each prediction ships a **90% interval** (q05\u2013q95 \u00b1 conformal
  adjustment {art['conformal']['adjustment']:.2f}). Its empirical out-of-fold
  coverage was {m['uncertainty']['empirical_coverage_oof']:.1%} with mean width
  {m['uncertainty']['mean_width']:.1f} mg/dL\u00b7h \u2014 wide, by design.
- The lower bound is **clipped at 0 for display only**; the stored interval is
  symmetric/additive and may extend slightly negative.
- The excursion classifier's raw probabilities are imperfectly calibrated
  (ECE {m['classification']['ece_raw']:.3f}); the dashboard shows the raw
  probability alongside the OOF max-F1 threshold
  ({m['classification']['threshold_max_f1_oof']:.3f}) and the 0.5 boundary.
- **Evidence labels** (high/medium/low) describe how much pre-meal context was
  available \u2014 they are about input completeness, not model confidence.
""")

    st.subheader("Drift & twin state")
    try:
        _bound = compute_drift_bound()
    except Exception:  # noqa: BLE001
        _bound = float("nan")
    st.markdown(f"""
The twin keeps honest per-patient state: observed-meal count, an EWMA
(\u03b1=0.3) of prediction residuals, a carbohydrate-sensitivity slope (needs
\u22653 observed meals), and a drift flag. Drift fires only when the rolling
mean |residual| over the last 5 meals exceeds **{_bound:.2f}** mg/dL\u00b7h
(1.5 \u00d7 overall OOF iAUC MAE, derived at runtime from
models/oof_predictions.csv) \u2014 a documented heuristic, not a clinical
threshold. **Simulated** what-if meals are logged for provenance but never
touch the profile; past predictions are immutable.
""")

    st.subheader("Data provenance")
    st.markdown(f"""
- Source: **CGMacros** continuous-glucose / meal / wearable dataset, DOI
  {DOI}. Processed meal table: data/processed/meals.csv
  (1,579 usable meals, 45 patients).
- **License note:** the dataset's redistribution/license terms were not
  re-verified while building this dashboard \u2014 consult the source
  repository (via the DOI) before any redistribution or public deployment.
- This dashboard runs fully offline: no data leaves the machine.
""")


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
PAGES = ["Overview", "Patient explorer", "Prediction", "Meal comparison",
         "Twin history", "Evaluation", "Safety"]


def main():
    st.set_page_config(page_title="AahaarTwin", layout="wide",
                       page_icon="\U0001F35A")
    st.sidebar.title("\U0001F35A AahaarTwin")
    st.sidebar.caption("Personalized meal-response digital twin "
                       "\u2014 research prototype")
    page = st.sidebar.radio("Page", PAGES)

    if not os.path.exists(DB_PATH):
        # Resilient startup: a fresh checkout has no twin_state.db (it is
        # git-ignored). Seed it on the fly from the committed processed meals
        # instead of erroring out.
        with st.spinner("Twin database not found — seeding from "
                        "data/processed/meals.csv (one-time, may take a "
                        "minute)..."):
            try:
                summary = seed_from_meals(DB_PATH)
            except Exception as exc:  # noqa: BLE001 - surface seed failure
                st.error("Auto-seeding twin_state.db failed: %s" % exc)
                st.stop()
        st.sidebar.success("Twin DB auto-seeded: %d patients, %d meals."
                           % (summary["n_patients"], summary["n_meals"]))
    try:
        art = load_artifacts()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.caption("Artifacts: models/*.json \u2014 all metrics shown are "
                       "loaded, never invented.")
    st.sidebar.caption(DISCLAIMER)

    if page == "Overview":
        page_overview(art)
    elif page == "Patient explorer":
        page_patient(art)
    elif page == "Prediction":
        page_prediction(art)
    elif page == "Meal comparison":
        page_comparison(art)
    elif page == "Twin history":
        page_history(art)
    elif page == "Evaluation":
        page_evaluation(art)
    elif page == "Safety":
        page_safety(art)


if __name__ == "__main__":
    main()
