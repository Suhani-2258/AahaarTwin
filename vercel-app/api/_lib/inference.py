"""Real-model inference for the AahaarTwin Vercel deployment.

Reimplements the prediction + ranking pipeline from src/twin_state.py and
src/ranking.py WITHOUT pandas, so it runs in a lightweight serverless
environment. Feature order, imputation medians, conformal adjustment and the
five LightGBM boosters are loaded from the same committed artifacts used by
the local app — predictions are produced by the real models, never mocked.

Parity is enforced by tests/vercel/test_inference_parity.py, which compares
this module's outputs against src.twin_state.predict_meal on fixed inputs.
"""

import json
import math
import os

import numpy as np
import lightgbm as lgb

_BASE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_BASE, "models")

MEAL_KEYS = ["carbs", "protein", "fat", "fiber", "calories"]
PRE_KEYS = ["g0", "slope_60min", "hr_mean_2h", "act_kcal_2h"]
STATIC_KEYS = [
    "age", "gender_m", "bmi", "hba1c", "fasting_glu",
    "fasting_insulin", "trig", "chol", "hdl", "ldl",
]
MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"]

MODEL_VERSION = "lgbm-population-v1"
# Carb p01/p99 of the training meals (computed from data/processed/meals.csv
# at build time; used for the out-of-distribution warning). See ranking.py.
CARBS_P01 = 0.0
CARBS_P99 = 148.44

_cache = {}


def _load():
    if "boosters" not in _cache:
        _cache["boosters"] = {
            key: lgb.Booster(model_file=os.path.join(_MODELS, f"lgbm_{key}.txt"))
            for key in ("iauc", "peak", "excursion", "iauc_q05", "iauc_q95")
        }
        with open(os.path.join(_MODELS, "feature_order.json")) as fh:
            _cache["order"] = json.load(fh)
        with open(os.path.join(_MODELS, "imputer_medians.json")) as fh:
            _cache["medians"] = json.load(fh)
        with open(os.path.join(_MODELS, "conformal.json")) as fh:
            _cache["adj"] = float(json.load(fh)["adjustment"])
        with open(os.path.join(_MODELS, "patient_statics.json")) as fh:
            _cache["statics"] = json.load(fh)
    return _cache


def _num(v):
    """Coerce to float; return None for missing/non-numeric/NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _hour_of_day(ts_iso):
    # Accept "YYYY-MM-DDTHH:MM" (or with seconds); mirrors pandas .dt.hour/.minute.
    t = ts_iso.replace("Z", "")
    date_part, _, time_part = t.partition("T")
    if not time_part:
        date_part, _, time_part = t.partition(" ")
    hh, mm = (time_part.split(":") + ["0", "0"])[:2]
    return float(hh) + float(mm) / 60.0


def build_features(pid, ts_iso, meal_type, macros, premeal, statics=None):
    """Build the 25-feature model row (median-imputed), as np.ndarray shape (1, 25)."""
    c = _load()
    macros = macros or {}
    premeal = premeal or {}
    hour = _hour_of_day(ts_iso)
    row = {}
    for k in MEAL_KEYS:
        row[k] = _num(macros.get(k))
    for k in PRE_KEYS:
        row[k] = _num(premeal.get(k))
    row["hour_sin"] = math.sin(2.0 * math.pi * hour / 24.0)
    row["hour_cos"] = math.cos(2.0 * math.pi * hour / 24.0)
    for mt in MEAL_TYPES:
        row[f"meal_type_{mt}"] = 1.0 if meal_type == mt else 0.0
    if statics is None:
        statics = c["statics"].get(str(pid), {})
    for k in STATIC_KEYS:
        row[k] = _num(statics.get(k))
    order, medians = c["order"], c["medians"]
    x = np.array(
        [row[f] if row[f] is not None else medians[f] for f in order],
        dtype=np.float64,
    ).reshape(1, -1)
    return x


def _evidence(premeal):
    present = sum(1 for k in PRE_KEYS if _num((premeal or {}).get(k)) is not None)
    score = present / len(PRE_KEYS)
    label = "high" if score == 1.0 else ("medium" if score >= 0.75 else "low")
    return score, label


def predict_one(pid, ts_iso, meal_type, macros, premeal=None, statics=None):
    """Full prediction dict for one candidate meal (mirrors twin_state.predict_meal)."""
    c = _load()
    x = build_features(pid, ts_iso, meal_type, macros, premeal, statics)
    b = c["boosters"]
    iauc = float(b["iauc"].predict(x)[0])
    peak = float(b["peak"].predict(x)[0])
    prob = float(b["excursion"].predict(x)[0])
    q05 = float(b["iauc_q05"].predict(x)[0])
    q95 = float(b["iauc_q95"].predict(x)[0])
    adj = c["adj"]
    score, label = _evidence(premeal)
    macros = macros or {}
    carbs = _num(macros.get("carbs"))
    ood = carbs is not None and (carbs < CARBS_P01 or carbs > CARBS_P99)
    return {
        "pid": int(pid),
        "ts": ts_iso,
        "meal_type": meal_type,
        "pred_iauc": iauc,
        "pred_peak": peak,
        "prob_excursion": prob,
        "iauc_lo": q05 - adj,
        "iauc_hi": q95 + adj,
        "evidence_score": score,
        "evidence_label": label,
        "ood_flag": bool(ood),
        "model_version": MODEL_VERSION,
    }


DISCLAIMER = (
    "Model projection (associational, not causal) — research prototype,"
    " not medical advice."
)


def _explanation(rank, n, p):
    """Plain-language explanation built ONLY from model outputs (mirrors ranking._explanation)."""
    iauc = p["pred_iauc"]
    risk = 100.0 * p["prob_excursion"]
    lo, hi = p["iauc_lo"], p["iauc_hi"]
    if rank == 1:
        lead = (f"Ranked #1 of {n}: lowest predicted 2h glucose load"
                f" ({iauc:.1f} mg/dL·h) and lowest excursion risk ({risk:.1f}%).")
    else:
        lead = (f"Ranked #{rank} of {n}: predicted 2h glucose load"
                f" ({iauc:.1f} mg/dL·h), excursion risk ({risk:.1f}%) —"
                f" higher predicted load than #{rank - 1}.")
    parts = [lead, f"90% interval [{lo:.1f}, {hi:.1f}]."]
    if p["ood_flag"]:
        parts.append("exploratory — outside observed data range"
                     " (carbs outside the training 1st–99th percentile).")
    if p["macros_missing"]:
        parts.append("exploratory — no nutritional information was provided,"
                     " so this prediction is fully median-imputed and should not"
                     " be compared as if it were equally reliable.")
    parts.append(DISCLAIMER)
    return " ".join(parts)


def rank_candidates(pid, ts_iso, meal_type, candidates, premeal=None, statics=None):
    """Rank candidate meals by predicted iAUC (mirrors ranking.rank_meals)."""
    scored = []
    for cand in candidates:
        macros = {
            k: _num((cand.get("macros") or {}).get(k)) for k in MEAL_KEYS
        }
        p = predict_one(pid, ts_iso, meal_type, macros, premeal, statics)
        macros_missing = all(v is None for v in macros.values())
        p["macros"] = macros
        p["macros_missing"] = macros_missing
        p["label"] = cand.get("label", "candidate")
        scored.append(p)
    scored.sort(key=lambda r: (r["pred_iauc"], r["prob_excursion"]))
    n = len(scored)
    for i, r in enumerate(scored, 1):
        r["rank"] = i
        r["explanation"] = _explanation(i, n, r)
    return scored
