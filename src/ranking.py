"""AahaarTwin Phase 4 — meal comparison / ranking.

Research prototype only — not a medical device; not for diagnosis or treatment.

Given a list of CANDIDATE meals supplied by the caller (the user's own input or
the patient's own logged meals — this module invents NO foods), rank them by
predicted 2h glucose load (ascending) using a caller-provided prediction
function. Each candidate gets a plain-language explanation built ONLY from
model outputs plus an out-of-distribution flag; every explanation carries the
associational/prototype disclaimer.

``predict_fn`` contract: predict_fn(ts_iso, meal_type, macros, premeal) -> dict
with keys pred_iauc, pred_peak, prob_excursion, iauc_lo, iauc_hi,
evidence_label. (``src.twin_state.predict_meal`` bound to a patient satisfies
this contract.)
"""

import os

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEALS_CSV = os.path.join(PROJ, "data", "processed", "meals.csv")

DISCLAIMER = ("Model projection (associational, not causal) — research prototype,"
              " not medical advice.")

MACRO_KEYS = ["carbs", "protein", "fat", "fiber", "calories"]
PRE_KEYS = ["g0", "slope_60min", "hr_mean_2h", "act_kcal_2h"]


def _training_carbs_range():
    """1st and 99th percentiles of usable carbs in the training meal table."""
    m = pd.read_csv(MEALS_CSV)
    u = m[m["usable"].astype(bool)]["carbs"]
    return float(u.quantile(0.01)), float(u.quantile(0.99))


CARBS_P01, CARBS_P99 = _training_carbs_range()


def _is_ood(carbs):
    """True when carbs fall outside the training 1st-99th percentile range."""
    if carbs is None or (isinstance(carbs, float) and np.isnan(carbs)):
        return False
    return float(carbs) < CARBS_P01 or float(carbs) > CARBS_P99


def _macros_missing(macros):
    """True when EVERY macro value is missing (None/NaN).

    Such a candidate's prediction is fully median-imputed by the prediction
    function; it must not be presented as equally reliable as a candidate
    with real nutritional information.
    """
    return all(v is None or (isinstance(v, float) and np.isnan(v))
               for v in macros.values())


def _explanation(rank, n, name, pred, ood_flag, macros_missing=False):
    """Plain-language explanation built ONLY from model outputs."""
    iauc = pred["pred_iauc"]
    risk = 100.0 * pred["prob_excursion"]
    lo, hi = pred["iauc_lo"], pred["iauc_hi"]
    if rank == 1:
        lead = (f"Ranked #1 of {n}: lowest predicted 2h glucose load"
                f" ({iauc:.1f} mg/dL\u00b7h) and lowest excursion risk ({risk:.1f}%).")
    else:
        lead = (f"Ranked #{rank} of {n}: predicted 2h glucose load"
                f" ({iauc:.1f} mg/dL\u00b7h), excursion risk ({risk:.1f}%) —"
                f" higher predicted load than #{rank - 1}.")
    parts = [lead, f"90% interval [{lo:.1f}, {hi:.1f}]."]
    if ood_flag:
        parts.append("exploratory \u2014 outside observed data range"
                     " (carbs outside the training 1st\u201399th percentile).")
    if macros_missing:
        parts.append("exploratory \u2014 no nutritional information was provided,"
                     " so this prediction is fully median-imputed and should not"
                     " be compared as if it were equally reliable.")
    parts.append(DISCLAIMER)
    return " ".join(parts)


def rank_meals(predict_fn, premeal_context, candidates):
    """Rank candidate meals by predicted 2h glucose load (ascending).

    predict_fn      : callable(ts_iso, meal_type, macros, premeal) -> pred dict.
    premeal_context : dict with optional g0, slope_60min, hr_mean_2h,
                      act_kcal_2h and optional ts_iso (defaults to now).
    candidates      : list of dicts {name, meal_type, carbs, protein, fat,
                      fiber, calories}; missing macros are NaN-imputed by the
                      prediction function.

    Returns a list of dicts {name, meal_type, pred_iauc, pred_peak,
    prob_excursion, iauc_lo, iauc_hi, evidence_label, ood_flag,
    macros_missing, explanation},
    sorted by pred_iauc ascending. Empty input -> [].
    """
    premeal_context = premeal_context or {}
    ts_iso = premeal_context.get("ts_iso")
    if ts_iso is None:
        ts_iso = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    premeal = {k: premeal_context.get(k) for k in PRE_KEYS}

    scored = []
    for c in candidates or []:
        macros = {k: c.get(k) for k in MACRO_KEYS}
        pred = predict_fn(ts_iso, c.get("meal_type", "snack"), macros, premeal)
        ood = _is_ood(c.get("carbs"))
        no_macros = _macros_missing(macros)
        scored.append({
            "name": c.get("name"),
            "meal_type": c.get("meal_type"),
            "pred_iauc": float(pred["pred_iauc"]),
            "pred_peak": float(pred["pred_peak"]),
            "prob_excursion": float(pred["prob_excursion"]),
            "iauc_lo": float(pred["iauc_lo"]),
            "iauc_hi": float(pred["iauc_hi"]),
            "evidence_label": pred["evidence_label"],
            "ood_flag": ood,
            "macros_missing": no_macros,
        })
    scored.sort(key=lambda r: r["pred_iauc"])  # stable: ties keep input order
    n = len(scored)
    for i, r in enumerate(scored):
        r["explanation"] = _explanation(i + 1, n, r["name"], r, r["ood_flag"],
                                        macros_missing=r["macros_missing"])
    return scored
