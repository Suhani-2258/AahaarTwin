"""AahaarTwin Phase 3 — stateful digital twin (SQLite-backed per-patient profile).

Research prototype only — not a medical device; not for diagnosis or treatment.

A per-patient *response profile* is maintained in SQLite: how many real meals have
been observed, a running EWMA of prediction residuals, a per-patient
carbohydrate-sensitivity estimate, and a drift flag. Predictions come from the
population LightGBM models (models/lgbm_*.txt); the profile is HONEST state —
a prediction for meal t is computed before the observation at t is recorded, so
no future information ever leaks into past predictions.

Conventions
-----------
- Timestamps are stored as 'YYYY-MM-DD HH:MM:SS' strings.
- pid is an integer patient id (de-identified).
- `source` is 'observed' (a real logged meal) or 'simulated' (a what-if).
  ONLY 'observed' rows update the profile; 'simulated' rows are logged for
  provenance but never touch n_meals_observed / ewma / sensitivity / drift.
- `record_observation` NEVER modifies rows in `predictions` — past predictions
  are immutable.
"""

import json
import math
import os
import sqlite3

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, STATIC_FEATURES, load_bio

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJ, "models")
BIO_CSV = os.path.join(PROJ, "..", "feas-data", "cgmacros_raw", "CGMacros", "bio.csv")
MEALS_CSV = os.path.join(PROJ, "data", "processed", "meals.csv")
OOF_CSV = os.path.join(MODELS_DIR, "oof_predictions.csv")
CONFORMAL_JSON = os.path.join(MODELS_DIR, "conformal.json")
MEDIANS_JSON = os.path.join(MODELS_DIR, "imputer_medians.json")

MODEL_VERSION = "lgbm-population-v1"
EWMA_ALPHA = 0.3
DRIFT_WINDOW = 5
PRE_FEATURES = ["g0", "slope_60min", "hr_mean_2h", "act_kcal_2h"]
MEAL_FEATURES = ["carbs", "protein", "fat", "fiber", "calories"]
EXCURSION_PEAK_THRESHOLD = 180.0  # mg/dL — matches the Phase-1 label rule (peak >= 180)

# Module-level lazy caches (models, bio table, imputation medians, conformal adj).
_BOOSTERS = None
_BIO = None
_MEDIANS = None
_CONFORMAL_ADJ = None
_DRIFT_BOUND = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm_ts(ts_iso):
    """Normalize any timestamp input to 'YYYY-MM-DD HH:MM:SS'."""
    return pd.Timestamp(ts_iso).strftime("%Y-%m-%d %H:%M:%S")


def _num(v):
    """Coerce to float; None/NaN -> np.nan."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f


def _connect(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _boosters():
    global _BOOSTERS
    if _BOOSTERS is None:
        _BOOSTERS = {
            key: lgb.Booster(model_file=os.path.join(MODELS_DIR, f"lgbm_{key}.txt"))
            for key in ["iauc", "peak", "excursion", "iauc_q05", "iauc_q95"]
        }
    return _BOOSTERS


def _bio():
    global _BIO
    if _BIO is None:
        _BIO = load_bio(BIO_CSV)
    return _BIO


def _medians():
    global _MEDIANS
    if _MEDIANS is None:
        _MEDIANS = pd.Series(json.load(open(MEDIANS_JSON)))
    return _MEDIANS


def _conformal_adj():
    global _CONFORMAL_ADJ
    if _CONFORMAL_ADJ is None:
        _CONFORMAL_ADJ = float(json.load(open(CONFORMAL_JSON))["adjustment"])
    return _CONFORMAL_ADJ


def compute_drift_bound(oof_csv=None):
    """Derive the drift-detection bound from out-of-fold performance.

    bound = 1.5 x (overall OOF iAUC MAE), computed from
    ``models/oof_predictions.csv`` at runtime.

    Rationale (documented choice, not a clinical threshold): the population
    model makes typical errors of ~1x OOF MAE on unseen meals. A *sustained*
    rolling mean |residual| above 1.5x that level over 5 consecutive meals is
    unlikely to be ordinary model noise and signals a systematic shift
    (changed physiology, logging change, or sensor drift) worth flagging for
    review. The flag requires a full 5-residual window so a single bad meal
    cannot trigger it.
    """
    oof = pd.read_csv(oof_csv or OOF_CSV)
    mae = float((oof["pred_iauc"] - oof["y_iauc"]).abs().mean())
    return 1.5 * mae


def _drift_bound_cached():
    global _DRIFT_BOUND
    if _DRIFT_BOUND is None:
        _DRIFT_BOUND = compute_drift_bound()
    return _DRIFT_BOUND


def _feature_row(pid, ts, meal_type, macros, premeal):
    """Build one raw (pre-imputation) feature dict in FEATURES order."""
    macros = macros or {}
    premeal = premeal or {}
    t = pd.Timestamp(ts)
    hour = t.hour + t.minute / 60.0
    bio = _bio()
    static = bio.loc[pid] if pid in bio.index else None

    row = {}
    for f in MEAL_FEATURES:
        row[f] = _num(macros.get(f))
    for f in PRE_FEATURES:
        row[f] = _num(premeal.get(f))
    row["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    row["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    for mt in ["breakfast", "lunch", "dinner", "snack"]:
        row[f"meal_type_{mt}"] = 1.0 if meal_type == mt else 0.0
    for f in STATIC_FEATURES:
        v = np.nan
        if static is not None:
            v = _num(static.get(f, np.nan))
        row[f] = v
    return row


def _impute(X):
    return X.fillna(_medians())


def _run_boosters(X):
    """Run the 5 app boosters on an imputed feature frame (FEATURES order).

    Returns dict of numpy arrays, one value per row.
    """
    b = _boosters()
    return {
        "iauc": b["iauc"].predict(X),
        "peak": b["peak"].predict(X),
        "prob_excursion": b["excursion"].predict(X),  # P(excursion), binary model
        "q05": b["iauc_q05"].predict(X),
        "q95": b["iauc_q95"].predict(X),
    }


def _evidence(premeal):
    """Evidence score = fraction of the 4 pre-meal context features present."""
    premeal = premeal or {}
    present = sum(1 for f in PRE_FEATURES
                  if premeal.get(f) is not None and not pd.isna(premeal.get(f)))
    score = present / len(PRE_FEATURES)
    if score == 1.0:
        label = "high"
    elif score >= 0.75:
        label = "medium"
    else:
        label = "low"
    return score, label


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
  pid INTEGER PRIMARY KEY,
  n_meals_observed INTEGER NOT NULL DEFAULT 0,
  carb_sensitivity REAL,
  ewma_residual REAL,
  last_update_ts TEXT,
  drift_flag INTEGER NOT NULL DEFAULT 0,
  drift_bound REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pid INTEGER NOT NULL,
  ts TEXT NOT NULL,
  meal_type TEXT,
  carbs REAL, protein REAL, fat REAL, fiber REAL, calories REAL,
  iauc_obs REAL, peak_obs REAL, excursion_obs INTEGER,
  source TEXT NOT NULL CHECK (source IN ('observed','simulated'))
);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pid INTEGER NOT NULL,
  ts TEXT NOT NULL,
  pred_iauc REAL, pred_peak REAL, prob_excursion REAL,
  iauc_lo REAL, iauc_hi REAL,
  evidence_score REAL,
  model_version TEXT,
  note TEXT
);
CREATE TABLE IF NOT EXISTS updates_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pid INTEGER NOT NULL,
  ts TEXT NOT NULL,
  residual REAL,
  ewma_before REAL, ewma_after REAL,
  drift_flag INTEGER
);
CREATE INDEX IF NOT EXISTS idx_obs_pid_ts ON observations(pid, ts);
CREATE INDEX IF NOT EXISTS idx_pred_pid_ts ON predictions(pid, ts);
CREATE INDEX IF NOT EXISTS idx_upd_pid_ts ON updates_log(pid, ts);
"""


def init_db(db_path):
    """Create the twin-state schema. Returns the drift bound used."""
    bound = _drift_bound_cached()
    con = _connect(db_path)
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()
    return bound


def _ensure_profile(con, pid):
    row = con.execute("SELECT * FROM profiles WHERE pid = ?", (pid,)).fetchone()
    if row is None:
        con.execute(
            "INSERT INTO profiles (pid, n_meals_observed, drift_flag, drift_bound)"
            " VALUES (?, 0, 0, ?)",
            (pid, _drift_bound_cached()),
        )
        con.commit()
        row = con.execute("SELECT * FROM profiles WHERE pid = ?", (pid,)).fetchone()
    return row


def get_profile(db_path, pid):
    """Return the patient's response profile as a dict (creates it if missing)."""
    con = _connect(db_path)
    try:
        row = _ensure_profile(con, pid)
        return {
            "pid": row["pid"],
            "n_meals_observed": row["n_meals_observed"],
            "carb_sensitivity": row["carb_sensitivity"],
            "ewma_residual": row["ewma_residual"],
            "last_update_ts": row["last_update_ts"],
            "drift_flag": row["drift_flag"],
            "drift_bound": row["drift_bound"],
        }
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _assemble_prediction(pid, ts, meal_type, outs, evidence_score, evidence_label, profile):
    adj = _conformal_adj()
    iauc = float(outs["iauc"])
    return {
        "pid": pid,
        "ts": ts,
        "meal_type": meal_type,
        "pred_iauc": iauc,
        "pred_peak": float(outs["peak"]),
        "prob_excursion": float(outs["prob_excursion"]),
        "iauc_lo": float(outs["q05"]) - adj,
        "iauc_hi": float(outs["q95"]) + adj,
        "evidence_score": evidence_score,
        "evidence_label": evidence_label,
        "model_version": MODEL_VERSION,
        # Profile snapshot at prediction time — provenance, not an input.
        "n_meals_observed": profile["n_meals_observed"],
        "ewma_residual": profile["ewma_residual"],
        "drift_flag": profile["drift_flag"],
    }


def _log_prediction(con, pred):
    con.execute(
        "INSERT INTO predictions (pid, ts, pred_iauc, pred_peak, prob_excursion,"
        " iauc_lo, iauc_hi, evidence_score, model_version, note)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (pred["pid"], pred["ts"], pred["pred_iauc"], pred["pred_peak"],
         pred["prob_excursion"], pred["iauc_lo"], pred["iauc_hi"],
         pred["evidence_score"], pred["model_version"],
         f"evidence:{pred['evidence_label']}"),
    )


def predict_meal(db_path, pid, ts_iso, meal_type, macros, premeal):
    """Predict post-meal response for one meal and log the prediction.

    Builds the 25-feature vector (meal macros, pre-meal CGM/wearable context,
    circadian + meal-type time features, static bio vars), median-imputes
    missing values, runs the 5 app boosters, and applies the conformal
    adjustment to the iAUC interval.

    The prediction uses only the POPULATION models plus static bio vars — the
    profile state is attached for provenance only. It is logged to the
    `predictions` table; later observations never modify this row.
    """
    con = _connect(db_path)
    try:
        _ensure_profile(con, pid)
        ts = _norm_ts(ts_iso)
        X = _impute(pd.DataFrame([_feature_row(pid, ts, meal_type, macros, premeal)],
                                 columns=FEATURES))
        outs = {k: float(v[0]) for k, v in _run_boosters(X).items()}
        score, label = _evidence(premeal)
        profile = get_profile(db_path, pid)
        pred = _assemble_prediction(pid, ts, meal_type, outs, score, label, profile)
        _log_prediction(con, pred)
        con.commit()
        return pred
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Observation recording + profile update
# --------------------------------------------------------------------------- #
def _update_profile_from_observations(con, pid, ts):
    """Recompute n_meals_observed and carb_sensitivity from observed rows.

    carb_sensitivity = least-squares slope of iauc_obs ~ carbs over the
    patient's observed meals; NULL until >= 3 observed meals (or zero carb
    variance).
    """
    rows = con.execute(
        "SELECT carbs, iauc_obs FROM observations"
        " WHERE pid = ? AND source = 'observed'"
        "   AND carbs IS NOT NULL AND iauc_obs IS NOT NULL"
        " ORDER BY ts",
        (pid,),
    ).fetchall()
    n = len(rows)
    sens = None
    if n >= 3:
        carbs = np.array([r["carbs"] for r in rows], dtype=float)
        iauc = np.array([r["iauc_obs"] for r in rows], dtype=float)
        if np.var(carbs) > 0:
            sens = float(np.polyfit(carbs, iauc, 1)[0])
    con.execute(
        "UPDATE profiles SET n_meals_observed = ?, carb_sensitivity = ?,"
        " last_update_ts = ? WHERE pid = ?",
        (n, sens, ts, pid),
    )


def _evaluate_drift(con, pid):
    """Rolling mean |residual| over the last DRIFT_WINDOW logged residuals.

    Requires a full window (a single bad meal must not trigger drift).
    Returns 1 if the rolling mean exceeds the profile's drift_bound, else 0.
    """
    rows = con.execute(
        "SELECT residual FROM updates_log WHERE pid = ? ORDER BY id DESC LIMIT ?",
        (pid, DRIFT_WINDOW),
    ).fetchall()
    if len(rows) < DRIFT_WINDOW:
        return 0
    mean_abs = float(np.mean([abs(r["residual"]) for r in rows]))
    prof = con.execute("SELECT drift_bound FROM profiles WHERE pid = ?",
                       (pid,)).fetchone()
    return 1 if mean_abs > prof["drift_bound"] else 0


def record_observation(db_path, pid, ts_iso, meal_type, macros, iauc_obs, peak_obs,
                       source="observed", prediction_ref=None):
    """Record a meal observation and update the patient profile.

    - Inserts into `observations` (excursion_obs derived as peak_obs >= 180,
      matching the Phase-1 label rule; NULL when peak_obs is missing).
    - Only `source='observed'` rows update the profile; 'simulated' rows are
      logged for provenance and leave the profile untouched.
    - If a prediction exists for the same (pid, ts), residual = obs - pred is
      computed, the EWMA (alpha=0.3) is updated, drift is evaluated, and the
      update is appended to `updates_log`.
    - Past `predictions` rows are NEVER modified.
    """
    if source not in ("observed", "simulated"):
        raise ValueError(f"source must be 'observed' or 'simulated', got {source!r}")
    macros = macros or {}
    ts = _norm_ts(ts_iso)
    excursion_obs = None
    if peak_obs is not None and not pd.isna(peak_obs):
        excursion_obs = int(float(peak_obs) >= EXCURSION_PEAK_THRESHOLD)

    con = _connect(db_path)
    try:
        _ensure_profile(con, pid)
        con.execute(
            "INSERT INTO observations (pid, ts, meal_type, carbs, protein, fat,"
            " fiber, calories, iauc_obs, peak_obs, excursion_obs, source)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, ts, meal_type, _num(macros.get("carbs")), _num(macros.get("protein")),
             _num(macros.get("fat")), _num(macros.get("fiber")), _num(macros.get("calories")),
             _num(iauc_obs), _num(peak_obs), excursion_obs, source),
        )
        result = {"pid": pid, "ts": ts, "source": source, "matched": False}

        if source != "observed":
            con.commit()
            return result  # simulated what-ifs never touch the profile

        _update_profile_from_observations(con, pid, ts)

        pred = con.execute(
            "SELECT pred_iauc FROM predictions WHERE pid = ? AND ts = ?"
            " ORDER BY id DESC LIMIT 1",
            (pid, ts),
        ).fetchone()
        if pred is not None and iauc_obs is not None and not pd.isna(iauc_obs):
            residual = float(iauc_obs) - float(pred["pred_iauc"])
            prof = con.execute("SELECT ewma_residual FROM profiles WHERE pid = ?",
                               (pid,)).fetchone()
            ewma_before = prof["ewma_residual"]
            ewma_after = (residual if ewma_before is None
                          else EWMA_ALPHA * residual + (1 - EWMA_ALPHA) * ewma_before)
            con.execute("UPDATE profiles SET ewma_residual = ? WHERE pid = ?",
                        (ewma_after, pid))
            con.execute(
                "INSERT INTO updates_log (pid, ts, residual, ewma_before, ewma_after)"
                " VALUES (?,?,?,?,?)",
                (pid, ts, residual, ewma_before, ewma_after),
            )
            # Evaluate drift AFTER the new residual is logged, so the rolling
            # window includes it.
            drift_flag = _evaluate_drift(con, pid)
            con.execute("UPDATE profiles SET drift_flag = ? WHERE pid = ?",
                        (drift_flag, pid))
            con.execute("UPDATE updates_log SET drift_flag = ?"
                        " WHERE pid = ? AND ts = ?",
                        (drift_flag, pid, ts))
            result.update({
                "matched": True,
                "residual": residual,
                "ewma_before": ewma_before,
                "ewma_after": ewma_after,
                "drift_flag": drift_flag,
            })
        con.commit()
        prof = get_profile(db_path, pid)
        result.update({
            "n_meals_observed": prof["n_meals_observed"],
            "carb_sensitivity": prof["carb_sensitivity"],
        })
        return result
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Demo seeding: honest chronological replay
# --------------------------------------------------------------------------- #
def seed_from_meals(db_path, max_meals_per_patient=None):
    """Seed the twin DB by replaying each patient's usable meals in order.

    DEMONSTRATION mechanism (honest replay): for each patient, meals are
    processed in chronological order; each meal is predicted FIRST (using the
    population models plus profile state built from PRIOR meals only) and then
    recorded as an 'observed' observation, which updates the profile. No
    prediction ever sees its own outcome.
    """
    init_db(db_path)
    meals = pd.read_csv(MEALS_CSV, parse_dates=["ts"])
    usable = meals[meals["usable"].astype(bool)].sort_values(["pid", "ts"])

    _boosters(); _bio(); _medians()  # warm caches
    bound = _drift_bound_cached()
    summary = {"n_patients": 0, "n_meals": 0, "n_predictions": 0, "n_updates": 0,
               "drift_bound": bound}

    con = _connect(db_path)
    try:
        for pid, g in usable.groupby("pid", sort=True):
            if max_meals_per_patient is not None:
                g = g.head(max_meals_per_patient)
            _ensure_profile(con, pid)
            summary["n_patients"] += 1

            # Batch booster inference for this patient's meals (one call each).
            vecs = []
            for r in g.itertuples():
                macros = {f: getattr(r, f) for f in MEAL_FEATURES}
                premeal = {f: getattr(r, f) for f in PRE_FEATURES}
                vecs.append(_feature_row(pid, _norm_ts(r.ts), r.meal_type, macros, premeal))
            X = _impute(pd.DataFrame(vecs, columns=FEATURES))
            b = _boosters()
            outs = {
                "iauc": b["iauc"].predict(X),
                "peak": b["peak"].predict(X),
                "prob_excursion": b["excursion"].predict(X),
                "q05": b["iauc_q05"].predict(X),
                "q95": b["iauc_q95"].predict(X),
            }

            for i, r in enumerate(g.itertuples()):
                ts = _norm_ts(r.ts)
                premeal = {f: getattr(r, f) for f in PRE_FEATURES}
                score, label = _evidence(premeal)
                profile = get_profile(db_path, pid)
                pred = _assemble_prediction(
                    pid, ts, r.meal_type,
                    {k: float(v[i]) for k, v in outs.items()},
                    score, label, profile,
                )
                _log_prediction(con, pred)
                summary["n_predictions"] += 1

                # Record as observed (the replay's honest observation step).
                macros = {f: getattr(r, f) for f in MEAL_FEATURES}
                exc = int(r.peak_2h >= EXCURSION_PEAK_THRESHOLD)
                con.execute(
                    "INSERT INTO observations (pid, ts, meal_type, carbs, protein,"
                    " fat, fiber, calories, iauc_obs, peak_obs, excursion_obs, source)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, ts, r.meal_type, float(r.carbs), float(r.protein),
                     float(r.fat), float(r.fiber), float(r.calories),
                     float(r.iauc_2h), float(r.peak_2h), exc, "observed"),
                )
                summary["n_meals"] += 1

                # Profile update (same logic as record_observation, inlined for
                # the batch path; prediction rows are only ever INSERTed).
                _update_profile_from_observations(con, pid, ts)
                residual = float(r.iauc_2h) - pred["pred_iauc"]
                prof = con.execute("SELECT ewma_residual, drift_bound FROM profiles"
                                   " WHERE pid = ?", (pid,)).fetchone()
                ewma_before = prof["ewma_residual"]
                ewma_after = (residual if ewma_before is None
                              else EWMA_ALPHA * residual + (1 - EWMA_ALPHA) * ewma_before)
                con.execute("UPDATE profiles SET ewma_residual = ? WHERE pid = ?",
                            (ewma_after, pid))
                con.execute(
                    "INSERT INTO updates_log (pid, ts, residual, ewma_before,"
                    " ewma_after) VALUES (?,?,?,?,?)",
                    (pid, ts, residual, ewma_before, ewma_after),
                )
                # Evaluate drift AFTER the new residual is logged, so the rolling
                # window includes it.
                drift_flag = _evaluate_drift(con, pid)
                con.execute("UPDATE profiles SET drift_flag = ? WHERE pid = ?",
                            (drift_flag, pid))
                con.execute("UPDATE updates_log SET drift_flag = ?"
                            " WHERE pid = ? AND ts = ?",
                            (drift_flag, pid, ts))
                summary["n_updates"] += 1
            con.commit()
    finally:
        con.close()
    return summary


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "twin_state.db")
    print("seeding", out)
    print(json.dumps(seed_from_meals(out), indent=2))
