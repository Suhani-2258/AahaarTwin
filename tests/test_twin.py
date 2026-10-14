"""Phase 3 (stateful digital twin) tests. Run with: python3 tests/test_twin.py

Each test raises AssertionError with a clear message on failure.
Uses throwaway SQLite DBs under /tmp; the seeded project DB is untouched.
"""
import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from twin_state import (init_db, get_profile, predict_meal, record_observation,
                        compute_drift_bound, seed_from_meals)

TMP = tempfile.mkdtemp(prefix="aahaartwin_twin_")


def fresh_db(name):
    p = os.path.join(TMP, name + ".db")
    if os.path.exists(p):
        os.remove(p)
    init_db(p)
    return p


def _macros(carbs=60.0):
    return {"carbs": carbs, "protein": 20.0, "fat": 12.0, "fiber": 5.0, "calories": 450.0}


def _premeal():
    return {"g0": 110.0, "slope_60min": -0.05, "hr_mean_2h": 80.0, "act_kcal_2h": 220.0}


def test_profile_initializes_empty():
    db = fresh_db("empty")
    p = get_profile(db, 999)
    assert p["n_meals_observed"] == 0, f"expected 0 meals, got {p['n_meals_observed']}"
    assert p["carb_sensitivity"] is None, "carb_sensitivity should be NULL initially"
    assert p["ewma_residual"] is None, "ewma_residual should be NULL initially"
    assert p["drift_flag"] == 0, "drift_flag should be 0 initially"
    assert p["drift_bound"] > 0, "drift_bound should be positive"


def test_ewma_moves_in_residual_direction():
    db = fresh_db("ewma")
    pid = 1
    t1, t2 = "2020-05-01 14:00:00", "2020-05-01 20:00:00"
    p1 = predict_meal(db, pid, t1, "lunch", _macros(), _premeal())
    r1 = record_observation(db, pid, t1, "lunch", _macros(),
                            iauc_obs=p1["pred_iauc"] + 50.0, peak_obs=150.0)
    assert r1["matched"] is True, "observation should match its prediction"
    assert abs(r1["residual"] - 50.0) < 1e-9, f"residual wrong: {r1['residual']}"
    prof1 = get_profile(db, pid)
    assert prof1["ewma_residual"] > 0, \
        f"ewma should move positive after +50 residual, got {prof1['ewma_residual']}"
    assert abs(prof1["ewma_residual"] - 50.0) < 1e-9, "first ewma should equal first residual"

    p2 = predict_meal(db, pid, t2, "dinner", _macros(), _premeal())
    r2 = record_observation(db, pid, t2, "dinner", _macros(),
                            iauc_obs=p2["pred_iauc"] + 80.0, peak_obs=160.0)
    prof2 = get_profile(db, pid)
    expected = 0.3 * 80.0 + 0.7 * 50.0
    assert abs(prof2["ewma_residual"] - expected) < 1e-9, \
        f"ewma wrong: {prof2['ewma_residual']} vs {expected}"
    assert prof2["ewma_residual"] > prof1["ewma_residual"], \
        "ewma should keep moving in the residual direction"

    # updates_log recorded both steps with before/after
    con = sqlite3.connect(db)
    rows = con.execute("SELECT ewma_before, ewma_after FROM updates_log WHERE pid=? "
                       "ORDER BY id", (pid,)).fetchall()
    con.close()
    assert len(rows) == 2, f"expected 2 log rows, got {len(rows)}"
    assert rows[0][0] is None and abs(rows[0][1] - 50.0) < 1e-9
    assert abs(rows[1][0] - 50.0) < 1e-9 and abs(rows[1][1] - expected) < 1e-9


def test_carb_sensitivity_null_until_three_meals():
    db = fresh_db("sens")
    pid = 2
    meals = [(30.0, 20.0), (60.0, 40.0), (90.0, 60.0)]  # slope ~ 2/3
    for i, (carbs, iauc) in enumerate(meals):
        ts = f"2020-05-0{i + 1} 12:00:00"
        pr = predict_meal(db, pid, ts, "lunch", _macros(carbs), _premeal())
        record_observation(db, pid, ts, "lunch", _macros(carbs),
                           iauc_obs=iauc, peak_obs=140.0)
        prof = get_profile(db, pid)
        if i < 2:
            assert prof["carb_sensitivity"] is None, \
                f"sensitivity must be NULL before 3 meals (meal {i + 1})"
    prof = get_profile(db, pid)
    assert prof["carb_sensitivity"] is not None, "sensitivity should be set after 3 meals"
    assert abs(prof["carb_sensitivity"] - 2.0 / 3.0) < 0.05, \
        f"least-squares slope wrong: {prof['carb_sensitivity']}"


def test_prediction_uses_only_prior_meals():
    # Feed 5 sequential meals; at each prediction, every updates_log row for the
    # patient must have ts < prediction ts (no future info in profile state).
    db = fresh_db("causal")
    pid = 7
    base = "2020-05-%02d %02d:00:00"
    con = sqlite3.connect(db)
    for i in range(5):
        ts = base % (i + 1, 8 + i * 3)
        pr = predict_meal(db, pid, ts, "lunch", _macros(50.0 + i * 5), _premeal())
        rows = con.execute("SELECT ts FROM updates_log WHERE pid=?", (pid,)).fetchall()
        for (uts,) in rows:
            assert uts < pr["ts"], \
                f"leakage: updates_log ts {uts} not before prediction ts {pr['ts']}"
        record_observation(db, pid, ts, "lunch", _macros(50.0 + i * 5),
                           iauc_obs=pr["pred_iauc"] + 10.0, peak_obs=150.0)
    n = con.execute("SELECT COUNT(*) FROM updates_log WHERE pid=?", (pid,)).fetchone()[0]
    con.close()
    assert n == 5, f"expected 5 update rows, got {n}"


def test_drift_flag_fires_after_five_huge_residuals():
    db = fresh_db("drift")
    pid = 3
    bound = compute_drift_bound()
    assert bound > 40, f"drift bound sanity check failed: {bound}"
    for i in range(4):
        ts = f"2020-05-0{i + 1} 12:00:00"
        pr = predict_meal(db, pid, ts, "lunch", _macros(), _premeal())
        record_observation(db, pid, ts, "lunch", _macros(),
                           iauc_obs=pr["pred_iauc"] + 1000.0, peak_obs=250.0)
        assert get_profile(db, pid)["drift_flag"] == 0, \
            f"drift must NOT fire before a full 5-residual window (meal {i + 1})"
    ts = "2020-05-05 12:00:00"
    pr = predict_meal(db, pid, ts, "lunch", _macros(), _premeal())
    record_observation(db, pid, ts, "lunch", _macros(),
                       iauc_obs=pr["pred_iauc"] + 1000.0, peak_obs=250.0)
    assert get_profile(db, pid)["drift_flag"] == 1, \
        "drift flag should fire after 5 consecutive huge residuals"


def test_evidence_label_low_when_premeal_missing():
    db = fresh_db("evidence")
    pid = 4
    pr_none = predict_meal(db, pid, "2020-05-01 12:00:00", "lunch", _macros(50.0), {})
    assert pr_none["evidence_label"] == "low", \
        f"expected 'low' with 0/4 premeal features, got {pr_none['evidence_label']}"
    assert pr_none["evidence_score"] == 0.0
    pr_half = predict_meal(db, pid, "2020-05-01 18:00:00", "dinner",
                           _macros(50.0), {"g0": 110.0, "slope_60min": 0.0})
    assert pr_half["evidence_label"] == "low", \
        f"expected 'low' with 2/4 missing, got {pr_half['evidence_label']}"
    pr_full = predict_meal(db, pid, "2020-05-02 12:00:00", "lunch", _macros(50.0), _premeal())
    assert pr_full["evidence_label"] == "high", \
        f"expected 'high' with 4/4 present, got {pr_full['evidence_label']}"


def test_source_labels_preserved():
    db = fresh_db("source")
    pid = 5
    pr = predict_meal(db, pid, "2020-05-01 12:00:00", "lunch", _macros(), _premeal())
    record_observation(db, pid, "2020-05-01 12:00:00", "lunch", _macros(),
                       iauc_obs=pr["pred_iauc"], peak_obs=150.0, source="observed")
    record_observation(db, pid, "2020-05-01 18:00:00", "snack", _macros(20.0),
                       iauc_obs=30.0, peak_obs=140.0, source="simulated")
    con = sqlite3.connect(db)
    rows = con.execute("SELECT ts, source FROM observations WHERE pid=? ORDER BY ts",
                       (pid,)).fetchall()
    con.close()
    assert [r[1] for r in rows] == ["observed", "simulated"], \
        f"source labels not preserved: {[r[1] for r in rows]}"
    # Simulated what-ifs must not touch the real profile.
    prof = get_profile(db, pid)
    assert prof["n_meals_observed"] == 1, \
        f"simulated observation leaked into profile: {prof['n_meals_observed']}"


def test_record_observation_never_alters_predictions():
    db = fresh_db("immutable")
    pid = 6
    ts = "2020-05-01 12:00:00"
    pr = predict_meal(db, pid, ts, "lunch", _macros(), _premeal())
    con = sqlite3.connect(db)
    before = con.execute("SELECT * FROM predictions").fetchall()
    con.close()
    record_observation(db, pid, ts, "lunch", _macros(),
                       iauc_obs=pr["pred_iauc"] + 500.0, peak_obs=250.0)
    con = sqlite3.connect(db)
    after = con.execute("SELECT * FROM predictions").fetchall()
    con.close()
    assert before == after, "record_observation modified a past prediction row"


def test_seed_replay_is_chronological_and_complete():
    db = fresh_db("seed")
    summary = seed_from_meals(db, max_meals_per_patient=6)
    assert summary["n_patients"] == 45, f"expected 45 patients, got {summary['n_patients']}"
    assert summary["n_meals"] == 45 * 6, f"expected 270 meals, got {summary['n_meals']}"
    assert summary["n_predictions"] == summary["n_meals"]
    assert summary["n_updates"] == summary["n_meals"]
    con = sqlite3.connect(db)
    # Per-patient: predictions and updates strictly chronological.
    bad = con.execute(
        "SELECT pid FROM (SELECT pid, ts, LAG(ts) OVER (PARTITION BY pid ORDER BY id)"
        " AS prev FROM predictions) WHERE prev IS NOT NULL AND ts < prev LIMIT 1"
    ).fetchone()
    assert bad is None, f"predictions out of order for pid {bad}"
    n_obs = con.execute("SELECT COUNT(*) FROM observations WHERE source='observed'").fetchone()[0]
    con.close()
    assert n_obs == summary["n_meals"], "seeded observations must all be 'observed'"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
