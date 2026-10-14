"""Phase 1 smoke tests (dependency-free; run with: python3 tests/test_phase1.py).

Each test raises AssertionError with a clear message on failure.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from splits import (time_ordered_within_patient, leave_one_patient_out,
                    assert_no_leakage_time_ordered, assert_no_leakage_lopo)

MEALS = os.path.join(PROJ, "data", "processed", "meals.csv")
METRICS = os.path.join(PROJ, "results", "phase1", "metrics.json")


def test_processed_file_exists():
    assert os.path.exists(MEALS), "meals.csv missing — run src/run_phase1.py first"
    assert os.path.exists(METRICS), "metrics.json missing — run src/run_phase1.py first"


def test_required_columns():
    df = pd.read_csv(MEALS, nrows=5)
    for c in ["pid", "ts", "meal_type", "carbs", "g0", "iauc_2h", "peak_2h",
              "excursion_2h", "usable"]:
        assert c in df.columns, f"missing column {c}"


def test_no_duplicate_meal_events():
    df = pd.read_csv(MEALS, parse_dates=["ts"])
    d = df.sort_values(["pid", "ts"]).groupby("pid")["ts"].diff().dt.total_seconds() / 60
    assert (d.dropna() >= 30).all(), "dedup failed: two meals <30 min apart remain"


def test_meal_types_normalized():
    df = pd.read_csv(MEALS)
    allowed = {"breakfast", "lunch", "dinner", "snack"}
    got = set(df["meal_type"].dropna().unique())
    assert got <= allowed, f"unnormalized meal types: {got - allowed}"


def test_usable_subset_sane():
    df = pd.read_csv(MEALS)
    u = df[df["usable"].astype(bool)]
    assert len(u) > 1500, f"too few usable meals: {len(u)}"
    assert u["iauc_2h"].notna().all() and u["carbs"].notna().all() and u["g0"].notna().all()
    assert u["pid"].nunique() == 45, "expected 45 patients"
    assert (u.groupby("pid").size() >= 10).all(), "some patient has <10 usable meals"


def test_excursion_label_consistent():
    df = pd.read_csv(MEALS)
    u = df[df["usable"].astype(bool) & df["peak_2h"].notna()]
    assert ((u["peak_2h"] >= 180) == u["excursion_2h"].astype(bool)).all(), \
        "excursion label inconsistent with peak>=180"


def test_splits_leakage_free():
    df = pd.read_csv(MEALS, parse_dates=["ts"])
    u = df[df["usable"].astype(bool)].sort_values(["pid", "ts"]).reset_index(drop=True)
    splits = time_ordered_within_patient(u, min_meals=10, train_fraction=0.5)
    assert len(splits) == 45, f"expected 45 patients in splits, got {len(splits)}"
    assert_no_leakage_time_ordered(u, splits)
    folds = leave_one_patient_out(u)
    assert len(folds) == 45
    assert_no_leakage_lopo(u, folds)


def test_metrics_reproduce_feasibility():
    m = json.load(open(METRICS))
    b = m["baselines"]
    # Canonical post-dedup values (measured by this pipeline; the feasibility
    # report predates the 30-min dedup rule — see docs/PHASE1_REPORT.md).
    expected = {
        "mae_patient_mean": 34.04, "mae_carb_only_linear": 34.21,
        "mae_population_linear": 38.22, "mae_personal_linear": 37.37,
        "frac_personal_beats_population": 0.67,
    }
    for k, v in expected.items():
        assert abs(b[k] - v) < 0.05, f"baseline {k} drifted: {b[k]:.3f} vs expected {v}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
