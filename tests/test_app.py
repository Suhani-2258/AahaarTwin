"""Phase 5+6 dashboard tests. Run with: python3 tests/test_app.py

Each test raises AssertionError with a clear message on failure.
Importing dashboard.app must be UI-free (no Streamlit runtime calls at
import time). Prediction/ranking tests use throwaway SQLite DBs under /tmp;
the seeded project DB is untouched.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, PROJ)

from twin_state import init_db, predict_meal  # noqa: E402
from ranking import rank_meals  # noqa: E402

TMP = tempfile.mkdtemp(prefix="aahaartwin_app_")


def fresh_db(name):
    p = os.path.join(TMP, name + ".db")
    if os.path.exists(p):
        os.remove(p)
    init_db(p)
    return p


def _macros(carbs=60.0):
    return {"carbs": carbs, "protein": 20.0, "fat": 12.0, "fiber": 5.0,
            "calories": 450.0}


def _premeal():
    return {"g0": 110.0, "slope_60min": -0.05, "hr_mean_2h": 80.0,
            "act_kcal_2h": 220.0}


def test_import_safe():
    # Importing the dashboard module must not execute any UI / require a
    # Streamlit runtime context.
    import dashboard.app as app
    for attr in ["main", "validate_macros", "validate_premeal", "risk_level",
                 "clip_interval_lo", "DISCLAIMER"]:
        assert hasattr(app, attr), f"dashboard.app missing {attr}"
    assert "not a medical device" in app.DISCLAIMER


def test_predict_path_returns_required_keys():
    db = fresh_db("predict")
    pred = predict_meal(db, 1, "2020-05-01 12:00:00", "lunch",
                        _macros(), _premeal())
    for key in ["pred_iauc", "pred_peak", "prob_excursion", "iauc_lo",
                "iauc_hi", "evidence_label", "evidence_score", "model_version"]:
        assert key in pred, f"prediction missing key {key}"
    assert pred["evidence_label"] == "high", \
        f"full premeal context -> 'high' evidence, got {pred['evidence_label']}"
    assert 0.0 <= pred["prob_excursion"] <= 1.0
    assert pred["iauc_lo"] <= pred["pred_iauc"] <= pred["iauc_hi"]


def test_validate_macros_rejects_negative_carbs():
    import dashboard.app as app
    bad = _macros(carbs=-10.0)
    errs = app.validate_macros(bad)
    assert errs, "negative carbs must produce validation errors"
    assert any("carbs" in e and "negative" in e for e in errs), \
        f"error should name carbs as negative: {errs}"


def test_validate_macros_rejects_out_of_range_and_nan():
    import dashboard.app as app
    assert app.validate_macros(_macros(carbs=600.0)), \
        "carbs > 500 must be rejected"
    bad = _macros()
    bad["protein"] = float("nan")
    assert app.validate_macros(bad), "NaN protein must be rejected"
    bad2 = _macros()
    bad2["fat"] = -0.5
    assert app.validate_macros(bad2), "negative fat must be rejected"


def test_validate_macros_accepts_valid():
    import dashboard.app as app
    assert app.validate_macros(_macros()) == [], \
        f"valid macros rejected: {app.validate_macros(_macros())}"


def test_risk_level_boundaries():
    import dashboard.app as app
    thr = 0.347
    assert app.risk_level(0.9, thr) == "HIGH"
    assert app.risk_level(0.5, thr) == "HIGH"
    assert app.risk_level(0.4, thr) == "MODERATE"
    assert app.risk_level(0.347, thr) == "MODERATE"
    assert app.risk_level(0.1, thr) == "LOW"


def test_clip_interval_lo():
    import dashboard.app as app
    assert app.clip_interval_lo(-5.0) == 0.0, "negative lo must clip to 0"
    assert app.clip_interval_lo(12.3) == 12.3


def test_metrics_jsons_load_with_expected_keys():
    expected = {
        "metrics": ["n_usable_meals", "n_patients", "regression",
                    "classification", "personalization_E1", "ranking_E2"],
        "metrics_ci": ["mae_iauc", "mae_peak", "pr_auc", "roc_auc"],
        "calibration": ["bins_raw", "bins_calibrated"],
        "personalization": ["per_patient", "summary"],
        "ranking": ["per_patient", "summary"],
        "feature_importance": ["lgbm_iauc.txt"],
        "conformal": ["adjustment", "method"],
    }
    for name, keys in expected.items():
        path = os.path.join(PROJ, "models", f"{name}.json")
        with open(path) as fh:
            d = json.load(fh)
        for k in keys:
            assert k in d, f"models/{name}.json missing key {k}"
    # Baselines artifact (Phase-1 time-ordered evaluation).
    with open(os.path.join(PROJ, "results", "phase1", "metrics.json")) as fh:
        b = json.load(fh)["baselines"]
    for k in ["mae_patient_mean", "mae_carb_only_linear",
              "mae_population_linear", "mae_personal_linear"]:
        assert k in b, f"results/phase1/metrics.json baselines missing {k}"


def test_ranking_integration_three_candidates():
    import dashboard.app as app  # noqa: F401  (import-safety covered elsewhere)
    db = fresh_db("rank")

    def _predict_fn(ts_iso, meal_type, macros, premeal):
        return predict_meal(db, 1, ts_iso, meal_type, macros, premeal)

    candidates = [
        {"name": "A", "meal_type": "lunch",
         **{k: v for k, v in _macros(carbs=70.0).items()}},
        {"name": "B", "meal_type": "lunch",
         **{k: v for k, v in _macros(carbs=30.0).items()}},
        {"name": "C", "meal_type": "snack",
         **{k: v for k, v in _macros(carbs=110.0).items()}},
    ]
    ranked = rank_meals(_predict_fn, {"ts_iso": "2020-05-01 12:00:00",
                                     **_premeal()}, candidates)
    assert len(ranked) == 3, f"expected 3 ranked, got {len(ranked)}"
    iaucs = [r["pred_iauc"] for r in ranked]
    assert iaucs == sorted(iaucs), "ranking must be ascending by pred_iauc"
    for r in ranked:
        for key in ["name", "pred_iauc", "pred_peak", "prob_excursion",
                    "iauc_lo", "iauc_hi", "evidence_label", "ood_flag",
                    "explanation"]:
            assert key in r, f"ranked candidate missing key {key}"
        assert "not medical advice" in r["explanation"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
