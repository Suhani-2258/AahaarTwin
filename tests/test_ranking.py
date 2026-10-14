"""Phase 4 (meal ranking) tests. Run with: python3 tests/test_ranking.py

Each test raises AssertionError with a clear message on failure.
Uses a real twin_state.predict_meal (population boosters) as predict_fn,
bound to a throwaway DB.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from twin_state import init_db, predict_meal
from ranking import rank_meals, DISCLAIMER, CARBS_P01, CARBS_P99

TMP = tempfile.mkdtemp(prefix="aahaartwin_rank_")
DB = os.path.join(TMP, "rank.db")
init_db(DB)
_PID = 1


def _predict_fn(ts_iso, meal_type, macros, premeal):
    return predict_meal(DB, _PID, ts_iso, meal_type, macros, premeal)


def _ctx(**kw):
    base = {"g0": 110.0, "slope_60min": -0.05, "hr_mean_2h": 80.0, "act_kcal_2h": 220.0}
    base.update(kw)
    return base


def test_ranking_sorted_by_pred_iauc_ascending():
    cands = [
        {"name": "High carb bowl", "meal_type": "lunch", "carbs": 120.0,
         "protein": 20.0, "fat": 10.0, "fiber": 5.0, "calories": 700.0},
        {"name": "Light salad", "meal_type": "lunch", "carbs": 15.0,
         "protein": 10.0, "fat": 8.0, "fiber": 6.0, "calories": 200.0},
        {"name": "Mid plate", "meal_type": "lunch", "carbs": 60.0,
         "protein": 25.0, "fat": 15.0, "fiber": 8.0, "calories": 500.0},
    ]
    ranked = rank_meals(_predict_fn, _ctx(), cands)
    assert len(ranked) == 3, f"expected 3 ranked, got {len(ranked)}"
    iaucs = [r["pred_iauc"] for r in ranked]
    assert iaucs == sorted(iaucs), f"not sorted ascending: {iaucs}"
    assert {r["name"] for r in ranked} == {"High carb bowl", "Light salad", "Mid plate"}
    for r in ranked:
        for k in ["pred_iauc", "pred_peak", "prob_excursion", "iauc_lo",
                  "iauc_hi", "evidence_label", "ood_flag", "explanation"]:
            assert k in r, f"missing key {k} in ranked result"


def test_ood_candidate_flagged_exploratory():
    assert CARBS_P99 < 500.0, f"sanity: 500g carbs should exceed P99 ({CARBS_P99})"
    cands = [
        {"name": "Monster shake", "meal_type": "dinner", "carbs": 500.0,
         "protein": 30.0, "fat": 40.0, "fiber": 2.0, "calories": 2500.0},
        {"name": "Normal dinner", "meal_type": "dinner", "carbs": 60.0,
         "protein": 25.0, "fat": 15.0, "fiber": 8.0, "calories": 500.0},
    ]
    ranked = rank_meals(_predict_fn, _ctx(), cands)
    monster = next(r for r in ranked if r["name"] == "Monster shake")
    normal = next(r for r in ranked if r["name"] == "Normal dinner")
    assert monster["ood_flag"] is True, "500g carbs must be flagged OOD"
    assert "exploratory \u2014 outside observed data range" in monster["explanation"], \
        "OOD explanation must say 'exploratory — outside observed data range'"
    assert normal["ood_flag"] is False, "60g carbs is within the training range"


def test_explanation_carries_disclaimer():
    ranked = rank_meals(_predict_fn, _ctx(),
                        [{"name": "A", "meal_type": "lunch", "carbs": 50.0,
                          "protein": 20.0, "fat": 10.0, "fiber": 5.0, "calories": 400.0},
                         {"name": "B", "meal_type": "lunch", "carbs": 90.0,
                          "protein": 20.0, "fat": 10.0, "fiber": 5.0, "calories": 600.0}])
    for r in ranked:
        assert DISCLAIMER in r["explanation"], \
            f"disclaimer sentence missing in explanation for {r['name']}"
    assert ranked[0]["explanation"].startswith("Ranked #1 of 2:"), \
        f"rank-1 explanation format wrong: {ranked[0]['explanation'][:40]}"
    assert ranked[1]["explanation"].startswith("Ranked #2 of 2:"), \
        f"rank-2 explanation format wrong: {ranked[1]['explanation'][:40]}"


def test_missing_optional_macros_no_exception():
    # Candidate with only carbs; missing macros -> NaN -> median imputation.
    ranked = rank_meals(_predict_fn, {}, [{"name": "Mystery snack", "meal_type": "snack",
                                          "carbs": 40.0}])
    assert len(ranked) == 1
    r = ranked[0]
    assert r["pred_iauc"] == r["pred_iauc"], "pred_iauc must not be NaN"
    assert r["evidence_label"] == "low", \
        f"no premeal context -> 'low' evidence, got {r['evidence_label']}"
    assert DISCLAIMER in r["explanation"]


def test_empty_candidates_returns_empty():
    assert rank_meals(_predict_fn, _ctx(), []) == []


def test_all_macros_missing_flagged_exploratory():
    # Candidate with NO nutritional information at all: prediction is fully
    # median-imputed and must carry an explicit exploratory indicator.
    ranked = rank_meals(_predict_fn, _ctx(), [
        {"name": "No info meal", "meal_type": "lunch"},
        {"name": "Normal lunch", "meal_type": "lunch", "carbs": 60.0,
         "protein": 25.0, "fat": 15.0, "fiber": 8.0, "calories": 500.0},
    ])
    noinfo = next(r for r in ranked if r["name"] == "No info meal")
    normal = next(r for r in ranked if r["name"] == "Normal lunch")
    assert noinfo["macros_missing"] is True, "all-missing macros must be flagged"
    assert "exploratory" in noinfo["explanation"], \
        "explanation must say 'exploratory' for all-missing macros"
    assert "median-imputed" in noinfo["explanation"], \
        "explanation must say the prediction is median-imputed"
    assert DISCLAIMER in noinfo["explanation"]
    # ood_flag semantics unchanged: missing carbs is not "outside the range"
    assert noinfo["ood_flag"] is False, "ood_flag must stay carbs-range based"
    # valid meal untouched
    assert normal["macros_missing"] is False
    assert "median-imputed" not in normal["explanation"]


def test_partial_macros_not_flagged_missing():
    # Only carbs given -> median imputation for the rest, but NOT the
    # all-missing exploratory flag.
    ranked = rank_meals(_predict_fn, _ctx(),
                        [{"name": "Carbs only", "meal_type": "snack", "carbs": 40.0}])
    assert len(ranked) == 1
    r = ranked[0]
    assert r["macros_missing"] is False
    assert "median-imputed" not in r["explanation"]
    assert r["pred_iauc"] == r["pred_iauc"], "pred_iauc must not be NaN"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
