"""Parity test: vercel-app inference must match src.twin_state internals exactly.

Compares the pure model-prediction path (feature row -> impute -> boosters ->
assemble), bypassing DB logging. Any deviation means the serverless API would
serve different numbers than the local app.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aahaartwin/
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, os.path.join(PROJ, "vercel-app", "api"))

import pandas as pd
import twin_state as ts
from _lib import inference as vi
from features import FEATURES

_STUB_PROFILE = {"n_meals_observed": 0, "ewma_residual": 0.0, "drift_flag": 0}


def ref_predict(pid, t, mt, macros, pre):
    ts_norm = ts._norm_ts(t)
    X = ts._impute(pd.DataFrame([ts._feature_row(pid, ts_norm, mt, macros, pre)],
                                columns=FEATURES))
    outs = {k: float(v[0]) for k, v in ts._run_boosters(X).items()}
    score, label = ts._evidence(pre)
    return ts._assemble_prediction(pid, ts_norm, mt, outs, score, label, _STUB_PROFILE)


CASES = [
    (7, "2026-10-20T13:00", "lunch",
     {"carbs": 80, "protein": 15, "fat": 10, "fiber": 3, "calories": 470},
     {"g0": 110.0, "slope_60min": 0.2, "hr_mean_2h": 72.0, "act_kcal_2h": 40.0}),
    (7, "2026-10-20T08:15", "breakfast",
     {"carbs": 45, "protein": 20, "fat": 12, "fiber": 6, "calories": 380},
     {}),
    (12, "2026-10-20T20:30", "dinner",
     {"carbs": 200, "protein": 30, "fat": 25, "fiber": 4, "calories": 900},
     {"g0": 140.0, "slope_60min": 1.1, "hr_mean_2h": 80.0, "act_kcal_2h": 10.0}),
    (3, "2026-10-20T16:00", "snack",
     {"carbs": None, "protein": None, "fat": None, "fiber": None, "calories": None},
     {}),
]

KEYS = ["pred_iauc", "pred_peak", "prob_excursion", "iauc_lo", "iauc_hi",
        "evidence_score", "evidence_label", "model_version"]

fails = 0
for pid, t, mt, macros, pre in CASES:
    ref = ref_predict(pid, t, mt, macros, pre)
    got = vi.predict_one(pid, t, mt, macros, pre)
    for k in KEYS:
        a, b = ref[k], got[k]
        ok = (a == b) if isinstance(a, str) else abs(a - b) < 1e-9
        if not ok:
            fails += 1
            print(f"MISMATCH pid={pid} {mt} {k}: twin_state={a!r} vercel={b!r}")

# OOD flag parity (carbs=200 > p99 -> True; others False)
for (pid, t, mt, macros, pre), want_ood in zip(CASES, [False, False, True, False]):
    got = vi.predict_one(pid, t, mt, macros, pre)
    if got["ood_flag"] != want_ood:
        fails += 1
        print(f"OOD MISMATCH pid={pid}: got {got['ood_flag']}, want {want_ood}")

print("FAILURES:", fails)
assert fails == 0, "parity broken"
print("PARITY OK")
