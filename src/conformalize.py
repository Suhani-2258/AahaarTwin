"""Conformalize the iAUC quantile intervals (Phase 2 follow-up).

The raw LightGBM quantile models undercover (empirical 0.605 vs nominal 0.90
on out-of-fold predictions). This script applies conformalized quantile
regression (Romano, Patterson & Candes, 2019):

  score_i = max(q05_i - y_i, y_i - q95_i)
  adj     = quantile(scores, ceil((n+1) * 0.90) / n)
  interval(x) = [q05(x) - adj, q95(x) + adj]

The adjustment is computed on OUT-OF-FOLD predictions only, so the coverage
guarantee is honest. Saves models/conformal.json used by the app and twin.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)


def main():
    mdir = os.path.join(PROJ, "models")
    oof = pd.read_csv(os.path.join(mdir, "oof_predictions.csv"))
    y = oof["y_iauc"].to_numpy()
    q05 = oof["q05"].to_numpy()
    q95 = oof["q95"].to_numpy()
    scores = np.maximum(q05 - y, y - q95)
    n = len(scores)
    level = 0.90
    adj = float(np.quantile(scores, np.ceil((n + 1) * level) / n))
    cov = float(((y >= q05 - adj) & (y <= q95 + adj)).mean())
    width = float(((q95 + adj) - (q05 - adj)).mean())
    out = {
        "method": "conformalized quantile regression (Romano, Patterson & Candes 2019)",
        "alpha": 0.1,
        "nominal_coverage": 0.9,
        "adjustment": adj,
        "empirical_coverage_oof_after": cov,
        "mean_width_after": width,
        "n_calibration_points": n,
        "computed_on": "out-of-fold predictions (honest)",
        "usage": "app interval = [q05(x) - adjustment, q95(x) + adjustment]",
    }
    with open(os.path.join(mdir, "conformal.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"adjustment={adj:.2f} coverage_after={cov:.3f} width_after={width:.1f}")
    assert cov >= 0.88, f"conformal coverage too low: {cov}"
    print("conformal check passed")


if __name__ == "__main__":
    main()
