"""Simple baselines for the meal-response task (Phase 1).

Reproduces the feasibility-report baselines on the leakage-safe
time-ordered within-patient split:

  A. patient-mean      : predict test iAUC with the patient's train mean
  B. carb-only linear  : least-squares fit of iAUC ~ carbs on train meals
  C. population linear : least-squares on all OTHER patients (leave-one-patient-out),
                         features = carbs, g0, slope_60min, protein, fat, fiber
  D. personal linear   : least-squares per patient on that patient's train meals,
                         features = carbs, g0, slope_60min, protein, fat

Plus: within-patient Spearman correlation between meal carbs and iAUC.

Metric: mean absolute error (MAE) on iAUC_2h. No accuracy; no invented targets.
"""
import numpy as np
import pandas as pd

FEATURES = ["carbs", "g0", "slope_60min", "protein", "fat", "fiber"]
FEATURES_PERSONAL = ["carbs", "g0", "slope_60min", "protein", "fat"]


def _lstsq(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _design(df, feats):
    med = df[feats].median(numeric_only=True)
    X = df[feats].fillna(med).to_numpy(dtype=float)
    return np.column_stack([np.ones(len(X)), X]), med


def compute_baselines(df, splits, min_meals=10):
    """df: usable meal table sorted by (pid, ts). splits: from time_ordered_within_patient."""
    recs = []
    for pid, (tr_idx, te_idx) in splits.items():
        tr = df.loc[tr_idx]
        te = df.loc[te_idx]
        y_tr = tr["iauc_2h"].to_numpy(dtype=float)
        y_te = te["iauc_2h"].to_numpy(dtype=float)

        # A. patient mean
        pred_mean = float(np.mean(y_tr))
        mae_mean = float(np.mean(np.abs(y_te - pred_mean)))

        # B. carb-only linear on this patient's train meals
        Xb = np.column_stack([np.ones(len(tr)), tr["carbs"].to_numpy(dtype=float)])
        beta = _lstsq(Xb, y_tr)
        pred_carb = beta[0] + beta[1] * te["carbs"].to_numpy(dtype=float)
        mae_carb = float(np.mean(np.abs(y_te - pred_carb)))

        # C. population linear, leave-one-patient-out
        pop = df[df["pid"] != pid]
        Xp, med_p = _design(pop, FEATURES)
        beta_p = _lstsq(Xp, pop["iauc_2h"].to_numpy(dtype=float))
        Xte = np.column_stack([np.ones(len(te)),
                               te[FEATURES].fillna(med_p).to_numpy(dtype=float)])
        pred_pop = Xte @ beta_p
        mae_pop = float(np.mean(np.abs(y_te - pred_pop)))

        # D. personal linear on this patient's train meals
        Xt, med_t = _design(tr, FEATURES_PERSONAL)
        beta_t = _lstsq(Xt, y_tr)
        Xte_t = np.column_stack([np.ones(len(te)),
                                 te[FEATURES_PERSONAL].fillna(med_t).to_numpy(dtype=float)])
        pred_pp = Xte_t @ beta_t
        mae_pp = float(np.mean(np.abs(y_te - pred_pp)))

        recs.append(dict(pid=pid, n_train=len(tr), n_test=len(te),
                         mae_patient_mean=mae_mean, mae_carb_only=mae_carb,
                         mae_population_linear=mae_pop, mae_personal_linear=mae_pp))
    out = pd.DataFrame(recs)
    return {
        "n_patients": len(out),
        "mae_patient_mean": float(out["mae_patient_mean"].mean()),
        "mae_carb_only_linear": float(out["mae_carb_only"].mean()),
        "mae_population_linear": float(out["mae_population_linear"].mean()),
        "mae_personal_linear": float(out["mae_personal_linear"].mean()),
        "frac_pop_beats_patient_mean": float((out["mae_population_linear"] < out["mae_patient_mean"]).mean()),
        "frac_carb_beats_patient_mean": float((out["mae_carb_only"] < out["mae_patient_mean"]).mean()),
        "frac_personal_beats_population": float((out["mae_personal_linear"] < out["mae_population_linear"]).mean()),
        "per_patient": out.to_dict(orient="records"),
    }


def carb_iauc_spearman(df):
    """Within-patient Spearman(carbs, iauc_2h)."""
    vals, n = [], 0
    for _, gdf in df.groupby("pid"):
        x = gdf["carbs"].to_numpy(dtype=float)
        y = gdf["iauc_2h"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            continue
        r = float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"))
        if np.isfinite(r):
            vals.append(r)
            n += int(ok.sum())
    vals = np.array(vals)
    return {
        "mean": float(vals.mean()), "median": float(np.median(vals)),
        "frac_positive": float((vals > 0).mean()),
        "n_patients": len(vals), "n_meals": n,
    }
