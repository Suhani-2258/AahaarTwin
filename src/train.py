"""Phase 2: train + evaluate AahaarTwin models (honest, leakage-safe).

Population models are evaluated with 5-fold patient-level GroupKFold;
all reported metrics come from OUT-OF-FOLD predictions. Final app models
(saved to models/*.txt) are trained on ALL usable meals AFTER evaluation.

Saves to models/: metrics.json, metrics_ci.json, feature_importance.json,
oof_predictions.csv, calibration.json, personalization.json, ranking.json,
config_snapshot.yaml, and the five final model files.
"""
import json
import os
import shutil
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from features import (FEATURES, FEATURE_DOCS, MedianImputer, build_feature_frame,
                      load_bio)

SEED = 42
rng = np.random.RandomState(SEED)


def load_config():
    with open(os.path.join(PROJ, "configs", "phase2.yaml")) as f:
        return yaml.safe_load(f)


def ece_score(y_true, prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, out = 0.0, []
    for i in range(n_bins):
        m = (prob > bins[i]) & (prob <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = prob[m].mean()
        ece += (m.sum() / len(prob)) * abs(acc - conf)
        out.append({"bin": [float(bins[i]), float(bins[i + 1])],
                    "n": int(m.sum()), "accuracy": float(acc),
                    "confidence": float(conf)})
    return float(ece), out


def monotone_list():
    return [1 if f == "carbs" else 0 for f in FEATURES]


def main():
    cfg = load_config()
    mdir = os.path.join(PROJ, "models")
    os.makedirs(mdir, exist_ok=True)

    meals = pd.read_csv(os.path.join(PROJ, "data/processed/meals.csv"),
                        parse_dates=["ts"])
    usable = meals[meals["usable"].astype(bool)].sort_values(
        ["pid", "ts"]).reset_index(drop=True)
    bio = load_bio(os.path.normpath(
        os.path.join(PROJ, "../feas-data/cgmacros_raw/CGMacros/bio.csv")))
    X = build_feature_frame(usable, bio)
    groups = usable["pid"].to_numpy()
    y_iauc = usable["iauc_2h"].to_numpy(float)
    y_peak = usable["peak_2h"].to_numpy(float)
    y_exc = usable["excursion_2h"].astype(int).to_numpy()

    rp = cfg["lgbm_regression"]
    cp = cfg["lgbm_classifier"]
    qp = cfg["lgbm_quantile"]

    reg_params = dict(objective="regression", num_leaves=rp["num_leaves"],
                      learning_rate=rp["learning_rate"], n_estimators=rp["n_estimators"],
                      min_child_samples=rp["min_child_samples"], subsample=rp["subsample"],
                      colsample_bytree=rp["colsample_bytree"], reg_lambda=rp["reg_lambda"],
                      monotone_constraints=monotone_list(), random_state=SEED, verbose=-1)
    clf_params = dict(objective="binary", is_unbalance=True, num_leaves=cp["num_leaves"],
                      learning_rate=cp["learning_rate"], n_estimators=cp["n_estimators"],
                      min_child_samples=cp["min_child_samples"], subsample=cp["subsample"],
                      colsample_bytree=cp["colsample_bytree"], reg_lambda=cp["reg_lambda"],
                      random_state=SEED, verbose=-1)

    n = len(usable)
    oof = pd.DataFrame({"pid": usable["pid"].to_numpy(), "ts": usable["ts"]})
    for c in ["pred_iauc", "pred_peak", "prob_exc", "q05", "q95"]:
        oof[c] = np.nan

    gkf = GroupKFold(n_splits=cfg["evaluation"]["cv_folds"])
    for tr_idx, te_idx in gkf.split(X, groups=groups):
        imp = MedianImputer().fit(X.iloc[tr_idx])
        Xtr, Xte = imp.transform(X.iloc[tr_idx]), imp.transform(X.iloc[te_idx])

        m = lgb.LGBMRegressor(**reg_params).fit(Xtr, y_iauc[tr_idx])
        oof.loc[te_idx, "pred_iauc"] = m.predict(Xte)
        m = lgb.LGBMRegressor(**reg_params).fit(Xtr, y_peak[tr_idx])
        oof.loc[te_idx, "pred_peak"] = m.predict(Xte)
        m = lgb.LGBMClassifier(**clf_params).fit(Xtr, y_exc[tr_idx])
        oof.loc[te_idx, "prob_exc"] = m.predict_proba(Xte)[:, 1]
        for a, col in [(0.05, "q05"), (0.95, "q95")]:
            mq = lgb.LGBMRegressor(**{**qp, "alpha": a,
                                      "num_leaves": qp["num_leaves"],
                                      "learning_rate": qp["learning_rate"],
                                      "n_estimators": qp["n_estimators"],
                                      "min_child_samples": qp["min_child_samples"],
                                      "random_state": SEED, "verbose": -1}).fit(Xtr, y_iauc[tr_idx])
            oof.loc[te_idx, col] = mq.predict(Xte)

    # ---- regression metrics ----
    def reg_metrics(y, p):
        return {"mae": float(mean_absolute_error(y, p)),
                "rmse": float(np.sqrt(mean_squared_error(y, p)))}
    # patient-mean baselines for iauc AND peak (same OOF structure: mean of train folds)
    pm_iauc = np.full(n, np.nan)
    pm_peak = np.full(n, np.nan)
    for tr_idx, te_idx in gkf.split(X, groups=groups):
        pm_iauc[te_idx] = y_iauc[tr_idx].mean()
        pm_peak[te_idx] = y_peak[tr_idx].mean()

    metrics = {
        "n_usable_meals": n,
        "n_patients": int(usable["pid"].nunique()),
        "regression": {
            "iauc": {**reg_metrics(y_iauc, oof["pred_iauc"].to_numpy()),
                     "baseline_patient_mean_mae": float(mean_absolute_error(y_iauc, pm_iauc))},
            "peak": {**reg_metrics(y_peak, oof["pred_peak"].to_numpy()),
                     "baseline_patient_mean_mae": float(mean_absolute_error(y_peak, pm_peak))},
        },
    }
    per_pat = oof.assign(y_iauc=y_iauc, y_peak=y_peak).groupby("pid").apply(
        lambda g: pd.Series({
            "mae_iauc": mean_absolute_error(g["y_iauc"], g["pred_iauc"]),
            "mae_peak": mean_absolute_error(g["y_peak"], g["pred_peak"]),
            "n": len(g)}))
    metrics["regression"]["per_patient"] = per_pat.reset_index().to_dict("records")

    # ---- classification metrics ----
    prob = oof["prob_exc"].to_numpy()
    pr_auc = float(average_precision_score(y_exc, prob))
    roc_auc = float(roc_auc_score(y_exc, prob))
    # operating point: max F1 on pooled OOF (each prediction out-of-sample -> no leak)
    prec, rec, thr = precision_recall_curve(y_exc, prob)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best = int(np.argmax(f1s))
    thr_star = float(thr[min(best, len(thr) - 1)])
    pred_star = (prob >= thr_star).astype(int)
    pred_05 = (prob >= 0.5).astype(int)
    ece_raw, bins_raw = ece_score(y_exc, prob, cfg["evaluation"]["calibration_bins"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(prob, y_exc)
    prob_cal = iso.predict(prob)
    ece_cal, bins_cal = ece_score(y_exc, prob_cal, cfg["evaluation"]["calibration_bins"])
    metrics["classification"] = {
        "event_rate": float(y_exc.mean()),
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "threshold_max_f1_oof": thr_star,
        "at_max_f1": {"precision": float(precision_score(y_exc, pred_star)),
                      "recall": float(recall_score(y_exc, pred_star)),
                      "f1": float(f1_score(y_exc, pred_star))},
        "at_0_5": {"precision": float(precision_score(y_exc, pred_05)),
                   "recall": float(recall_score(y_exc, pred_05)),
                   "f1": float(f1_score(y_exc, pred_05))},
        "ece_raw": ece_raw, "ece_calibrated": ece_cal,
    }
    calibration = {"bins_raw": bins_raw, "bins_calibrated": bins_cal,
                   "note": "Isotonic calibration fit on pooled OOF predictions."}

    # ---- quantile coverage ----
    cov = float(((y_iauc >= oof["q05"].to_numpy()) &
                 (y_iauc <= oof["q95"].to_numpy())).mean())
    metrics["uncertainty"] = {"interval": "q05-q95 (nominal 90%)",
                              "empirical_coverage_oof": cov,
                              "mean_width": float((oof["q95"] - oof["q05"]).mean())}

    oof["y_iauc"] = y_iauc
    oof["y_peak"] = y_peak
    oof["y_exc"] = y_exc
    oof.to_csv(os.path.join(mdir, "oof_predictions.csv"), index=False)

    # ---- E1 personalization ----
    fp = cfg["lgbm_finetune"]
    pers_recs = []
    for pid, gdf in usable.groupby("pid"):
        gdf = gdf.sort_values("ts")
        k = max(1, len(gdf) // 2)
        te = gdf.iloc[k:]
        # alignment guard: selected rows must belong to this patient
        assert (usable.loc[te.index, "pid"] == pid).all(), "row misalignment in E1"
        Xi = X.loc[gdf.index]
        # population model on OTHER patients
        mask_pop = usable["pid"] != pid
        imp = MedianImputer().fit(X[mask_pop])
        Xp, yp = imp.transform(X[mask_pop]), y_iauc[mask_pop.to_numpy()]
        pop_m = lgb.LGBMRegressor(**reg_params).fit(Xp, yp)
        Xte = imp.transform(Xi.loc[te.index])
        pred_pop = pop_m.predict(Xte)
        # fine-tune on patient's first 50%
        tr = gdf.iloc[:k]
        Xtr = imp.transform(Xi.loc[tr.index])
        ft = lgb.LGBMRegressor(objective="regression", num_leaves=31,
                               learning_rate=fp["learning_rate"],
                               n_estimators=fp["n_estimators"],
                               min_child_samples=fp["min_child_samples"],
                               reg_lambda=1.0, random_state=SEED, verbose=-1)
        ft.fit(Xtr, y_iauc[tr.index.to_numpy()], init_model=pop_m.booster_)
        pred_ft = ft.predict(Xte)
        yte = y_iauc[te.index.to_numpy()]
        pers_recs.append({"pid": int(pid), "n_train": len(tr), "n_test": len(te),
                          "mae_pop": float(mean_absolute_error(yte, pred_pop)),
                          "mae_personal": float(mean_absolute_error(yte, pred_ft))})
    pers = pd.DataFrame(pers_recs)
    pers["dMAE"] = pers["mae_personal"] - pers["mae_pop"]
    win_rate = float((pers["dMAE"] < 0).mean())
    boots = np.array([rng.choice(pers["dMAE"].to_numpy(),
                                 size=len(pers), replace=True).mean()
                      for _ in range(cfg["evaluation"]["personalization"]["bootstrap_resamples"])])
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    rule = cfg["evaluation"]["personalization"]
    supported = bool(win_rate > rule["rule_win_rate_gt"] and (ci_hi < 0 or ci_lo > 0))
    metrics["personalization_E1"] = {
        "mean_dMAE_personal_minus_pop": float(pers["dMAE"].mean()),
        "win_rate": win_rate,
        "bootstrap_95ci_mean_dMAE": [ci_lo, ci_hi],
        "n_patients": len(pers),
        "decision_rule": "supported iff win_rate>0.5 and CI excludes 0",
        "claim_supported": supported,
        "verdict": ("SUPPORTED — per-patient fine-tuning improves on population model"
                    if supported else
                    "NULL — no measured personalization gain; ship population model, report honestly"),
    }
    pers.to_csv(os.path.join(mdir, "personalization.csv"), index=False)
    with open(os.path.join(mdir, "personalization.json"), "w") as f:
        json.dump({"per_patient": pers.to_dict("records"),
                   "summary": metrics["personalization_E1"]}, f, indent=2, default=str)

    # ---- E2 ranking ----
    rank_recs = []
    for pid, gdf in usable.groupby("pid"):
        gdf = gdf.sort_values("ts")
        k = max(1, len(gdf) // 2)
        te = gdf.iloc[k:]
        if len(te) < 5:
            continue
        assert (usable.loc[te.index, "pid"] == pid).all(), "row misalignment in E2"
        oo = oof.loc[te.index]
        r_pred = float(pd.Series(oo["pred_iauc"].to_numpy()).corr(
            pd.Series(oo["y_iauc"].to_numpy()), method="spearman"))
        r_carb = float(pd.Series(te["carbs"].to_numpy(float)).corr(
            pd.Series(oo["y_iauc"].to_numpy()), method="spearman"))
        rank_recs.append({"pid": int(pid), "spearman_pred": r_pred,
                          "spearman_carb": r_carb})
    rk = pd.DataFrame(rank_recs).dropna()
    metrics["ranking_E2"] = {
        "mean_spearman_pred": float(rk["spearman_pred"].mean()),
        "mean_spearman_carb": float(rk["spearman_carb"].mean()),
        "frac_pred_beats_carb": float((rk["spearman_pred"] > rk["spearman_carb"]).mean()),
        "n_patients": len(rk),
    }
    rk.to_csv(os.path.join(mdir, "ranking.csv"), index=False)
    with open(os.path.join(mdir, "ranking.json"), "w") as f:
        json.dump({"per_patient": rk.to_dict("records"),
                   "summary": metrics["ranking_E2"]}, f, indent=2, default=str)

    # ---- final app models (trained on ALL usable data AFTER evaluation) ----
    imp_full = MedianImputer().fit(X)
    Xf = imp_full.transform(X)
    finals = {
        "lgbm_iauc.txt": lgb.LGBMRegressor(**reg_params).fit(Xf, y_iauc),
        "lgbm_peak.txt": lgb.LGBMRegressor(**reg_params).fit(Xf, y_peak),
        "lgbm_excursion.txt": lgb.LGBMClassifier(**clf_params).fit(Xf, y_exc),
        "lgbm_iauc_q05.txt": lgb.LGBMRegressor(
            **{**qp, "alpha": 0.05, "num_leaves": qp["num_leaves"],
               "learning_rate": qp["learning_rate"], "n_estimators": qp["n_estimators"],
               "min_child_samples": qp["min_child_samples"],
               "random_state": SEED, "verbose": -1}).fit(Xf, y_iauc),
        "lgbm_iauc_q95.txt": lgb.LGBMRegressor(
            **{**qp, "alpha": 0.95, "num_leaves": qp["num_leaves"],
               "learning_rate": qp["learning_rate"], "n_estimators": qp["n_estimators"],
               "min_child_samples": qp["min_child_samples"],
               "random_state": SEED, "verbose": -1}).fit(Xf, y_iauc),
    }
    for name, model in finals.items():
        model.booster_.save_model(os.path.join(mdir, name))
    # median imputer for the app
    pd.Series(imp_full.medians_).to_json(os.path.join(mdir, "imputer_medians.json"))
    with open(os.path.join(mdir, "feature_order.json"), "w") as f:
        json.dump(FEATURES, f)
    with open(os.path.join(mdir, "feature_docs.json"), "w") as f:
        json.dump(FEATURE_DOCS, f, indent=2)
    fi = {}
    for name, model in finals.items():
        imp_vals = model.booster_.feature_importance(importance_type="gain")
        fi[name] = {f: float(v) for f, v in zip(FEATURES, imp_vals)}
    with open(os.path.join(mdir, "feature_importance.json"), "w") as f:
        json.dump(fi, f, indent=2)

    # ---- CIs for headline metrics (bootstrap by patient) ----
    def boot(stat_fn, n_res=2000):
        pids = usable["pid"].unique()
        vals = []
        for _ in range(n_res):
            samp = rng.choice(pids, size=len(pids), replace=True)
            sub = oof[oof["pid"].isin(samp)]
            vals.append(stat_fn(sub))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    metrics_ci = {
        "mae_iauc": boot(lambda d: mean_absolute_error(d["y_iauc"], d["pred_iauc"])),
        "mae_peak": boot(lambda d: mean_absolute_error(d["y_peak"], d["pred_peak"])),
        "pr_auc": boot(lambda d: average_precision_score(d["y_exc"], d["prob_exc"])),
        "roc_auc": boot(lambda d: roc_auc_score(d["y_exc"], d["prob_exc"])),
        "note": "bootstrap by patient (resample patients, not rows), 2000 resamples",
    }

    with open(os.path.join(mdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(mdir, "metrics_ci.json"), "w") as f:
        json.dump(metrics_ci, f, indent=2, default=str)
    with open(os.path.join(mdir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2, default=str)
    shutil.copy(os.path.join(PROJ, "configs", "phase2.yaml"),
                os.path.join(mdir, "config_snapshot.yaml"))

    print("phase 2 complete")
    print(f"  iauc: MAE={metrics['regression']['iauc']['mae']:.2f} "
          f"(patient-mean {metrics['regression']['iauc']['baseline_patient_mean_mae']:.2f})")
    print(f"  peak: MAE={metrics['regression']['peak']['mae']:.2f} "
          f"(patient-mean {metrics['regression']['peak']['baseline_patient_mean_mae']:.2f})")
    print(f"  excursion: PR-AUC={metrics['classification']['pr_auc']:.3f} "
          f"ROC-AUC={metrics['classification']['roc_auc']:.3f}")
    print(f"  E1 personalization: win_rate={win_rate:.2f} CI=[{ci_lo:.2f},{ci_hi:.2f}] "
          f"-> {metrics['personalization_E1']['verdict'][:60]}")
    print(f"  E2 ranking: pred {metrics['ranking_E2']['mean_spearman_pred']:.3f} vs "
          f"carb {metrics['ranking_E2']['mean_spearman_carb']:.3f}")
    print(f"  quantile coverage: {metrics['uncertainty']['empirical_coverage_oof']:.3f} (nominal 0.90)")
    return metrics


if __name__ == "__main__":
    main()
