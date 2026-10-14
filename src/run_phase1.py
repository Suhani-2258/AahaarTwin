"""Phase 1 orchestrator: preprocess -> splits -> baselines -> save artifacts.

Writes:
  data/processed/meals.csv          meal-event table (reproducible)
  results/phase1/metrics.json       all phase-1 metrics + data report
  results/phase1/config_snapshot.yaml  exact config used
"""
import json
import os
import shutil
import sys

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from preprocess import build_meal_table, load_config
from splits import (time_ordered_within_patient, leave_one_patient_out,
                    assert_no_leakage_time_ordered, assert_no_leakage_lopo)
from baselines import compute_baselines, carb_iauc_spearman


def main():
    cfg = load_config()
    eval_cfg = cfg["evaluation"]

    meals, report = build_meal_table(cfg)

    out_dir = os.path.normpath(os.path.join(PROJ, cfg["data"]["processed_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    meals.to_csv(os.path.join(out_dir, "meals.csv"), index=False)

    usable = meals[meals["usable"]].sort_values(["pid", "ts"]).reset_index(drop=True)

    splits = time_ordered_within_patient(
        usable,
        min_meals=eval_cfg["min_meals_per_patient"],
        train_fraction=eval_cfg["train_fraction"],
    )
    assert_no_leakage_time_ordered(usable, splits)

    lopo = leave_one_patient_out(usable)
    assert_no_leakage_lopo(usable, lopo)

    baseline_metrics = compute_baselines(usable, splits)
    spearman = carb_iauc_spearman(usable)

    res_dir = os.path.normpath(os.path.join(PROJ, cfg["data"]["results_dir"]))
    os.makedirs(res_dir, exist_ok=True)

    metrics = {
        "data_report": report,
        "splits": {
            "strategy": "time_ordered_within_patient",
            "min_meals_per_patient": eval_cfg["min_meals_per_patient"],
            "train_fraction": eval_cfg["train_fraction"],
            "n_patients_in_splits": len(splits),
            "population_eval": eval_cfg["population_eval"],
            "n_lopo_folds": len(lopo),
            "leakage_checks": "passed (time-ordered + LOPO patient disjointness)",
        },
        "baselines": baseline_metrics,
        "carb_iauc_spearman": spearman,
        "iauc_overall_std": float(usable["iauc_2h"].std()),
    }
    with open(os.path.join(res_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    shutil.copy(os.path.join(PROJ, "configs", "phase1.yaml"),
                os.path.join(res_dir, "config_snapshot.yaml"))

    print("phase 1 complete")
    print(f"  usable meals: {report['usable_meals']} | patients in splits: {len(splits)}")
    print(f"  baselines MAE: mean={baseline_metrics['mae_patient_mean']:.2f} "
          f"carb={baseline_metrics['mae_carb_only_linear']:.2f} "
          f"pop={baseline_metrics['mae_population_linear']:.2f} "
          f"personal={baseline_metrics['mae_personal_linear']:.2f}")
    return metrics


if __name__ == "__main__":
    main()
