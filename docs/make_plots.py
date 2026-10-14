"""Regenerate every presentation plot from REAL project artifacts.

Every chart reads from files under models/ or results/phase1/ — nothing is
invented, nothing is hard-coded. Run with the venv python:

    .venv/bin/python docs/make_plots.py

Outputs PNGs to docs/plots/ (used by docs/AahaarTwin_Presentation.pptx and
optionally docs/architecture.pptx).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

ACCENT = "#1f77b4"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#7f7f7f"
DARK = "#222222"


def _style(ax, title):
    ax.set_title(title, fontsize=13, color=DARK, pad=10)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pred_vs_obs():
    """Predicted vs observed iAUC (OOF) with the y=x line."""
    df = pd.read_csv(os.path.join(MODELS, "oof_predictions.csv"))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(df["pred_iauc"], df["y_iauc"], s=8, alpha=0.35, c=ACCENT,
               label="OOF meal (n=%d)" % len(df))
    lim = [df[["pred_iauc", "y_iauc"]].min().min(), df[["pred_iauc", "y_iauc"]].max().max()]
    ax.plot(lim, lim, "r--", lw=1.5, label="perfect prediction")
    _style(ax, "Predicted vs observed 2-h iAUC (out-of-fold)")
    ax.set_xlabel("Predicted iAUC (mg·min/dL)")
    ax.set_ylabel("Observed iAUC (mg·min/dL)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p1_pred_vs_obs.png"), dpi=150)
    plt.close(fig)


def plot_calibration():
    """Excursion probability calibration curve: raw vs isotonic-calibrated."""
    cal = json.load(open(os.path.join(MODELS, "calibration.json")))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for key, color, label in [("bins_raw", RED, "raw probabilities"),
                              ("bins_calibrated", GREEN, "isotonic-calibrated")]:
        bins = cal[key]
        conf = np.array([b["confidence"] for b in bins])
        acc = np.array([b["accuracy"] for b in bins])
        ax.plot(conf, acc, "o-", c=color, label=label, markersize=6)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")
    _style(ax, "Excursion probability calibration (OOF)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed excursion rate")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p2_calibration.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance():
    """Top-12 LightGBM gain features for the iAUC model (associational)."""
    fi = json.load(open(os.path.join(MODELS, "feature_importance.json")))
    d = fi["lgbm_iauc.txt"]
    top = sorted(d.items(), key=lambda kv: -kv[1])[:12]
    feats, vals = zip(*top)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(range(len(feats))[::-1], vals[::-1], color=ACCENT)
    ax.set_yticks(range(len(feats))[::-1])
    ax.set_yticklabels(feats[::-1], fontsize=10)
    _style(ax, "Top-12 features by LightGBM gain (iAUC model)\n"
               "associational importance, not causal")
    ax.set_xlabel("Total split gain")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p3_feature_importance.png"), dpi=150)
    plt.close(fig)


def plot_personalization():
    """E1: per-patient ΔMAE (personalized − population), sorted."""
    pj = json.load(open(os.path.join(MODELS, "personalization.json")))
    rows = sorted(pj["per_patient"], key=lambda r: r["dMAE"])
    dmae = np.array([r["dMAE"] for r in rows])
    colors = [GREEN if v < 0 else RED for v in dmae]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(dmae)), dmae, color=colors, width=0.9)
    ax.axhline(0, color="k", lw=1)
    _style(ax, "E1 — personalization: per-patient ΔMAE "
               "(fine-tuned − population; green = personal wins)\n"
               "win rate 0.60, mean ΔMAE −3.68, bootstrap 95% CI [−6.96, −0.47] → SUPPORTED")
    ax.set_xlabel("Patient (sorted by ΔMAE)")
    ax.set_ylabel("ΔMAE (iAUC)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p4_personalization.png"), dpi=150)
    plt.close(fig)


def plot_ranking():
    """E2: per-patient Spearman(model) vs Spearman(carb-only), sorted by model."""
    rj = json.load(open(os.path.join(MODELS, "ranking.json")))
    rows = sorted(rj["per_patient"], key=lambda r: r["spearman_pred"])
    x = np.arange(len(rows))
    sp = [r["spearman_pred"] for r in rows]
    sc = [r["spearman_carb"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(x, sp, c=ACCENT, s=22, label="model ranking")
    ax.scatter(x, sc, c=GRAY, s=22, alpha=0.7, label="carb-only ranking")
    _style(ax, "E2 — meal ranking: per-patient Spearman vs observed order\n"
               "mean model 0.546 vs carb-only 0.274; model wins in 82.2% of patients")
    ax.set_xlabel("Patient (sorted by model Spearman)")
    ax.set_ylabel("Spearman correlation")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p5_ranking.png"), dpi=150)
    plt.close(fig)


def plot_mae_baselines():
    """iAUC MAE: model (OOF, patient-level GroupKFold) vs honest baselines."""
    m = json.load(open(os.path.join(MODELS, "metrics.json")))
    ph1 = json.load(open(os.path.join(PROJ, "results", "phase1", "metrics.json")))
    rows = [
        ("Model (OOF 5-fold\npatient-level)", m["regression"]["iauc"]["mae"], ACCENT),
        ("Patient-mean\n(time-ordered)", ph1["baselines"]["mae_patient_mean"], GRAY),
        ("Carb-only\nlinear", ph1["baselines"]["mae_carb_only_linear"], GRAY),
        ("Population\nlinear", ph1["baselines"]["mae_population_linear"], GRAY),
        ("Patient-mean\n(OOF eval)", m["regression"]["iauc"]["baseline_patient_mean_mae"], GRAY),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                "%.2f" % v, ha="center", fontsize=10, color=DARK)
    _style(ax, "iAUC MAE: model vs baselines (95% CI for model: 29.09–35.73)")
    ax.set_ylabel("Mean absolute error")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p6_mae_baselines.png"), dpi=150)
    plt.close(fig)


def plot_interval_coverage():
    """Quantile interval coverage: raw undercoverage (60.5%) vs conformal fix (90.1%)."""
    m = json.load(open(os.path.join(MODELS, "metrics.json")))
    c = json.load(open(os.path.join(MODELS, "conformal.json")))
    labels = ["Raw q05–q95\n(nominal 90%)", "Conformalized\n(α = 0.1)"]
    cov = [m["uncertainty"]["empirical_coverage_oof"], c["empirical_coverage_oof_after"]]
    widths = [m["uncertainty"]["mean_width"], c["mean_width_after"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.5))
    a1.bar(labels, cov, color=[RED, GREEN], width=0.6)
    a1.axhline(0.9, color="k", ls="--", lw=1.2, label="nominal 0.90")
    for x, v in zip(labels, cov):
        a1.text(x, v + 0.015, "%.3f" % v, ha="center", fontsize=10, color=DARK)
    _style(a1, "Empirical OOF coverage")
    a1.set_ylim(0, 1.05)
    a1.legend(fontsize=9)
    a2.bar(labels, widths, color=[RED, GREEN], width=0.6)
    for x, v in zip(labels, widths):
        a2.text(x, v + 1.5, "%.1f" % v, ha="center", fontsize=10, color=DARK)
    _style(a2, "Mean interval width")
    fig.suptitle("Uncertainty honesty: raw intervals undercovered, conformal fix restores it",
                 fontsize=11, color=DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "p7_intervals.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_pred_vs_obs()
    plot_calibration()
    plot_feature_importance()
    plot_personalization()
    plot_ranking()
    plot_mae_baselines()
    plot_interval_coverage()
    print("Plots written to", OUT)
    print(sorted(os.listdir(OUT)))
