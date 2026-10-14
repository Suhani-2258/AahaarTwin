"""Architecture flowchart for AahaarTwin — matplotlib -> docs/architecture.pdf
and docs/plots/architecture.png (PNG is embedded in docs/architecture.pptx).

Every box carries the real component/file name. Evaluation feedback loops are
dashed arrows.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT_PDF = os.path.join(HERE, "architecture.pdf")
OUT_PNG = os.path.join(HERE, "plots", "architecture.png")
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)

# (x, y, width, height) in 0..100 canvas coords; y=0 is bottom.
STAGES = [
    ("1 · DATA SOURCES\n"
     "CGMacros v1.0.0 — DOI 10.13026/3z8q-x658\n"
     "Dexcom/Libre CGM · Fitbit (HR, activity)\n"
     "meal logs (carbs/protein/fat/fiber/calories)\n"
     "bio.csv static labs (age, BMI, HbA1c,\n"
     "fasting glucose/insulin, lipids)",
     "#dfe9f5", (8, 88, 84, 10)),
    ("2 · PREPROCESSING — src/preprocess.py\n"
     "schema normalization (8 raw variants) → dedupe 1,706 → 1,613 events\n"
     "meal–CGM alignment: valid 2-h window required → 1,579 usable events, 45 patients\n"
     "labels: iAUC, peak, excursion (2-h peak ≥ 180 mg/dL, rate 39.5%)",
     "#fef3d8", (8, 76, 84, 10)),
    ("3 · FEATURES — src/features.py · 25 features\n"
     "5 meal macros | 4 pre-meal dynamic (g0, 60-min glucose slope, mean HR 2 h,\n"
     "activity kcal 2 h) | cyclical time-of-day | meal-type one-hot | 8 static biometrics",
     "#e8f5e9", (8, 64, 84, 10)),
    ("4 · EVALUATION HARNESS — src/splits.py, src/baselines.py\n"
     "5-fold patient-level GroupKFold (no leakage: patient never in train+eval)\n"
     "baselines: patient-mean · carb-only linear · population linear",
     "#f3e5f5", (8, 52, 84, 10)),
    ("5 · MODELS — src/train.py → 5 LightGBM boosters (models/)\n"
     "lgbm_iauc (iAUC) · lgbm_peak (peak) · lgbm_excursion (excursion classifier)\n"
     "lgbm_iauc_q05 / lgbm_iauc_q95 (quantile bounds)",
     "#ffe0b2", (8, 40, 84, 10)),
    ("6 · UNCERTAINTY — src/conformalize.py\n"
     "conformalized quantile regression (α=0.1): OOF coverage 0.901 (raw: 0.605),\n"
     "mean width 119.2 · isotonic calibration of excursion probs (ECE 0.122 → ~0)",
     "#ffccbc", (8, 28, 84, 10)),
    ("7 · TWIN STATE — src/twin_state.py → twin_state.db (SQLite)\n"
     "seeded by replaying 1,579 meals in order (predict-first, then observe)\n"
     "per-patient profile: running residual stats, drift bound 48.58 (iAUC units)\n"
     "every prediction logged with its later observed outcome",
     "#d7f0f7", (8, 16, 84, 10)),
    ("8 · MEAL RANKING — src/ranking.py  →  9 · DASHBOARD — dashboard/app.py\n"
     "rank candidate meals by predicted response  ·  7-page Streamlit app:\n"
     "E2: model Spearman 0.546 vs carb-only 0.274  ·  Overview · Patient explorer ·\n"
     "Prediction · Meal comparison · Twin history · Evaluation · Safety\n"
     "all metrics loaded from models/*.json artifacts; disclaimers on every page",
     "#e0e7ff", (8, 4, 84, 10)),
]

FEEDBACK = [
    # (from box idx, to box idx, label)
    (4, 3, "OOF evaluation →\nmodels/metrics.json,\nmetrics_ci.json"),
    (6, 4, "predicted-vs-observed →\nprofile updates"),
    (6, 7, "drift warnings →\nSafety page"),
]


def draw():
    fig, ax = plt.subplots(figsize=(11, 15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    for text, color, (x, y, w, h) in STAGES:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                             facecolor=color, edgecolor="#333333", lw=1.2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.2, color="#111111", linespacing=1.35)

    # Solid pipeline arrows (down the left edge).
    for i in range(len(STAGES) - 1):
        _, _, (x, y, w, h) = STAGES[i]
        _, _, (nx, ny, nw, nh) = STAGES[i + 1]
        arr = FancyArrowPatch((x + 6, y), (nx + 6, ny + nh),
                              arrowstyle="-|>", mutation_scale=14,
                              color="#333333", lw=1.4)
        ax.add_patch(arr)

    # Dashed feedback-loop arrows on the right edge.
    for frm, to, label in FEEDBACK:
        _, _, (x, y, w, h) = STAGES[frm]
        _, _, (tx, ty, tw, th) = STAGES[to]
        ax.annotate("", xy=(tx + tw - 6, ty + th / 2), xytext=(x + w - 6, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#b71c1c",
                                    ls="dashed", lw=1.3, mutation_scale=12,
                                    connectionstyle="arc3,rad=0.18"))
        ax.text((x + w - 6 + tx + tw - 6) / 2 + 4.2, (y + h / 2 + ty + th / 2) / 2,
                label, fontsize=6.6, color="#b71c1c", ha="center", va="center",
                linespacing=1.2,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.text(50, 99.4, "AahaarTwin — system architecture (static + dynamic fusion "
                      "→ adverse-event prediction → doctor dashboard)",
            ha="center", fontsize=11.5, weight="bold", color="#111111")
    ax.text(50, 1.0, "Research prototype — not a medical device; not for diagnosis or treatment.",
            ha="center", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print("wrote", OUT_PDF, "and", OUT_PNG)


if __name__ == "__main__":
    draw()
