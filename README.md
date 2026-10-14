# AahaarTwin — A Personalized Meal-Response Digital Twin

> **Research prototype only — not a medical device; not for diagnosis or treatment.**

AahaarTwin is a digital-twin system that predicts how *your* blood glucose
responds to *your* meals. It fuses **static clinical data** (HbA1c, fasting
glucose/insulin, lipids, BMI, age) with **dynamic streams** (continuous glucose
monitoring, heart rate, physical activity, time of day) to predict an
**adverse glycemic event** — a post-meal glucose excursion (2-hour peak
≥ 180 mg/dL) — for any candidate meal, and ranks candidate meals so a patient
or clinician can compare options before eating.

Built for the **Happiest Health Digital Twin Challenge 2026**.

---

## 1. Team details

| Field | Value |
|---|---|
| Developer | **Sujal Giri**, B.Tech CSE (2nd year) |
| Team name | **[TO BE CONFIRMED]** — not shown on the Unstop registration mail |
| College / institution | **Indian Institute of Technology (IIT), Patna** — verified from the Unstop registration mail, 13 Oct 2026 |
| Other members | **Suhani Kumari** (IIT Patna, registration verified) · **Aradhya Srivastava** (IIT Patna, confirmation pending — must confirm on Unstop by **18 Oct 2026, 7:00 PM IST**) |

## 2. Project title

**AahaarTwin — A Personalized Meal-Response Digital Twin** (aahaar = food).
A per-patient digital twin of postprandial glucose dynamics.

## 3. Problem statement

Generic dashboards show the same glycemic-index chart to everyone, but the
*same meal can spike glucose in one person and not in another* — responses
depend on the individual's physiology, pre-meal state, and context. Clinicians
and patients need **personalized, prospective, and uncertainty-aware**
predictions: *what will happen to MY glucose if I eat THIS meal right now,
given MY current state?* — not population averages.

## 4. Healthcare use case

- **Use case:** Pre-meal decision support for glycemic control — predict the
  2-hour glucose response (iAUC, peak, excursion risk) for candidate meals and
  rank them best-to-worst for a specific patient at a specific pre-meal moment.
- **Adverse event predicted:** glucose excursion, defined as post-meal 2-hour
  peak **≥ 180 mg/dL** (excursion rate in our dataset: 39.5%).
- **Intended users (conceptual doctor dashboard):** clinicians reviewing a
  patient's twin state, predicted-vs-observed calibration, and drift warnings;
  and the patient choosing among meals.
- **Explicitly out of scope:** diagnosis, treatment recommendations, insulin
  dosing, real-time closed-loop control. This is a research prototype with no
  clinical validation.

## 5. Technical stack

- **ML framework:** LightGBM 4.7.0 (gradient boosting); scikit-learn 1.9.1
  (GroupKFold, metrics, isotonic calibration); scipy; NumPy; pandas
- **Uncertainty:** quantile LightGBM (5th/95th percentile) + **conformalized
  quantile regression** (Romano–Patterson–Candès), α = 0.1
- **Twin state:** SQLite (`twin_state.db`) — per-patient profiles with running
  residual statistics and drift bounds
- **App:** Streamlit 1.65.0 + Plotly 7.1.0 (7-page interactive dashboard)
- **Visualization (docs):** matplotlib 3.11.2, python-pptx 1.0.2
- **Config:** YAML (`configs/`); Python 3.10+

## 6. AI/ML model + framework details

- **Inputs (25 features):** 5 meal macros (carbs, protein, fat, fiber, calories),
  4 pre-meal dynamic features (pre-meal glucose g0, 60-min glucose slope, mean
  HR over prior 2 h, activity calories over prior 2 h), cyclical time-of-day,
  one-hot meal type, 8 static biometrics (age, gender, BMI, HbA1c, fasting
  glucose/insulin, triglycerides, cholesterol, HDL, LDL). No raw CGM traces
  leave the dataset — only engineered aggregates enter the model.
- **Models (5 LightGBM boosters, `models/`):** `lgbm_iauc` (2-h iAUC point
  prediction), `lgbm_peak` (2-h peak point prediction), `lgbm_excursion`
  (binary excursion classifier), `lgbm_iauc_q05` / `lgbm_iauc_q95` (quantile
  bounds).
- **Evaluation protocol:** 5-fold **patient-level GroupKFold** on 1,579 usable
  meal events from 45 patients — every reported metric is out-of-fold (no
  prediction ever sees its own patient's data in training). 95% CIs by
  patient-level bootstrap (2,000 resamples).
- **Personalization (E1):** per-patient fine-tuning vs population model —
  tested with a pre-registered decision rule (win rate > 0.5 AND bootstrap CI
  excludes 0). **Result: SUPPORTED** (win rate 0.60, mean ΔMAE −3.68, CI
  [−6.96, −0.47]). Fine-tuning is evaluated but NOT in the live prediction
  path.
- **Ranking (E2):** model-predicted meal ordering vs carb-only ordering against
  observed ordering. **Result: model Spearman 0.546 vs carb-only 0.274;**
  model beats carb-only in **82.2% of patients** — personalized ranking adds
  real value beyond "just count carbs".
- **Uncertainty honesty:** raw quantile intervals undercovered badly (nominal
  90% → empirical **60.5%**). We did not hide this: we applied conformalized
  quantile regression; OOF coverage is now **0.901** at mean width 119.2
  (adjustment +24.04). The app's displayed interval lower bound is clipped at 0
  (glucose can't be negative); the stored interval remains symmetric/additive.
- **Calibration:** excursion probabilities are isotonic-calibrated (ECE raw
  0.122 → ~0.0 on the calibration set); the calibration curve is in the
  Evaluation page.

### Measured results (out-of-fold, n = 1,579, 45 patients)

| Target | Model | Baseline | 95% CI |
|---|---|---|---|
| iAUC MAE | **32.39** | train-fold global mean 39.09 (OOF) / carb-only 34.21 / patient-mean 34.04 (time-ordered) | 29.09–35.73 |
| Peak MAE | **27.10** | patient-mean 40.37 | 24.67–29.46 |
| Excursion PR-AUC (39.5% positive) | **0.798** | — | 0.73–0.84 |
| Excursion ROC-AUC | **0.837** | — | 0.80–0.87 |
| Excursion @ thr 0.347 (OOF max-F1) | P 0.707 / R 0.745 / F1 0.726 | — | — |
| Excursion @ thr 0.5 | P 0.730 / R 0.676 / F1 0.702 | — | — |
| Conformal 90% interval | coverage **0.901**, width 119.2 | raw quantile coverage 0.605 | — |

Every number in the dashboard is loaded from a real artifact in `models/*.json`
at startup; the app refuses to start if an artifact is missing.

### Measured vs predicted vs simulated
- **Measured:** CGM, heart-rate, activity, and meal-log values from the CGMacros
  dataset; all evaluation metrics (OOF).
- **Model predictions:** pre-meal iAUC/peak/excursion-risk estimates from the
  trained LightGBM boosters, with conformalized 90% intervals — associational
  projections, not causal claims.
- **Demonstration examples:** the candidate meals in the demo script and the
  Meal comparison page are hypothetical inputs typed by the user, not dataset
  meals; their outputs are predictions, not observations.
- **Simulated:** the twin database tags every logged observation as
  `observed` or `simulated`; simulated rows never update the patient profile.

## 7. Demo video link

**[PENDING — to be recorded by Sujal]** — planned as an unlisted YouTube video,
~20 minutes. Narration script and recording checklist: `docs/DEMO_SCRIPT.md`.

## 8. Open-source license details

- **Our code:** MIT License — see `LICENSE` (Copyright 2026 Sujal Giri). This
  covers only the code in this repository.
- **Dataset — CGMacros v1.0.0** (Gutierrez-Osuna et al., PhysioNet),
  DOI: [10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658),
  <https://physionet.org/content/cgmacros/>.
  Raw data is **NOT included** in this repository (challenge data rules; the
  shipped `LICENSE.txt` is empty — terms unverified, so **check the source
  before redistributing**). Reproduce locally: download CGMacros from
  PhysioNet and place the extracted `CGMacros/` folder where
  `configs/phase1.yaml` expects it. The derived, anonymized
  `data/processed/meals.csv` (1,613 de-duplicated meal events; patient IDs are
  numeric pseudonyms) is generated locally by `src/run_phase1.py` and is
  intentionally not committed to the repository.

## 9. Architecture diagram

- `docs/architecture.pdf` — flowchart (matplotlib): data sources →
  preprocessing → features → models → twin state → meal ranking → dashboard,
  with evaluation feedback loops (OOF evaluation, predicted-vs-observed, drift).
- `docs/architecture.pptx` — same diagram in slide form.
- Full technical narrative: `docs/TECHNICAL_REPORT.md`.

## 10. Presentation

- `docs/AahaarTwin_Presentation.pptx` — 12–14 slides: problem, motivation,
  solution, twin architecture, data + methodology, measured results (plots
  regenerated from real artifacts by `docs/make_plots.py`), twin workflow,
  differentiators with evidence, limitations / future work, submission
  checklist.

## 11. Quickstart

```bash
# 1. Create the environment (needs ~1 min)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Reproduce the full pipeline (needs CGMacros from PhysioNet, see §8)
.venv/bin/python src/run_phase1.py        # preprocess → features → splits → baselines
.venv/bin/python src/train.py             # 5 LightGBM boosters + OOF metrics
.venv/bin/python src/conformalize.py      # conformal intervals + calibration

# 3. Launch the doctor/patient dashboard
.venv/bin/streamlit run dashboard/app.py
# twin_state.db is auto-seeded on first launch if missing.
```

The app's 7 pages: **Overview · Patient explorer · Prediction ·
Meal comparison · Twin history · Evaluation · Safety**. All metrics load from
`models/*.json`; a missing artifact stops startup with a named error.

## 12. Data provenance

- Source: **CGMacros v1.0.0** (Gutierrez-Osuna et al., PhysioNet), DOI
  10.13026/3z8q-x658 — 45 participants, CGM + Fitbit + meal macros.
- Pipeline: `src/preprocess.py` (dedupe, CGM windowing, meal–CGM alignment) →
  `src/splits.py` (patient-level GroupKFold) → `src/features.py` (25 static +
  dynamic features) → `src/baselines.py` → `src/train.py` → `src/conformalize.py`.
- Real stats: 1,613 meal events post-dedup → **1,579 usable** (require valid
  2-h CGM window + macros) across **45 patients** (min 17, median 36, max 52
  meals/patient). Excursion rate 39.5%. See `docs/PHASE1_REPORT.md` and
  `docs/TECHNICAL_REPORT.md` for full provenance.

---

## Limitations (honest)

- **No clinical validation** — research prototype; not a medical device.
- 45 patients / 1,579 meals from one research dataset (CGMacros, published 2025)
  — no claim of generalization to India or any other population.
- Peak MAE 27 mg/dL and interval width ~119 mean predictions are directional,
  not precise; excursion P/R (~0.71/0.74) leaves substantial false alarms and
  misses.
- Personal fine-tuning (E1) is evaluated but NOT in the live prediction path.
- CGM/HR trend plots in the app need the optional raw directory; the app
  degrades gracefully to a notice when absent.
- Every Prediction/Comparison interaction appends rows to the seeded
  `twin_state.db` (git-ignored); copy the DB before heavy demoing.
- Demo patient labels ("diabetes range" via A1c ≥ 6.5%) are illustrative, not
  clinical classifications.

## Submission checklist

- [ ] Public GitHub repo — **[PENDING user action: create repo + push]**
- [x] README with all required items (this file)
- [ ] 20-min unlisted YouTube demo video — **[PENDING recording by Sujal]**
- [x] Architecture diagram (PDF + PPTX in `docs/`)
- [x] Presentation PPTX in `docs/`
- [x] Static + dynamic fusion → adverse-event prediction
- [x] Conceptual doctor dashboard (7-page Streamlit app)
- [x] MIT license for our code; dataset cited, raw data excluded
- [ ] Judging criteria — **[not published by organizer]**

Registration closes **18 Oct 2026, 7:00 PM IST** · Submission closes
**20 Oct 2026, 7:00 PM IST**.

## Project layout

```
aahaartwin/
├── src/                      # pipeline: preprocess, splits, baselines, features,
│                             #   train, conformalize, twin_state, ranking, run_phase1
├── dashboard/app.py          # 7-page Streamlit app (conceptual doctor dashboard)
├── models/                   # 5 LightGBM boosters + all metrics JSONs + OOF CSV
├── data/processed/meals.csv  # derived, anonymized meal events (raw data excluded)
├── configs/                  # YAML experiment configs
├── results/phase1/           # phase-1 metrics snapshot
├── docs/                     # architecture.pdf/.pptx, TECHNICAL_REPORT.md,
│                             #   AahaarTwin_Presentation.pptx, make_plots.py,
│                             #   DEMO_SCRIPT.md, SUBMISSION_AUDIT.md, DEV_LOG.md
├── tests/                    # 31 tests (test_phase1, test_twin, test_ranking, test_app)
└── twin_state.db             # runtime twin DB (git-ignored; auto-seeded on launch)
```

Tests: 31/31 passing (`test_phase1` 8, `test_twin` 9, `test_ranking` 5,
`test_app` 9). Run any file directly with the venv python, e.g.
`.venv/bin/python tests/test_phase1.py` — plain asserts, no pytest needed.
