# AahaarTwin — Technical Report

**A Personalized Meal-Response Digital Twin**
Happiest Health Digital Twin Challenge 2026 · Developer: Sujal Giri, B.Tech CSE (2nd year)
Team: Sujal Giri (team lead), Suhani Kumari (verified), Aradhya Srivastava (confirmation pending) — Indian Institute of Technology (IIT), Patna, verified from the Unstop registration mail dated 13 Oct 2026. Team name: **[TO BE CONFIRMED]**.

> Research prototype only — not a medical device; not for diagnosis or treatment.
> Every number below is measured from a real project artifact (files named inline).
> Nothing is projected or imputed.

---

## 1. Problem

Postprandial glucose response is individual: the same meal spikes glucose in one
person and not in another, depending on physiology, pre-meal state, and context.
Generic glycemic-index charts ignore this. Clinicians and patients need a
**personalized, prospective, uncertainty-aware** answer to: *what will happen
to MY glucose if I eat THIS meal right now, given MY current state?*

## 2. Solution

AahaarTwin is a per-patient digital twin that:

1. **Fuses static + dynamic data** — 8 static biometrics (age, gender, BMI,
   HbA1c, fasting glucose/insulin, triglycerides, cholesterol, HDL, LDL) with
   dynamic streams (CGM traces, heart rate, activity, time of day) into 25
   engineered features per meal event.
2. **Predicts an adverse glycemic event** — glucose excursion (2-h post-meal
   peak ≥ 180 mg/dL) plus the 2-h incremental AUC and peak as continuous targets.
3. **Quantifies uncertainty honestly** — quantile models + conformalized
   intervals, isotonic probability calibration.
4. **Maintains twin state** — a SQLite twin DB replays each patient's meals
   (predict-first, then observe), keeps running residual statistics, and flags
   drift when the rolling mean |residual| exceeds 1.5× the OOF iAUC MAE
   (48.58 iAUC units) over 5 consecutive meals.
5. **Ranks candidate meals** — compares any set of candidate meals for a
   specific patient at a specific pre-meal moment (evaluation E2).
6. **Surfaces everything in a conceptual doctor dashboard** — a 7-page
   Streamlit app (Overview, Patient explorer, Prediction, Meal comparison,
   Twin history, Evaluation, Safety) whose every metric loads from a real
   artifact at startup and whose every page carries the research-prototype
   disclaimer.

## 3. Dataset

- **CGMacros v1.0.0** (Gutierrez-Osuna et al., PhysioNet),
  DOI 10.13026/3z8q-x658, <https://physionet.org/content/cgmacros/>.
  Raw data is **not redistributed** in this repository (the DOI landing page's
  `LICENSE.txt` is empty; terms unverified — check the source before
  redistributing).
- Raw intake: 45 participant CSVs with 8 schema variants; 1,706 raw meal
  events (`results/phase1/metrics.json`: `data_report`).
- Post-dedup (30-min minimum inter-meal gap): **1,613 events**.
- Usable (require valid 2-h CGM window + complete macros): **1,579 events**
  from **45 patients** (min 17, median 36, max 52 meals/patient).
- Adverse-event rate: **39.5%** of usable meals are excursions
  (`metrics.json`: `classification.event_rate` = 0.3946).
- Missingness handled by median imputation (fitted on training folds only):
  g0 1.2%, hr_mean_2h 3.0%, act_kcal_2h 1.4%, iauc_2h 2.1%, peak_2h 1.8%.
- Derived artifact `data/processed/meals.csv` is committed; patient IDs are
  numeric pseudonyms (no PII).

## 4. Architecture

Full diagram: `docs/architecture.pdf` / `docs/architecture.pptx` (regenerable
from `docs/make_architecture.py`). Pipeline stages:

1. **Data sources** — CGMacros (CGM + Fitbit + meal logs) + bio.csv static labs.
2. **Preprocessing** (`src/preprocess.py`) — schema normalization, dedup,
   meal–CGM alignment, label computation (iAUC / peak / excursion).
3. **Features** (`src/features.py`) — 25 features: 5 meal macros · 4 pre-meal
   dynamic (g0, 60-min glucose slope, mean HR over prior 2 h, activity kcal
   over prior 2 h) · cyclical time-of-day · meal-type one-hot · 8 static
   biometrics.
4. **Evaluation harness** (`src/splits.py`, `src/baselines.py`) — 5-fold
   patient-level GroupKFold (a patient is never in train and eval at once);
   baselines: patient-mean, carb-only linear, population linear.
5. **Models** (`src/train.py`) — 5 LightGBM boosters saved to `models/`:
   `lgbm_iauc.txt`, `lgbm_peak.txt`, `lgbm_excursion.txt`,
   `lgbm_iauc_q05.txt`, `lgbm_iauc_q95.txt`. Hyperparameters fixed a priori
   (`configs/phase2.yaml`): num_leaves 31, lr 0.05, 400 estimators, monotone
   +1 constraint on carbs.
6. **Uncertainty** (`src/conformalize.py`) — conformalized quantile regression
   (α = 0.1), isotonic calibration of excursion probabilities.
7. **Twin state** (`src/twin_state.py`) — `twin_state.db` (SQLite), seeded by
   honest replay; per-patient profiles, predicted-vs-observed logging, drift
   flags.
8. **Meal ranking** (`src/ranking.py`) — candidate-meal ordering by predicted
   response.
9. **Dashboard** (`dashboard/app.py`) — 7-page Streamlit app.

**Feedback loops:** (a) OOF evaluation artifacts (`models/metrics.json`,
`metrics_ci.json`) feed the Evaluation page; (b) predicted-vs-observed pairs
update twin profiles meal by meal; (c) drift flags surface on the Safety page.

## 5. Implementation

- `src/run_phase1.py` orchestrates preprocess → features → splits → baselines
  (config snapshot in `results/phase1/`).
- `src/train.py` trains final boosters on all 1,579 usable meals AND records
  cross-validated out-of-fold predictions to `models/oof_predictions.csv`.
  **All reported metrics come from OOF predictions, never from the final
  models.**
- `src/conformalize.py` fits the conformal adjustment on OOF residuals and
  the isotonic calibrator on OOF excursion probabilities.
- `dashboard/app.py` is import-safe (import performs zero Streamlit UI calls);
  all 31 tests pass, including a smoke test (`streamlit run` → HTTP 200).
- If `twin_state.db` is missing at startup (it is git-ignored), the app
  auto-seeds it from `data/processed/meals.csv` via
  `src.twin_state.seed_from_meals` instead of erroring out.

## 6. Methodology

- **Evaluation protocol:** 5-fold patient-level GroupKFold; out-of-fold
  predictions pooled (n = 1,579). 95% CIs by patient-level bootstrap
  (resample patients, not rows; 2,000 resamples) — `models/metrics_ci.json`.
- **E1 — personalization:** for each patient, a population model trained on
  the other 44 patients, fine-tuned on that patient's first 50% time-ordered
  meals, evaluated on the last 50%; paired comparison vs the population model
  on the same test meals. **Pre-registered decision rule:** claim supported
  iff win rate > 0.5 AND bootstrap 95% CI of mean ΔMAE excludes 0.
- **E2 — ranking:** within-patient Spearman(predicted order, observed order)
  vs Spearman(carb-only order, observed order) on each patient's held-out
  test meals.
- **Uncertainty:** quantile LightGBM (α = 0.05/0.95) → conformalized with the
  Romano–Patterson–Candès adjustment computed on honest OOF residuals.
- **No tuning on test data:** hyperparameters fixed a priori in
  `configs/phase2.yaml` (seed 42 throughout).

## 7. Evaluation — measured results

All values from `models/metrics.json`, `models/metrics_ci.json`,
`models/calibration.json`, `models/personalization.json`,
`models/ranking.json`, `models/conformal.json`, `models/oof_predictions.csv`.

### 7.1 Regression

| Target | Model OOF | Baselines (artifact-grounded) | 95% CI |
|---|---|---|---|
| iAUC MAE | **32.39** | train-fold global mean 39.09 (OOF eval — the test patient's own meals are entirely in the test fold, so the OOF "patient-mean" is the mean of the other patients' train folds); patient-mean 34.04 (time-ordered); carb-only linear 34.21; population linear 38.22 | 29.09–35.73 |
| iAUC RMSE | 44.25 | — | — |
| Peak MAE | **27.10** | patient-mean 40.37 | 24.67–29.46 |
| Peak RMSE | 35.79 | — | — |

The model beats the train-fold global-mean baseline on iAUC by ~17% (32.39 vs 39.09)
and on peak by ~33% (27.10 vs 40.37). Note the gap is narrower against the
time-ordered patient-mean baseline (34.04) — reported honestly.

### 7.2 Excursion classification (adverse event; 39.5% positive rate)

| Metric | Value | 95% CI |
|---|---|---|
| PR-AUC | **0.798** | 0.73–0.84 |
| ROC-AUC | **0.837** | 0.80–0.87 |
| @ threshold 0.347 (OOF max-F1): P / R / F1 | 0.707 / 0.745 / 0.726 | — |
| @ threshold 0.5: P / R / F1 | 0.730 / 0.676 / 0.702 | — |
| ECE (raw probabilities) | 0.122 → ~0.0 after isotonic calibration | — |

### 7.3 E1 — personalization verdict

| Statistic | Value |
|---|---|
| Win rate (personal beats population) | **0.60** |
| Mean ΔMAE (personal − population) | **−3.68** |
| Bootstrap 95% CI | **[−6.96, −0.47]** |
| n patients | 45 |
| Decision (pre-registered rule) | **SUPPORTED** — per-patient fine-tuning improves on the population model |

Honest caveat: E1 fine-tuning is **evaluated but not in the live prediction
path**; the app serves the population models plus twin-state profiles.

### 7.4 E2 — ranking verdict

| Statistic | Value |
|---|---|
| Mean Spearman (model ranking) | **0.546** |
| Mean Spearman (carb-only ranking) | **0.274** |
| Fraction of patients where model beats carb-only | **82.2%** |
| n patients | 45 |

Personalized ranking adds real value beyond "just count carbs".

### 7.5 Uncertainty honesty — the undercoverage finding

Raw quantile intervals (nominal 90%) achieved only **0.605** empirical OOF
coverage — a serious undercoverage we reported rather than hid. After
conformalized quantile regression (adjustment +24.04): **OOF coverage 0.901**
at mean width 119.2. The app clips the displayed interval lower bound at 0
(glucose cannot be negative); the stored interval remains symmetric/additive.

## 8. Limitations

1. No clinical validation; not a medical device.
2. 45 patients / 1,579 meals from one research dataset (CGMacros, published 2025)
   — no generalization claim to India or any other population.
3. Peak MAE 27 mg/dL and interval width ~119 make predictions directional,
   not precise; excursion P/R ≈ 0.71/0.74 leaves substantial false alarms and
   misses.
4. E1 fine-tuning is evaluated but not shipped in the prediction path.
5. CGM/HR trend plots need the optional raw directory; the app degrades
   gracefully when absent.
6. Display clipping of the interval lower bound at 0 is documented in-app.
7. Demo patient labels ("diabetes range" via A1c ≥ 6.5%) are illustrative,
   not clinical classifications.

## 9. Future work

- Ship E1 personal fine-tuning behind a data-sufficiency gate (e.g. ≥20
  observed meals) with the same pre-registered decision rule per patient.
- Add nutritional goal constraints (protein/fiber targets) to meal ranking.
- Prospective silent-shadow evaluation: log predictions without showing them,
  then measure real predicted-vs-observed calibration.
- Expand beyond one dataset; multi-site validation before any clinical claim.

---

## Appendix A — key files

| Path | Contents |
|---|---|
| `src/preprocess.py`, `splits.py`, `baselines.py`, `features.py`, `train.py`, `conformalize.py`, `twin_state.py`, `ranking.py`, `run_phase1.py` | Full pipeline |
| `dashboard/app.py` | 7-page Streamlit app |
| `models/` | 5 LightGBM boosters, `metrics.json`, `metrics_ci.json`, `calibration.json`, `personalization.json`, `ranking.json`, `feature_importance.json`, `conformal.json`, `oof_predictions.csv` |
| `data/processed/meals.csv` | 1,613 de-duplicated meal events (1,579 usable) |
| `configs/phase1.yaml`, `configs/phase2.yaml` | Fixed experiment configs (seed 42) |
| `results/phase1/` | Phase-1 metrics snapshot |
| `docs/` | architecture.pdf/.pptx, AahaarTwin_Presentation.pptx, DEMO_SCRIPT.md, SUBMISSION_AUDIT.md, DEV_LOG.md, make_plots.py, make_architecture.py |
| `tests/` | 31 tests (8 + 9 + 5 + 9), all passing |
