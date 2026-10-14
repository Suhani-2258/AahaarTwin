# Phase 1 Report — Reproducible data & baseline pipeline

**Date:** 2026-10-13 · **Status:** COMPLETE · **Tests:** 8/8 passing
**Artifacts:** `data/processed/meals.csv`, `results/phase1/metrics.json`, `results/phase1/config_snapshot.yaml`

> Research prototype only — not a medical device; not for diagnosis or treatment.

## 1. Preprocessing results

| Step | Result |
|---|---|
| Participant CSVs found | 45 |
| Raw meal-event rows | 1,706 |
| Schema variants across files | 9 distinct column sets (documented in `metrics.json`: 2 files lack Amount Consumed; 1 has Steps instead of activity calories; 11 use Intensity instead of METs; 1 has extra Sugar column; trailing-space variants) |
| Column normalization | whitespace stripped; variant names unified (`dexcom_gl`, `meal_type`, `amount_consumed`, `activity_kcal`, `image_path`) |
| Meal-type normalization | case variants unified → breakfast (436), lunch (434), dinner (445), snack (298) |
| Dedup (<30 min gap, same patient) | **93 rows dropped** → 1,613 meal events |
| Usable meals (valid iAUC + macros + pre-meal glucose) | **1,579** (98% of deduped) |
| Patients | 45, all with ≥10 usable meals (min 17, median 38, max 63) |
| Excursion rate P(peak ≥180 mg/dL in 2h) | **39.5%** |

Missingness on pre-meal features: glucose 1.2%, HR 2.9%, activity 1.3%.

## 2. Data-split details (leakage-safe)

- **Within-patient time-ordered split:** per patient with ≥10 usable meals, first 50% of meals
  chronologically → reference set, last 50% → test set. No shuffling. All 45 patients qualify.
- **Population evaluation:** leave-one-patient-out (45 folds); test patient IDs never appear in training.
- **Leakage checks (asserted in code, all passing):** every test meal timestamp is strictly after
  every reference meal timestamp per patient; no index overlap; no patient appears in both
  train and test of any LOPO fold.

## 3. Reproduced baselines (iAUC_2h MAE, mg/dL·h)

| Baseline | MAE | Notes |
|---|---|---|
| A. Patient-mean | 34.04 | predicts test iAUC with patient's own reference mean |
| B. Carb-only linear (per patient) | 34.21 | beats patient-mean in 47% of patients |
| C. Population linear (LOPO, 6 features) | 38.22 | beats patient-mean in only 33% of patients |
| D. Personal linear (per patient, 5 features) | 37.37 | beats population linear in **67% of patients** |

Carb–iAUC Spearman: mean 0.326, median 0.297, positive in all 45 patients (n=1,579 meals).
Overall iAUC std: 51.2 — baselines leave large headroom, as expected for linear models.

## 4. Discrepancies vs the feasibility report (2026-10-12)

All differences trace to **one deliberate change**: the feasibility script had no dedup rule;
Phase 1 drops 93 near-duplicate meal rows (<30 min apart, same patient; 13 pairs were <5 min
apart — clear double-logged meals). Keeping them would leak near-duplicates across the
time-ordered split.

| Metric | Feasibility (no dedup) | Phase 1 (dedup) |
|---|---|---|
| Usable meals | 1,672 | 1,579 |
| Patient-mean MAE | 33.79 | 34.04 |
| Carb-only MAE | 34.16 | 34.21 |
| Population linear MAE | 37.94 | 38.22 |
| Personal linear MAE | 36.28 | 37.37 |
| Personal beats population | 71% | 67% |
| Excursion rate | 38.8% | 39.5% |
| Carb–iAUC Spearman mean | 0.308 | 0.326 |

Direction of conclusions is unchanged: carbs carry real signal in every patient;
personal linear already beats population linear in two-thirds of patients;
nonlinear models have clear headroom. The deduped numbers above are the new canonical values.

## 5. Test results

8/8 passing (`python3 tests/test_phase1.py`):
processed file exists · required columns · no duplicate meal events (≥30 min gap) ·
meal types normalized · usable subset sane (45 patients, ≥10 meals each) ·
excursion label consistent with peak ≥180 · splits leakage-free (time-ordered + LOPO) ·
metrics match canonical values.

## 6. Blockers

None. Phase 1 is complete and gated: **awaiting approval before Phase 2** (population model,
excursion classifier, personalization experiment).
