# Data Quality Report — AahaarTwin Phase 1

**Date:** 2026-10-13 · **Source:** CGMacros v1.0.0 (PhysioNet, DOI 10.13026/3z8q-x658)
**Pipeline:** `src/preprocess.py` → `data/processed/meals.csv` (reproducible, deterministic)

## 1. Schema normalization (audit)
- 45 participant CSVs contain **9 distinct column sets**. Variants handled:
  - trailing whitespace in column names (`'Amount Consumed '`, `'Body weight '`);
  - 2 files lack `Amount Consumed` entirely → filled as NaN;
  - 1 file has `Steps` + `RecordIndex` instead of `Calories (Activity)`;
  - 11 files use `Intensity` instead of `METs`;
  - 1 file has an extra `Sugar` column (ignored);
  - 1 file swaps Dexcom/Libre column order (name-based access, unaffected).
- All files normalized to one schema; missing columns → NaN. Verified by
  `test_required_columns` and per-file column-set logging in `metrics.json`.

## 2. Deduplication (audit)
- 1,706 raw meal-event rows → **93 rows dropped** by the 30-minute minimum-gap rule
  (same patient, <30 min after the previous kept meal) → **1,613 events**.
- Independent check: 99 adjacent pairs <30 min apart, 13 pairs <5 min apart
  (clear double-logged meals). Greedy keep-first dedup drops 93 (chains of 3+
  close meals explain the 99 vs 93 difference).
- Rationale: near-duplicate rows would leak across the time-ordered split.

## 3. Timestamp ordering
- All timestamps parsed with `errors="coerce"`; rows sorted by (pid, ts) before
  any feature/label computation. CGMacros dates are shifted ±365–720 days for PHI;
  ordering within patient is preserved, absolute dates are meaningless (documented).

## 4. Feature construction (pre-meal, leakage-safe)
- `g0`: nearest non-null Dexcom glucose within 15-min lookback (1.2% missing).
- `slope_60min`: least-squares slope of prior 60-min glucose (needs ≥12 points).
- `hr_mean_2h`, `act_kcal_2h`: 2-hour context windows (2.9% / 1.3% missing).
- Meal macros (carbs/protein/fat/fiber/calories) from the meal row itself;
  meal-type normalized to {breakfast, lunch, dinner, snack}.
- No post-meal information is used in any pre-meal feature (asserted by construction:
  all windows are `iloc[i-window : i+1]`, i.e. up to and including meal onset).

## 5. Label construction (post-meal)
- Window: 0–120 min after meal onset on the 1-min grid.
- `iauc_2h`: trapezoidal integral of max(glucose − g0, 0), in mg/dL·h; requires
  ≥80% window coverage (else NaN). Mean 51.0, std 51.2.
- `peak_2h`: max glucose in window; requires ≥50% coverage. Mean 177.1 mg/dL.
- `excursion_2h`: 1 if peak ≥ 180 mg/dL else 0. Event rate **39.5%** (balanced).
- Usable meals (valid iAUC + carbs + g0): **1,579 of 1,613 (98%)**;
  per-patient min 17, median 38, max 63; all 45 patients ≥ 10.

## 6. Missing-data handling
- CGM gaps ≤ interpolation: labels use linear interpolation only when coverage
  thresholds are met; otherwise NaN → meal excluded from `usable`.
- Pre-meal features: NaN preserved in the table; models median-impute at train time
  (documented in `src/baselines.py`; Phase 2 models handle natively).
- No imputation of meal macros; meals without macros are excluded from `usable`.

## 7. Leakage checks (all asserted in code, 8/8 tests passing)
- Time-ordered within-patient split: first 50% of meals (chronological) → reference,
  last 50% → test. Asserted: every test timestamp strictly after every reference
  timestamp per patient; no index overlap.
- Leave-one-patient-out (45 folds): asserted zero patient overlap between train/test.
- Population baselines never see the test patient; personalization never sees future
  meals of the test patient.

## 8. Data provenance & licensing
- Downloaded 2026-10-12 from PhysioNet (primary) / AWS Open Data mirror;
  SHA256-verified: `05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d`.
- Shipped `LICENSE.txt` is **empty (0 bytes)** — license text unverified from official
  files. Raw data is NOT redistributed in this repo; users download from PhysioNet.
  Citation + DOI recorded in `configs/phase1.yaml` and README.

## 9. Known limitations
- n=45 participants; US cohort (no Indian CGM data openly available).
- 10-day window per participant; dinners self-reported; steps effectively in 1/45 files.
- No detailed medication data; controlled breakfasts/lunches, free dinners.
