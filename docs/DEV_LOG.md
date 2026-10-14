# AahaarTwin Development Log

> Research prototype only — not a medical device; not for diagnosis or treatment.

## 2026-10-13 — Phase 0/1: validation
- Created isolated venv `.venv` (python3 -m venv): pandas 2.1.4, numpy 1.26.4, pyyaml.
- Deleted `data/processed/meals.csv` + `results/phase1/metrics.json`; re-ran
  `src/run_phase1.py` from scratch in the fresh venv to confirm reproducibility.
- Re-ran `tests/test_phase1.py` (8/8 expected).
- Wrote `docs/DATA_QUALITY.md` (schema audit, dedup audit, leakage checks, provenance).
- Baselines (canonical, post-dedup): patient-mean MAE 34.04; carb-only 34.21;
  population-linear 38.22; personal-linear 37.37 (beats population in 67% of patients).
- Carb–iAUC Spearman mean 0.326, positive in all 45 patients. Excursion rate 39.5%.
## 2026-10-13 — Phase 1 re-run (fresh venv) PASSED
- Fixed: fresh venv lacked scipy (pandas needs it for Spearman); installed scipy==1.14.1
  with numpy pinned to 1.26.4 (scipy latest had pulled numpy 2.x, breaking pandas 2.1.4).
- Re-ran full pipeline from scratch in venv: 1,579 usable meals, 45 patients —
  numbers EXACTLY reproduce the canonical Phase 1 values.
- tests/test_phase1.py: 8/8 passing in the venv.

## 2026-10-13 — Phase 2: models COMPLETE (acceptance: honest evaluation, no invented numbers)
- Installed in .venv: lightgbm==4.7.0, scikit-learn==1.9.1 (pinned in requirements.txt).
- `src/features.py`: 25 documented features (meal macros, pre-meal CGM/slope, HR/activity,
  circadian sin/cos, meal-type one-hots, 10 bio statics joined on pid — verified 45/45 ID match).
  Median imputation fit on train only. `configs/phase2.yaml`: fixed hyperparams, no tuning on test.
- **BUG FOUND & FIXED during E1/E2:** groupby+reset_index misaligned row indices, so the first
  E1/E2 run compared models on wrong rows (reported NULL win_rate 0.40). Root cause identified,
  fixed (kept original index labels), added alignment assertions. Re-ran clean.
- **Measured results (5-fold patient-level GroupKFold, out-of-fold predictions, n=1,579):**
  - iAUC regression: LightGBM MAE **32.39** (95% CI 29.09–35.73) vs patient-mean 39.09 (OOF) / 34.04 (time-ordered) / population-linear 38.22.
  - Peak regression: MAE **27.10** (CI 24.67–29.46) vs patient-mean 40.37.
  - Excursion classifier (39.5% positive): PR-AUC **0.798** (CI 0.73–0.84), ROC-AUC **0.837** (CI 0.80–0.87);
    at OOF-max-F1 threshold 0.347: P 0.707 / R 0.745 / F1 0.726; at 0.5: P 0.730 / R 0.676 / F1 0.702;
    ECE 0.122 raw → ~0.0 after isotonic calibration (fit on OOF).
  - **E1 personalization: SUPPORTED** — per-patient fine-tune (init_model on first 50% meals) vs
    population model on last 50%: mean ΔMAE **−3.68**, win rate **0.60**, bootstrap 95% CI [−6.96, −0.47]
    (excludes 0; pre-registered rule met, n=45 patients).
  - **E2 ranking: model beats carb-only** — mean Spearman 0.546 vs 0.274; predicted ranking beats
    carb ranking in 82.2% of patients.
  - Uncertainty: raw quantile intervals undercovered (0.605); applied conformalized quantile
    regression (`src/conformalize.py`): adjustment 24.04 → OOF coverage **0.901** at mean width 119.2.
- Artifacts in `models/`: 5 final LightGBM models (trained on ALL usable meals AFTER evaluation),
  metrics.json, metrics_ci.json, feature_importance.json, oof_predictions.csv, calibration.json,
  personalization.json/.csv, ranking.json/.csv, conformal.json, imputer_medians.json,
  feature_order.json, feature_docs.json, config_snapshot.yaml.
- NOTE: final .txt models are for the app only; every reported metric is from OOF predictions.

## 2026-10-13 — Phases 3+4: stateful twin + meal ranking COMPLETE
- **`src/twin_state.py`** — SQLite-backed per-patient response profile.
  Tables: `profiles` (pid, n_meals_observed, carb_sensitivity, ewma_residual,
  last_update_ts, drift_flag, drift_bound), `observations` (source in
  {'observed','simulated'}), `predictions` (immutable once logged),
  `updates_log` (residual, ewma_before/after, drift_flag).
  `predict_meal` builds the 25-feature vector (bio statics via
  `features.load_bio`, median imputation, circadian + one-hot time features),
  runs the 5 app boosters, applies conformal adjustment 24.04 to the iAUC
  interval, and logs evidence score (fraction of 4 pre-meal features present →
  high/medium/low). `record_observation` inserts the observation, and only for
  `source='observed'` + a matching prediction computes residual = obs − pred,
  updates EWMA (α=0.3), least-squares carb_sensitivity (NULL until ≥3 observed
  meals), and evaluates drift. `seed_from_meals` replays each patient's usable
  meals chronologically: predict (prior-meals state only) → record as observed.
- **Drift bound derived:** 1.5 × OOF iAUC MAE from `models/oof_predictions.csv`
  = 1.5 × 32.3858 = **48.5787**, computed at runtime in `init_db` via
  `compute_drift_bound()` and stored per profile. Rationale documented in the
  docstring: sustained rolling mean |residual| > 1.5× typical error over a full
  5-residual window signals systematic shift; a single bad meal cannot trigger
  it. This is a pragmatic multiple, not a clinical threshold.
- **`src/ranking.py`** — `rank_meals(predict_fn, premeal_context, candidates)`
  ranks caller-supplied candidates by predicted iAUC ascending; OOD flag when
  carbs fall outside the training 1st–99th percentile (0.0–148.44 g, computed
  at import from usable meals); explanations built ONLY from model outputs with
  the exact disclaimer "Model projection (associational, not causal) —
  research prototype, not medical advice." and "exploratory — outside observed
  data range" for OOD. Module invents no foods.
- **`twin_state.db`** (project root, 577 KB) seeded by full replay: 45
  patients, 1,579 observations + 1,579 predictions + 1,579 updates. All 45
  carb sensitivities positive (consistent with Phase-1 carb–iAUC finding);
  0 drift flags on the replay (expected: replay residuals are in-sample for
  the final models — demonstration, not evaluation).
- **BUG FOUND & FIXED:** `_evaluate_drift` originally ran *before* the new
  residual was inserted into `updates_log`, so the 5th huge residual never saw
  a full 5-residual window and the drift flag could never fire. Fixed by
  inserting the log row first, then evaluating drift (in both
  `record_observation` and `seed_from_meals`); caught by the new test suite.
- **Tests: 22/22 PASS** — `tests/test_phase1.py` 8/8 (unchanged, re-run),
  `tests/test_twin.py` 9/9 (empty profile init, EWMA direction + exact α=0.3
  arithmetic, sensitivity NULL until 3 meals with slope≈2/3 check, no-future-
  leakage over 5 sequential meals, drift fires only after full window of 5
  huge residuals, evidence low/medium/high, source labels preserved +
  simulated doesn't touch profile, predictions immutable), `tests/test_ranking.py`
  5/5 (ascending sort, 500g-carbs OOD flagged exploratory, disclaimer in every
  explanation, missing macros via imputation, empty input). Run with the venv
  python; plain asserts, no pytest.
- **Limitations (honest):**
  - `predict_meal` uses the population models only; the profile is state +
    provenance — the E1 personal fine-tuning is NOT yet wired into the app
    prediction path.
  - The seed replay is a demonstration of the state machine, not an
    evaluation: final boosters were trained on all usable meals, so replay
    residuals are in-sample.
  - Conformal intervals are symmetric/additive and NOT clipped at 0 — the
    lower bound can go slightly negative (physiologically impossible).
  - excursion_obs is derived from the peak ≥ 180 mg/dL rule (Phase-1 label
    definition), not measured independently.
  - `record_observation`'s `prediction_ref` parameter is accepted but unused;
    matching is by (pid, ts).

## 2026-10-13 — Phases 5+6: dashboard + app tests COMPLETE
- **Built `dashboard/app.py`** (~1,020 lines): single Streamlit app, sidebar
  page radio, fully offline. Pages:
  1. **Overview** — project summary; dataset status (1,579 usable meals /
     45 patients / CGMacros DOI 10.13026/3z8q-x658, from models/metrics.json);
     5-booster status cards (loaded via `st.cache_resource`); OOF metrics with
     95% CIs (from models/metrics.json + metrics_ci.json); limitations warning;
     demo patient selector (pid 35: A1c 8.5%, diabetes range — pid 31: A1c
     4.6%, healthy range; selection illustrative, from bio.csv).
  2. **Patient explorer** — bio statics via `src/features.load_bio`; CGM trend
     (Dexcom GL) + HR trend from raw 1-min CSVs downsampled to 15-min means for
     a selected day (notice shown when the raw dir/file/day is unavailable);
     meal history table (observations LEFT JOIN predictions on (pid, ts) with
     residual column); evidence-quality bar chart.
  3. **Prediction** — patient, meal date/time (default now), meal type, macro
     inputs (validated: `validate_macros` rejects negatives/NaN/out-of-range,
     carbs 0–500); pre-meal context auto-filled from the patient's latest
     usable meal row with editable overrides; outputs pred_iAUC, pred_peak,
     excursion probability, 90% interval (lower bound clipped at 0 for display
     only — documented), evidence label, excursion-risk badge (HIGH ≥ 0.5;
     MODERATE ≥ 0.347 OOF max-F1 threshold; else LOW).
  4. **Meal comparison** — 3 candidate forms prefilled with values explicitly
     labeled "demonstration example" (not nutritional facts); ranks via
     `src/ranking.rank_meals` bound to `twin_state.predict_meal`; plotly bars
     with asymmetric interval error bars + excursion-probability bars with
     threshold lines; per-candidate explanations; OOD exploratory warnings
     (training carbs 1st–99th percentile).
  5. **Twin history** — predicted-vs-observed scatter (+ diagonal) and
     over-time overlay, updates_log table (residual, EWMA α=0.3, drift flag),
     drift status with bound 48.58 (= 1.5 × OOF iAUC MAE) and plain-language
     meaning; observed vs simulated labels everywhere.
  6. **Evaluation** — baseline table (patient-mean 34.04 / carb-only 34.21 /
     population-linear 38.22 / personal-linear 37.37 from
     results/phase1/metrics.json; LightGBM 32.39 CI 29.09–35.73 from
     models/metrics.json + metrics_ci.json; "not computed" CI labels where
     unavailable); classification metrics (PR-AUC 0.798, ROC-AUC 0.837 with
     CIs; thresholds 0.347/0.5 with P/R/F1; ECE 0.122); calibration curve
     from calibration.json; E1 card (win rate 0.60, mean ΔMAE −3.68, CI
     [−6.96,−0.47], verdict SUPPORTED); E2 card (Spearman 0.546 vs 0.274,
     beats carb-only in 82.2%); feature-importance bar (split gain, with
     associational-not-causal note); limitations list.
  7. **Safety** — plain-language explainability (feature groups, what split
     gain means, associational-not-causal), uncertainty, evidence quality,
     drift-bound meaning, full disclaimers, provenance (DOI) + license note
     (terms not re-verified — check source before redistribution).
- Honesty engineering: every displayed metric loads from a real artifact at
  startup (`load_artifacts`, crash-safe RuntimeError listing the missing
  file); `validate_macros`/`validate_premeal` reject negatives/NaN; model
  calls wrapped in try/except with friendly errors; internal absolute paths
  never shown (relative display names only); research-prototype disclaimer in
  the sidebar footer (every page) + Safety page; import of `dashboard.app`
  executes zero UI calls (all pages are functions; `main()` under
  `if __name__ == "__main__"`).
- **Tests: `tests/test_app.py` 9/9 PASS** — import-safe module import,
  predict path returns all required keys on a fixture patient (temp DB),
  negative carbs rejected, carbs>500/NaN/negative-fat rejected, valid macros
  accepted, risk-badge boundaries, interval clipping at 0, all 7 metrics
  JSONs load with expected keys (+ baseline keys in results/phase1), 3-candidate
  ranking integration (ascending, keys, disclaimer in explanation).
- **Full suite: 31/31 PASS** — test_phase1 8/8, test_twin 9/9, test_ranking 5/5,
  test_app 9/9 (venv python, plain asserts, no pytest).
- **Smoke test: PASS** — `streamlit run dashboard/app.py --server.headless true
  --server.port 8507` → `curl localhost:8507` returned **HTTP 200**,
  `/_stcore/health` returned `ok`; server killed afterward, port confirmed free.
- **Limitations (honest):**
  - CGM/HR trend plots depend on the optional raw directory
    (`../feas-data/cgmacros_raw/CGMacros/CGMacros-NNN/CGMacros-NNN.csv`); the
    app degrades gracefully to a notice when it is absent.
  - Every Prediction/Comparison interaction appends rows to the seeded
    `twin_state.db` predictions table (provenance by design); demo use grows
    the DB — consider a copy for heavy demoing.
  - Personal fine-tuning (E1) is evaluated but NOT in the prediction path.
  - Demo patient labels ("diabetes range" via A1c ≥ 6.5%) are illustrative,
    not clinical classifications.
  - Display clipping of the interval lower bound at 0 is documented in-app;
    the stored interval remains symmetric/additive.

## 2026-10-13 — Phases 9–12: submission documentation & audit (final docs agent)

### Files created
- `LICENSE` — MIT, Copyright 2026 Sujal Giri; code-only scope note (does NOT cover CGMacros).
- `docs/make_plots.py` — regenerates 7 presentation plots from real artifacts
  (oof_predictions.csv, calibration.json, feature_importance.json,
  personalization.json, ranking.json, metrics.json, results/phase1/metrics.json).
- `docs/plots/` — p1_pred_vs_obs, p2_calibration, p3_feature_importance,
  p4_personalization, p5_ranking, p6_mae_baselines, p7_intervals (+ architecture.png).
- `docs/make_architecture.py` → `docs/architecture.pdf` + `docs/architecture.pptx`
  (2 slides: title + embedded diagram).
- `docs/TECHNICAL_REPORT.md` — problem, solution, dataset, architecture,
  implementation, methodology, measured results with CIs, limitations, future work.
- `docs/AahaarTwin_Presentation.pptx` — 13 slides, every chart from a real artifact.
- `docs/DEMO_SCRIPT.md` — timed 20-min narration (10 beats), recording checklist,
  fixed demo scenario (pid 7: 17 usable meals, OOF iAUC MAE 28.10 / peak MAE 20.88;
  3 labeled demonstration meals), forbidden-claims list, upload instructions.
  Video status: **[PENDING — to be recorded by Sujal]**.
- `docs/SUBMISSION_AUDIT.md` — 18-row requirement table with status/evidence/
  verification/remaining-issue + outstanding actions.
- `README.md` — rewritten to cover all 12 submission items + quickstart + honest
  results summary + limitations + disclaimer.

### Files modified
- `.gitignore` — added `.venv/`, `*.pyo`, `twin_state.db`, `results/train_phase2.log`,
  `.streamlit/` (kept `data/raw/`, `*.zip`, `__pycache__/`, `*.pyc`).
- `dashboard/app.py` — startup now auto-seeds `twin_state.db` via
  `src.twin_state.seed_from_meals` when missing (spinner + sidebar confirmation)
  instead of erroring; `seed_from_meals` added to the existing twin_state import.

### Test counts (re-run 2026-10-13, venv python, plain asserts)
- `tests/test_phase1.py`: **8/8 PASS**
- `tests/test_twin.py`: **9/9 PASS**
- `tests/test_ranking.py`: **5/5 PASS**
- `tests/test_app.py`: **9/9 PASS**
- **Full suite: 31/31 PASS** (includes the app.py auto-seed edit).

### Issues found + fixed
1. `make_architecture.py` feedback-loop tuple referenced stage index 8 (only 0–7
   exist) → IndexError; fixed to index 7 (dashboard stage) and regenerated.
2. App hard-failed on missing `twin_state.db` → replaced with auto-seed; verified
   on a temp DB (45 patients / 1,579 meals / 1,579 predictions / drift bound 48.58);
   temp file removed.
3. Old README covered only Phase 1 and mis-described the title gloss → rewritten
   for all 12 submission items; added PhysioNet URL from the verified citation.
4. Baseline numbers for the MAE plot initially only partially artifact-sourced →
   confirmed all four (34.04 / 34.21 / 38.22 from `results/phase1/metrics.json`;
   39.09 OOF patient-mean from `models/metrics.json`).
5. No app screenshots possible (no chromium in this environment) → honestly
   documented in SUBMISSION_AUDIT.md row 17 and DEMO_SCRIPT checklist instead of
   faking any.

### Remaining work (user actions)
1. Confirm team name / college / other members → replace `[TO BE CONFIRMED]` in README §1.
2. Record + upload the 20-min unlisted YouTube video (`docs/DEMO_SCRIPT.md`) →
   paste link into README §7.
3. `git remote add origin https://github.com/<user>/<repo>.git` → `git branch -M main` →
   `git push -u origin main` (verify `git status` shows no db/venv/pycache/raw first).
4. Register by 18 Oct 2026 7PM IST; submit by 20 Oct 2026 7PM IST.
5. Optional: PPTX→PDF export, real app screenshots, watch for judging criteria.

### Git status (local only, no remote)
- `git init` done 2026-10-13; one local commit "AahaarTwin submission-ready prototype".
- No remote configured (push boundary respected).

## 2026-10-14 — Targeted improvements (post-audit)
1. README §8 + SUBMISSION_AUDIT row 8: corrected stale "meals.csv is committed" — derived table is generated locally, intentionally untracked.
2. Baseline relabel: "patient-mean 39.09" → "train-fold global mean 39.09 (OOF)" in README results table, TECHNICAL_REPORT §7.1 (table + prose), DEMO_SCRIPT narration. Metric values unchanged.
3. src/ranking.py: new `macros_missing` flag + explicit exploratory explanation when ALL macros missing; surfaced in app Meal comparison page (table column + warning). ood_flag semantics unchanged. 2 new regression tests.
4. Presentation: 7 genuine screenshots captured 2026-10-14 from the running app (headless Chromium via playwright, installed in .venv for this task only — not added to requirements.txt); 6 embedded on 2 new slides (15 slides total), visually verified before embedding.
Tests: 33/33 pass (8+9+7+9). Streamlit health 200. No retraining, no protocol changes.

## 2026-10-14 — Team details verified from Unstop registration mail
- Unstop mail ("Registration Complete | Confirmation pending from member | Digital Twin Challenge 2026", 13 Oct 2026 22:35 IST to sujalgiri5@gmail.com): registration successful. Team = Sujal Giri (leader, verified), Suhani Kumari (verified), Aradhya Srivastava (NOT verified — confirmation pending, deadline 18 Oct 2026 7:00 PM IST). All listed as IIT Patna.
- README §1 updated: college + members filled (team name stays [TO BE CONFIRMED] — not in the mail). TECHNICAL_REPORT §1 and SUBMISSION_AUDIT row 15 + outstanding action 1 updated to match.
- Registration checklist item "register by 18 Oct 7PM IST" is now DONE; remaining team-side action is Aradhya's confirmation.
