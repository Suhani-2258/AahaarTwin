# AahaarTwin — FINAL VERIFICATION AUDIT (independent)

**Date:** 2026-10-14 · **Auditor:** independent verification pass (not the build agent)
**Scope:** `~/workspace/digital-twin-challenge/aahaartwin/` — all 12 build phases, 31 tests, app, models, docs.
**Method:** inspected files directly, re-ran every test, recalculated metrics from raw OOF artifacts,
functionally exercised the twin state machine (predict → observe → update → drift), executed all 7
dashboard pages headlessly, and read every submission document. Nothing was taken on trust.

> Research prototype — not a medical device; not for diagnosis or treatment.

---

## PHASE 1 — Project verification

### Inventory (all present, git clean, 4 commits, no remote)
- **Source (9 modules):** `src/preprocess.py`, `splits.py`, `baselines.py`, `features.py` (25 documented
  features), `train.py`, `conformalize.py`, `twin_state.py`, `ranking.py`, `run_phase1.py`
- **App:** `dashboard/app.py` (1,031 lines, 7 pages) — verified below
- **Models:** 5 LightGBM boosters (`lgbm_iauc/peak/excursion/iauc_q05/iauc_q95.txt`) + 11 artifact files
  (`metrics.json`, `metrics_ci.json`, `calibration.json`, `conformal.json`, `personalization.json/.csv`,
  `ranking.json/.csv`, `feature_importance.json`, `feature_docs.json`, `feature_order.json`,
  `imputer_medians.json`, `oof_predictions.csv`, `config_snapshot.yaml`)
- **Tests (4 files):** `test_phase1.py`, `test_twin.py`, `test_ranking.py`, `test_app.py`
- **Docs:** `README.md` (12 submission items), `LICENSE` (MIT, code-only), `docs/TECHNICAL_REPORT.md`,
  `docs/architecture.pdf` + `.pptx`, `docs/AahaarTwin_Presentation.pptx` (13 slides, opens, real plots),
  `docs/DEMO_SCRIPT.md` (timed 20-min), `docs/SUBMISSION_AUDIT.md` (18 rows), `docs/DATA_QUALITY.md`,
  `docs/DEV_LOG.md`, `docs/PHASE1_REPORT.md`, `configs/phase1.yaml`, `configs/phase2.yaml`,
  `requirements.txt` (pinned: lightgbm 4.7.0, sklearn 1.9.1, streamlit 1.65.0, …), isolated `.venv/`
- **Git:** clean tree, `0c329ca` HEAD, **no remote** (boundary respected). 62 tracked files.
  No raw CGMacros data, no zips, no secrets, no `.env` in tracking (verified by scan).
  `data/processed/meals.csv` is deliberately UNTRACKED (derived, regenerable; safer boundary).

### Tests — re-ran myself, all pass
| Suite | Result |
|---|---|
| `tests/test_phase1.py` | 8/8 pass |
| `tests/test_twin.py` | 9/9 pass |
| `tests/test_ranking.py` | 5/5 pass |
| `tests/test_app.py` | 9/9 pass |
| **Total** | **31/31 pass** |

### Application — verified by execution
- `streamlit run dashboard/app.py` → **HTTP 200** on `/` and `/_stcore/health` (checked with curl).
- All 7 pages executed headlessly (Streamlit AppTest), **0 exceptions**:
  Overview, Patient explorer, Prediction, Meal comparison, Twin history, Evaluation, Safety.
- Startup loads artifacts from `models/*.json` + `results/phase1/metrics.json` and **refuses to start**
  if any is missing/corrupt ("Metrics are never invented"). No hardcoded metric values in `app.py`
  (grep for 32.39/0.798/0.837/27.10/−3.68 → empty).
- Disclaimer ("not a medical device…") appears 9× in the app (sidebar + Safety page).
- `twin_state.db` auto-seeds from `meals.csv` on first launch if missing (fresh-checkout path).
- ⚠️ Visual page-by-page browser walkthrough was attempted but the live-browser task could not reach
  the local server (it failed on an unrelated page) — pages are verified by headless execution, not
  screenshots. No app screenshots exist in the presentation (honestly documented as omitted).

---

## PHASE 2 — ML verification (independent recalculation from `models/oof_predictions.csv`, n=1,579)

| Metric | Reported | Independently recalculated | Verdict |
|---|---|---|---|
| iAUC MAE (LightGBM) | 32.3858 | **32.3858** | ✅ exact |
| Peak MAE (LightGBM) | 27.1026 | **27.1026** | ✅ exact |
| Excursion event rate | 0.3946 | **0.3946** | ✅ exact |
| Excursion PR-AUC | 0.7975 | **0.7975** | ✅ exact |
| Excursion ROC-AUC | 0.8366 | **0.8366** | ✅ exact |
| P/R/F1 @ thr 0.3469 | 0.7073/0.7448/0.7256 | 0.7069/0.7432/0.7246 | ✅ trivial threshold-rounding diff |
| 95% CI iAUC MAE | 29.09–35.73 | [29.094, 35.731] in `metrics_ci.json` (bootstrap by patient, 2000 resamples) | ✅ |
| Conformal 90% coverage | 0.901 | **0.9006** (raw quantile was 0.605 → honest conformal adjustment 24.04) | ✅ |
| ECE calibration | 0.122 → ~0 | 0.1224 → 0.0 (isotonic) | ✅ |
| E1 personalization ΔMAE / win / CI | −3.68 / 0.60 / [−6.96,−0.47] | exact in `personalization.json`; **protocol audited**: population model trained on *other patients only*, fine-tune on patient's first 50% time-ordered meals, test on last 50%; alignment assertions present; pre-registered rule (win>0.5 AND CI∌0) met → **SUPPORTED** | ✅ |
| E2 ranking Spearman model/carb, win frac | 0.546/0.274/0.822 | exact in `ranking.json` | ✅ |
| Demo patient pid 7 | 17 meals, MAE 28.10/20.88 | `per_patient` records: pid 7 → n=17, mae_iauc=28.1, mae_peak=20.88 | ✅ |

**Leakage:** GroupKFold groups by patient (no patient overlap across folds — asserted in tests);
time-ordered splits never shuffle; E1 excludes the test patient from population training.
**One labeling issue (non-blocking):** `metrics.json` calls the OOF baseline "patient-mean MAE 39.09",
but in GroupKFold-OOF it is computed as the mean of the *train folds* (other patients) — effectively a
global-mean baseline. It is leakage-free and honestly computed; the *name* is misleading.
The true patient-mean baseline (34.04, leakage-free time-ordered protocol) is also beaten by 32.39,
so the "beats every baseline" conclusion stands — but the README table should relabel 39.09 as
train-fold/global mean. My own leaky recalculation (32.93, using test rows in the mean) confirmed
why the honest 39.09 is the correct comparator.

**Build-process note:** one genuine bug was found and fixed mid-build (groupby/reset_index row
misalignment that had produced a false E1 null) — root-caused, fixed, alignment assertions added,
clean re-run. This is evidence the evaluation was actually stress-tested, not rubber-stamped.

---

## PHASE 3 — Digital Twin requirements (functionally exercised, not just read)

| Requirement | Status | Evidence |
|---|---|---|
| Historical health records | ✅ implemented | statics in feature vector: age, gender, BMI, HbA1c, fasting glucose/insulin, triglycerides, cholesterol, HDL, LDL (`STATIC_FEATURES`) |
| Linked dynamic CGM/HR/activity | ✅ implemented | `g0`, `slope_60min`, `hr_mean_2h`, `act_kcal_2h` (`PRE_FEATURES`); fused with statics in 25-feature vector |
| Adverse-event prediction model | ✅ implemented | excursion classifier (peak ≥ 180 mg/dL), PR-AUC 0.798, calibrated |
| Patient-specific response profile | ✅ implemented | SQLite `profiles`: `carb_sensitivity`, `ewma_residual`, `n_meals_observed`, `drift_flag`, `drift_bound`; updated on each observed meal (verified 3→4, EWMA 0.88) |
| Meal comparison + ranking | ✅ implemented | `rank_meals()` — sorted ascending, plain-language explanations, verified (paneer salad 26.5 < high-carb thali 75.8) |
| Prediction-vs-observation tracking | ✅ implemented | `predictions` + `observations` tables join on (pid, ts); `updates_log` records residual/EWMA/drift per event (verified) |
| Uncertainty + evidence-quality | ✅ implemented | conformalized 90% intervals on live predictions (verified `iauc_lo/hi`); evidence score/label from pre-meal context completeness |
| Drift monitoring | ✅ implemented | rolling EWMA vs bound 48.58; **verified tripping to 1** after bad observations |
| Observed vs simulated distinction | ✅ implemented | `source ∈ ('observed','simulated')`; **verified simulated rows leave the profile untouched** |

**Partial / documented limitations:**
1. Live predictions use the **population boosters**; E1 fine-tuning is *evaluated* (SUPPORTED) but not
   wired into the live path — the module docstring states this explicitly. Optional future work, not a blocker.
2. OOD flag fires only when carbs are outside the training 1st–99th percentile; a candidate with
   **all-missing macros is imputed and NOT flagged exploratory** — minor gap, worth one line in limitations.

---

## PHASE 4 — Submission files

| File | Verdict |
|---|---|
| `README.md` | ✅ all 12 submission items; results table matches artifacts exactly; honest `[TO BE CONFIRMED]` (team/college) and `[PENDING]` (video) placeholders — **nothing invented** |
| `LICENSE` | ✅ MIT, code-only, Copyright 2026 Sujal Giri; dataset explicitly excluded |
| Dataset license handling | ✅ CGMacros cited with DOI + PhysioNet URL; raw data NOT in repo; empty shipped `LICENSE.txt` → terms documented as unverified |
| `requirements.txt` | ✅ pinned |
| `docs/architecture.pdf` / `.pptx` | ✅ exist, open |
| `docs/AahaarTwin_Presentation.pptx` | ✅ 13 slides, opens, real artifact plots only, no faked screenshots |
| `docs/DEMO_SCRIPT.md` | ✅ timed 20:00, fixed pid-7 scenario (numbers verified above), forbidden-claims list, upload checklist; states no video exists yet |
| `docs/SUBMISSION_AUDIT.md` | ✅ 18-row table, honest (includes PENDING user actions + unknown judging criteria) |
| PII/secrets | ✅ none in repo (scan clean; pids are numeric pseudonyms) |

**⚠️ Stale sentence found by this audit (proposed fix, not applied):** `README.md` §8 and
`SUBMISSION_AUDIT.md` row 8 say `data/processed/meals.csv` "is committed" — it is now **untracked**
(deliberate, post-audit tightening). Proposed one-line fix: *"Only the derived, anonymized
`data/processed/meals.csv` (regenerable via `src/run_phase1.py`) is used locally; it is not committed."*

---

### Addendum — 2026-10-14 (targeted improvements applied)
1. README §8 + SUBMISSION_AUDIT row 8: stale "meals.csv is committed" corrected — the derived
   `data/processed/meals.csv` is generated locally by `src/run_phase1.py` and intentionally not committed.
2. Baseline relabel: "patient-mean 39.09" → "train-fold global mean 39.09 (OOF)" in README results table,
   `docs/TECHNICAL_REPORT.md` (§7.1 table + prose), and `docs/DEMO_SCRIPT.md` narration. The OOF
   evaluation is patient-level GroupKFold, so the test patient's own meals are never in training and the
   baseline is the mean of the other patients' train folds — leakage-free, just mislabeled before.
3. `src/ranking.py`: new `macros_missing` flag + explicit "exploratory — no nutritional information was
   provided, so this prediction is fully median-imputed" explanation sentence when ALL macros are missing;
   surfaced in the app's Meal comparison page (new "macros" column + warning block). Behavior for valid
   meals unchanged (`ood_flag` semantics untouched). 2 new regression tests → suite now 33/33.
4. Presentation: 7 genuine screenshots captured from the running app (headless Chromium via playwright,
   visually verified) — 6 embedded on 2 new slides (15 slides total). Captions state date + provenance.

## PHASE 5 — Final report

### Verified completed work
Reproducible pipeline (1,579 usable meals, 45 patients, leakage-safe) → trained/evaluated LightGBM
models (all headline numbers independently recalculated and matching) → stateful twin (predict/observe/
update/drift verified by execution) → meal ranking (verified) → 7-page Streamlit app (launches, all pages
execute, artifacts-or-crash, disclaimers) → 31/31 tests passing (re-ran) → full docs (README/LICENSE/
report/architecture/presentation/demo script/audit) → local git, clean, no remote, no raw data, no secrets.

### Failed tests and bugs
**None outstanding.** 31/31 pass. One historical bug (E1 row misalignment) was found and fixed during the
build with regression assertions added.

### Unverified claims
- **None in the code/docs I could falsify.** The one labeling imprecision: "patient-mean 39.09" is
  train-fold global mean (recommend relabel; conclusion unaffected).
- Visual browser walkthrough of pages: not possible in this environment (headless execution verified instead).
- Judging criteria: still unpublished by organizer (documented as unknown, correctly).

### Missing submission requirements (all require Sujal — nothing technical missing)
1. Team details → replace `[TO BE CONFIRMED]` in README §1
2. 20-min unlisted YouTube video → record per `docs/DEMO_SCRIPT.md`, paste link in README §7
3. GitHub push → `git remote add origin <url> && git branch -M main && git push -u origin main`
4. Register by **18 Oct 2026 7PM IST**; submit by **20 Oct 2026 7PM IST**

### Exact commands to run the application
```bash
cd ~/workspace/digital-twin-challenge/aahaartwin
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # fresh env (~1 min)
.venv/bin/python src/run_phase1.py --config configs/phase1.yaml     # needs CGMacros from PhysioNet
.venv/bin/python src/train.py --config configs/phase2.yaml          # trains + evaluates
.venv/bin/streamlit run dashboard/app.py                            # launch (http://localhost:8501)
# tests:
.venv/bin/python tests/test_phase1.py && .venv/bin/python tests/test_twin.py \
  && .venv/bin/python tests/test_ranking.py && .venv/bin/python tests/test_app.py
```

### Prioritized remaining tasks
1. **P0 (done 2026-10-14):** stale "meals.csv is committed" sentence fixed in README §8 +
   SUBMISSION_AUDIT row 8; "patient-mean 39.09" relabeled as train-fold/global mean in README results
   table, TECHNICAL_REPORT §7.1, and DEMO_SCRIPT narration.
2. **P0:** confirm team details → replace placeholders.
3. **P0:** record + upload 20-min video → README link.
4. **P0:** create GitHub repo + push (commands above).
5. **P0:** register (≤18 Oct 7PM IST) and submit (≤20 Oct 7PM IST).
6. **P1 (optional):** export presentation to PDF; add real app screenshots to the deck.
7. **P1 (optional):** wire E1 fine-tuned models into the live prediction path (differentiator upgrade).
8. **P2:** watch for judging-criteria publication.

### Recommendation: **GO (with the P0 doc touch-ups)**
The prototype is real, tested, and honest — every headline number was independently recalculated from
artifacts and matched, the twin mechanics were functionally exercised (drift trips, simulated rows are
isolated, pred-vs-obs joins), and all 31 tests pass on re-run. No fabricated metrics, no leaked labels,
no raw data in the repo, disclaimers everywhere. The only content fixes are two stale sentences; the rest
is genuinely in Sujal's hands (team details, video, push, register, submit).
