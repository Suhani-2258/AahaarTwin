# AahaarTwin — Submission Audit

**Date:** 2026-10-13 · **Auditor:** final documentation/audit agent
Challenge: Happiest Health Digital Twin Challenge 2026
Registration closes **18 Oct 2026, 7:00 PM IST** · Submission closes **20 Oct 2026, 7:00 PM IST**

> Research prototype — not a medical device; not for diagnosis or treatment.

## Requirement-by-requirement table

| # | Requirement | Status | Evidence / file path | Test / verification | Remaining issue |
|---|---|---|---|---|---|
| 1 | Registration before deadline | ⏳ PENDING (user) | — | Calendar dates verified from challenge info | Register by **18 Oct 2026 7PM IST** |
| 2 | Submission before deadline | ⏳ PENDING (user) | — | — | Submit by **20 Oct 2026 7PM IST** |
| 3 | Public GitHub repo | ⏳ PENDING (user) | Local git repo initialized 2026-10-13 (commit hash in DEV_LOG); **no remote configured** | `git status` + `git remote -v` (empty) | Create repo + push (commands in §Outstanding actions) |
| 4 | README with all 12 items | ✅ DONE | `README.md` — team, title, problem, use case, stack, model details, video link slot, license, architecture pointer, presentation pointer, quickstart, provenance | All referenced doc paths exist; links checked | Video link placeholder until recorded |
| 5 | 15–20 min unlisted YouTube video | ⏳ PENDING (user) | `docs/DEMO_SCRIPT.md` — timed 20-min script, recording checklist, fixed demo scenario (pid 7 + 3 demo meals), forbidden-claims list, upload instructions | Script reviewed; no video claimed to exist | **Record + upload (Sujal), then paste link into README §7** |
| 6 | Architecture diagram PDF + PPTX | ✅ DONE | `docs/architecture.pdf` (matplotlib), `docs/architecture.pptx` (title + embedded diagram); regenerable via `docs/make_architecture.py` | PDF/PPTX open and render (visual check of PNG) | None |
| 7 | Presentation PDF/PPTX | ✅ DONE (PPTX) | `docs/AahaarTwin_Presentation.pptx` — 13 slides, real plots only | PPTX re-opens; valid zip; 13 slides | PDF export of slides is optional (not required — "PDF/PPT" means either) |
| 8 | Data rules: anonymized/open/synthetic; no raw redistribution; citation | ✅ DONE | Raw CGMacros NOT in repo (`data/raw/` absent, no zips); `data/processed/meals.csv` derived + pseudonymized, generated locally, intentionally not committed; `README.md` §8 + `LICENSE` scope note cite CGMacros v1.0.0, DOI 10.13026/3z8q-x658, PhysioNet URL; shipped LICENSE.txt empty → terms unverified, flagged | `find` for *.zip/raw CSVs (only derived artifacts present) | None |
| 9 | Mandatory: static + dynamic fusion → adverse-event prediction | ✅ DONE | 25 features (8 static biometrics + dynamic CGM/HR/activity/time); excursion classifier (peak ≥180) PR-AUC 0.798 | `models/metrics.json`, 31/31 tests | None |
| 10 | Mandatory: conceptual doctor dashboard | ✅ DONE | `dashboard/app.py` — 7 pages, artifacts-or-crash startup, disclaimers every page, auto-seeds DB if missing | 31/31 tests; smoke test HTTP 200 | None |
| 11 | No secrets / PII | ✅ DONE | No credentials in repo; patient IDs are numeric pseudonyms; `.gitignore` excludes venv/db/pycache | `git status` shows no `.env`/keys staged | None |
| 12 | Install + startup instructions | ✅ DONE | `README.md` §11 quickstart (venv, pip install, run_phase1, train, streamlit run); `requirements.txt` pinned | Auto-seed path verified on temp DB (45 patients, 1,579 meals) | Clean-env reinstall not re-run today (venv pre-existed) |
| 13 | Clean-env reproducibility | ⚠️ PARTIAL | All pipeline scripts + fixed configs (seed 42) + pinned requirements committed; `twin_state.db` auto-seeds on first launch | Full suite re-run 2026-10-13 (see DEV_LOG) | End-to-end rerun from raw CGMacros not possible here (raw data out of repo scope) |
| 14 | Demo video link in README | ⏳ PENDING | `README.md` §7 = **[PENDING — to be recorded by Sujal]** | — | Record, upload unlisted, paste link |
| 15 | Team name / college | ⚠️ PARTIAL | `README.md` §1 now uses verified values from the Unstop registration mail (13 Oct 2026): IIT Patna; Suhani Kumari verified; Aradhya Srivastava confirmation pending (deadline 18 Oct 7:00 PM IST) | Unstop registration mail to sujalgiri5@gmail.com | Team name still unconfirmed; Aradhya's Unstop confirmation still pending |
| 16 | Judging criteria | ❓ UNKNOWN | Not published by organizer (per verified challenge facts) | — | Watch organizer announcements |
| 17 | App screenshots in presentation | ✅ DONE | `docs/screenshots/*.png` — 7 genuine screenshots captured 2026-10-14 from the running Streamlit app via headless Chromium (playwright); 6 embedded on 2 new slides ("Live dashboard — …") in `docs/AahaarTwin_Presentation.pptx` (now 15 slides) | Screenshots visually verified (real metrics, no errors) before embedding | None |
| 18 | Disclaimer present | ✅ DONE | "not a medical device; not for diagnosis or treatment" in README (top), TECHNICAL_REPORT, DEMO_SCRIPT, app sidebar + Safety page, every presentation slide, LICENSE-adjacent docs | Grep counts verified | None |

## Fixes applied during this audit

1. **`.gitignore` extended** — added `.venv/`, `*.pyo`, `twin_state.db`, `results/train_phase2.log`, `.streamlit/` (kept existing `data/raw/`, `*.zip`, `__pycache__/`, `*.pyc`).
2. **App resilience** — `dashboard/app.py` `main()` no longer errors when `twin_state.db` is missing; it auto-seeds via `src.twin_state.seed_from_meals` with a spinner and a sidebar confirmation. Verified on a temp DB: 45 patients / 1,579 meals / 1,579 predictions, drift bound 48.58. Temp file removed.
3. **README rewritten** — now covers all 12 submission items + quickstart + honest results + limitations + submission checklist; corrected the project-title gloss and added the PhysioNet URL from the previously verified citation.
4. **`LICENSE` added** — MIT, Copyright 2026 Sujal Giri, code-only scope note (does not cover CGMacros).
5. **Old demo-patient stats double-checked** — pid 7: 17 usable meals, OOF iAUC MAE 28.10 / peak MAE 20.88 (from `models/metrics.json` `per_patient`) — used in `docs/DEMO_SCRIPT.md`.
6. **Architecture feedback-loop index bug fixed** — `docs/make_architecture.py` referenced a non-existent stage index; corrected (drift loop → dashboard stage) and regenerated both outputs.

## Outstanding actions (for Sujal / parent agent)

1. **Confirm team name** → the only remaining `[TO BE CONFIRMED]` in `README.md` §1; college (IIT Patna) and members (Suhani Kumari verified, Aradhya Srivastava pending) are now filled from the Unstop registration mail.
1b. **Aradhya Srivastava must confirm on Unstop by 18 Oct 2026, 7:00 PM IST** — registration is complete, her confirmation is the pending step.
2. **Record + upload demo video** → follow `docs/DEMO_SCRIPT.md`; paste the unlisted YouTube link into `README.md` §7 and mark row 5/14 done here.
3. **Create the public GitHub repo and push** (exact commands):
   ```bash
   cd ~/workspace/digital-twin-challenge/aahaartwin
   git remote add origin https://github.com/<user>/<repo>.git
   git branch -M main
   git push -u origin main
   ```
   ⚠️ Verify `git status` shows no `twin_state.db`, `.venv/`, `__pycache__/`, or raw data before pushing.
4. **Register** by 18 Oct 2026 7PM IST; **submit** by 20 Oct 2026 7PM IST.
5. Optional: export `docs/AahaarTwin_Presentation.pptx` to PDF; add real app screenshots to the deck; watch for judging-criteria publication.
