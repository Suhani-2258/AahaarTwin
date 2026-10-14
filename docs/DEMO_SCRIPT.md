# AahaarTwin — 20-Minute Demo Video Script

**Status: [PENDING — to be recorded by Sujal]**
Target: one unlisted YouTube video, ~20 minutes. No video exists yet; do not
claim otherwise.

> Research prototype only — not a medical device; not for diagnosis or
> treatment. Say this on camera in the first 60 seconds.

## Reproducible demo scenario (fixed before recording)

Use **one real patient throughout: pid 7** (17 usable meals; OOF iAUC MAE
28.10, peak MAE 20.88 — mid-pack, representative). Same patient on every page
so the story stays coherent.

**3 candidate meals** (all labeled **demonstration examples**, not real meals):

| # | Label | Carbs (g) | Protein (g) | Fat (g) | Fiber (g) | Calories |
|---|---|---|---|---|---|---|
| A | "White-rice lunch (demo)" | 80 | 15 | 10 | 3 | 470 |
| B | "Lentil + veg lunch (demo)" | 55 | 22 | 12 | 9 | 420 |
| C | "Fried snack plate (demo)" | 45 | 8 | 28 | 2 | 470 |

Enter these on the **Meal comparison** page for pid 7, keep pre-meal context
at the app's auto-filled values, and show the ranked order + excursion-risk
badges + conformal intervals. Then open **Twin history** for pid 7 and point
at one predicted-vs-observed pair.

## Minute-by-minute narration (20:00 total)

| Time | Beat | Screens to show | Say (key lines) |
|---|---|---|---|
| 0:00–1:30 | 1. Problem | Title slide / README | "Generic dashboards give everyone the same glycemic chart. But the same meal spikes glucose in one person and not another. This is a research prototype — not a medical device." |
| 1:30–3:30 | 2. Twin concept | Slides 3–4 (solution, architecture) | "AahaarTwin is a per-patient digital twin: static data like HbA1c fused with dynamic streams — CGM, heart rate, activity, time of day — to predict YOUR response to a meal before you eat it." |
| 3:30–6:00 | 3. Dataset + methodology | Slide 5 (data+method) | "We used CGMacros v1.0.0, 45 patients, 1,579 usable meal events. Evaluation is 5-fold patient-level — a patient is never in train and test at once. Every number I'll show is out-of-fold." |
| 6:00–8:30 | 4. App walkthrough | Streamlit: Overview → Patient explorer (pid 7) | "The dashboard has 7 pages. Here's patient 7 — 17 meals. Every metric loads from a real artifact file; disclaimers on every page." |
| 8:30–11:00 | 5. Patient-specific prediction | Prediction page, pid 7 | "For patient 7 right now, a meal's predicted 2-hour glucose load, peak, excursion probability with a conformalized 90% interval — not just a point guess." |
| 11:00–13:30 | 6. Meal comparison | Meal comparison: meals A, B, C | "Three demonstration meals, same patient, same moment. The twin ranks them and flags excursion risk — this is the pre-meal decision the product exists for." |
| 13:30–15:00 | 7. Predicted-vs-observed | Twin history, pid 7 | "The twin keeps score: every prediction is paired with what actually happened. Drift flags fire when errors shift systematically over 5 meals." |
| 15:00–17:00 | 8. State updates | Twin history / Safety page | "Profiles update meal by meal. The Safety page explains uncertainty and evidence quality in plain language — including what the model does NOT know." |
| 17:00–19:00 | 9. Evaluation | Evaluation page + slides 6–10 | "iAUC MAE 32.39 vs a 39.09 train-fold-mean baseline — in our patient-level cross-validation the test patient's own meals are never in training, so this baseline is the mean of the other patients' meals. Excursion PR-AUC 0.798. Our raw intervals undercovered at 60.5% — we fixed it with conformalization to 90.1% and we're showing you the fix, not hiding it. Personalization supported by a pre-registered rule; ranking beats carb-only in 82% of patients." |
| 19:00–20:00 | 10. Limitations + future | Slide 13 | "Limits: no clinical validation, 45 patients from one dataset, no generalization claim, directional not precise. Next: personal fine-tuning in the live path, prospective shadow evaluation. Thank you." |

## Recording checklist

- [ ] Fresh terminal: `.venv/bin/streamlit run dashboard/app.py` (DB auto-seeds if missing)
- [ ] Browser at 1280×800 or larger; zoom 100%; hide bookmarks bar
- [ ] App sidebar shows "Artifacts: models/*.json" caption on every page (proof of real metrics)
- [ ] Say the disclaimer in the first 60 seconds AND show it on screen
- [ ] Use pid 7 and meals A/B/C exactly as scripted — don't improvise new claims mid-take
- [ ] Label candidate meals as "demonstration examples" verbally and on screen
- [ ] No app screenshots needed in slides (none available — noted honestly in the audit)
- [ ] Record in one or two takes; total runtime 18:00–21:00 (target 20:00)
- [ ] Export 1080p; check audio levels before uploading

## Claims that must NOT be made

- ❌ Any clinical validation, diagnostic, or treatment claim
- ❌ Generalization to India / any population beyond the 45-patient dataset
- ❌ Real-time / live-patient monitoring claims (the twin replays historical meals; inference is per-meal, not streaming)
- ❌ Superiority beyond measured CIs (e.g. don't claim "beats patient-mean" without noting the narrower time-ordered gap)
- ❌ That E1 personal fine-tuning is in the live prediction path (it is evaluated, not shipped)
- ❌ That a video exists before it is recorded and uploaded

## Upload instructions

1. Upload to YouTube as **Unlisted** (not public, not private).
2. Title: `AahaarTwin — Personalized Meal-Response Digital Twin (Happiest Health Digital Twin Challenge 2026)`.
3. Description: one-line summary + "Research prototype — not a medical device" + repo link (after push).
4. Duration check: 18–21 min. Captions: auto-generate, then skim for the disclaimer line.
5. Paste the video link into README §7 (replacing **[PENDING]**) and into `docs/SUBMISSION_AUDIT.md`.
