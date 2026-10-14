# AahaarTwin — Vercel serverless demo

A stateless, Vercel-compatible companion to the full Streamlit application.
The Streamlit app cannot run on Vercel (it needs a long-running server process),
so this directory exposes the **same trained models** through serverless API
functions plus a static single-page frontend.

## What runs here

| Route | Description |
|---|---|
| `GET /` | Static demo page (`public/index.html`) |
| `POST /api/predict` | Single-meal prediction from the real LightGBM boosters |
| `POST /api/rank` | Rank candidate meals (mirrors `src/ranking.py`) |
| `GET /api/metrics` | Measured out-of-fold evaluation metrics (static) |

## Honest scope

- Predictions are produced by the **real** population boosters
  (`lgbm-population-v1`) with the same feature order, median imputation and
  conformal adjustment as the local app. Parity is enforced by
  `tests/test_vercel_parity.py` (bit-identical outputs).
- This demo is **stateless**: no twin database, no drift monitoring, no profile
  updates. Those live in the Streamlit app (`dashboard/app.py`).
- Patient statics (`api/_lib/models/patient_statics.json`) are a derived,
  de-identified subset (10 model features × 45 pseudonymous pids) — no raw
  dataset files are shipped.

## Deploy

Set the Vercel project's **Root Directory** to `vercel-app`. No environment
variables are required. Python 3.12 runtime; `requirements.txt` pins the exact
inference dependencies (lightgbm 4.7.0, numpy 1.26.4, scipy 1.14.1).
