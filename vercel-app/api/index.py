"""AahaarTwin Vercel serverless API — WSGI application entrypoint.

Single WSGI app serving all three endpoints with the REAL LightGBM models:
  POST /api/predict  — single-meal prediction
  POST /api/rank     — rank candidate meals
  GET  /api/metrics  — measured out-of-fold evaluation metrics (static)

Uses the same inference module (api/_lib/inference.py) as the local pipeline —
predictions are bit-identical to src/twin_state.py (enforced by
tests/test_vercel_parity.py). No mocks, no hardcoded responses.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib.inference import (  # noqa: E402
    predict_one,
    rank_candidates,
    MEAL_TYPES,
    MODEL_VERSION,
)

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lib", "models")
with open(os.path.join(_MODELS_DIR, "metrics.json")) as _fh:
    _METRICS = json.load(_fh)

_DISCLAIMER = (
    "Model projection (associational, not causal) — research prototype, "
    "not a medical device, not medical advice."
)
_JSON = [("Content-Type", "application/json"),
         ("Access-Control-Allow-Origin", "*")]


def _body(environ):
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        length = 0
    if not length:
        return {}
    data = environ["wsgi.input"].read(length).decode("utf-8") or "{}"
    return json.loads(data)


def _resp(start_response, status, payload):
    body = json.dumps(payload).encode("utf-8")
    start_response(status, _JSON + [("Content-Length", str(len(body)))])
    return [body]


def _handle_predict(req):
    pid = int(req.get("pid", 7))
    ts = str(req.get("ts", "2026-10-20T13:00"))
    meal_type = str(req.get("meal_type", "lunch")).lower()
    if meal_type not in MEAL_TYPES:
        return 400, {"error": f"meal_type must be one of {MEAL_TYPES}"}
    macros = req.get("macros") or {}
    if not isinstance(macros, dict):
        return 400, {"error": "macros must be an object"}
    pred = predict_one(pid, ts, meal_type, macros,
                       req.get("premeal") or {}, req.get("statics"))
    pred["disclaimer"] = _DISCLAIMER
    return 200, pred


def _handle_rank(req):
    pid = int(req.get("pid", 7))
    ts = str(req.get("ts", "2026-10-20T13:00"))
    meal_type = str(req.get("meal_type", "lunch")).lower()
    if meal_type not in MEAL_TYPES:
        return 400, {"error": f"meal_type must be one of {MEAL_TYPES}"}
    candidates = req.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return 400, {"error": "candidates must be a non-empty list"}
    if len(candidates) > 12:
        return 400, {"error": "at most 12 candidates per request"}
    ranked = rank_candidates(pid, ts, meal_type, candidates,
                             req.get("premeal") or {}, req.get("statics"))
    return 200, {"pid": pid, "meal_type": meal_type,
                 "results": ranked, "disclaimer": _DISCLAIMER}


def app(environ, start_response):
    """WSGI application callable — Vercel Python entrypoint."""
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")

    if method == "OPTIONS":
        start_response("204 No Content", [
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "POST, GET, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type"),
        ])
        return [b""]

    try:
        if path == "/api/predict" and method == "POST":
            status, payload = _handle_predict(_body(environ))
        elif path == "/api/rank" and method == "POST":
            status, payload = _handle_rank(_body(environ))
        elif path == "/api/metrics" and method == "GET":
            status, payload = 200, _METRICS
        else:
            status, payload = 404, {"error": "not found"}
    except Exception as exc:  # never leak a stack trace
        status, payload = 500, {"error": "request failed",
                                "detail": str(exc)[:200]}

    code = {200: "200 OK", 400: "400 Bad Request",
            404: "404 Not Found", 500: "500 Internal Server Error"}[status]
    return _resp(start_response, code, payload)


# Vercel also accepts `application` as the WSGI callable name.
application = app
