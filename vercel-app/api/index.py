"""AahaarTwin Vercel serverless API — single function router.

Routes all three endpoints through one BaseHTTPRequestHandler (Vercel's
native Python function format):
  POST /api/predict  — single-meal prediction
  POST /api/rank     — rank candidate meals
  GET  /api/metrics  — measured out-of-fold evaluation metrics (static)

Uses the real LightGBM models via api/_lib/inference.py — predictions are
bit-identical to the local pipeline (tests/test_vercel_parity.py).
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib.inference import (  # noqa: E402
    predict_one,
    rank_candidates,
    MEAL_TYPES,
)

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lib", "models")
with open(os.path.join(_MODELS_DIR, "metrics.json")) as _fh:
    _METRICS = json.load(_fh)

_DISCLAIMER = (
    "Model projection (associational, not causal) — research prototype, "
    "not a medical device, not medical advice."
)


class handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/api/metrics", "/api/index/metrics"):
            self._send(200, _METRICS)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            req = self._read_json()
        except Exception:
            self._send(400, {"error": "invalid JSON body"})
            return
        try:
            if path in ("/api/predict", "/api/index/predict"):
                self._handle_predict(req)
            elif path in ("/api/rank", "/api/index/rank"):
                self._handle_rank(req)
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": "request failed",
                             "detail": str(exc)[:200]})

    def _handle_predict(self, req):
        pid = int(req.get("pid", 7))
        ts = str(req.get("ts", "2026-10-20T13:00"))
        meal_type = str(req.get("meal_type", "lunch")).lower()
        if meal_type not in MEAL_TYPES:
            self._send(400, {"error": f"meal_type must be one of {MEAL_TYPES}"})
            return
        macros = req.get("macros") or {}
        if not isinstance(macros, dict):
            self._send(400, {"error": "macros must be an object"})
            return
        pred = predict_one(pid, ts, meal_type, macros,
                           req.get("premeal") or {}, req.get("statics"))
        pred["disclaimer"] = _DISCLAIMER
        self._send(200, pred)

    def _handle_rank(self, req):
        pid = int(req.get("pid", 7))
        ts = str(req.get("ts", "2026-10-20T13:00"))
        meal_type = str(req.get("meal_type", "lunch")).lower()
        if meal_type not in MEAL_TYPES:
            self._send(400, {"error": f"meal_type must be one of {MEAL_TYPES}"})
            return
        candidates = req.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            self._send(400, {"error": "candidates must be a non-empty list"})
            return
        if len(candidates) > 12:
            self._send(400, {"error": "at most 12 candidates per request"})
            return
        ranked = rank_candidates(pid, ts, meal_type, candidates,
                                 req.get("premeal") or {}, req.get("statics"))
        self._send(200, {"pid": pid, "meal_type": meal_type,
                         "results": ranked, "disclaimer": _DISCLAIMER})
