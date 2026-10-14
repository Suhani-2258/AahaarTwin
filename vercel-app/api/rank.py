"""POST /api/rank — rank candidate meals by predicted 2h glucose load.

Request JSON:
  { "pid": 7, "ts": "2026-10-20T13:00", "meal_type": "lunch",
    "candidates": [
      {"label": "White-rice lunch (demo)", "macros": {"carbs": 80, ...}},
      {"label": "Roti + dal (demo)", "macros": {"carbs": 45, ...}}
    ],
    "premeal": {...} }

Returns candidates sorted by predicted iAUC ascending, each with an
output-based explanation, OOD flag and macros-missing flag — mirroring
src/ranking.py. Predictions come from the real LightGBM boosters.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib.inference import rank_candidates, MEAL_TYPES  # noqa: E402


def _read_json(handler):
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if not length:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8") or "{}")


def _send(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            req = _read_json(self)
        except Exception:
            _send(self, 400, {"error": "invalid JSON body"})
            return
        try:
            pid = int(req.get("pid", 7))
            ts = str(req.get("ts", "2026-10-20T13:00"))
            meal_type = str(req.get("meal_type", "lunch")).lower()
            if meal_type not in MEAL_TYPES:
                _send(self, 400, {"error": f"meal_type must be one of {MEAL_TYPES}"})
                return
            candidates = req.get("candidates") or []
            if not isinstance(candidates, list) or not candidates:
                _send(self, 400, {"error": "candidates must be a non-empty list"})
                return
            if len(candidates) > 12:
                _send(self, 400, {"error": "at most 12 candidates per request"})
                return
            premeal = req.get("premeal") or {}
            statics = req.get("statics")
            ranked = rank_candidates(pid, ts, meal_type, candidates, premeal, statics)
            _send(self, 200, {
                "pid": pid,
                "meal_type": meal_type,
                "results": ranked,
                "disclaimer": (
                    "Model projections (associational, not causal) — research prototype, "
                    "not a medical device, not medical advice."
                ),
            })
        except Exception as exc:
            _send(self, 500, {"error": "ranking failed", "detail": str(exc)[:200]})

    def log_message(self, *args):
        pass
