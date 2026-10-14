"""POST /api/predict — single-meal prediction from the real AahaarTwin models.

Request JSON:
  { "pid": 7, "ts": "2026-10-20T13:00", "meal_type": "lunch",
    "macros": {"carbs": 80, "protein": 15, "fat": 10, "fiber": 3, "calories": 470},
    "premeal": {"g0": 110, "slope_60min": 0.2, "hr_mean_2h": 72, "act_kcal_2h": 40} }

All fields except pid/ts/meal_type/macros are optional; missing values are
median-imputed exactly as in the local pipeline. statics default to the
patient's derived profile (45 de-identified patients, pids 1..45).
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib.inference import predict_one, MEAL_TYPES  # noqa: E402


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
            macros = req.get("macros") or {}
            if not isinstance(macros, dict):
                _send(self, 400, {"error": "macros must be an object"})
                return
            premeal = req.get("premeal") or {}
            statics = req.get("statics")  # optional override; None -> patient profile
            pred = predict_one(pid, ts, meal_type, macros, premeal, statics)
            pred["disclaimer"] = (
                "Model projection (associational, not causal) — research prototype, "
                "not a medical device, not medical advice."
            )
            _send(self, 200, pred)
        except Exception as exc:  # never leak a stack trace
            _send(self, 500, {"error": "prediction failed", "detail": str(exc)[:200]})

    def log_message(self, *args):  # keep serverless logs quiet
        pass
