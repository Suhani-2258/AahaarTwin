"""POST /api/rank — rank candidate meals with real LightGBM models."""
import json, os, sys
from http.server import BaseHTTPRequestHandler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib.inference import rank_candidates, MEAL_TYPES
_DISCLAIMER = "Model projection (associational, not causal) — research prototype, not a medical device, not medical advice."
class handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, s, p):
        b = json.dumps(p).encode()
        self.send_response(s)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    def do_POST(self):
        try: req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)).decode() or "{}")
        except Exception: self._send(400, {"error": "invalid JSON"}); return
        try:
            mt = str(req.get("meal_type", "lunch")).lower()
            if mt not in MEAL_TYPES: self._send(400, {"error": f"meal_type must be one of {MEAL_TYPES}"}); return
            cands = req.get("candidates") or []
            if not isinstance(cands, list) or not cands: self._send(400, {"error": "candidates must be non-empty list"}); return
            ranked = rank_candidates(int(req.get("pid", 7)), str(req.get("ts", "2026-10-20T13:00")), mt, cands, req.get("premeal") or {}, req.get("statics"))
            self._send(200, {"results": ranked, "disclaimer": _DISCLAIMER})
        except Exception as e: self._send(500, {"error": "failed", "detail": str(e)[:200]})
