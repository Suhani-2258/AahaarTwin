"""GET /api/metrics — measured evaluation metrics (static, from models/metrics.json).

These are the out-of-fold results reported in the README; served as-is so the
frontend can display the honest numbers without recomputation.
"""

import json
import os
from http.server import BaseHTTPRequestHandler

_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_lib", "models", "metrics.json"
)
with open(_METRICS_PATH) as fh:
    _METRICS = json.load(fh)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(_METRICS).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass
