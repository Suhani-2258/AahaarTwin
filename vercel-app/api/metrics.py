"""GET /api/metrics — measured evaluation metrics (static JSON)."""
import json, os
from http.server import BaseHTTPRequestHandler
_d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lib", "models")
with open(os.path.join(_d, "metrics.json")) as _f: _M = json.load(_f)
class handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps(_M).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
