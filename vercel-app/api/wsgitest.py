"""Minimal WSGI test — no imports beyond stdlib, no file I/O."""


def app(environ, start_response):
    start_response("200 OK", [("Content-Type", "application/json")])
    return [b'{"wsgi": "works"}']


application = app
