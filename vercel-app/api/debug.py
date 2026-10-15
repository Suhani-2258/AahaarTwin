"""Minimal debug endpoint — no lightgbm, no _lib imports."""


def app(environ, start_response):
    start_response("200 OK", [("Content-Type", "application/json")])
    return [b'{"debug": "wsgi works", "path": "%s"}' % environ.get("PATH_INFO", "/").encode()]


application = app
