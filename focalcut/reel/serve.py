"""A local server for the review pages: serves the footage folder (with byte ranges, so the browser can seek the
exports) and lets the pages read and write focalcut/reel/picks.json directly -- the editor's picks land in the file
as they make them, with no download or copy step.

    focalcut reel serve <folder> [--port 8765]   ->  http://localhost:8765/   (the source browser)

GET/PUT /api/picks reads / replaces picks.json (the previous file is kept as picks.json.bak). Only localhost.
"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


def handler(project):
    root = os.path.realpath(project.folder)
    picks = project.path("picks.json")
    start = "/" + os.path.relpath(project.path("review", "sources.html"), root).replace(os.sep, "/")

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store" if self.path.endswith((".html", "/api/picks")) else "max-age=3600")
            super().end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(302)
                self.send_header("Location", start)
                self.end_headers()
                return
            if path == "/api/picks":
                body = open(picks, "rb").read() if os.path.exists(picks) else b"{}"
                return self.reply(200, body)
            f = os.path.realpath(os.path.join(root, unquote(path).lstrip("/")))
            if not f.startswith(root + os.sep) or not os.path.isfile(f):
                return self.reply(404, b'{"error": "not found"}')
            self.send_file(f)

        def do_PUT(self):
            if urlparse(self.path).path != "/api/picks":
                return self.reply(404, b'{"error": "not found"}')
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                assert isinstance(data, dict)
            except (ValueError, AssertionError):
                return self.reply(400, b'{"error": "picks must be a JSON object"}')
            if os.path.exists(picks):
                shutil.copyfile(picks, picks + ".bak")
            tmp = picks + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=1)
            os.replace(tmp, picks)
            self.reply(200, b'{"ok": true}')

        def reply(self, code, body, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, f):
            size = os.path.getsize(f)
            kind = mimetypes.guess_type(f)[0] or "application/octet-stream"
            a, b, code = 0, size - 1, 200
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                lo, _, hi = rng[6:].split(",")[0].partition("-")
                if lo:
                    a, b = int(lo), int(hi) if hi else size - 1
                else:
                    a = max(0, size - int(hi))
                b = min(b, size - 1)
                if a > b:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(b - a + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.end_headers()
            with open(f, "rb") as fh:
                fh.seek(a)
                left = b - a + 1
                try:
                    while left > 0:
                        chunk = fh.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass                                 # the browser seeks by dropping the request
    return H


def run(project, port=8765, host="127.0.0.1"):
    if not os.path.exists(project.path("review", "sources.html")):
        from . import browse
        browse.run(project)
    srv = ThreadingHTTPServer((host, port), handler(project))
    srv.daemon_threads = True
    print(f"[reel] serving {project.folder}\n[reel] open http://localhost:{port}/  (picks save to "
          f"{project.path('picks.json')}; Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
