"""A local server for the review pages: serves footage folders (with byte ranges, so the browser can seek the
exports) and lets the pages read and write each project's focalcut/reel/picks.json directly -- the editor's picks
land in the file as they make them, with no download or copy step. One process serves any number of projects.

    focalcut reel serve <folder> [<folder> ...] [--root <dir>] [--port 8765]   ->  http://localhost:8765/

Each project lives under /p/<slug>/ (the slug from its folder name: "SSMC 2026" -> ssmc-2026); its files resolve
against its own folder, and under that prefix:
    GET/PUT api/picks    reads / replaces picks.json (the previous file is kept as picks.json.bak)
    GET     api/crop     ?id=<export file>&a=<s>&b=<s>&fmt=portrait: the crop the planner derives for that stretch,
                         before any focal point the editor set (path times are seconds from a)
/ is an index of the projects and their pages. GET /api/projects lists them; POST /api/projects {"folder": ...}
adds one to the running server; --root <dir> also picks up any <dir>/*/focalcut/reel/reel.json as it appears.
With a single folder (and no --root) the project is also served at the top level, as before: / opens its sources
page and /api/picks is its picks. Localhost only.
"""
from __future__ import annotations

import html
import json
import mimetypes
import os
import re
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

from .project import Project

LOCAL = {"localhost", "127.0.0.1", "[::1]"}


def slug(folder: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", os.path.basename(os.path.normpath(folder)).lower()).strip("-") or "project"


def is_project(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, "focalcut", "reel", "reel.json"))


class Site:
    """The projects one server holds, by slug; more can join while it runs (add, or a scan of the root folder)."""

    def __init__(self, projects=(), root=None):
        self.projects, self.root, self.units = {}, root and os.path.realpath(root), {}
        self.lock = threading.Lock()
        for p in projects:
            self.add(p)
        self.scan()

    def add(self, project: Project) -> str:
        real = os.path.realpath(project.folder)
        with self.lock:
            for s, p in self.projects.items():
                if os.path.realpath(p.folder) == real:
                    return s
            base = s = slug(real)
            n = 2
            while s in self.projects:
                s, n = f"{base}-{n}", n + 1
            self.projects[s] = project
            return s

    def scan(self):
        if not self.root or not os.path.isdir(self.root):
            return
        for d in sorted(os.listdir(self.root)):
            f = os.path.join(self.root, d)
            if is_project(f):
                self.add(Project(f))

    def get(self, s: str) -> Project | None:
        if s not in self.projects:
            self.scan()
        return self.projects.get(s)

    @property
    def single(self) -> Project | None:
        """The project served at the top level too: the only one, when there's no root to grow from."""
        return next(iter(self.projects.values())) if len(self.projects) == 1 and not self.root else None


def pages(project: Project) -> list[tuple[str, str]]:
    """(label, path relative to the project folder) of its review pages that exist: sources first, then each cut."""
    rev = project.path("review")
    if not os.path.isdir(rev):
        return []
    rel = lambda f: os.path.relpath(os.path.join(rev, f), project.folder).replace(os.sep, "/")
    out = [("sources", rel("sources.html"))] if os.path.exists(os.path.join(rev, "sources.html")) else []
    order = {"landscape": 0, "portrait": 1}
    cuts = sorted((f[:-5] for f in os.listdir(rev) if f.endswith(".html") and f != "sources.html"),
                  key=lambda c: (order.get(c, 2), c))
    return out + [(c, rel(c + ".html")) for c in cuts]


def index(site: Site) -> bytes:
    site.scan()
    rows = []
    for s, p in sorted(site.projects.items()):
        links = " · ".join(f'<a href="/p/{quote(s)}/{quote(path)}">{html.escape(label)}</a>' for label, path in pages(p))
        rows.append(f"<li><b>{html.escape(os.path.basename(os.path.normpath(p.folder)))}</b> "
                    f"<code>/p/{html.escape(s)}/</code><br>{links or '<i>no review pages yet</i>'}</li>")
    return INDEX.replace("%ROWS%", "".join(rows) or "<li><i>no projects</i></li>").encode()


INDEX = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>focalcut projects</title><style>
:root{--bg:#f3f3f5;--fg:#15161a;--mut:#5d6270;--a:#087ea4;--line:#d9dbe0}
@media (prefers-color-scheme:dark){:root{--bg:#0e0f12;--fg:#e8e8ea;--mut:#8b8f99;--a:#4cc9f0;--line:#2c303a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif;padding:24px 16px}
main{max-width:720px;margin:auto}h1{font-size:20px}ul{list-style:none;padding:0}
li{padding:12px 0;border-bottom:1px solid var(--line)}a{color:var(--a)}code{color:var(--mut);font-size:12px}
</style><main><h1>focalcut — projects</h1><ul>%ROWS%</ul></main>"""


def host_name(h: str) -> str:
    """The host part of a Host header: "localhost:8765" -> localhost, "[::1]:8765" -> [::1]."""
    return h.split("]")[0] + "]" if h.startswith("[") else h.rsplit(":", 1)[0]


def handler(site):
    if isinstance(site, Project):
        site = Site([site])

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            p = urlparse(self.path).path
            self.send_header("Cache-Control", "no-store" if "/api/" in p or p.endswith((".html", "/")) else "max-age=3600")
            super().end_headers()

        def local(self):
            """Only requests addressed to this machine by name (a page on another site that rebinds its DNS to
            127.0.0.1 still sends its own host name)."""
            if host_name(self.headers.get("Host") or "") in LOCAL:
                return True
            self.reply(403, b'{"error": "localhost only"}')
            return False

        def route(self):
            """(project, its slug, the path within it) for this request, or (None, None, path) for the server's own."""
            path = unquote(urlparse(self.path).path)
            m = re.match(r"^/p/([^/]+)(/.*)?$", path)
            if m:
                p = site.get(m.group(1))
                return (p, m.group(1), m.group(2) or "") if p else (None, None, None)   # (no such project)
            if site.single and path != "/api/projects":
                return site.single, None, path
            return None, None, path

        def do_GET(self):
            if not self.local():
                return
            project, s, path = self.route()
            if project is None:
                if path == "/":
                    return self.reply(200, index(site), "text/html; charset=utf-8")
                if path == "/api/projects":
                    site.scan()
                    return self.reply(200, json.dumps({k: p.folder for k, p in site.projects.items()}).encode())
                return self.reply(404, b'{"error": "not found"}')
            if path in ("", "/"):
                first = pages(project)
                if not first:
                    return self.reply(404, b'{"error": "no review pages yet: run the review step"}')
                return self.redirect((f"/p/{quote(s)}" if s else "") + "/" + quote(first[0][1]))
            if path == "/api/picks":
                f = project.path("picks.json")
                return self.reply(200, open(f, "rb").read() if os.path.exists(f) else b"{}")
            if path == "/api/crop":
                return self.crop(project, parse_qs(urlparse(self.path).query))
            root = os.path.realpath(project.folder)
            f = os.path.realpath(os.path.join(root, path.lstrip("/")))
            if not f.startswith(root + os.sep) or not os.path.isfile(f):
                return self.reply(404, b'{"error": "not found"}')
            self.send_file(f)

        def do_PUT(self):
            if not self.local():
                return
            project, _, path = self.route()
            if project is None or path != "/api/picks":
                return self.reply(404, b'{"error": "not found"}')
            data = self.body()
            if not isinstance(data, dict):
                return self.reply(400, b'{"error": "picks must be a JSON object"}')
            picks = project.path("picks.json")
            if os.path.exists(picks):
                shutil.copyfile(picks, picks + ".bak")
            tmp = picks + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=1)
            os.replace(tmp, picks)
            self.reply(200, b'{"ok": true}')

        def do_POST(self):
            if not self.local():
                return
            project, _, path = self.route()
            if project is not None or path != "/api/projects":
                return self.reply(404, b'{"error": "not found"}')
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):   # a cross-site form
                return self.reply(415, b'{"error": "send JSON"}')                         # can't send JSON
            data = self.body()
            folder = data.get("folder") if isinstance(data, dict) else None
            if not isinstance(folder, str) or not is_project(folder):
                return self.reply(400, b'{"error": "folder must hold focalcut/reel/reel.json"}')
            s = site.add(Project(folder))
            self.reply(200, json.dumps({"slug": s, "url": f"/p/{s}/"}).encode())

        def body(self):
            try:
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except ValueError:
                return None

        def crop(self, project, q):
            from .plan import reframe
            key = os.path.realpath(project.folder)
            units = site.units.get(key)
            if units is None and os.path.exists(project.path("units.json")):
                units = site.units[key] = {u["id"]: u for u in project.load("units.json")}
            try:
                u = (units or {})[q["id"][0]]
                a, b = float(q["a"][0]), float(q["b"][0])
                fmt = q.get("fmt", ["portrait"])[0]
                assert fmt in ("portrait", "landscape") and b > a
            except (KeyError, ValueError, AssertionError):
                return self.reply(400, b'{"error": "id, a < b and fmt"}')
            c = reframe(project, u, u.get("export_t0", 0.0) + a, b - a, fmt, focus=False)
            self.reply(200, json.dumps(c).encode())

        def redirect(self, to):
            self.send_response(302)
            self.send_header("Location", to)
            self.send_header("Content-Length", "0")
            self.end_headers()

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


def run(projects, port=8765, host="127.0.0.1", root=None):
    projects = [projects] if isinstance(projects, Project) else list(projects)
    for p in projects:
        if not os.path.exists(p.path("review", "sources.html")):
            from . import browse
            browse.run(p)
    site = Site(projects, root)
    srv = ThreadingHTTPServer((host, port), handler(site))
    srv.daemon_threads = True
    print(f"[reel] open http://localhost:{port}/  (Ctrl+C to stop)")
    for s, p in site.projects.items():
        print(f"[reel]   {p.folder}  ->  http://localhost:{port}/p/{s}/  (picks save to {p.path('picks.json')})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
