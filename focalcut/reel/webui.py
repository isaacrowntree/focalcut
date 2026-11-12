"""The review pages' front end: plain HTML, CSS and JavaScript files in focalcut/reel/web/, put together into one
self-contained page per review page (no build step, nothing fetched): a page works served by `focalcut reel serve`
and opened as a file alike.

    web/ui.css, app.js, picks.js     shared: the look, saving and undo, the picks model (per-reel tags, selections)
    web/sources.{html,css,js}        the source browser (Clips, Cut & picks) and web/frame.js (portrait framing)
    web/review.{html,css,js}         a reel's review page

Each page carries the version of the files it was made from (<meta name=focalcut-ui>), so the server can tell a page
made by an older focalcut and make the sources page again.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_cache: dict = {}


def asset(name: str) -> str:
    """A front-end file's text (read again when it changes, so an edit shows on the next page made)."""
    f = os.path.join(WEB, name)
    key = os.stat(f).st_mtime_ns
    hit = _cache.get(name)
    if not hit or hit[0] != key:
        with open(f, encoding="utf-8") as fh:
            hit = _cache[name] = (key, fh.read())
    return hit[1]


def version() -> str:
    """A short hash of every front-end file: the stamp a page carries."""
    h = hashlib.sha1()
    for name in sorted(os.listdir(WEB)):
        if name.endswith((".js", ".css", ".html")):
            h.update(name.encode() + b"\0" + asset(name).encode() + b"\0")
    return h.hexdigest()[:12]


def stamp_of(path: str) -> str | None:
    """The front-end version a page on disk was made from (None: no page, or one from before pages had a stamp)."""
    try:
        with open(path, encoding="utf-8") as fh:
            m = re.search(r'<meta name=focalcut-ui content="?([0-9a-f]+)', fh.read(2048))
    except OSError:
        return None
    return m.group(1) if m else None


def document(title: str, body: str, data: dict, css=(), js=(), subs: dict | None = None) -> str:
    """One page: the stylesheets and scripts inline, in order, the page's data as `D` ahead of the scripts. subs:
    {"NAME": text} filled into the body's (and the stylesheets') %NAME% places as given."""
    style = "\n".join(asset(c) for c in css)
    body = asset(body)
    for k, v in (subs or {}).items():
        body, style = body.replace(f"%{k}%", v), style.replace(f"%{k}%", v)
    script = "\n".join(asset(j) for j in js)
    if "</script" in script.lower():
        raise ValueError("a front-end script holds '</script': it would end the page's inline script")
    blob = json.dumps(data).replace("</", "<\\/")
    return (f'<!doctype html><html lang=en><meta charset=utf-8>'
            f'<meta name=viewport content="width=device-width,initial-scale=1">'
            f'<meta name=focalcut-ui content="{version()}"><title>{html.escape(title)}</title>\n'
            f'<style>\n{style}</style>\n{body}\n<script>const D={blob};</script>\n<script>\n{script}</script>\n')
