"""A local review page per reel: every shot with a frame, its export file and the time range it uses in that file --
exactly what picks take, so a bad shot can be banned or a better moment suggested. Each shot opens a scrubber on
the export it comes from, with an overview strip and a zoomed strip around the playhead (drag either to scrub, scroll
to zoom) and editor keys -- J/K/L shuttle, frame steps, I/O, M markers, [ ] to hop between marks: mark extra shots
anywhere, swap the shot to another moment or range, ban it, or ask for more from that video. The page writes the
picks.json to save (it remembers unsaved changes in the browser).

    review/<fmt>.html  (+ review/<fmt>/NNN.jpg)

Frames come from the rendered reel when it exists, else from the shot's source.
"""
from __future__ import annotations

import html
import os
import subprocess
from urllib.parse import quote

from . import browse, webui
from .project import Project


def mmss(x):
    return f"{int(x // 60)}:{x % 60:04.1f}"


def pick_line(s):
    """The shot as a picks entry: its export file and the range it plays, in that file's own timeline."""
    u = s["unit"]
    x0 = u.get("export_t0", 0.0)
    a = s["src_in"] - x0
    return f'"{u["id"]}": {{"ban": ["{mmss(a)}-{mmss(a + s["length"])}"]}}'


def page(plan, fmt, thumbs, sources=None, picks=None, key="review", root="", nav="", project="", reel="main", reels=None):
    """The review page's HTML; thumbs[i] is shot i's image path relative to the page (or None). sources: per unit id,
    {"video": its export file relative to the page, "moments": [[t, kind, rank], ...]} in the export's own timeline;
    picks: the current picks.json, which the page starts from; root: the project folder relative to the page, where
    `focalcut reel serve` answers api/picks (served, the page saves picks.json itself); reel: the reel this cut is
    (picks made here are for it alone), reels: [{"id", "label"}] of every reel the picks serve."""
    w = 320 if fmt == "landscape" else 180
    cards, shots = [], []
    for i, s in enumerate(plan["shots"]):
        u = s["unit"]
        a = s["src_in"] - u.get("export_t0", 0.0)
        tags = " ".join(filter(None, [s["act"], s["kind"], "+".join(s["fx"]),
                                      s.get("transition") and f"⇢ {s['transition']}"]))
        img = f'<img src="{html.escape(quote(thumbs[i]))}" loading="lazy" alt="">' if thumbs[i] else ""
        cards.append(f'<figure data-i="{i}">{img}<figcaption><b>{i + 1}</b> {html.escape(u["id"])} '
                     f'<span class=t>{mmss(a)}–{mmss(a + s["length"])}</span><br><small>{html.escape(tags)}</small>'
                     f'<code title="copy into picks.json">{html.escape(pick_line(s))}</code>'
                     f'<span class=bar><button data-act=scrub>Scrub source ▸</button>'
                     f'<button data-act=ban>Ban</button><button data-act=more>More +</button></span>'
                     f'<span class=st></span></figcaption></figure>')
        shots.append(dict(id=u["id"], a=round(a, 2), b=round(a + s["length"], 2), kind=s["kind"]))
    data = dict(shots=shots, sources=sources or {}, picks=picks or {}, key=key, root=root, reel=reel,
                reels=reels or [])
    return webui.document(f"{fmt} reel review", "review.html", data, css=("review.css",), js=("picks.js", "review.js"),
                          subs={"W": str(w), "FMT": html.escape(fmt), "N": str(len(shots)), "CARDS": "".join(cards),
                                "NAV": nav, "PROJECT": html.escape(project)})


def sources(project, plan, fmt, page_dir):
    """Per unit in the plan: its export file (relative to the page; browsers play the H.264 exports, not the cameras'
    10-bit 4:2:2 HEVC) and the moments the planner found in it, in the export's own timeline, best first."""
    moments = {r["unit"]["id"]: r for r in project.load("moments.json")} if os.path.exists(
        project.path("moments.json")) else {}
    key = "score_p" if fmt == "portrait" else "score_l"
    out = {}
    for s in plan["shots"]:
        u = s["unit"]
        if u["id"] in out:
            continue
        x0 = u.get("export_t0", 0.0)
        f = os.path.join(project.export_dir, u.get("category", ""), u["id"].split("#")[0] + ".mp4")
        r = moments.get(u["id"], {})
        ms = sorted(r.get("moments", []), key=lambda m: -m.get(key, m.get("score", 0)))
        # a joined B-roll piece ("#n") has no export of its own: its times are its own clip's, not the joined file's
        ok = os.path.exists(f) and "#" not in u["id"]
        out[u["id"]] = dict(video=quote(os.path.relpath(f, page_dir).replace(os.sep, "/")) if ok else None,
                            moments=[[round(m["t"] - x0, 2), m["kind"], round(m.get(key, m.get("score", 0)), 2)]
                                     for m in ms if m["t"] - x0 >= 0])
    return out


def run(project, fmt, cut=None) -> str:
    cut = cut or fmt
    plan = project.load(f"plan_{cut}.json")
    fmt = plan.get("format", fmt)
    video = project.path("out", f"{project.out_stem(cut, fmt)}.mp4")
    out = project.path("review", cut)
    os.makedirs(out, exist_ok=True)
    w = 320 if fmt == "landscape" else 180
    thumbs = []
    for i, s in enumerate(plan["shots"]):
        jpg = os.path.join(out, f"{i + 1:03d}.jpg")
        if os.path.exists(video):
            src, t = video, s["rec_in"] + s["length"] * 0.5
        else:
            src, t = project.decode_from(s["unit"]["source"], s["src_in"] + s["length"] * 0.5)
            src = src if os.path.isabs(src) else os.path.join(project.folder, src)
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1",
                            "-vf", f"scale={w}:-2", "-q:v", "4", jpg])
        thumbs.append(f"{cut}/{i + 1:03d}.jpg" if r.returncode == 0 and os.path.exists(jpg) else None)
    path = project.path("review", f"{cut}.html")
    with open(path, "w") as f:
        main = Project(project.folder)
        rs = browse.reels(main, {f: main.load(f"plan_{f}.json") for f in ("landscape", "portrait")
                                 if os.path.exists(main.path(f"plan_{f}.json"))})
        f.write(page(plan, fmt, thumbs, sources(project, plan, fmt, os.path.dirname(path)), project.picks(reel=False),
                     key=f"{project.name}/{cut}", root=browse.api_root(project),
                     nav=browse.menu(project, cut), project=browse.title_of(project), reel=project.reel_id,
                     reels=[{"id": r["id"], "label": r["label"]} for r in rs]))
    print(f"[reel] review: {path}")
    return path
