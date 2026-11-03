"""A look book of one style library (fx/looks.py: glitch, carnival): every subject effect, transition and accent in it
played on the reel's own footage, and each LOOK as a short sequence, on one local HTML page -- to choose a look before
it goes in the film. (The edm looks are overlays; they show in a render, not here.)

    focalcut fx lookbook <footage folder> [--library glitch|carnival] [--fmt landscape] [--width 960]
      -> <folder>/focalcut/reel/review/<library>/index.html  (+ one mp4 per style)

Clips: the strongest moves the plan found (spins and dips with velocity edits first), one per export file, 2.4 s
around each peak. Subject effects play on the first, accents on the second, transitions cut from one to the next.
"""
from __future__ import annotations

import html
import json
import os
import subprocess

import numpy as np

from . import EFFECTS, TRANSITIONS, Context
from .looks import LIBRARIES

FPS, PRE, POST = 25, 1.0, 1.4             # look book frame rate; seconds before / after each clip's peak


def clips_from_plan(project, fmt, n=5):
    """The plan's best moves, one per export file: [(unit, source time of the peak)]."""
    shots = project.load(f"plan_{fmt}.json")["shots"]
    cand = [s for s in shots if s["unit"]["role"] in ("show", "dance") and s.get("peak") is not None
            and s["kind"] in ("spin", "dip", "trick", "shake")]
    cand.sort(key=lambda s: (not s["fx"], -s["length"]))
    out, seen = [], set()
    for s in cand:
        if s["unit"]["id"] not in seen:
            seen.add(s["unit"]["id"]); out.append((s["unit"], s["src_in"] + max(s["peak"], 0.0)))
        if len(out) == n:
            break
    return out


def grab(project, unit, t, width):
    """2.4 s of the unit's export file around t (source seconds) at the look book's size and rate."""
    from ..reel.shake import export_of
    path = export_of(project, unit) or project.decode_from(unit["source"], t)[0]
    a = t - PRE - unit.get("export_t0", 0.0)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", path], capture_output=True, text=True).stdout.split(",")
    W, H = int(r[0]), int(r[1])
    w, h = width, int(round(width * H / W / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(a, 0):.3f}", "-t", f"{PRE + POST:.3f}", "-i", path,
                          "-vf", f"fps={FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3))


def write(frames, path):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()


def fit(frames, h, w):
    """Frames letterboxed (or pillarboxed) to h x w, so clips of both shapes can be cut together."""
    import cv2
    fh, fw = frames[0].shape[:2]
    if (fh, fw) == (h, w):
        return frames
    s = min(w / fw, h / fh)
    nw, nh = int(fw * s) // 2 * 2, int(fh * s) // 2 * 2
    out = []
    for f in frames:
        c = np.zeros((h, w, 3), np.uint8)
        y, x = (h - nh) // 2, (w - nw) // 2
        c[y:y + nh, x:x + nw] = cv2.resize(f, (nw, nh))
        out.append(c)
    return out


def cut(a, b, name, beat, seed=0):
    """Clip a then clip b, with transition `name` replacing the frames either side of the cut."""
    ctx = Context(fps=FPS, beat=beat, seed=seed)
    t = TRANSITIONS[name](); k = t.frames(ctx)
    ha = k // 2
    return a[:len(a) - ha] + t.apply(a[len(a) - ha:], b[:k - ha], ctx) + b[k - ha:]


def effect(frames, names, beat, seed=0):
    ctx = Context(fps=FPS, beat=beat, peak=PRE, seed=seed)
    for n in names:
        nm, kw = n if isinstance(n, tuple) else (n, {})
        frames = EFFECTS[nm](**kw).apply(frames, ctx)
    return frames


def doc(cls):
    return " ".join((cls.__doc__ or "").split())


def run(project, fmt="landscape", width=960, library="glitch"):
    lib = LIBRARIES[library]
    if not lib["prefix"]:
        raise SystemExit(f"the {library} looks are overlays: they show in a render (reel.json \"look\"), not a look book")
    LOOKS, pre = lib["looks"], lib["prefix"]
    beat = float(project.load("music.json")["beat"]) if os.path.exists(project.path("music.json")) else 0.6
    while beat < 0.4:
        beat *= 2
    out_dir = project.path("review", library)
    os.makedirs(out_dir, exist_ok=True)
    picks = clips_from_plan(project, fmt)
    raw = [grab(project, u, t, width) for u, t in picks]
    H, W = raw[0][0].shape[:2]
    clips = [fit(c, H, W) for c in raw]
    A, B = clips[0], clips[1 % len(clips)]
    cards = {"Subject": [], "Transitions": [], "Accents": [], "Looks": []}
    subj = [k for k, c in EFFECTS.items() if k.startswith(pre) and c.needs_masks]
    acc = [k for k, c in EFFECTS.items() if k.startswith(pre) and not c.needs_masks]
    tr = [k for k in TRANSITIONS if k.startswith(pre)]

    def done(group, name, frames, text, src):
        f = f"{name}.mp4"
        write(frames, os.path.join(out_dir, f))
        cards[group].append(dict(name=name, file=f, text=text, src=src))
        print(f"[fx] lookbook {group.lower()}: {name}", flush=True)

    label = lambda i: f"{picks[i][0]['id']} @ {picks[i][1]:.1f}s"
    for name in subj:
        done("Subject", name, effect(A, [name], beat), doc(EFFECTS[name]), label(0))
    for name in acc:
        done("Accents", name, effect(B, [name], beat), doc(EFFECTS[name]), label(1 % len(clips)))
    for name in tr:
        done("Transitions", name, cut(A, B, name, beat), doc(TRANSITIONS[name]), f"{label(0)} → {label(1 % len(clips))}")
    for key, lk in LOOKS.items():                # each look: three or four clips, its subject effect, two of its
        seq = [effect(c, lk["base"], beat, seed=i) for i, c in enumerate(clips[:4])]   # transitions, an accent
        seq[0] = effect(seq[0], lk["subject"][:1], beat)
        seq[1] = effect(seq[1], lk["accents"][:1], beat)
        if len(seq) > 2:
            seq[2] = effect(seq[2], lk["accents"][1:2] or lk["accents"][:1], beat)
        film = seq[0]
        for i, c in enumerate(seq[1:]):
            film = cut(film, c, lk["transitions"][i % len(lk["transitions"])], beat, seed=i) if i < 2 else film + c
        names = lk["subject"] + lk["transitions"] + lk["accents"] + [b if isinstance(b, str) else b[0] for b in lk["base"]]
        done("Looks", f"look_{key}", film, f"<b>{html.escape(lk['title'])}</b> — {html.escape(lk['mood'])}. "
             f"Uses {', '.join(names)}. <i>{html.escape(lk['use'])}</i>", ", ".join(label(i) for i in range(len(seq))))
    page = os.path.join(out_dir, "index.html")
    with open(page, "w") as fh:
        fh.write(PAGE.replace("__DATA__", json.dumps(dict(cards=cards, title=project.name, lib=lib["title"])).replace("</", "<\\/")))
    print(f"[fx] lookbook: {page}")
    return page


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Look book</title><style>
:root{--bg:#0d0d10;--fg:#ece9f1;--mut:#9a95a6;--card:#17161c;--acc:#7cf2e0;--acc2:#ff4fa3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{padding:28px 20px 8px;max-width:1400px;margin:auto}h1{margin:0;font-size:26px;letter-spacing:.02em}
h1 span{color:var(--acc)}p.sub{color:var(--mut);margin:6px 0 0}nav{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
nav a{color:var(--fg);text-decoration:none;border:1px solid #333;padding:4px 12px;border-radius:99px;font-size:13px}
nav a:hover{border-color:var(--acc)}section{max-width:1400px;margin:auto;padding:6px 20px 24px}
h2{font-size:18px;margin:18px 0 10px;color:var(--acc2)}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.looks .grid{grid-template-columns:repeat(auto-fill,minmax(460px,1fr))}
.card{background:var(--card);border-radius:10px;overflow:hidden;border:1px solid #24222b}
video{width:100%;display:block;background:#000}.card div{padding:10px 12px}.card h3{margin:0 0 4px;font-size:14px;font-family:ui-monospace,monospace;color:var(--acc)}
.card p{margin:0;color:var(--mut);font-size:13px}.card small{display:block;margin-top:6px;color:#6f6a7a;font-size:11px}
@media (max-width:520px){.looks .grid{grid-template-columns:1fr}}
</style></head><body><header><h1><b id=lib></b> <span>look book</span></h1><p class=sub id=sub></p><nav id=nav></nav></header>
<main id=main></main><script>
const D=__DATA__,order=["Looks","Subject","Transitions","Accents"],blurb={Looks:"Ways to dress the whole film: pick one, then its pieces go in on the hits.",
Subject:"On the dancer only (SAM 3 masks); the room stays clean.",Transitions:"Across a cut, on the beat.",Accents:"Whole frame, on a hit or as a base texture."};
document.getElementById("lib").textContent=D.lib;document.title=D.lib+" look book";
document.getElementById("sub").textContent=D.title+" — every "+D.lib.toLowerCase()+" style on this reel's own footage. Click a clip to restart it.";
const esc=s=>s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"})[c]);
document.getElementById("nav").innerHTML=order.map(g=>`<a href="#${g}">${g} (${D.cards[g].length})</a>`).join("");
document.getElementById("main").innerHTML=order.map(g=>`<section id=${g} class=${g.toLowerCase()}><h2>${g}</h2><p class=sub>${blurb[g]}</p><div class=grid>`+
 D.cards[g].map(c=>`<div class=card><video src="${c.file}" muted loop autoplay playsinline preload=metadata onclick="this.currentTime=0;this.play()"></video>
 <div><h3>${esc(c.name)}</h3><p>${g=="Looks"?c.text:esc(c.text)}</p><small>${esc(c.src)}</small></div></div>`).join("")+`</div></section>`).join("");
</script></body></html>"""
