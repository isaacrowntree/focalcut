"""Dust on the lens (or the sensor): found in the footage itself, taken out exactly -- reel.json "dust".

A speck of dust sits in the same place in every frame a lens shoots, darkening what is behind it by the same
share: it is a multiplication, so it comes out with a division, not by painting anything in. That makes the fix
deterministic and safe -- the picture behind the speck is the picture, only brighter by what the speck took.

Finding it: every frame of the clips named (a tripod's, where it shows on bright water and sky) is divided by a
blur of itself, which leaves only fine local darkening; people are masked out of each frame (the score step's
person boxes, padded), so that over each clip the median is the lens's own pattern; a pixel darker than `below`
in most clips (`votes`) is dust -- the people and the waves move, the dust doesn't. Its depth there is how much
light it takes (the transmission map).

Taking it out (render.py): the frame is divided by that map (blend dodge: A / (1 - B), B = 1 - transmission) at
the camera's own resolution, before anything else. A speck a person passes over during a shot is left out of that
shot's map ("if there's a person there don't try to fix them"), so nobody is touched.

reel.json "dust": {"clips": ["baptisms"] (categories or clip patterns), "below": 0.95, "votes": 0.6,
                   "every": 2.0 (s between frames), "gain": 1.2 (how much deeper than measured to take a speck
                   out), "people": true}
Writes <reel>/looks/dust.npz (the transmission map at 1920x1080 and the specks: x, y, r in frame fractions).
"""
from __future__ import annotations

import fnmatch
import os
import subprocess

import numpy as np

W, H = 1920, 1080


def _clips(project, units):
    pats = (project.config.get("dust") or {}).get("clips") or []
    return [u for u in units if any(u["category"] == p or fnmatch.fnmatch(u["clip"], p) for p in pats)]


def applies(project, u):
    """Is this footage shot through the dusty lens?"""
    return bool(project.config.get("dust")) and bool(_clips(project, [u])) and os.path.exists(project.path("looks", "dust.npz"))


def build(project):
    """Find the dust (see the module notes) and write looks/dust.npz."""
    import cv2
    cfg = project.config.get("dust") or {}
    below, votes, every = float(cfg.get("below", 0.95)), float(cfg.get("votes", 0.6)), float(cfg.get("every", 2.0))
    units = _clips(project, project.load("units.json"))
    per, per_c = [], []
    for u in units:
        path, x0 = project.decode_from(u["source"], u["t0"])
        z = np.load(project.path("scores", u["score"] + ".npz"))
        pt, bx = z["pose_t"], z["boxes"]
        aw = 640.0
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{x0:.3f}", "-t", f"{u['t1'] - u['t0']:.3f}", "-i", path,
                              "-vf", f"fps=1/{every},scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                             capture_output=True).stdout
        frames = np.frombuffer(raw, np.uint8)[: len(raw) // (W * H) * W * H].reshape(-1, H, W).astype(np.float32)
        rs, rc = [], []
        for i, g in enumerate(frames):
            r = g / (cv2.GaussianBlur(g, (0, 0), 6) + 1)            # found on the fine scale; how deep, on the
            c = g / (cv2.GaussianBlur(g, (0, 0), 16) + 1)           # coarse one too (a long hair's full depth)
            t = u["t0"] + i * every
            j = int(np.argmin(np.abs(pt - t)))
            for b in bx[max(j - 3, 0):j + 4].reshape(-1, bx.shape[-1]):
                if np.isnan(b[0]):
                    continue
                x0_, y0, x1, y1 = (b[:4] * W / aw).astype(int)
                pw, ph = int((x1 - x0_) * 0.25), int((y1 - y0) * 0.1)
                r[max(y0 - ph, 0):y1 + ph, max(x0_ - pw, 0):x1 + pw] = np.nan
                c[max(y0 - ph, 0):y1 + ph, max(x0_ - pw, 0):x1 + pw] = np.nan
            rs.append(r)
            rc.append(c)
        if rs:
            with np.errstate(all="ignore"):
                per.append(np.nanmedian(np.array(rs), axis=0))
                per_c.append(np.nanmedian(np.array(rc), axis=0))
    P = np.array(per)
    with np.errstate(all="ignore"):
        hits, seen = np.nansum(below > P, 0), np.sum(~np.isnan(P), 0)
        R = np.fmin(np.nanmedian(P, 0), np.nanmedian(np.array(per_c), 0))   # depth: the deeper of the two
    mask = ((hits >= np.maximum(3, votes * seen)) & (seen >= 3)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    grown = np.zeros_like(mask)                              # each speck's area grown with its size: a hair's
    for i in range(1, n):                                    # soft halo is wider than a dot's
        k = 5 + 2 * int(min(4, np.sqrt(st[i][4]) / 6))
        grown |= cv2.dilate((lab == i).astype(np.uint8), np.ones((k, k), np.uint8))
    mask = grown
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    trans = np.ones((H, W), np.float32)
    specks = []
    for i in range(1, n):
        if st[i][4] < 6 or st[i][4] > 4000:
            continue
        m = lab == i
        depth = 1.0 - np.nan_to_num(R[m], nan=1.0)               # (a speck's own blur hides a little of its
        trans[m] = np.clip(1.0 - depth * float(cfg.get("gain", 1.0)), 0.5, 1.0)   # depth from the ratio: gain)
        r = float(np.sqrt(st[i][4] / np.pi))
        specks.append([round(cen[i][0] / W, 4), round(cen[i][1] / H, 4), round(r / W, 4),
                       round(float(np.nanmin(R[m])), 3)])
    trans = cv2.GaussianBlur(trans, (0, 0), 0.8)               # the speck's soft edge
    np.savez_compressed(project.path("looks", "dust.npz"), trans=trans.astype(np.float16), specks=np.array(specks))
    print(f"[reel] dust: {len(specks)} specks from {len(units)} clips -> looks/dust.npz")
    return specks


def for_shot(project, s, length):
    """The shot's own correction map (a one-frame y4m at the camera's size and layout, each plane 1 - transmission
    in 10 bits, each speck as deep as this shot's own frames show it; with "people": false, every speck a person
    passes over during the shot left out); None when the footage has no dust map."""
    import cv2
    u = s["unit"]
    if not applies(project, u):
        return None
    d = np.load(project.path("looks", "dust.npz"))
    trans, specks = d["trans"].astype(np.float32), d["specks"]
    keep = np.ones(len(specks), bool)
    if (project.config.get("dust") or {}).get("people", True):   # specks over people are taken out too
        z = None
    else:
        z = np.load(project.path("scores", u["score"] + ".npz"))
    boxes = np.zeros((0, 5))
    if z is not None:
        m = (z["pose_t"] >= s["src_in"] - 0.3) & (z["pose_t"] <= s["src_in"] + length + 0.3)
        boxes = z["boxes"][m].reshape(-1, z["boxes"].shape[-1])
    aw, ah = 640.0, 640.0 * H / W
    for b in boxes:
        if np.isnan(b[0]):
            continue
        x0, y0, x1, y1 = b[0] / aw, b[1] / ah, b[2] / aw, b[3] / ah
        for k, (x, y, r, _) in enumerate(specks):
            if x0 - r - 0.01 <= x <= x1 + r + 0.01 and y0 - r - 0.01 <= y <= y1 + r + 0.01:
                keep[k] = False
    key = f"{u['id']}_{s['src_in']:.2f}_{length:.2f}_{VERSION}"
    path = project.path("work", "dust", key.replace("#", "-") + ".y4m")
    if os.path.exists(path):                                  # (made already: this shot, this map)
        return path
    out = trans.copy()
    for k, (x, y, r, _) in enumerate(specks):
        if not keep[k]:
            rr = int(r * W * 2.2) + 3
            cx, cy = int(x * W), int(y * H)
            out[max(cy - rr, 0):cy + rr, max(cx - rr, 0):cx + rr] = 1.0
    loss = 1.0 - out
    own = _measured(project, u, s["src_in"], length, specks, loss > 0)   # each speck as deep as this shot shows
    if own is not None:                                       # it (the camera's curve squeezes a speck's shadow
        loss = np.minimum(own, loss * OWN_MAX + 0.01)         # on bright water: measured, not guessed) -- never
                                                              # far past the lens's own depth (OWN_MAX)
    sw, sh, sub = source_layout(project, u)
    m = np.clip(loss * 1023, 0, 1023).astype(np.float32)      # 10-bit, the same in Y, Cb and Cr
    y = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_CUBIC).clip(0, 1023).astype("<u2")
    cw, ch = (sw // 2, sh) if sub == "422" else (sw // 2, sh // 2) if sub == "420" else (sw, sh)
    c = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_AREA).clip(0, 1023).astype("<u2")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as fh:                    # one frame in the camera's own layout: no scaling
        fh.write(f"YUV4MPEG2 W{sw} H{sh} F25:1 Ip A1:1 C{sub}p10 XYSCSS={sub}P10\nFRAME\n".encode())
        fh.write(y.tobytes() + c.tobytes() + c.tobytes())
    os.replace(path + ".part", path)
    return path


def source_layout(project, u):
    """(width, height, chroma layout "422" | "420" | "444") of a unit's camera file."""
    src = os.path.join(project.folder, u["source"])
    w, h, pf = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=width,height,pix_fmt", "-of", "csv=p=0", src]).decode().strip().split(",")[:3]
    return int(w), int(h), "422" if "422" in pf else "444" if "444" in pf else "420"


def pix_fmt(sub):
    return f"yuv{sub}p10le"


OWN_MAX = 1.5              # a shot's own measure of a speck: at most this times the map's depth (the stack over every
                           # clip, people masked, is the speck's real depth; a shot's frames can be fooled)
VERSION = 6                # bumps every shot's cache key (render.key) when the way specks are taken out changes


def version(project):
    """The dust map's identity for render caches: the map's content (render.file_id: a map measured again to the
    same specks, or copied with the project, re-renders nothing) and VERSION."""
    from .render import file_id
    h = file_id(project.path("looks", "dust.npz"))
    return f"{VERSION}:{h}" if h else None


def _measured(project, u, t0, length, specks, where, n=9):
    """The darkening (0..1, 1 - transmission) of each speck in this shot's own frames, inside the dust map's
    areas (`where`): n frames across the shot, each divided by its blur at two scales (the deeper), the median
    over frames. None when the frames can't be read."""
    import cv2

    from .render import proxy_of
    path, x0 = proxy_of(project, u["source"], t0) or project.decode_from(u["source"], t0)   # (1080p is plenty)
    rate = max(n / max(length, 0.2), 0.5)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{x0:.3f}", "-t", f"{length:.3f}", "-i", path, "-vf",
                          f"fps={rate:.3f},scale={W}:{H},format=gray", "-frames:v", str(n), "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    k = len(raw) // (W * H)
    if k < 2:
        return None
    fr = np.frombuffer(raw, np.uint8)[: k * W * H].reshape(k, H, W).astype(np.float32)
    rs = [np.minimum(g / (cv2.GaussianBlur(g, (0, 0), 6) + 1), g / (cv2.GaussianBlur(g, (0, 0), 16) + 1)) for g in fr]
    R = np.percentile(np.array(rs), 80, axis=0)              # (a speck darkens every frame alike; something dark
    out = np.zeros((H, W), np.float32)                         # passing over it -- a person, who then goes under --
    out[where] = np.clip(1.0 - R[where], 0.0, 0.5)             # only darkens it more: its lightest reading is the
    return cv2.GaussianBlur(out, (0, 0), 0.8)                  # speck's own, the median a white hole left behind)


def rolloff(v, knee=0.6, slope=1.6, least=0.3):
    """How much of a speck's measured darkening to take out over a background of brightness v (0..1): all of it up
    to the knee, less above it, where the camera's highlight curve squeezes the speck's shadow."""
    return float(np.clip(1.0 - slope * (v - knee), least, 1.0))


def _backgrounds(project, u, t, specks):
    """Each speck's background brightness (0..1) at source time t: the median of a ring around it in one frame."""
    path, x0 = project.decode_from(u["source"], t)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{x0:.3f}", "-i", path, "-frames:v", "1", "-vf",
                          f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    if len(raw) < W * H:
        return [0.5] * len(specks)
    g = np.frombuffer(raw, np.uint8)[: W * H].reshape(H, W).astype(np.float32) / 255
    out = []
    for x, y, r, _ in specks:
        cx, cy, rr = int(x * W), int(y * H), int(r * W * 3) + 8
        yy, xx = np.ogrid[-cy:H - cy, -cx:W - cx]
        ring = (xx * xx + yy * yy <= (2 * rr) ** 2) & (xx * xx + yy * yy > rr * rr)
        out.append(float(np.median(g[ring])) if ring.any() else 0.5)
    return out


def _read_map(path):
    """A shot's dust map (for_shot's y4m) as a (H, W) array, 0 where nothing is taken out."""
    import cv2
    raw = open(path, "rb").read()
    hdr = raw[: raw.index(b"\n")].decode().split()
    sw, sh = int(hdr[1][1:]), int(hdr[2][1:])
    head = raw.index(b"FRAME\n") + 6
    y = np.frombuffer(raw[head:head + sw * sh * 2], "<u2").reshape(sh, sw)
    return cv2.resize(y, (W, H), interpolation=cv2.INTER_AREA)


def run(project):
    build(project)
    page = review(project)
    if page:
        print(f"[reel] dust: before / after -> {page}")


FILTER = ("[1:v]setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709[dm];"
          "[0:v]format={fmt}[ds];[ds][dm]lut2=c0='(x-64)*1023/max(1023-y,1)+64'"
          ":c1='(x-512)*1023/max(1023-y,1)+512':c2='(x-512)*1023/max(1023-y,1)+512'")
# The division in the camera's own 10-bit Y'CbCr (lut2): luma above black and chroma around neutral, each divided by
# the speck's transmission (the map's value y: 1 - transmission, written in all three planes at the camera's own size
# and chroma layout, tagged as the camera's colour so nothing is converted to meet it) -- every pixel outside a speck
# passes untouched (y = 0), superwhites included; no RGB round trip. The map is one frame, which lut2 repeats for
# every frame of the shot (its eof_action) -- looping the input instead floods lut2's queue at 4K.
INPUT: list = []


def _frame(src, t, w, h, extra=(), graph=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", src, *extra]
    if graph:
        cmd += ["-filter_complex", graph + ",format=rgb24"]
    cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def review(project, frames=None, crops=6):
    """A before / after page of the dust fix (review/dust/index.html): whole frames of the original camera file
    through the render's own filter, the specks found circled, and the worst specks at 2x. frames: [(unit id,
    source seconds), ...]; default: the start and middle of the first dusty clip."""
    import json

    import cv2
    units = _clips(project, project.load("units.json"))
    if not units:
        return None
    frames = frames or [(units[0]["id"], units[0]["t0"] + 3.0),
                        (units[0]["id"], (units[0]["t0"] + units[0]["t1"]) / 2)]
    byid = {u["id"]: u for u in units}
    specks = np.load(project.path("looks", "dust.npz"))["specks"]
    out = project.path("review", "dust")
    os.makedirs(out, exist_ok=True)
    meta = []
    for uid, t in frames:
        u = byid[uid]
        src = os.path.join(project.folder, u["source"])
        w, h = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                 "stream=width,height", "-of", "csv=p=0", src]).decode().split(",")[:2])
        png = for_shot(project, dict(unit=u, src_in=t, length=1.0), 1.0)
        before = _frame(src, t, w, h)
        after = _frame(src, t, w, h, [*INPUT, "-i", png], FILTER.format(fmt=pix_fmt(source_layout(project, u)[2])))
        corr = _read_map(png)
        circ = before.copy()
        tag = f"{uid}_{int(t)}"
        rows = []
        for k, (x, y, r, mn) in enumerate(specks):
            fixed = bool(corr[min(int(y * H), H - 1), min(int(x * W), W - 1)] > 0)
            cv2.circle(circ, (int(x * w), int(y * h)), int(r * w * 3) + w // 140, (255, 40, 40) if fixed else (255, 200, 0),
                       max(w // 640, 2))
            rows.append((mn, k, x, y, fixed))
        for nm, im in (("before", before), ("after", after), ("circles", circ)):
            cv2.imwrite(os.path.join(out, f"{tag}_{nm}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        worst = []
        for mn, k, x, y, fixed in sorted(rows)[:crops]:
            cx, cy, half = int(x * w), int(y * h), w // 35
            x0, y0 = min(max(cx - half, 0), w - 2 * half), min(max(cy - half, 0), h - 2 * half)
            for nm, im in (("before", before), ("after", after)):
                c = cv2.resize(im[y0:y0 + 2 * half, x0:x0 + 2 * half], (440, 440), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(out, f"{tag}_crop{k}_{nm}.png"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
            worst.append(dict(k=int(k), depth=round((1 - mn) * 100), fixed=fixed))
        meta.append(dict(tag=tag, label=f"{uid} at {t:.0f} s", crops=worst, n=len(specks),
                         fixed=sum(1 for *_, f in rows if f)))
    page = os.path.join(out, "index.html")
    with open(page, "w") as fh:
        fh.write(_page(meta))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    return page


def _page(meta):
    def sec(m):
        crops = "".join(
            f'<figure><div class="pair"><img src="{m["tag"]}_crop{c["k"]}_before.png" alt="">'
            f'<img src="{m["tag"]}_crop{c["k"]}_after.png" alt=""></div><figcaption>speck {c["k"] + 1} · '
            f'{c["depth"]}% darker{"" if c["fixed"] else " · left as shot"}</figcaption></figure>' for c in m["crops"])
        return (f'<section><h2>{m["label"]}</h2><p class="mut">{m["fixed"]} of {m["n"]} specks taken out. Red: '
                f'taken out · amber: left as shot.</p><div class="tabs" data-tag="{m["tag"]}"><button data-v="before" '
                f'class="on">Before</button><button data-v="after">After</button><button data-v="circles">Detected '
                f'specks</button></div><img class="full" id="img-{m["tag"]}" src="{m["tag"]}_before.jpg" alt="">'
                f'<h3>Worst specks at 2× (left before, right after)</h3><div class="crops">{crops}</div></section>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>Lens Dust Fix</title><style>'
            ':root{--bg:#f7f6f4;--fg:#1d1b1c;--mut:#6c6668;--line:#dedad7;--on:#2d3c4b;--onfg:#fff}'
            '@media (prefers-color-scheme:dark){:root{--bg:#151416;--fg:#ece8e6;--mut:#9c9799;--line:#333036;'
            '--on:#9fb4c8;--onfg:#111}}body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;'
            'margin:0}main{max-width:1400px;margin:0 auto;padding:20px 16px 60px}h1{font-size:24px;margin:4px 0}'
            'h2{font-size:18px;margin:34px 0 4px}h3{font-size:15px;margin:18px 0 8px}.mut{color:var(--mut)}'
            '.tabs{display:flex;gap:6px;margin:10px 0}button{font:inherit;padding:6px 14px;border:1px solid var(--line);'
            'background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}button.on{background:var(--on);'
            'color:var(--onfg);border-color:var(--on)}img.full{width:100%;border-radius:6px;display:block}'
            '.crops{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}.pair{display:grid;'
            'grid-template-columns:1fr 1fr;gap:4px}.pair img{width:100%;image-rendering:pixelated;border-radius:4px}'
            'figure{margin:0}figcaption{font-size:13px;color:var(--mut);margin-top:4px}</style></head><body><main>'
            '<h1>Lens dust</h1><p class="mut">Found by stacking the clips with people masked out: what stays dark in '
            'the same pixel across clips is on the lens. The fix divides out each speck\'s darkening (nothing is '
            'painted in). Original camera frames through the render\'s own filter. Keys: <b>1</b> before · <b>2</b> '
            'after · <b>3</b> detected.</p>' + "".join(sec(m) for m in meta) +
            '</main><script>document.querySelectorAll(".tabs").forEach(t=>t.addEventListener("click",e=>{const b='
            'e.target.closest("button");if(!b)return;t.querySelectorAll("button").forEach(x=>x.classList.toggle("on",'
            'x===b));document.getElementById("img-"+t.dataset.tag).src=t.dataset.tag+"_"+b.dataset.v+".jpg";}));'
            'document.addEventListener("keydown",e=>{const v={"1":"before","2":"after","3":"circles"}[e.key];if(!v)'
            'return;document.querySelectorAll(".tabs").forEach(t=>t.querySelector("[data-v="+v+"]").click());});'
            '</script></body></html>')
