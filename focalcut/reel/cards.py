"""Text and logo cards drawn for a reel: a title laid over a shot, and an end card (reference mode, template.py).

    title    a shot's own (plan "title": {"box": [...], "text": [lines], "color": ..., "sizes": [...]}), else
             reel.json "title": {"lines": ["November 21st, 2026", "MALABAR BEACH"], "font": "<ttf>", "weight":
             "SemiBold", "color": "#161616", "sizes": [0.46, 0.2], "tracking": [0, 0.12]} -- drawn into the box the
             reference's title sat in (sizes: each line's font size as a share of that box's height; tracking:
             letter spacing in ems), a transparent PNG the size of the frame
    endcard  reel.json "endcard": {"logo": "<png>", "stack": true} (default: the project's "logo") -- the logo
             at the size and place the reference's logo had, a transparent layer render.card_shot lays over moving
             footage softened into a ground ("ground": "still" puts it on the reference card's own still ground
             instead: its frame with the logo taken out, a heavy median). A wide logo (a mark, a gap, then a wordmark) is stacked, mark over
             wordmark, when the reference's logo is about as tall as it is wide.
"""
from __future__ import annotations

import os

import numpy as np


def _font(path, size, weight=None):
    from PIL import ImageFont
    f = ImageFont.truetype(path, max(int(size), 6))
    if weight:
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
    return f


def _text_width(draw, text, font, track):
    return sum(draw.textlength(ch, font=font) for ch in text) + track * max(len(text) - 1, 0)


def title(project, fmt, spec, size):
    """The title PNG for a shot (spec: the reference's title -- its box), from reel.json "title"."""
    from PIL import Image, ImageDraw
    cfg = {**(project.config.get("title") or {}), **{k: v for k, v in (spec or {}).items()
                                                      if k in ("text", "sizes", "tracking", "color", "font", "weight",
                                                               "weight2")}}
    lines = cfg.get("text") or cfg.get("lines") or []      # (a shot's own title: its "text" lines and style)
    if not lines:
        return None
    W, H = size
    font = cfg.get("font")
    if font and not os.path.isabs(font):
        font = os.path.join(project.folder, font)
    if not font or not os.path.exists(font):
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    sizes = cfg.get("sizes") or [0.46, 0.2]
    tracks = cfg.get("tracking") or [0.0, 0.12]
    colour = cfg.get("color", "#161616")
    bx, by, bw, bh = (spec or {}).get("box") or [0.3, 0.25, 0.4, 0.13]
    cx = (bx + bw / 2) * W
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = by * H
    for k, text in enumerate(lines):
        fs = sizes[min(k, len(sizes) - 1)] * bh * H
        f = _font(font, fs, cfg.get("weight", "SemiBold") if k == 0 else cfg.get("weight2", "Regular"))
        tr = tracks[min(k, len(tracks) - 1)] * fs
        w = _text_width(d, text, f, tr)
        x = cx - w / 2
        asc, desc = f.getmetrics()
        for ch in text:
            d.text((x, y), ch, font=f, fill=colour)
            x += d.textlength(ch, font=f) + tr
        y += asc + desc * 0.35
    import hashlib
    tag = hashlib.md5(repr((lines, sizes, tracks, colour, font, spec and spec.get("box"))).encode()).hexdigest()[:8]
    out = project.path("cards", f"title_{fmt}_{W}x{H}_{tag}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    return out


def ground(frame):
    """A card's ground with what sits on it taken out: a heavy median at low resolution, scaled back up."""
    from scipy.ndimage import median_filter, zoom
    small = frame[::8, ::8].astype(float)
    g = median_filter(small, size=(21, 21, 1))
    g = zoom(g, (frame.shape[0] / small.shape[0], frame.shape[1] / small.shape[1], 1), order=1)
    return np.clip(g[: frame.shape[0], : frame.shape[1]], 0, 255)


def stack(logo):
    """A wide logo (mark | gap | wordmark) as mark over wordmark; the logo unchanged when it has no clear gap."""
    from PIL import Image
    a = np.array(logo)[..., 3] > 20
    on = a.any(0)
    xs = np.nonzero(on)[0]
    if not len(xs):
        return logo
    gaps, start = [], None
    for x in range(xs[0], xs[-1] + 1):
        if not on[x] and start is None:
            start = x
        elif on[x] and start is not None:
            gaps.append((x - start, start, x)); start = None
    if not gaps:
        return logo
    g, a0, a1 = max(gaps)
    if g < 0.03 * logo.width:
        return logo
    mark = logo.crop((xs[0], 0, a0, logo.height))
    word = logo.crop((a1, 0, xs[-1] + 1, logo.height))
    mark, word = mark.crop(mark.getbbox()), word.crop(word.getbbox())
    mw = mark.width
    word = word.resize((int(mw * 1.45), int(word.height * mw * 1.45 / word.width)), Image.LANCZOS)
    gap = int(mark.height * 0.16)
    out = Image.new("RGBA", (max(mark.width, word.width), mark.height + gap + word.height), (0, 0, 0, 0))
    out.alpha_composite(mark, ((out.width - mark.width) // 2, 0))
    out.alpha_composite(word, ((out.width - word.width) // 2, mark.height + gap))
    return out


def endcard(project, fmt, shot, size):
    """The end card PNG: the logo on the reference card's ground, where and as big as the reference's was."""
    from PIL import Image

    from .reference import frame
    W, H = size
    ref = project.load("reference.json")
    spec = shot.get("card") or {}
    cfg = project.config.get("endcard") or {}
    if cfg.get("ground") == "still":                         # the reference's own still ground (asked for by name)
        f = frame(ref["video"], (shot["t0"] + shot["t1"]) / 2)
        g = Image.fromarray(ground(f).astype(np.uint8)).resize((W, H), Image.LANCZOS).convert("RGBA")
    else:                                                    # the logo alone: render.card_shot lays it over footage
        g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    path = cfg.get("logo") or project.config.get("logo")
    if path and not os.path.isabs(path):
        path = os.path.join(project.folder, path)
    if path and os.path.exists(path):
        logo = Image.open(path).convert("RGBA")
        bx, by, bw, bh = spec.get("box") or [0.33, 0.2, 0.34, 0.58]
        if cfg.get("stack", bh * H > 0.6 * bw * W) and logo.width > 2.2 * logo.height:
            logo = stack(logo)
        k = min(bw * W / logo.width, bh * H / logo.height)
        logo = logo.resize((max(int(logo.width * k), 1), max(int(logo.height * k), 1)), Image.LANCZOS)
        g.alpha_composite(logo, (int((bx + bw / 2) * W - logo.width / 2), int((by + bh / 2) * H - logo.height / 2)))
    out = project.path("cards", f"endcard_{fmt}_{W}x{H}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    (g.convert("RGB") if cfg.get("ground") == "still" else g).save(out)
    return out


def caption(project, spec, size):
    """A caption PNG (transparent, the frame's size): spec {"text": words, "ref": a source line or None} -- the words
    in quotation marks, wrapped to `width` of the frame, the reference small and spaced beneath, bottom left, white
    over a soft shadow; the font is reel.json "title"'s (or "captions_style" {"font", "size", "width", "x", "y"})."""
    import hashlib

    from PIL import Image, ImageDraw, ImageFilter
    st = {**{"size": 0.040, "width": 0.44, "x": 0.075, "y": 0.86, "color": "#FAF7F0"},
          **(project.config.get("captions_style") or {})}
    font = st.get("font") or (project.config.get("title") or {}).get("font")
    if font and not os.path.isabs(font):
        font = os.path.join(project.folder, font)
    if not font or not os.path.exists(font):
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    W, H = size
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    fq = _font(font, H * float(st["size"]), "Regular")
    fr = _font(font, H * float(st["size"]) * 0.6, "SemiBold")
    lines, cur = [], ""
    for wd in f"\u201c{spec['text']}\u201d".split():
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=fq) > W * float(st["width"]) and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = t
    lines.append(cur)
    ref = spec.get("ref")
    x = int(W * float(st["x"]))
    y = int(H * float(st["y"])) - int(len(lines) * fq.size * 1.25) - (int(fr.size * 1.8) if ref else 0)
    ops = []
    for ln in lines:
        ops.append((x, y, ln, fq))
        y += int(fq.size * 1.25)
    if ref:
        tx = x
        for ch in ref.upper():
            ops.append((tx, y + int(fr.size * 0.5), ch, fr))
            tx += d.textlength(ch, font=fr) + fr.size * 0.2
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ds = ImageDraw.Draw(sh)
    for ox, oy, t, f in ops:
        ds.text((ox + 2 * W // 1920, oy + 2 * W // 1920), t, font=f, fill=(0, 0, 0, 180))
    sh = sh.filter(ImageFilter.GaussianBlur(H * 0.006))
    for ox, oy, t, f in ops:
        d.text((ox, oy), t, font=f, fill=st["color"])
    img = Image.alpha_composite(sh, lay)
    tag = hashlib.md5(repr((spec, st, font, W, H)).encode()).hexdigest()[:10]
    out = project.path("cards", f"caption_{tag}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    return out


def glow_under(png, strength=0.22, colour=(250, 247, 240)):
    """The logo layer with a soft light under it: an ellipse about the logo's own bounds, `strength` opaque at its
    middle and gone well before the frame's edges -- a vignette for the logo only, the ground left as it is."""
    from PIL import Image
    im = Image.open(png).convert("RGBA")
    a = np.asarray(im)[..., 3].astype(np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.05)
    if not len(xs):
        return png
    H, W = a.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    rx, ry = max(xs.max() - xs.min(), 1) * 0.85, max(ys.max() - ys.min(), 1) * 0.85
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = strength * np.exp(-2.0 * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
    under = np.zeros((H, W, 4), np.uint8)
    under[..., :3] = colour
    under[..., 3] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    out = Image.alpha_composite(Image.fromarray(under, "RGBA"), im)
    dst = png.replace(".png", f"_glow{int(strength * 100)}.png")
    out.save(dst)
    return dst
