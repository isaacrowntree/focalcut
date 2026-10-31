"""An audio-reactive light show over the whole cut, for EDM and festival reels: MilkDrop-style presets of geometric,
particle and light effects in changing palettes, screened over the footage, plus beat-locked blinks and drop
flashes. Off unless reel.json has "overlay".

Everything is driven by the mix the reel plays (video time t is song time plan["music_start"] + t). Like MilkDrop,
the show changes preset every 8 bars (a phrase): each preset is two or three effects and a palette, and the
effects cross-fade over a beat as one preset hands over to the next.

  geometric  tunnel (circles, hexagons, squares or stars flying out of the centre, one ring per beat), mandala
             (a turning flower-of-life with nested polygons, breathing on the bass), kaleido (the footage folded
             into a turning 8-way kaleidoscope), lasers (festival scanners sweeping from the edges), grid (a
             synthwave perspective floor)
  particles  starfield (stars streaking out of the centre, faster on the bass), embers (sparks rising and
             flickering), confetti (bursts from the centre on drops, falling), bokeh (soft drifting discs), sparks
             (short streaks bursting outwards on every beat)
  light      plasma (a slow colour field), leaks (big soft light leaks drifting), flare (an anamorphic streak on the
             kicks), flame (fire rising from the bottom edge, fed by the spectrum, in the palette's colours)
  colour     palettes: neon, synthwave, vaporwave, acid, halloween, ice, fire, rainbow; duotone recolours the
             footage into the palette on drops
  beat       kick (a brightness blink and zoom punch on every beat), split (RGB split on each bar), drops (a white
             flash, then one blink per beat through the drop's first bar, with a confetti / sparks / kaleidoscope /
             duotone burst; song joins count as drops)

Each act has a strength (day sets lighter, night and climax full) and calmer presets in the daytime; shots under
the logo ease it back so the logo reads. Every effect fades out towards the centre of the frame, where the subject
is, so the subject is always seen clean (reel.json "clear": the clear radius). With MilkDrop the footage also fades to
near black towards the edges ("vignette": how dark) and MilkDrop fills them in, rendered at full size; pattern
presets are zoomed in 2x ("milkdrop_zoom"). Light goes mostly into the dark parts of the frame, so faces stay clean.
Full-frame flashes never come faster than one per beat (photosensitivity: about 2 a second at 125 BPM).

reel.json: "overlay": true, or {"strength": {act: 0..1}, "opacity": 0..1, "presets": [[effects], palette], ...],
"beat": [names from kick/split/drops]}.
"""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np

STRENGTH = dict(teaser=0.75, arrival=0.45, shows=0.7, night=0.9, climax=1.0, finale=0.85)
NB = 48                                            # spectrum bands (log-spaced, 40 Hz .. 14 kHz)
HEX = dict(
    neon=["FF2DAA", "00E5FF", "7C4DFF", "FF6EC7", "7DF9FF"],
    synthwave=["DD517F", "E68E36", "556DC8", "7998EE", "FFD166"],
    vaporwave=["A653F5", "8F8CF2", "65B8BF", "F96CFF", "FA92FB"],
    acid=["CCFF00", "FF00CC", "00FFAA", "FFFF33", "BC13FE"],
    halloween=["FF7518", "8A2BE2", "39FF14", "FFB347", "B026FF"],
    ice=["7DF9FF", "00B3FF", "FFFFFF", "6A5CFF", "B8F3FF"],
    fire=["B41400", "FF6E00", "FFD23C", "FFFFE6", "FF3C00"],
)
PRESETS = [                                        # (effects, palette); "calm" ones suit the daytime
    (["tunnel", "sparks"], "neon"),
    (["flame", "embers"], "fire"),
    (["mandala", "bokeh"], "vaporwave"),
    (["lasers", "starfield"], "acid"),
    (["kaleido", "sparks"], "rainbow"),
    (["grid", "leaks", "flare"], "synthwave"),
    (["tunnel", "embers", "plasma"], "halloween"),
    (["plasma", "starfield"], "ice"),
    (["lasers", "flame"], "halloween"),
    (["mandala", "flare", "sparks"], "neon"),
]
CALM = [(["leaks", "bokeh"], "synthwave"), (["mandala", "bokeh"], "vaporwave"), (["leaks", "flare"], "fire"),
        (["plasma", "bokeh"], "ice")]
DAY = ("arrival",)
MILKDROP = [                                       # real MilkDrop presets (Butterchurn's packs), loud ones for the night
    "Cope - The Neverending Explosion of Red Liquid Fire", "ech0 - liquid firesticks I", "martin - ice flames",
    "Rovastar + Geiss - Hyperspace - kaleidoscope", "Rovastar + Geiss - Hyperkaleidoscope Glow 2 motion blur (Jelly)",
    "martin - tunnel race", "martin - cope - laser dome", "Phat+fiShbRaiN+Eo.S_Mandala_Chasers_remix",
    "flexi - jelly fish mandala", "Rovastar - Explosive Minds", "martin - dark galaxy", "martin - into the fireworks",
]
MILKDROP_ZOOM = {                                  # pattern-filling presets read better zoomed in: bigger, bolder shapes
    "Cope - The Neverending Explosion of Red Liquid Fire": 2, "ech0 - liquid firesticks I": 2, "martin - ice flames": 2,
    "Rovastar + Geiss - Hyperspace - kaleidoscope": 2, "Rovastar + Geiss - Hyperkaleidoscope Glow 2 motion blur (Jelly)": 2,
    "Phat+fiShbRaiN+Eo.S_Mandala_Chasers_remix": 2, "flexi - jelly fish mandala": 2, "Rovastar - Explosive Minds": 2,
}
MILKDROP_CALM = ["martin - dark galaxy", "martin - into the fireworks",
                 "Rovastar + Geiss - Hyperkaleidoscope Glow 2 motion blur (Jelly)", "martin - ice flames"]
BUTTERCHURN = ["butterchurn@2.6.7/lib/butterchurn.min.js", "butterchurn-presets@2.4.7/lib/butterchurnPresets.min.js",
               "butterchurn-presets@2.4.7/lib/butterchurnPresetsExtra.min.js",
               "butterchurn-presets@2.4.7/lib/butterchurnPresetsExtra2.min.js"]


def options(cfg) -> dict | None:
    if not cfg:
        return None
    cfg = {} if cfg is True else dict(cfg)
    return dict(strength={**STRENGTH, **cfg.get("strength", {})}, opacity=float(cfg.get("opacity", 1.0)), clear=float(cfg.get("clear", 0.42)),
                presets=[(list(e), p) for e, p in cfg.get("presets", PRESETS)],
                beat=tuple(cfg.get("beat", ("kick", "split", "drops"))), engine=cfg.get("engine", "builtin"),
                milkdrop=list(cfg.get("milkdrop", MILKDROP)), milkdrop_calm=list(cfg.get("milkdrop_calm", MILKDROP_CALM)),
                milkdrop_zoom={**MILKDROP_ZOOM, **cfg.get("milkdrop_zoom", {})}, vignette=float(cfg.get("vignette", 0.85)))


class Palette:
    def __init__(self, name):
        self.name = name
        self.c = None if name == "rainbow" else np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)]
                                                           for h in HEX[name]], np.float32)

    def pick(self, i, t=0.0):
        if self.c is None:
            return hsv((i * 0.17 + t * 0.05) % 1)
        return tuple(int(v) for v in self.c[int(i) % len(self.c)])

    def ramp(self):
        """256-entry gradient from black up through the palette (for intensity maps: fire, plasma, duotone)."""
        stops = np.array([hsv(i / 5) for i in range(5)], np.float32) if self.c is None else self.c[:4]
        if self.name == "fire":
            stops = self.c[[0, 1, 2, 3]]
        stops = np.vstack([[0, 0, 0], stops])
        x = np.linspace(0, 1, len(stops))
        v = np.linspace(0, 1, 256)
        return np.stack([np.interp(v, x, stops[:, j]) for j in range(3)], 1).astype(np.float32)


def hsv(h, s=1.0, v=1.0):
    c = cv2.cvtColor(np.uint8([[[int(h * 180) % 180, int(s * 255), int(v * 255)]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(x) for x in c)


def features(song: str, start: float, n: int, fps: int) -> dict:
    """Per output frame: NB band levels (0..1), bass level, and the waveform (256 samples, -1..1)."""
    sr = 48000                                     # decoded with ffmpeg: no audio library needed in the render's env
    y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", song, "-ac", "1", "-ar", str(sr), "-f", "f32le",
                                      "-"], capture_output=True, check=True).stdout, np.float32)
    win = 2048
    hann = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    edges = np.geomspace(40, 14000, NB + 1)
    idx = [np.where((freqs >= a) & (freqs < b))[0] for a, b in zip(edges[:-1], edges[1:])]
    idx = [i if len(i) else np.array([np.argmin(np.abs(freqs - a))]) for i, a in zip(idx, edges[:-1])]
    bass_i = np.where((freqs > 30) & (freqs < 150))[0]
    bands, bass, wave = np.zeros((n, NB), np.float32), np.zeros(n, np.float32), np.zeros((n, 256), np.float32)
    for k in range(n):
        c = int((start + k / fps) * sr)
        seg = y[max(c - win // 2, 0):c + win // 2]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        S = np.abs(np.fft.rfft(seg * hann))
        bands[k] = [S[i].mean() for i in idx]
        bass[k] = S[bass_i].mean()
        wave[k] = seg[win // 2 - 512:win // 2 + 512:4]
    bands = np.log1p(bands * 4)
    bands /= np.percentile(bands, 98, axis=0) + 1e-6           # each band against its own loud level
    bass = bass / (np.percentile(bass, 95) + 1e-6)
    wave /= np.percentile(np.abs(wave), 99.5) + 1e-6
    return dict(bands=np.clip(bands, 0, 1.2), bass=np.clip(bass, 0, 1.5), wave=np.clip(wave, -1.2, 1.2))


def strength(plan: dict, n: int, opts: dict) -> np.ndarray:
    out, pos = np.zeros(n, np.float32), 0
    for s in plan["shots"]:
        v = opts["strength"].get(s.get("act"), 0.7) * (0.35 if s.get("logo") else 1.0)
        out[pos:pos + s["frames"]] = v
        pos += s["frames"]
    out[pos:] = out[pos - 1] if pos else 0.7
    k = np.ones(13) / 13                                         # ease between acts rather than jump
    return np.convolve(np.pad(out, 6, mode="edge"), k, "valid") * opts["opacity"]



def grid(music: dict, plan: dict, n: int, fps: int) -> dict:
    """Per frame: time since the last beat (s), whether it was a downbeat, the bar number, and the drop pulse (beats
    since a drop started, or -1)."""
    off = plan["music_start"]
    beats = np.array(music["beats"]) - off
    bar_t = np.array(music["bars"]) - off
    bars = set(np.round(bar_t, 3))
    acc = np.array(music.get("beat_energy") or np.ones(len(beats)))
    drop_beats = set()
    for s in (music.get("sections") or [])[1:]:                  # each song join
        drop_beats.add(int(np.argmin(np.abs(beats - (s["at"] - off)))))
    bar_i = [i for i, b in enumerate(beats) if round(b, 3) in bars]
    e = acc / (np.median(acc) + 1e-9)
    for j in range(4, len(bar_i) - 4):                          # the low end jumping back in after a build
        i = bar_i[j]
        after, before = e[i:i + 16].mean(), e[max(i - 16, 0):i].mean()
        if after > 1.7 * before and after > 0.9:
            drop_beats.add(i)
    t = np.arange(n) / fps
    k = np.searchsorted(beats, t, side="right") - 1
    since = np.where(k >= 0, t - beats[np.clip(k, 0, None)], 9.0)
    down = np.array([round(beats[i], 3) in bars if i >= 0 else False for i in k])
    drop = np.full(n, -1)
    for d in sorted(drop_beats):
        for b in range(4):                                       # the drop's first bar
            m = k == d + b
            drop[m & (drop < 0)] = b
    bar = np.clip(np.searchsorted(bar_t, t + 1e-3, side="right") - 1, 0, None)
    return dict(since=since, down=down, drop=drop, bar=bar)


def acts(plan: dict, n: int) -> list:
    out = []
    for s in plan["shots"]:
        out += [s.get("act")] * s["frames"]
    return (out + [out[-1] if out else None] * n)[:n]


def phrases(G: dict, A: list, pool: list, calm: list, seed: int = 7) -> list:
    """One choice per frame: a new one every 8 bars, never the same twice running, calm ones in the daytime."""
    rng = np.random.default_rng(seed)
    queues, last, out, cur = {False: [], True: []}, None, [], {}
    for k in range(len(A)):
        ph = int(G["bar"][k]) // 8
        day = A[k] in DAY
        if (ph, day) not in cur:
            src, q = (calm if day else pool), queues[day]
            if not q:
                q.extend(rng.permutation(len(src)).tolist())
            choice = src[q.pop(0)]
            if choice == last and q:
                choice = src[q.pop(0)]
            cur[(ph, day)] = last = choice
        out.append(cur[(ph, day)])
    return out


def schedule(G: dict, A: list, opts: dict, seed: int = 7) -> list:
    """The built-in preset (effects, palette) for each frame."""
    return phrases(G, A, opts["presets"], CALM, seed)


class Show:
    """The effects at half resolution, each keeping its own state; a weight per effect eases it in and out."""

    def __init__(self, W, H, fps, beat, rng):
        self.W, self.H, self.fps, self.beat, self.rng = W, H, fps, beat, rng
        self.w, self.h = W // 2, H // 2
        w, h = self.w, self.h
        self.R = float(np.hypot(w, h) / 2)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.xx, self.yy = xx, yy
        dx, dy = xx - w / 2, yy - h / 2
        self.r, self.th = np.hypot(dx, dy), np.arctan2(dy, dx)
        q = np.mgrid[0:h // 4, 0:w // 4].astype(np.float32) / max(h, w) * 4
        self.qy, self.qx = q[0], q[1]
        self.weight = {}
        self.edge = None                                         # full-size centre mask (set by apply)
        self.zoom = None                                         # MilkDrop zoom per frame (set by apply)
        self.pal, self.ramp = Palette("neon"), Palette("neon").ramp()
        self.fire = np.zeros((h, w), np.float32)
        self.burst = np.zeros((h, w, 3), np.float32)
        self.rings, self.stars, self.parts, self.flakes = [], None, [], []
        self.bok = None

    def palette(self, name):
        if name != self.pal.name:
            self.pal, self.ramp = Palette(name), Palette(name).ramp()

    def glow(self, img, sigma=3, gain=0.9):
        return img + cv2.GaussianBlur(img, (0, 0), sigma) * gain

    # geometric ----------------------------------------------------------------------------------------------------
    def tunnel(self, c):
        w, h, R = self.w, self.h, self.R
        if c["beat_now"]:
            self.rings.append(dict(r=0.03 * R, sides=[0, 6, 4, 5][int(c["bar"]) % 4], col=self.pal.pick(len(self.rings))))
        img = np.zeros((h, w, 3), np.float32)
        keep = []
        for g in self.rings:
            g["r"] *= 1.045 + 0.03 * c["bass"]
            if g["r"] > 1.6 * R:
                continue
            keep.append(g)
            th = int(1 + 5 * g["r"] / R)
            rot = c["t"] * 0.4
            if g["sides"] == 0:
                cv2.circle(img, (w // 2, h // 2), int(g["r"]), g["col"], th, cv2.LINE_AA)
            else:
                n = g["sides"] * (2 if g["sides"] == 5 else 1)
                a = rot + np.arange(n) * 2 * np.pi / n
                rr = g["r"] * (np.where(np.arange(n) % 2, 0.5, 1.0) if g["sides"] == 5 else 1.0)
                pts = np.c_[w / 2 + rr * np.cos(a), h / 2 + rr * np.sin(a)].astype(np.int32)
                cv2.polylines(img, [pts], True, g["col"], th, cv2.LINE_AA)
        self.rings = keep
        return self.glow(img, 4, 1.2)

    def mandala(self, c):
        w, h = self.w, self.h
        img = np.zeros((h, w, 3), np.float32)
        base = min(w, h) * (0.12 + 0.05 * c["bass"])
        cx, cy, t = w / 2, h / 2, c["t"]
        for ring in range(3):
            m = 6 * (ring + 1)
            d = base * (ring + 1) * 0.9
            for j in range(m):
                a = t * (0.3 if ring % 2 else -0.25) + j * 2 * np.pi / m
                cv2.circle(img, (int(cx + d * np.cos(a)), int(cy + d * np.sin(a))), int(base * 0.9),
                           self.pal.pick(ring + j % 2, t), 1, cv2.LINE_AA)
        for p, sides in enumerate((3, 4, 6, 8)):
            rr = base * (0.6 + p * 0.85) * (1 + 0.15 * c["kick"])
            a = t * (0.5 - 0.25 * p) + np.arange(sides) * 2 * np.pi / sides
            pts = np.c_[cx + rr * np.cos(a), cy + rr * np.sin(a)].astype(np.int32)
            cv2.polylines(img, [pts], True, self.pal.pick(p + 2, t), 2, cv2.LINE_AA)
        return self.glow(img, 3, 1.0)

    def kaleido(self, c, n=8):
        small = cv2.resize(c["frame"], (self.w, self.h), interpolation=cv2.INTER_AREA)
        wedge = 2 * np.pi / n
        a = np.mod(self.th + c["t"] * 0.35, wedge)
        a = np.where(a > wedge / 2, wedge - a, a) - np.pi / 2    # mirror inside the wedge; sample from the top
        r = self.r * 0.8
        mx = (self.w / 2 + r * np.cos(a)).astype(np.float32)
        my = (self.h / 2 + r * np.sin(a)).astype(np.float32)
        return cv2.remap(small, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    def lasers(self, c):
        w, h, t = self.w, self.h, c["t"]
        img = np.zeros((h, w, 3), np.float32)
        on = 0.35 + 0.65 * c["kick"]
        L = 2.2 * max(w, h)
        for j, (ox, oy) in enumerate(((0, h), (w, h), (w / 2, h), (0, 0), (w, 0))):
            for b in range(3):
                base = np.arctan2(h / 2 - oy, w / 2 - ox)
                a = base + 0.6 * np.sin(t * (0.9 + 0.2 * j) + b * 0.5 + j)
                col = np.array(self.pal.pick(j + int(c["bar"]), t), np.float32) * on
                cv2.line(img, (int(ox), int(oy)), (int(ox + L * np.cos(a)), int(oy + L * np.sin(a))),
                         col.tolist(), 2, cv2.LINE_AA)
        return self.glow(img, 5, 1.5)

    def grid(self, c):
        w, h, t = self.w, self.h, c["t"]
        img = np.zeros((h, w, 3), np.float32)
        hz = int(h * 0.62)
        col = self.pal.pick(0, t)
        for i in range(14):
            z = ((i + (t * 1.2) % 1) / 14) ** 2
            y = int(hz + (h - hz) * z)
            cv2.line(img, (0, y), (w, y), col, 1 + int(2 * z), cv2.LINE_AA)
        for i in range(-12, 13):
            cv2.line(img, (w // 2, hz), (int(w / 2 + i * w / 8), h), col, 1, cv2.LINE_AA)
        cv2.line(img, (0, hz), (w, hz), self.pal.pick(1, t), 2, cv2.LINE_AA)
        return self.glow(img, 4, 1.3) * (0.6 + 0.6 * c["kick"])

    # particles ----------------------------------------------------------------------------------------------------
    def starfield(self, c):
        w, h = self.w, self.h
        if self.stars is None:
            self.stars = np.c_[self.rng.uniform(-1, 1, (300, 2)), self.rng.uniform(0.1, 1, 300)]
        S = self.stars
        f = 0.35 * min(w, h)
        p0 = S[:, :2] / S[:, 2:] * f
        S[:, 2] -= 0.008 + 0.03 * c["bass"]
        dead = S[:, 2] < 0.05
        S[dead] = np.c_[self.rng.uniform(-1, 1, (dead.sum(), 2)), np.ones(dead.sum())]
        p1 = S[:, :2] / S[:, 2:] * f
        img = np.zeros((h, w, 3), np.float32)
        for i, (a, b) in enumerate(zip(p0, p1)):
            if dead[i]:
                continue
            cv2.line(img, (int(w / 2 + a[0]), int(h / 2 + a[1])), (int(w / 2 + b[0]), int(h / 2 + b[1])),
                     self.pal.pick(i), 1 + int(1.2 / S[i, 2] > 6), cv2.LINE_AA)
        return self.glow(img, 2, 0.8)

    def embers(self, c):
        w, h = self.w, self.h
        for _ in range(int(3 + 10 * c["bass"])):
            self.parts.append([self.rng.uniform(0, w), h + 2, self.rng.normal(0, 0.6), -self.rng.uniform(1.5, 4.5),
                               self.rng.uniform(40, 110), self.pal.pick(self.rng.integers(5))])
        img = np.zeros((h, w, 3), np.float32)
        keep = []
        for p in self.parts:
            p[0] += p[2] + 0.8 * np.sin(p[1] / 23); p[1] += p[3]; p[4] -= 1
            if p[4] <= 0 or p[1] < -5:
                continue
            keep.append(p)
            fl = 0.5 + 0.5 * self.rng.random()
            cv2.circle(img, (int(p[0]), int(p[1])), 2, [v * fl * min(1, p[4] / 30) for v in p[5]], -1, cv2.LINE_AA)
        self.parts = keep[-900:]
        return self.glow(img, 3, 1.4)

    def confetti(self, c):
        w, h = self.w, self.h
        if c["drop"] == 0 and c["beat_now"]:
            for _ in range(260):
                a, v = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(4, 14)
                self.flakes.append([w / 2, h / 2, v * np.cos(a), v * np.sin(a) - 3, 90,
                                      self.pal.pick(self.rng.integers(5)), self.rng.uniform(0, np.pi)])
        img = np.zeros((h, w, 3), np.float32)
        keep = []
        for p in self.flakes:
            p[0] += p[2]; p[1] += p[3]; p[2] *= 0.97; p[3] = p[3] * 0.97 + 0.35; p[4] -= 1; p[6] += 0.3
            if p[4] <= 0 or p[1] > h + 5:
                continue
            keep.append(p)
            dx, dy = 5 * np.cos(p[6]), 3 * abs(np.sin(p[6] * 1.7)) + 1
            cv2.rectangle(img, (int(p[0] - dx), int(p[1] - dy)), (int(p[0] + dx), int(p[1] + dy)), p[5], -1)
        self.flakes = keep
        return self.glow(img, 2, 0.6)

    def bokeh(self, c):
        w, h, t = self.w, self.h, c["t"]
        if self.bok is None:
            self.bok = np.c_[self.rng.uniform(0, 1, (22, 2)), self.rng.uniform(0.02, 0.06, 22),
                             self.rng.uniform(0, 6.3, 22)]
        img = np.zeros((h // 2, w // 2, 3), np.float32)
        for i, (x, y, r, ph) in enumerate(self.bok):
            px = (x + 0.02 * t * (1 + i % 3)) % 1.1 - 0.05
            py = (y + 0.03 * np.sin(t * 0.5 + ph)) % 1
            lv = 0.35 + 0.4 * (0.5 + 0.5 * np.sin(t * 2 + ph)) + 0.4 * c["bass"]
            col = [v * lv for v in self.pal.pick(i, t)]
            cv2.circle(img, (int(px * w / 2), int(py * h / 2)), int(r * max(w, h) / 2), col, -1, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (0, 0), 2.5)
        return cv2.resize(img, (w, h)) * 0.5

    def sparks(self, c):
        w, h = self.w, self.h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.8, 1.06)
        b = cv2.warpAffine(self.burst, M, (w, h), borderMode=cv2.BORDER_CONSTANT) * 0.78
        if c["beat_now"]:
            n = 40 if c["drop"] == 0 else 12 + int(10 * min(c["bass"], 1))
            R = self.R
            for a in self.rng.uniform(0, 2 * np.pi, n):
                r0 = self.rng.uniform(0.05, 0.3) * R; r1 = r0 + self.rng.uniform(0.06, 0.18) * R   # sparks, not beams
                col = (255, 255, 255) if c["drop"] == 0 and self.rng.random() < 0.4 else \
                    self.pal.pick(self.rng.integers(5), c["t"])
                cv2.line(b, (int(w / 2 + r0 * np.cos(a)), int(h / 2 + r0 * np.sin(a))),
                         (int(w / 2 + r1 * np.cos(a)), int(h / 2 + r1 * np.sin(a))), col, 2, cv2.LINE_AA)
        self.burst = b
        return self.glow(b, 4, 1.2)

    # light --------------------------------------------------------------------------------------------------------
    def plasma(self, c):
        t, x, y = c["t"], self.qx, self.qy
        v = (np.sin(x * 6 + t) + np.sin(y * 7 - t * 1.3) + np.sin((x + y) * 5 + t * 0.7)
             + np.sin(np.hypot(x - 0.5, y - 0.4) * 12 - t * 2))
        v = ((v + 4) / 8 * 255 * (0.55 + 0.3 * c["bass"])).clip(0, 255).astype(np.uint8)
        return cv2.resize(self.ramp[v], (self.w, self.h), interpolation=cv2.INTER_LINEAR) * (0.2 + 0.25 * c["bass"])

    def leaks(self, c):
        w, h, t = self.w, self.h, c["t"]
        img = np.zeros((h // 4, w // 4, 3), np.float32)
        for i in range(3):
            px = 0.5 + 0.55 * np.sin(t * (0.21 + 0.07 * i) + i * 2.1)
            py = 0.5 + 0.45 * np.sin(t * (0.17 + 0.05 * i) + i)
            cv2.circle(img, (int(px * w / 4), int(py * h / 4)), int(max(w, h) / 10), self.pal.pick(i, t), -1)
        img = cv2.GaussianBlur(img, (0, 0), max(w, h) / 30)
        return cv2.resize(img, (w, h)) * (0.3 + 0.3 * c["bass"])

    def flare(self, c):
        w, h = self.w, self.h
        img = np.zeros((h, w, 3), np.float32)
        k = c["kick"]
        if k > 0.05:
            y = int(h * (0.3 + 0.2 * np.sin(c["bar"] * 1.7)))
            cv2.line(img, (0, y), (w, y), [v * k for v in self.pal.pick(int(c["bar"]))], 3)
            cv2.circle(img, (int(w * (0.3 + 0.4 * ((c["bar"] * 0.37) % 1))), y), 6, (255 * k,) * 3, -1)
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=25, sigmaY=2) * 3
        return img

    def flame(self, c):
        w = self.w
        bands = c["bands"]
        half = np.interp(np.linspace(0, 1, w // 2), np.linspace(0, 1, NB), bands)
        row = np.r_[half[::-1], half]
        row = np.pad(row, (0, w - len(row)), mode="edge")
        seed = np.clip(row * (0.9 + 0.8 * c["bass"]), 0, 1.4) * 255
        clumps = cv2.GaussianBlur(self.rng.random((1, w)).astype(np.float32), (0, 0), 6)[0]
        clumps = np.clip((clumps - clumps.min()) / (np.ptp(clumps) + 1e-6) * 1.6 - 0.3, 0, 1)   # separate tongues
        fire = self.fire
        fire[-4:] = np.maximum(fire[-4:], seed * clumps * 1.5)
        mx = self.xx + 3.0 * np.sin(self.yy / 17 + c["t"] * 5.0) + 2.0 * np.sin(self.yy / 7 - c["t"] * 9.0)
        my = self.yy + 4 + 4 * c["bass"]
        fire = cv2.remap(fire, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        self.fire = np.clip(cv2.GaussianBlur(fire, (0, 0), 1.0) * 0.972 - 0.3, 0, 320)
        return self.ramp[np.clip(self.fire, 0, 255).astype(np.uint8)]

    def duotone(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return self.ramp[g]


LIGHT = ("tunnel", "mandala", "lasers", "grid", "starfield", "embers", "confetti", "bokeh", "sparks", "plasma",
         "leaks", "flare", "flame")


def key(md, W, H):
    """Alpha for a MilkDrop frame (it has none: every preset draws a full opaque frame): transparent below about a
    fifth of full brightness, solid from about half, and grey, low-saturation textures mostly keyed out so only its
    vivid light shows. Returned at W x H, shape (H, W, 1)."""
    m = md.astype(np.float32) / 255
    lum = m.mean(-1)
    sat = m.max(-1) - m.min(-1)
    a = np.clip((lum - 0.2) / 0.35, 0, 1) ** 1.5 * (0.3 + 0.7 * np.clip((sat - 0.12) / 0.35, 0, 1))
    a = cv2.GaussianBlur(a, (0, 0), 1.0)                         # soft edges
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]


def edge_mask(W, H, clear=0.42):
    """Where the effects may show, (H, W, 1) in 0..1: nothing in the centre of the frame, where the subject is (the
    crop keeps them there), fading in to full strength towards the edges. clear: the radius of the clear centre as a
    fraction of the half-frame (an ellipse, so it fits both formats); the fade takes another 0.45."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    e = np.hypot((x - W / 2) / (W / 2), (y - H / 2) / (H / 2))
    m = np.clip((e - clear) / 0.45, 0, 1)
    return (m * m * (3 - 2 * m))[..., None]                       # smoothstep: no visible ring


def zoom(f, z):
    H, W = f.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def frame(f, k, show, F, G, S, preset, opts, md=None):
    """One output frame: the beat layer, the effects (built-in, or the MilkDrop layer md at full size), all clear of
    the centre; with MilkDrop the footage also fades to near black at the edges and MilkDrop fills them."""
    s, beat, fps = float(S[k]), show.beat, show.fps
    since, down, drop = float(G["since"][k]), bool(G["down"][k]), int(G["drop"][k])
    kick = float(np.exp(-since / (0.22 * beat)))                 # 1 on the beat, gone within half a beat
    bass = min(float(F["bass"][k]), 1.3)
    effects, pal = preset
    show.palette(pal)
    active = set(effects) | ({"confetti", "sparks"} if drop >= 0 or show.flakes else set())
    ease = 1 / max(1, beat * fps)
    for e in set(show.weight) | active:
        show.weight[e] = min(1.0, show.weight.get(e, 0) + ease) if e in active else max(0.0, show.weight.get(e, 0) - ease)
    c = dict(t=k / fps, bass=bass, bands=F["bands"][k], kick=kick, since=since, down=down, drop=drop,
             bar=int(G["bar"][k]), beat_now=since < 1 / fps, frame=f)
    B = opts["beat"]
    if "kick" in B:
        f = zoom(f, 1 + (0.035 if down else 0.018) * s * kick)
    if "split" in B and down and since < 4 / fps:
        sh = int((4 - since * fps) * 4 * s * show.W / 1920)
        if sh:
            f = f.copy(); f[..., 0] = np.roll(f[..., 0], sh, 1); f[..., 2] = np.roll(f[..., 2], -sh, 1)
    x = f.astype(np.float32)
    up = (show.W, show.H)
    kw = show.weight.get("kaleido", 0)
    a = (0.08 + 0.25 * bass ** 2) * kw
    if down and kw:
        a = max(a, 0.7 * kw * float(np.exp(-since / (0.5 * beat))))
    if drop >= 0:                                                # every drop floods in the kaleidoscope
        a = max(a, (0.85 if drop == 0 else 0.55) * float(np.exp(-since / (0.6 * beat))) * (0.6 if md is not None else 1))
    a *= s
    if a > 0.02:
        c["frame"] = f
        x = x * (1 - a * show.edge) + cv2.resize(show.kaleido(c), up) * (a * show.edge)
    if drop >= 0:                                                # and recolours the footage into the palette
        d = 0.6 * s * float(np.exp(-since / (0.8 * beat))) * (1.0 if drop == 0 else 0.6) * (0.5 if md is not None else 1)
        if d > 0.02:
            x = x * (1 - d * show.edge) + show.duotone(f) * (d * show.edge)
    light = None
    for e in LIGHT:
        wgt = show.weight.get(e, 0)
        if wgt <= 0:
            continue
        img = getattr(show, e)(c) * wgt
        light = img if light is None else light + img
    if light is not None:
        dark = 1.15 - x.mean(-1, keepdims=True) / 255            # light goes mostly into the dark: faces stay clean
        light = cv2.resize(np.clip(light, 0, 255), up, interpolation=cv2.INTER_LINEAR)
        x = 255 - (255 - x) * (255 - light * np.minimum(1.0, s * dark) * show.edge) / 255
    if md is not None:                                           # the MilkDrop layer
        z = float(show.zoom[k]) if show.zoom is not None else 1.0
        if z > 1.01:                                             # zoomed: the centre of it, scaled up
            h, w = md.shape[:2]
            cw, ch = int(w / z), int(h / z)
            md = md[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
        mdu = cv2.resize(md, up, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        e = show.edge
        x = x * (1 - opts["vignette"] * min(1.0, 1.2 * s) * e)   # the footage fades to near black at the edges...
        k_ = key(md, show.W, show.H)                             # ...and MilkDrop fills them in: whole at the edges,
        am = (k_ + (1 - k_) * e) * e * min(1.0, s * (0.75 + 0.25 * kick))   # keyed nearer the middle, none in it
        x = x * (1 - am) + mdu * am
    blink = (0.14 if down else 0.07) * s * kick if "kick" in B else 0.0
    if "drops" in B and drop >= 0:
        blink = max(blink, (0.85 if drop == 0 else 0.35) * float(np.exp(-since / (0.12 * beat))))
    if blink:
        x = x + (255 - x) * blink
    return np.clip(x, 0, 255).astype(np.uint8)


def chrome() -> str:
    """A Chromium for the MilkDrop layer: FOCALCUT_CHROME, else Playwright's download."""
    import glob
    c = os.environ.get("FOCALCUT_CHROME") or next(iter(sorted(glob.glob(os.path.expanduser(
        "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")), reverse=True)), None)
    if not c:
        raise RuntimeError("the MilkDrop layer needs Chromium: set FOCALCUT_CHROME or `playwright install chromium`")
    return c


def butterchurn() -> str:
    """Butterchurn (MilkDrop in WebGL) and its preset packs, downloaded once to ~/.cache/focalcut/milkdrop."""
    import urllib.request
    lib = os.path.expanduser("~/.cache/focalcut/milkdrop")
    os.makedirs(lib, exist_ok=True)
    for f in BUTTERCHURN:
        dst = os.path.join(lib, os.path.basename(f))
        if not os.path.exists(dst):
            urllib.request.urlretrieve("https://cdn.jsdelivr.net/npm/" + f, dst + ".part")
            os.replace(dst + ".part", dst)
    return lib


def milkdrop_layer(work, song, start, n, fps, w, h, per_frame, beat) -> str:
    """Render the MilkDrop layer (cached in work/): the presets per frame, each change blended over a beat."""
    import hashlib
    import json
    sched = [[k, p, round(beat, 3)] for k, p in enumerate(per_frame) if k == 0 or p != per_frame[k - 1]]
    sched[0][2] = 0
    key = hashlib.md5(json.dumps([song, os.path.getmtime(song), start, n, fps, w, h, sched]).encode()).hexdigest()[:10]
    out = os.path.join(work, f"milkdrop_{key}.mp4")
    if os.path.exists(out):
        return out
    job = dict(song=song, start=start, frames=n, fps=fps, width=w, height=h, out=out + ".part.mp4", schedule=sched,
               chrome=chrome(), lib=butterchurn())
    jf = os.path.join(work, f"milkdrop_{key}.json")
    with open(jf, "w") as f:
        json.dump(job, f)
    subprocess.run(["node", os.path.join(os.path.dirname(__file__), "milkdrop", "render.mjs"), jf], check=True)
    os.replace(out + ".part.mp4", out)
    return out


def apply(src: str, out: str, plan: dict, music: dict, song: str, opts: dict, enc: list, fps: int) -> None:
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", src]).decode().split(",")[:2])
    n = sum(s["frames"] for s in plan["shots"])
    F = features(song, plan["music_start"], n, fps)
    G = grid(music, plan, n, fps)
    S = strength(plan, n, opts)
    A = acts(plan, n)
    show = Show(W, H, fps, plan["beat"], np.random.default_rng(7))
    show.edge = edge_mask(W, H, opts["clear"])
    layer = None
    if opts["engine"] == "milkdrop":                             # drop bursts only; the visuals are MilkDrop's
        pals = list(HEX) + ["rainbow"]
        P = [([], pals[(int(G["bar"][k]) // 8) % len(pals)]) for k in range(n)]
        names = phrases(G, A, opts["milkdrop"], opts["milkdrop_calm"])
        path = milkdrop_layer(os.path.dirname(out), song, plan["music_start"], n, fps, W, H, names, plan["beat"])
        zk = np.ones(max(1, int(plan["beat"] * fps))) / max(1, int(plan["beat"] * fps))   # zoom eases over a beat
        z = np.array([opts["milkdrop_zoom"].get(nm, 1) for nm in names], np.float32)
        show.zoom = np.convolve(np.pad(z, (len(zk) // 2, len(zk) - 1 - len(zk) // 2), mode="edge"), zk, "valid")
        layer = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                 stdout=subprocess.PIPE)
    else:
        P = schedule(G, A, opts)
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                          stdout=subprocess.PIPE)
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", *enc, out], stdin=subprocess.PIPE)
    size, k = W * H * 3, 0
    while True:
        b = rd.stdout.read(size)
        if len(b) < size:
            break
        f = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        md = None
        if layer is not None and k < n:
            lb = layer.stdout.read(W * H * 3)
            md = np.frombuffer(lb, np.uint8).reshape(H, W, 3) if len(lb) == W * H * 3 else None
        g = frame(f, k, show, F, G, S, P[k], opts, md) if k < n else f
        wr.stdin.write(np.ascontiguousarray(g).tobytes())
        k += 1
        if k % 250 == 0:
            print(f"\r[reel] overlay: {k}/{n}", end="", flush=True)
    print()
    rd.wait(); wr.stdin.close(); wr.wait()
    if layer is not None:
        layer.stdout.close(); layer.wait()
    if wr.returncode:
        raise RuntimeError("overlay encode failed")
