"""An audio-reactive layer over the whole cut, for EDM and festival reels: a Winamp / MilkDrop-style visualiser
screened over the footage plus beat-locked blinks, kicks and flashes. Off unless reel.json has "overlay".

Everything is driven by the mix the reel plays (video time t is song time plan["music_start"] + t):
  trails   the oscilloscope waveform and a radial spectrum drawn into a feedback buffer that zooms, turns and fades
           into itself (MilkDrop's tunnel), hue cycling; screened over the footage at an opacity that pumps on the bass
  bars     a translucent spectrum analyser with falling peak caps, blinking white on the kicks
  kick     a brightness blink and a small zoom punch on every beat, harder on the downbeat
  split    an RGB split on every bar's downbeat
  drops    a white flash, then one blink per beat through the drop's first bar (song joins count as drops too)
Each act has a strength (day sets lighter, night and climax full); shots under the logo ease it back so the logo
reads. Full-frame flashes never come faster than one per beat (photosensitivity: about 2 a second at 125 BPM).

reel.json: "overlay": true, or {"strength": {act: 0..1}, "layers": [names], "opacity": 0..1}.
"""
from __future__ import annotations

import subprocess

import cv2
import numpy as np

STRENGTH = dict(teaser=0.7, arrival=0.4, shows=0.65, night=0.9, climax=1.0, finale=0.85)
LAYERS = ("trails", "bars", "kick", "split", "drops")
NB = 48                                            # spectrum bands (log-spaced, 40 Hz .. 14 kHz)


def options(cfg) -> dict | None:
    if not cfg:
        return None
    cfg = {} if cfg is True else dict(cfg)
    return dict(strength={**STRENGTH, **cfg.get("strength", {})}, layers=tuple(cfg.get("layers", LAYERS)),
                opacity=float(cfg.get("opacity", 1.0)))


def features(song: str, start: float, n: int, fps: int) -> dict:
    """Per output frame: NB band levels (0..1), bass level, and the waveform (256 samples, -1..1)."""
    import soundfile as sf
    y, sr = sf.read(song, dtype="float32", always_2d=True)
    y = y.mean(1)
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


def grid(music: dict, plan: dict, n: int, fps: int) -> dict:
    """Per frame: time since the last beat (s), whether it was a downbeat, and the drop pulse (beats since a drop
    started, or -1)."""
    off = plan["music_start"]
    beats = np.array(music["beats"]) - off
    bars = set(np.round(np.array(music["bars"]) - off, 3))
    acc = np.array(music.get("beat_energy") or np.ones(len(beats)))
    drop_beats = set()
    sec = music.get("sections") or []
    for s in sec[1:]:                                            # each song join
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
    return dict(since=since, down=down, drop=drop)


def strength(plan: dict, n: int, opts: dict) -> np.ndarray:
    out, pos = np.zeros(n, np.float32), 0
    for s in plan["shots"]:
        v = opts["strength"].get(s.get("act"), 0.7) * (0.35 if s.get("logo") else 1.0)
        out[pos:pos + s["frames"]] = v
        pos += s["frames"]
    out[pos:] = out[pos - 1] if pos else 0.7
    k = np.ones(13) / 13                                         # ease between acts rather than jump
    return np.convolve(np.pad(out, 6, mode="edge"), k, "valid") * opts["opacity"]


def hsv(h, s=1.0, v=1.0):
    c = cv2.cvtColor(np.uint8([[[int(h * 180) % 180, int(s * 255), int(v * 255)]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(x) for x in c)


class Visualiser:
    def __init__(self, W, H, fps, rng):
        self.W, self.H, self.fps, self.rng = W, H, fps, rng
        self.w, self.h = W // 2, H // 2
        self.buf = np.zeros((self.h, self.w, 3), np.float32)
        self.caps = np.zeros(NB, np.float32)
        self.hue = rng.random()

    def trails(self, k, F, s, kick):
        w, h, b = self.w, self.h, self.buf
        bass = float(F["bass"][k])
        self.hue = (self.hue + 0.0015 + 0.004 * bass) % 1
        ang = 0.6 * np.sin(k / self.fps * 0.7) + 1.5 * kick
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.025 + 0.03 * bass)
        b = cv2.warpAffine(b, M, (w, h), borderMode=cv2.BORDER_CONSTANT) * (0.90 - 0.04 * (1 - s))
        wave = F["wave"][k]
        xs = np.linspace(0.08 * w, 0.92 * w, len(wave))
        ys = h / 2 + wave * h * 0.22
        col = np.array(hsv(self.hue), np.float32)
        cv2.polylines(b, [np.c_[xs, ys].astype(np.int32)], False, col.tolist(), 2, cv2.LINE_AA)
        r0 = min(w, h) * (0.30 + 0.05 * bass)                   # radial spectrum, mirrored
        bands = F["bands"][k]
        th = np.linspace(0, np.pi, NB)
        for sgn in (1, -1):
            for i in range(0, NB, 2):
                a = np.pi / 2 + sgn * th[i]
                L = r0 + bands[i] * min(w, h) * 0.2
                p0 = (int(w / 2 + r0 * np.cos(a)), int(h / 2 + r0 * np.sin(a)))
                p1 = (int(w / 2 + L * np.cos(a)), int(h / 2 + L * np.sin(a)))
                cv2.line(b, p0, p1, hsv(self.hue + 0.25 + i / NB * 0.5), 2, cv2.LINE_AA)
        self.buf = b
        glow = cv2.GaussianBlur(b, (0, 0), 3) * 0.8 + b
        return cv2.resize(glow, (self.W, self.H), interpolation=cv2.INTER_LINEAR), 0.12 + 0.4 * min(bass, 1.2) ** 2

    def bars(self, k, F, s, kick, layer):
        W, H = self.W, self.H
        bands = F["bands"][k]
        self.caps = np.maximum(self.caps - 0.02, bands)
        base = H * (0.95 if W > H else 0.76)                    # portrait: above the app's caption area
        bw = W * 0.8 / NB
        x0, top = W * 0.1, H * (0.16 if W > H else 0.1)
        for i in range(NB):
            x = int(x0 + i * bw)
            y = int(base - bands[i] * top)
            c = (255, 255, 255) if kick > 0.6 else hsv(self.hue + i / NB * 0.6, 0.85)
            cv2.rectangle(layer, (x, y), (int(x + bw * 0.7), int(base)), c, -1)
            yc = int(base - self.caps[i] * top) - 4
            cv2.rectangle(layer, (x, yc - 3), (int(x + bw * 0.7), yc), (255, 255, 255), -1)


def zoom(f, z):
    H, W = f.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def frame(f, k, vis, F, G, S, opts, beat):
    s = float(S[k])
    L = opts["layers"]
    since, down, drop = G["since"][k], bool(G["down"][k]), int(G["drop"][k])
    kick = float(np.exp(-since / (0.22 * beat)))                 # 1 on the beat, gone within half a beat
    if "kick" in L:
        f = zoom(f, 1 + (0.035 if down else 0.018) * s * kick)
    if "split" in L and down and since < 4 / vis.fps:
        sh = int((4 - since * vis.fps) * 4 * s * vis.W / 1920)
        if sh:
            f = f.copy(); f[..., 0] = np.roll(f[..., 0], sh, 1); f[..., 2] = np.roll(f[..., 2], -sh, 1)
    x = f.astype(np.float32)
    if "trails" in L:
        tr, a = vis.trails(k, F, s, kick)
        a = a * s * (1.15 - x.mean(-1, keepdims=True) / 255)     # mostly into the dark: faces stay clean
        x = 255 - (255 - x) * (255 - tr * a) / 255               # screen
    if "bars" in L:
        layer = np.zeros_like(f)
        vis.bars(k, F, s, kick, layer)
        a = (0.35 + 0.35 * kick) * s
        m = layer.any(-1, keepdims=True)
        x = np.where(m, x * (1 - a) + layer * a, x)
    blink = 0.0
    if "kick" in L:
        blink = (0.14 if down else 0.07) * s * kick
    if "drops" in L and drop >= 0:
        blink = max(blink, (0.85 if drop == 0 else 0.35) * float(np.exp(-since / (0.12 * beat))))
    if blink:
        x = x + (255 - x) * blink
    return np.clip(x, 0, 255).astype(np.uint8)


def apply(src: str, out: str, plan: dict, music: dict, song: str, opts: dict, enc: list, fps: int) -> None:
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", src]).decode().split(",")[:2])
    n = sum(s["frames"] for s in plan["shots"])
    F = features(song, plan["music_start"], n, fps)
    G = grid(music, plan, n, fps)
    S = strength(plan, n, opts)
    vis = Visualiser(W, H, fps, np.random.default_rng(7))
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
        g = frame(f, k, vis, F, G, S, opts, plan["beat"]) if k < n else f
        wr.stdin.write(np.ascontiguousarray(g).tobytes())
        k += 1
        if k % 250 == 0:
            print(f"\r[reel] overlay: {k}/{n}", end="", flush=True)
    print()
    rd.wait(); wr.stdin.close(); wr.wait()
    if wr.returncode:
        raise RuntimeError("overlay encode failed")
