"""Events: the moments a shoot is about, named and found the way the shoot needs (reel.json "events").

A recap reel's moments are moves (moments.py: spins, dips, tricks). Other shoots turn on events: a baptism's dunk, a
graduate taking the stage, a bouquet toss, a goal. Each shoot names its own and says how to find them:

    "events": {
      "dunk":  {"detect": ["dip", "cheer"], "in": ["baptisms"]},
      "toss":  {"detect": ["cheer"], "in": ["reception"], "cheer": {"over": 24}}
    }

Every event found becomes a moment of that kind (moments.json: kind = the event's name, t = its peak, start = when
it began, by = the detectors that saw it), which the planner can then place: reference mode (template.py) puts one
event (reel.json "reference_event", default the first) on the reference's long shots of its footage, every one once,
in the order they happened.

Detectors (each with its own settings under its name in the event; the defaults were measured on the 2026 Grace
City Church baptisms):

  dip    someone in a settled group goes out of sight and comes back where they were, while the others stoop with
         them -- a dunk, a kneel, a bow, a lift that drops. From the people found at 5 fps (score step): the count
         of people at least `tall` of the frame high drops below the group's own for `least`..`most` s, the same
         count holding for `settle` s before and after, nobody at the frame's edge, and the boxes of those still seen
         `stoop` shorter with the highest head `drop` of the frame lower. A hug or a passer-by in front of the lens
         takes someone away too, but nobody stoops. The peak is the return (the rise).
  cheer  a crowd breaking into cheering or clapping, in the camera's own sound: the level `over` dB above the
         clip's murmur at its peak, `gap` s apart, from when it crossed half that. Alone it makes an event at
         `alone` dB or more with at least `people` there; beside a dip it confirms it (within `confirm` s after
         the dip's peak).

"in": the export categories the event can happen in (default: all). "detect": the detectors, any of the above.
"""
from __future__ import annotations

import subprocess

import numpy as np

DEFAULTS = {
    "dip": dict(least=0.4, most=3.5, settle=2.0, tall=0.08, stoop=0.18, drop=0.08),
    "cheer": dict(over=20.0, gap=6.0, skip=1.0, alone=28.0, people=2, confirm=[-1.0, 3.0], lead=1.2),
}
EVENTS: dict = {}


def configure(cfg):
    """The shoot's events from reel.json "events" (and the older "immersion": [categories], read as a dunk)."""
    EVENTS.clear()
    ev = dict((cfg or {}).get("events") or {})
    old = (cfg or {}).get("immersion")
    if old and not ev:
        ev["dunk"] = {"detect": ["dip", "cheer"], "in": old if isinstance(old, list) else None}
    for name, spec in ev.items():
        spec = dict(spec or {})
        spec.setdefault("detect", ["dip"])
        for d in spec["detect"]:
            if d not in DEFAULTS:
                raise ValueError(f'reel.json "events" {name!r}: unknown detector {d!r} (choose from {sorted(DEFAULTS)})')
            spec[d] = {**DEFAULTS[d], **(spec.get(d) or {})}
        EVENTS[name] = spec
    return EVENTS


def names():
    return list(EVENTS)


def dip(pt_, boxes, ah, t10=None, subj=None, aw=640.0, least=0.4, most=3.5, settle=2.0, tall=0.08, stoop=0.18,
        drop=0.08):
    """[(start, peak, strength)]: someone in a settled group goes out of sight and comes back while the others stoop
    (see the module notes). strength: how much the picture moved meanwhile (a splash), 0..1."""
    if len(pt_) < 10:
        return []
    from scipy.ndimage import median_filter
    h = boxes[..., 3] - boxes[..., 1]
    ok = np.nan_to_num(h) > tall * ah
    n = median_filter(ok.sum(1).astype(float), size=3, mode="nearest")
    cx = np.where(ok, (boxes[..., 0] + boxes[..., 2]) / 2, np.nan) / aw
    hh = np.where(ok, h, np.nan)
    top = np.where(ok, boxes[..., 1], np.nan)
    dt = float(np.median(np.diff(pt_)))
    k = max(int(round(settle / dt)), 2)
    out, i = [], k
    while i < len(n) - 1:
        before = n[i - k:i]
        base = float(np.median(before))
        if not (base >= 2 and n[i] < base and np.mean(before == base) >= 0.8):
            i += 1
            continue
        j, cap = i, min(len(n), i + int(most / dt) + 2)
        while j < cap and n[j] < base:
            j += 1
        if j >= cap:                                         # too long for the event (or the clip ends): not one,
            i += 1                                           # and the gap may still hide a real one further on
            continue
        dur = pt_[j] - pt_[i]
        after = n[j:j + k]
        if least <= dur <= most and len(after) >= k // 2 and np.mean(after == base) >= 0.8:
            pre = (pt_ >= pt_[i] - 1.5) & (pt_ < pt_[i] - 0.3)
            with np.errstate(all="ignore"):
                bent = np.nanmedian(hh[i:j]) < (1 - stoop) * np.nanmedian(hh[pre])
                fell = np.nanmean(np.nanmin(top[i:j], 1)) - np.nanmean(np.nanmin(top[pre], 1)) > drop * ah
                inside = np.nanmin(cx[pre]) > 0.04 and np.nanmax(cx[pre]) < 0.96
            if bent and fell and inside:
                st = 0.5
                if t10 is not None and subj is not None:
                    m = (t10 >= pt_[i] - 0.5) & (t10 <= pt_[j] + 0.5)
                    if m.any():
                        st = float(np.clip(np.max(subj[m]) / (np.percentile(subj, 90) + 1e-6), 0, 1))
                out.append((round(float(pt_[i]), 2), round(float(pt_[j]), 2), round(st, 2)))
        i = max(j, i + 1)
    return out


def level(path, rate=16000, hop=0.25):
    """A file's sound level, dB every `hop` s; None when it has none."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    n = int(rate * hop)
    if len(a) < n * 8:
        return None
    m = len(a) // n * n
    return 20 * np.log10(np.sqrt(np.add.reduceat(a[:m] ** 2, np.arange(0, m, n)) / n) + 1e-9)


def cheer(db, hop=0.25, over=20.0, gap=6.0, skip=1.0, **_):
    """[(onset s, dB over the murmur)]: the crowd's cheers in a level series (see the module notes)."""
    if db is None:
        return []
    sm = np.convolve(db, np.ones(4) / 4, "same") - float(np.median(db))
    first = int(skip / hop) + 2
    peaks = []
    for i in np.argsort(-sm):
        if sm[i] < over:
            break
        if i >= first and all(abs(i - j) * hop >= gap for j in peaks):
            peaks.append(int(i))
    out = []
    for i in sorted(peaks):
        j = i
        while j > first and sm[j - 1] >= over / 2:
            j -= 1
        out.append((round(j * hop, 2), round(float(sm[i]), 1)))
    return out


_LEVELS: dict = {}


def find(u, S, audio=None):
    """The shoot's events in one unit: [{"name", "start", "t", "strength", "by"}], peaks inside the unit."""
    out = []
    if S is None:
        return out
    for name, spec in EVENTS.items():
        cats = spec.get("in")
        if cats and u["category"] not in cats:
            continue
        found = []
        raw = S.get("raw") or {}
        if "dip" in spec["detect"] and raw:
            p = spec["dip"]
            found = [dict(start=a, t=b, strength=st, by=["dip"]) for a, b, st in
                     dip(S["t"], raw["boxes"], raw["ah"], raw.get("t10"), raw.get("subj"), raw.get("aw", 640.0),
                         **{k: p[k] for k in DEFAULTS["dip"]})]
        if "cheer" in spec["detect"] and audio:
            p = spec["cheer"]
            if audio not in _LEVELS:
                _LEVELS[audio] = level(audio)
            for c, loud in cheer(_LEVELS[audio], **p):
                near = [e for e in found if p["confirm"][0] <= c - e["t"] <= p["confirm"][1]]
                if near:                                     # heard as well as seen: surer
                    near[0]["strength"] = 1.0
                    near[0]["by"].append("cheer")
                elif loud >= p["alone"] and _people(S, raw, c) >= p["people"]:
                    found.append(dict(start=round(c - p["lead"], 2), t=round(c - 0.3, 2), strength=0.6, by=["cheer"]))
        for e in found:
            if u["t0"] + 0.3 <= e["t"] <= u["t1"] - 0.3:
                out.append(dict(e, name=name))
    return sorted(out, key=lambda e: e["t"])


def _people(S, raw, t, half=2.0, tall=0.08):
    """How many people stood in the picture around t (median over `half` s either side)."""
    if not raw:
        return 0
    m = np.abs(S["t"] - t) <= half
    if not m.any():
        return 0
    n = np.sum(np.nan_to_num(raw["boxes"][m][..., 3] - raw["boxes"][m][..., 1]) > tall * raw["ah"], axis=1)
    return float(np.median(n))
