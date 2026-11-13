"""portrait_eval.py [--truth truth.json] [--freeze] [--json out.json] <project folder>[:cut] ...
portrait_eval.py --track <plan.json> <project folder>[:cut]

How close the AUTOMATIC portrait strip comes to the editor's own corrections. Every "portrait" focal point and
"follow" click in a project's picks.json is a label: the editor looked at the strip the planner cut and moved it.
For each labelled pick (the "add" range a label falls in), the strip is derived again with the labels hidden
(plan._reframe, as a pick) and compared with where the editor put it:

    err    mean distance of the strip's centre from the editor's, across the shot (frame widths)
    worst  the largest distance in the shot
    held   share of the shot the editor's point is comfortably inside the strip (its middle 70%)

The editor's centre over a shot: the focal points, interpolated (plan.focus_override), or for a follow click the
clicked person's track (plan.follow_override). --freeze writes those paths to --truth, so a later change to how a
click is followed doesn't move the target; with --truth and no --freeze they are read from it. Read-only on the
projects (the score caches aside). A follow click says who, where they were when clicked: such a shot is judged
within FOLLOW_AT seconds of the click (how the click is then followed is the code under test, not the truth).

--track measures how well a plan's own strips stay on their subjects while they move, whoever they are: per shot,
the person the strip is on (the one whose head it holds most), and how far their head is from the strip's centre
at every pose frame, in strip widths (0 centred, 0.5 at its edge) -- mean, 95th percentile, the share of frames
their head is out of the middle 70%, and the share the frame's own top cuts their head off (no crop fixes that:
another part of the clip does)."""
from __future__ import annotations

import json
import sys

import numpy as np

from focalcut.reel import frame as FR
from focalcut.reel import moments as MO
from focalcut.reel import plan as PL
from focalcut.reel.project import Project

HELD = 0.35                                       # the label within this share of the strip's width of its centre
FOLLOW_AT = 0.3                                   # a click held as a point is the truth this near the click (s)


def hidden(real):
    """A project's picks with the editor's focal points and follow clicks left out."""
    def picks(*a, **k):
        return {i: ({f: v for f, v in e.items() if f not in ("portrait", "follow")} if isinstance(e, dict) else e)
                for i, e in real(*a, **k).items()}
    return picks


def labelled(project):
    """[(unit, t0, length, kind)]: the picks ("add" ranges) of this project that carry a label."""
    picks = project.picks()
    units = {u["id"]: u for u in project.load("units.json")}
    out = []
    for uid, p in picks.items():
        if not isinstance(p, dict) or uid not in units:
            continue
        u = units[uid]
        x0 = u.get("export_t0", 0.0)
        for kind in ("portrait", "follow"):
            keys = [x0 + PL._secs(k[0]) for k in p.get(kind) or []]
            for v in p.get("add") or ():
                n = PL._norm(v)
                if isinstance(n, tuple) and any(x0 + n[0] - PL.FOCUS_REACH <= k <= x0 + n[1] + PL.FOCUS_REACH for k in keys):
                    out.append((u, x0 + n[0], n[1] - n[0], kind))
    return out


def path_x(crop, ts):
    p = np.array(crop["path"])
    return np.interp(ts, p[:, 0], p[:, 1]), np.interp(ts, p[:, 0], p[:, 2])


def run(specs, truth=None, freeze=False):
    rows, truth = [], dict(truth or {})
    for spec in specs:
        folder, _, cut = spec.partition(":")
        P = Project(folder, cut or None)
        PL.portrait_rules(PL.config(P, "portrait"))
        FR.configure(P.config)
        MO.configure(P.config)
        shots = labelled(P)
        real = P.picks
        for u, t0, L, kind in shots:
            key = f"{P.name}/{u['id']}@{t0:.2f}"
            ts = np.arange(0, L + 1e-6, 0.1)
            if freeze or key not in truth:
                c = PL.reframe(P, u, t0, L, "portrait")
                x, y = path_x(c, ts)
                at = [round(k[0] - t0, 2) for k in PL.follow_keys(real(), u)] if kind == "follow" else None
                truth[key] = dict(kind=kind, x=[round(float(v), 4) for v in x], y=[round(float(v), 4) for v in y],
                                  at=at)
            P.picks = hidden(real)
            try:
                c = PL.reframe(P, u, t0, L, "portrait")
            finally:
                P.picks = real
            x, _ = path_x(c, ts)
            tx = np.interp(ts, np.arange(len(truth[key]["x"])) * 0.1, truth[key]["x"])
            d = np.abs(x - tx)
            if truth[key].get("at"):
                near = np.min(np.abs(ts[:, None] - np.clip(truth[key]["at"], 0, L)[None, :]), 1) <= FOLLOW_AT
                d = d[near] if near.any() else d
            rows.append(dict(key=key, kind=kind, t0=round(t0, 2), len=round(L, 2), err=round(float(d.mean()), 3),
                             worst=round(float(d.max()), 3), held=round(float((d <= HELD * c["w"]).mean()), 2),
                             auto=[round(float(x[0]), 2), round(float(x[-1]), 2)],
                             want=[round(float(tx[0]), 2), round(float(tx[-1]), 2)],
                             how=c.get("on") or ("facetrack" if c.get("facetrack") else "body")))
    return rows, truth


def report(rows):
    print(f"{'shot':44s} {'kind':8s} {'err':>6s} {'worst':>6s} {'held':>5s}  auto x      want x      how")
    for r in rows:
        print(f"{r['key']:44s} {r['kind']:8s} {r['err']:6.3f} {r['worst']:6.3f} {r['held']:5.2f}  "
              f"{r['auto'][0]:.2f}>{r['auto'][1]:.2f}   {r['want'][0]:.2f}>{r['want'][1]:.2f}   {r['how']}")
    if rows:
        e = np.array([r["err"] for r in rows])
        h = np.array([r["held"] for r in rows])
        print(f"\n{len(rows)} labelled shots: mean err {e.mean():.3f}, median {np.median(e):.3f}, mean held "
              f"{h.mean():.2f}, shots held >= 0.8: {(h >= 0.8).sum()}/{len(rows)}")


def track(spec, planf):
    """How a plan's strips hold their subjects (see the module's notes): rows per shot, and the totals."""
    from focalcut.reel import lead as LD
    folder, _, cut = spec.partition(":")
    P = Project(folder, cut or None)
    with open(planf) as fh:
        plan = json.load(fh)
    rows, off = [], []
    for i, s in enumerate(plan["shots"]):
        c, u = s.get("crop"), s["unit"]
        if not c or c["w"] >= 0.5:
            continue
        a, L = PL.played(s)
        stem = P.path("scores", u["score"])
        d = PL._cached(stem + ".npz", FR.load, (stem + "_focus.npz", stem + "_subject.npz"))
        if d is None:
            continue
        trs = LD.tracks(dict(d), None, a, a + L)
        p = np.array(c["path"])
        best, share = None, 0.0
        for tr in trs:
            sx = np.interp(tr["t"] - a, p[:, 0], p[:, 1])
            on = float((np.abs(tr["head"][:, 0] - sx) <= 0.4 * c["w"]).sum()) / max(len(tr["t"]), L / 0.2)
            if on > share:
                best, share = tr, on
        if best is None or share < 0.5:
            rows.append(dict(i=i, unit=u["id"], t0=round(a, 2), subject=False))
            continue
        sx = np.interp(best["t"] - a, p[:, 0], p[:, 1])
        o = np.abs(best["head"][:, 0] - sx) / c["w"]
        cutoff = float(((best["box"][:, 1] <= 0.01) & (best["head"][:, 1] - 0.5 * best["hs"] < 0)).mean())
        off += o.tolist()
        rows.append(dict(i=i, unit=u["id"], t0=round(a, 2), subject=True, mean=round(float(o.mean()), 3),
                         worst=round(float(o.max()), 3), cut=round(cutoff, 2), on=c.get("on")))
    off = np.array(off)
    have = [r for r in rows if r["subject"]]
    tot = dict(shots=len(rows), on_someone=len(have), mean=round(float(off.mean()), 3) if len(off) else None,
               p95=round(float(np.percentile(off, 95)), 3) if len(off) else None,
               off_centre=round(float((off > HELD).mean()), 3) if len(off) else None,
               head_cut=[f"#{r['i']} {r['unit']}" for r in have if r["cut"] >= 0.3])
    return rows, tot


def _dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = {k: a[a.index(k) + 1] for k in ("--truth", "--json", "--track") if k in a}
    freeze = "--freeze" in a
    specs = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in opt)]
    if "--track" in opt:
        rows, tot = track(specs[0], opt["--track"])
        for r in rows:
            print(f"#{r['i']:3d} {r['unit'][:28]:28s} {r['t0']:7.2f}  " + (
                f"mean {r['mean']:.3f} worst {r['worst']:.3f} cut {r['cut']:.2f} {r['on'] or ''}" if r["subject"]
                else "on nobody"))
        print(json.dumps(tot))
        if "--json" in opt:
            _dump(dict(rows=rows, total=tot), opt["--json"])
        sys.exit(0)
    truth = None
    if "--truth" in opt and not freeze:
        try:
            with open(opt["--truth"]) as fh:
                truth = json.load(fh)
        except FileNotFoundError:
            truth = None
    rows, truth = run(specs, truth, freeze)
    if "--truth" in opt:
        _dump(truth, opt["--truth"])
    if "--json" in opt:
        _dump(rows, opt["--json"])
    report(rows)
