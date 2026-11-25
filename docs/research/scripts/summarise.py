"""Stage C (focalcut venv): merge stage A + B per film and compute the timing statistics.
usage: summarise.py <outdir> <index.json> <references.json>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/focalcut"))
from focalcut.reel import moves  # noqa: E402

NAMES = ["aerial", "wide", "medium", "close"]
TOL = 0.07                 # s: a cut within this of a beat is "on" it (about 2 frames at 25-30 fps)
SOFT, SOFT2 = 0.80, 0.72   # SigLIP similarity of frames 0.5 s / 1.0 s apart inside one detected shot under which
                           # the picture changed without a hard cut (a dissolve the scene score missed)
rng = np.random.default_rng(0)
r = lambda x, n=3: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def pct(x, q):
    return r(np.percentile(x, q)) if len(x) else None


def dist(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), median=pct(x, 50), q25=pct(x, 25), q75=pct(x, 75), p10=pct(x, 10), p90=pct(x, 90),
                mean=r(x.mean()) if len(x) else None, max=r(x.max()) if len(x) else None)


def spearman(a, b):
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return r(np.corrcoef(ra, rb)[0, 1])


LAT = {}


def film(slug, outdir, meta):
    A = json.load(open(os.path.join(outdir, slug + ".a.json")))
    B = np.load(os.path.join(outdir, slug + ".b.npz"))
    ref, mu, mo = A["ref"], A["music"], A["motion"]
    dur, fps = ref["duration"], ref["fps"]
    mend = ref["music_end"] or dur
    bt_, emb, cls = B["t"], B["emb"].astype(np.float32), B["cls"]

    # -- shots: the analyser's hard cuts, split again where the picture changed with no hard cut (dissolves) ------
    shots = []
    for s in ref["shots"]:
        a, b = s["t0"], s["t1"]
        idx = np.where((bt_ > a + 0.15) & (bt_ < b - 0.15))[0]
        splits = []
        if len(idx) >= 3 and "card" not in s and "title" not in s:
            s1 = (emb[idx[:-1]] * emb[idx[1:]]).sum(1)
            s2 = (emb[idx[:-2]] * emb[idx[2:]]).sum(1) if len(idx) > 2 else np.array([])
            low = np.zeros(len(idx) - 1, bool)
            low |= s1 < SOFT
            low[:-1] |= s2 < SOFT2
            low[1:] |= s2 < SOFT2
            k = 0
            while k < len(low):
                if low[k]:
                    j = k
                    while j < len(low) and low[j]:
                        j += 1
                    kk = k + int(np.argmin(s1[k:j]))
                    splits.append(float((bt_[idx[kk]] + bt_[idx[kk + 1]]) / 2))
                    k = j
                else:
                    k += 1
        edges = [a] + [x for x in splits if x - a > 0.6 and b - x > 0.6] + [b]
        for n, (x, y) in enumerate(zip(edges, edges[1:])):
            d = dict(t0=round(x, 3), t1=round(y, 3), length=round(y - x, 3), hard_in=n == 0)
            for key in ("stutter", "split", "card", "title"):
                if key in s:
                    d[key] = True
            shots.append(d)
    rate = mo["rate"]
    mv = np.array(mo["mv"])
    luma = np.array(mo["luma"])
    speed = (np.hypot(mv[:, 0], mv[:, 1]) + np.abs(mv[:, 2]) * 0.5 + np.abs(mv[:, 3])) * rate   # per second
    for s in shots:
        i0, i1 = int(np.ceil(s["t0"] * rate)) + 1, int(np.floor(s["t1"] * rate))    # (skip the frame pair over the cut)
        seg = mv[i0:i1]
        s["move"] = moves.label(seg)[0] if len(seg) >= 3 else None
        idx = np.where((bt_ >= s["t0"]) & (bt_ < s["t1"]))[0]
        if not len(idx):
            idx = [int(np.argmin(np.abs(bt_ - (s["t0"] + s["t1"]) / 2)))]
        s["scale"] = NAMES[int(cls[idx].mean(0).argmax())]
        s["dark"] = bool(luma[i0:i1].mean() < 12) if i1 > i0 else False
        sp = speed[i0:i1]
        if len(sp) >= 20 and s["move"] not in ("held", "wobble", None):
            med = float(np.median(sp)) + 1e-9
            s["in_ratio"], s["out_ratio"] = r(sp[:4].mean() / med, 2), r(sp[-4:].mean() / med, 2)
    body = [s for s in shots if not (s.get("card") or s.get("title") or s["dark"]) and s["t1"] <= mend + 0.5]
    L = np.array([s["length"] for s in body])

    # -- music grid ----------------------------------------------------------------------------------------------
    beats, beat = np.array(mu["beats"]), mu["beat"]
    phase = mu["bar_phase"]
    env_rate = mu["env_rate"]
    rms, onset = np.array(mu["rms"]), np.array(mu["onset"])
    cuts_all = np.array([s["t0"] for s in shots[1:] if s["t0"] < mend])
    hard = np.array([s["t0"] for s in shots[1:] if s["hard_in"] and s["t0"] < mend and not s.get("stutter")])
    # the systematic lead/lag of the cuts against the tracked beat: the shift (s) that puts most cuts on a beat
    # (within +-LOCK), against the same search over cuts jittered at random (null: 95th percentile of its best)
    LOCK = 0.045
    ds = np.round(np.arange(-0.20, 0.2001, 0.01), 2)
    on_share = lambda c: float(np.mean(np.abs(c - beats[np.clip(np.searchsorted(beats, c), 1, len(beats) - 1) - 1]) <= LOCK) if False else
                               np.mean(np.min(np.abs(c[:, None] - beats[None, :]), 1) <= LOCK)) if len(c) else 0.0
    curve = np.array([on_share(hard + d) for d in ds])
    sm = np.convolve(curve, [0.25, 0.5, 0.25], "same")
    k = int(np.argmax(sm))
    nullmax = [max(np.convolve([on_share(hard + rng.uniform(-beat, beat, len(hard)) + d) for d in ds[::2]], [0.25, 0.5, 0.25], "same"))
               for _ in range(60)] if len(hard) > 5 else [1.0]
    locked = bool(len(hard) > 5 and sm[k] > np.percentile(nullmax, 95))
    lead = float(ds[k]) if locked else 0.0                      # > 0: the cuts come this long BEFORE the tracked beat
    lock = dict(lock_tolerance_s=LOCK, cuts_lead_beat_s=r(float(ds[k]), 2), cuts_lead_beat_frames=r(float(ds[k]) * fps, 1),
                on_beat_at_lead=r(curve[k]), on_beat_at_zero=r(curve[len(ds) // 2]), chance=r(min(2 * LOCK / beat, 1.0)),
                null95_of_best=r(np.percentile(nullmax, 95)), locked=locked,
                curve={str(d): r(c, 2) for d, c in zip(ds[::2].tolist(), curve[::2].tolist())})
    lat = LAT.get(slug, {}).get("attack_start_minus_beat_s")
    lock["tracker_latency_s"] = r(-lat, 3) if lat is not None else None     # the tracked beat sits this long after the attack
    lock["cuts_lead_attack_s"] = r(float(ds[k]) + lat, 3) if lat is not None else None
    lock["cuts_lead_attack_frames"] = r((float(ds[k]) + lat) * fps, 1) if lat is not None else None
    hard_raw, hard = hard, hard + lead                          # from here the cuts are measured at their own lead
    nb = np.array([int(np.argmin(np.abs(beats - c))) for c in hard]) if len(hard) and len(beats) else np.array([], int)
    off = hard - beats[nb] if len(nb) else np.array([])
    on = np.abs(off) <= TOL
    bar_pos = (nb - phase) % 4                                  # 0 = downbeat under the music-derived phase
    # 4-bar phrase phase: where the bar-to-bar change of loudness is greatest (music only, not the cuts)
    bars = beats[phase::4]
    bar_rms = np.array([rms[int(a * env_rate):max(int(b * env_rate), int(a * env_rate) + 1)].mean() for a, b in zip(bars[:-1], bars[1:])])
    nov = np.abs(np.diff(np.log(bar_rms + 1e-6)))               # novelty at bar k+1
    pphase = int(np.argmax([nov[(p - 1) % 4::4].mean() if len(nov[(p - 1) % 4::4]) else 0 for p in range(4)])) if len(nov) > 8 else 0
    bar_i = (nb - phase) // 4
    share = lambda m: r(float(m.mean())) if len(m) else None
    sync = dict(
        tolerance_s=TOL, hard_cuts=int(len(hard)), beat_lock=lock,
        on_beat=share(on), chance_beat=r(min(2 * TOL / beat, 1.0)),
        on_downbeat=share(on & (bar_pos == 0)), chance_downbeat=r(2 * TOL / (4 * beat)),
        downbeat_by_phase=[share(on & (bar_pos == p)) for p in range(4)],
        on_beat3=share(on & (bar_pos == 2)),
        on_phrase4=share(on & (bar_pos == 0) & (bar_i % 4 == pphase)), chance_phrase4=r(2 * TOL / (16 * beat)),
        phrase4_by_phase=[share(on & (bar_pos == 0) & (bar_i % 4 == p)) for p in range(4)],
        phrase8_best=max([share(on & (bar_pos == 0) & (bar_i % 8 == p)) or 0 for p in range(8)]) if len(on) else None,
        chance_phrase8=r(2 * TOL / (32 * beat)),
        on_half_beat=share(np.abs(np.abs(off) - beat / 2) <= TOL) if len(off) else None,
        off_beat=share(~on),
    )
    fr = (off - lead) * fps                                              # frames after (+) / before (-) the nearest beat
    near = np.abs(fr) <= 3.5
    sync["offset_frames_hist"] = {str(k): int((np.round(fr) == k).sum()) for k in range(-6, 7)}
    sync["offset_frames_median_near"] = r(np.median(fr[near]), 2) if near.any() else None
    sync["early_1to3"], sync["on_0"], sync["late_1to3"] = (share((np.round(fr) <= -1) & near), share(np.round(fr) == 0),
                                                           share((np.round(fr) >= 1) & near))
    # onsets (no grid): cuts within TOL of a strong onset peak, against the same cuts moved at random
    pk, ps = np.array(mu["onset_peaks"]), np.array(mu["onset_peak_strength"])
    strong = pk[ps >= np.median(ps)] if len(pk) else pk
    hit = lambda c: float((np.abs(strong[np.clip(np.searchsorted(strong, c), 1, len(strong) - 1) - 1] - c) <= TOL).mean() if False else
                          np.mean([np.min(np.abs(strong - x)) <= TOL for x in c])) if len(c) and len(strong) else None
    null = [hit((hard_raw + rng.uniform(0.3, 3.0)) % mend) for _ in range(200)] if len(hard) and len(strong) else [None]
    sync["on_strong_onset"], sync["chance_strong_onset"] = r(hit(hard_raw)), r(np.mean(null)) if null[0] is not None else None
    # shot length against the grid
    Lb = L / beat
    whole_bar = np.abs(Lb / 4 - np.round(Lb / 4)) <= 0.25 / 4
    grid = dict(length_beats=dist(Lb), length_bars=dist(Lb / 4),
                whole_bars_share=share(whole_bar & (np.round(Lb / 4) >= 1)), chance_whole_bars=0.125,
                shots_4_bars_or_more=share(Lb >= 16), shots_2_bars_or_more=share(Lb >= 8), shots_under_1_bar=share(Lb < 4),
                shots_under_2_beats=share(Lb < 2))

    # -- pace curve, 10 s windows over the music ------------------------------------------------------------------
    W = 10.0
    nwin = int(mend // W)
    cw = np.array([((cuts_all >= k * W) & (cuts_all < (k + 1) * W)).sum() for k in range(nwin)])
    ew = np.array([rms[int(k * W * env_rate):int((k + 1) * W * env_rate)].mean() for k in range(nwin)])
    ow = np.array([onset[int(k * W * env_rate):int((k + 1) * W * env_rate)].mean() for k in range(nwin)])
    erank = np.argsort(np.argsort(ew)) / max(nwin - 1, 1)
    pace = dict(window_s=W, cuts=cw.tolist(), mean_shot_s=[r(W / c, 2) if c else None for c in cw],
                rms_rel=[r(x, 2) for x in ew / (np.median(ew) + 1e-9)], onset_rel=[r(x, 2) for x in ow / (np.median(ow) + 1e-9)],
                corr_cuts_rms=spearman(cw, ew), corr_cuts_onset=spearman(cw, ow),
                fastest_at=r((int(cw.argmax()) + 0.5) * W / mend, 2) if nwin else None,
                fastest_energy_rank=r(erank[int(cw.argmax())], 2) if nwin else None,
                slowest_at=r((int(cw.argmin()) + 0.5) * W / mend, 2) if nwin else None,
                slowest_energy_rank=r(erank[int(cw.argmin())], 2) if nwin else None,
                loudest_at=r((int(ew.argmax()) + 0.5) * W / mend, 2) if nwin else None)
    fifths = []
    for k in range(5):
        a, b = mend * k / 5, mend * (k + 1) / 5
        x = [s["length"] for s in body if a <= (s["t0"] + s["t1"]) / 2 < b]
        fifths.append(r(np.median(x), 2) if x else None)
    pace["median_shot_by_fifth"] = fifths

    # -- scale and move -----------------------------------------------------------------------------------------
    by_scale = {n: dist([s["length"] for s in body if s["scale"] == n]) for n in NAMES}
    tot = float(L.sum()) or 1.0
    scale_time = {n: r(sum(s["length"] for s in body if s["scale"] == n) / tot) for n in NAMES}
    seq = [s["scale"] for s in body]
    trans = {a: {b: 0 for b in NAMES} for a in NAMES}
    for a, b in zip(seq, seq[1:]):
        trans[a][b] += 1
    runs, k = [], 0
    while k < len(seq):
        j = k
        while j < len(seq) and seq[j] == seq[k]:
            j += 1
        runs.append((seq[k], j - k))
        k = j
    aer = [s for s in body if s["scale"] == "aerial"]
    kinds = list(moves.KINDS)
    mv_all = {k_: dist([s["length"] for s in body if s["move"] == k_]) for k_ in kinds if any(s["move"] == k_ for s in body)}
    mv_aer = {k_: dist([s["length"] for s in aer if s["move"] == k_]) for k_ in kinds if any(s["move"] == k_ for s in aer)}
    heldish = lambda s: s["move"] in ("held", "wobble", "drift")
    moving = [s for s in body if "in_ratio" in s]
    edge = dict(n=len(moving),
                in_already_moving=share(np.array([s["in_ratio"] >= 0.6 for s in moving])),
                in_from_rest=share(np.array([s["in_ratio"] < 0.3 for s in moving])),
                out_still_moving=share(np.array([s["out_ratio"] >= 0.6 for s in moving])),
                out_at_rest=share(np.array([s["out_ratio"] < 0.3 for s in moving])),
                in_ratio_median=pct([s["in_ratio"] for s in moving], 50), out_ratio_median=pct([s["out_ratio"] for s in moving], 50))
    scale = dict(
        length_by_scale=by_scale, screen_time=scale_time, shots={n: seq.count(n) for n in NAMES},
        cut_changes_scale=share(np.array([a != b for a, b in zip(seq, seq[1:])])),
        transitions=trans, run_length_mean={n: r(np.mean([c for m_, c in runs if m_ == n]), 2) if any(m_ == n for m_, _ in runs) else None for n in NAMES},
        aerial_positions=[r((s["t0"] + s["t1"]) / 2 / mend, 2) for s in aer],
        first_shot=seq[0] if seq else None, last_shot=seq[-1] if seq else None,
        sequence="".join(x[0].upper() for x in seq),
    )
    return dict(
        slug=slug, **meta, duration=dur, fps=fps, music_end=r(mend, 2), size=ref["size"],
        cuts=dict(hard=int(sum(s["hard_in"] for s in shots[1:])), soft_dissolves=int(sum(not s["hard_in"] for s in shots)),
                  stutter_shots=int(sum(bool(s.get("stutter")) for s in shots)), split_screens=int(sum(bool(s.get("split")) for s in shots)),
                  cards_titles=int(sum(bool(s.get("card") or s.get("title")) for s in shots))),
        shot_length_s=dist(L), cuts_per_minute=r(len(body) / (tot / 60), 1),
        music=dict(tempo=r(mu["tempo"], 1), beat_s=r(beat, 4), grid_confidence=r(mu["grid_conf"], 2), bar_phase=phase, phrase4_phase=pphase),
        grid=grid, sync=sync, pace=pace, scale=scale,
        moves=dict(all=mv_all, aerial=mv_aer, share_held_like=share(np.array([heldish(s) for s in body])), cut_edges=edge,
                   aerial_held_like=dist([s["length"] for s in aer if heldish(s)]),
                   aerial_moving=dist([s["length"] for s in aer if not heldish(s)]),
                   ground_held_like=dist([s["length"] for s in body if s["scale"] != "aerial" and heldish(s)]),
                   ground_moving=dist([s["length"] for s in body if s["scale"] != "aerial" and not heldish(s)])),
        shots=shots,
    )


outdir, index, dst = sys.argv[1:4]
LAT = json.load(open(os.path.join(outdir, 'latency.json'))) if os.path.exists(os.path.join(outdir, 'latency.json')) else {}
idx = json.load(open(index))
films = []
for e in idx:
    if os.path.exists(os.path.join(outdir, e["slug"] + ".a.json")) and os.path.exists(os.path.join(outdir, e["slug"] + ".b.npz")):
        films.append(film(e["slug"], outdir, {k: e[k] for k in ("title", "channel", "url", "kind", "views") if k in e}))
        f = films[-1]
        print(f"{f['slug'][:34]:34s} {f['kind']:6s} n={f['shot_length_s']['n']:3d} med={f['shot_length_s']['median']:5.2f}s "
              f"{f['grid']['length_beats']['median']:5.1f}b tempo={f['music']['tempo']:5.1f} conf={f['music']['grid_confidence']:.2f} "
              f"lead={f['sync']['beat_lock']['cuts_lead_beat_frames']}f lock={f['sync']['beat_lock']['on_beat_at_lead']}/{f['sync']['beat_lock']['null95_of_best']}{'*' if f['sync']['beat_lock']['locked'] else ' '} beat={f['sync']['on_beat']}/{f['sync']['chance_beat']} down={f['sync']['on_downbeat']}/{f['sync']['chance_downbeat']} "
              f"soft={f['cuts']['soft_dissolves']} corr={f['pace']['corr_cuts_rms']}")
json.dump(dict(
    method=dict(
        cuts="focalcut.reel.reference.analyse (ffmpeg scene score > 0.25); shots split again where SigLIP 2 frames 0.5 s apart inside one "
             f"shot fall under {SOFT} cosine similarity (1.0 s apart: {SOFT2}) -- dissolves, timed to about +-0.5 s and left out of the beat statistics",
        music="focalcut.reel.music.analyse on the film's own audio (librosa beat_track, straight-line grid, downbeat phase = the bar phase "
              "with the most low-end onset energy); grid_confidence = onset strength on the grid's beats over half-way between them",
        sync=f"beat_lock: the constant lead of the cuts over the tracked beats that puts most cuts within +-0.045 s of a beat, tested against "
             "randomly jittered cuts; when locked, every later figure is measured with the cuts moved by that lead. The tracker's own latency "
             "(librosa places a beat on the onset envelope's peak, some 20-50 ms after the attack) is inside that lead. "
             f"A hard cut is 'on' a beat within +-{TOL} s; chance_* is the share a random cut would score; phrase phase from bar-loudness novelty",
        scale="SigLIP 2 zero-shot at 2 fps (reference.SCALES prompts + generic ones), averaged over the shot; not hand-checked",
        moves="focalcut.reel.moves.frames_moves on 10 fps grey frames of the whole film, label() per shot; in/out_ratio = camera speed in the "
              "first/last 0.4 s of a moving shot of 2 s or more, over the shot's median speed",
        body="statistics are over 'body' shots: no title, card or black, and before the music ends",
    ), films=films), open(dst, "w"), indent=1)
print("wrote", dst, len(films))
