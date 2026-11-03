"""Beat grid for the reel's track: tempo, beats, downbeats (bar starts), and the window the reel uses.

Salsa and similar music usually tracks at half-time, so one detected "bar" (4 beats) is one 8-count -- the unit
dancers phrase in, which makes it the natural grid for cuts.

The window starts on a bar downbeat:
    start = seconds          the bar nearest that time
    start_hint = seconds     the bar nearest that time (same thing; kept for configs that set both)
    start = "auto"           the first bar after the intro: where the track's energy, averaged over two bars,
                             first reaches 70% of its typical (median) level -- i.e. the band coming in
and ends on the last bar boundary that keeps the whole reel (music + stinger) within the target length, on an
8-bar phrase where possible.
"""
from __future__ import annotations

import os

import numpy as np


def window(bars: np.ndarray, start_idx: int, max_len: float, phrase: bool = True) -> int:
    """Index of the bar that ends the window: an 8-bar phrase boundary within max_len if one is close (phrase), else
    the last bar that fits."""
    start = bars[start_idx]
    cands = [i for i in range(start_idx + 1, len(bars)) if bars[i] - start <= max_len]
    if not cands:
        return len(bars) - 1
    phr = [i for i in cands if (i - start_idx) % 8 == 0]
    return phr[-1] if phrase and phr and bars[phr[-1]] - start > max_len - 8 else cands[-1]


def auto_start(bars: np.ndarray, rms: np.ndarray, rms_t: np.ndarray) -> int:
    """First bar where the two-bar average energy reaches 70% of the track's median energy."""
    med = float(np.median(rms))
    for i in range(len(bars) - 2):
        seg = rms[(rms_t >= bars[i]) & (rms_t < bars[i + 2])]
        if len(seg) and seg.mean() >= 0.7 * med:
            return i
    return 0


def fit_beats(bt: np.ndarray, max_jitter: float = 0.03) -> tuple[np.ndarray, float]:
    """Steady tempo: a straight line through the beats gives the exact period and phase. The tracker's own intervals
    are quantised to its ~23 ms hop -- enough to drift about a second off the beat over three minutes. If the tempo
    really drifts (residuals above max_jitter), the tracked beats are kept as they are."""
    k = np.arange(len(bt))
    period, phase0 = np.polyfit(k, bt, 1)
    if np.std(bt - (phase0 + period * k)) < max_jitter:
        return phase0 + period * k, float(period)
    return bt, float(np.median(np.diff(bt)))


def accents(onset: np.ndarray, onset_t: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """How hard each beat hits: the peak onset strength right at the beat against the median of the 8 beats either
    side -- horn stabs and breaks, where a flash on the cut feels earned."""
    hit = np.array([onset[(onset_t >= b - 0.04) & (onset_t <= b + 0.04)].max(initial=0) for b in beats])
    local = np.array([np.median(hit[max(0, i - 8):i + 9]) for i in range(len(hit))])
    return (hit / (local + 1e-6)).round(3)


def analyse(song: str, start="auto", start_hint=None, max_len: float = 180.0, phrase: bool = True) -> dict:
    import librosa
    y, sr = librosa.load(song, sr=22050, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, units="frames", tightness=120)
    bt = librosa.frames_to_time(beats, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    # downbeat phase: of the 4 possible bar phases, the one whose beats carry the most low-end onset energy
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, fmax=200)
    low = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=sr)
    strength = low[np.clip(beats, 0, len(low) - 1)]
    phase = int(np.argmax([strength[p::4].mean() for p in range(4)]))
    bt, period = fit_beats(bt)
    tempo = 60 / period
    bars = bt[phase::4]
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    beat_energy = [float(rms[(rms_t >= a) & (rms_t < b)].mean()) for a, b in zip(bt[:-1], bt[1:])]
    accent = accents(onset, librosa.frames_to_time(np.arange(len(onset)), sr=sr), bt)
    hint = start if isinstance(start, (int, float)) else start_hint
    s_idx = int(np.argmin(np.abs(bars - float(hint)))) if hint is not None else auto_start(bars, rms, rms_t)
    e_idx = window(bars, s_idx, max_len, phrase)
    return dict(song=song, tempo=tempo, beat=float(period), beats=bt.tolist(), bars=bars.tolist(), beat_energy=beat_energy,
                start=float(bars[s_idx]), end=float(bars[e_idx]), length=float(bars[e_idx] - bars[s_idx]),
                bar_phase=phase, beat_accent=accent.tolist())


def to_end(m: dict, max_len: float) -> dict:
    """Play the song out to its last sample (reel.json "to_end": true), past the last bar the beat tracker found (an
    outro or fade), within max_len: the beat grid carries on at the song's period, so cuts keep landing on it."""
    import soundfile as sf
    dur = sf.info(m["song"]).duration
    end = min(dur, m["start"] + max_len)
    bt, bars, p = list(m["beats"]), list(m["bars"]), m["beat"]
    while bt[-1] + p <= end + 1e-6:
        bt.append(bt[-1] + p)
    k = bt.index(bars[-1]) if bars[-1] in bt else None
    if k is not None:
        bars += bt[k + 4::4]
    energy = list(m["beat_energy"]) + [m["beat_energy"][-1]] * (len(bt) - 1 - len(m["beat_energy"]))
    accent = list(m.get("beat_accent", [])) + [0.0] * (len(bt) - len(m.get("beat_accent", [])))
    return dict(m, beats=bt, bars=bars, beat_energy=energy, beat_accent=accent, end=float(end),
                length=float(end - m["start"]))


def drops(y: np.ndarray, sr: int, bars: np.ndarray) -> np.ndarray:
    """How hard each bar drops: the low end (below 150 Hz) over the next 4 bars against the 4 before. A house or EDM
    drop is the kick and bass coming back after a build that took them out, so this peaks on the drop's first bar."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
    low = S[librosa.fft_frequencies(sr=sr, n_fft=4096) < 150].sum(0)
    t = librosa.frames_to_time(np.arange(len(low)), sr=sr, hop_length=1024)
    per = np.array([low[(t >= a) & (t < b)].mean() if ((t >= a) & (t < b)).any() else 0.0
                    for a, b in zip(bars[:-1], bars[1:])])
    med = float(np.median(per)) + 1e-9
    out = np.zeros(len(bars))
    for i in range(4, len(per) - 3):
        after, before = per[i:i + 4].mean(), per[i - 4:i].mean()
        out[i] = (after / (before + 0.1 * med)) * min(after / med, 1.5)   # a real drop is loud as well as a jump
    return out


def song_window(m: dict, drop: np.ndarray, n: int | None, pre: int = 8, tail: int = 1) -> tuple[int, int]:
    """(first, end) bar of an n-bar window: `pre` bars of build, then the song's strongest drop (bars counted in the
    song's own grid; n None: from there to the last bar that leaves `tail` bars for a crossfade)."""
    nb = len(m["bars"])
    last = nb - 1 - tail
    d = int(np.argmax(drop[: last + 1]))
    pre = min(pre, d, (n or nb) // 2)
    s = d - pre
    if n is not None and s + n > last:                     # a drop near the end: keep the window inside the song
        s = max(last - n, 0)
    return s, (s + n if n is not None else last)


MIX = dict(bars=4, ramp=8, max_ramp=0.15, split=150.0)   # reel.json "mix" overrides; "mix": false cuts between songs


def ramp(native: np.ndarray, target: float, n: int) -> np.ndarray:
    """Beat lengths for the last n beats before a mix: from the song's own beat to the next song's (target), a
    little each beat, the way a DJ rides the pitch fader into the next record."""
    f = (np.arange(n) + 0.5) / n
    return native * (1 - f) + target * f


def join(songs: list[dict], max_len: float, preroll: float = 0.5, mix: dict | bool | None = None) -> tuple[dict, list]:
    """One reel's music from several songs, in order, mixed into each other the way a DJ would.

    songs: [{"song": path, "bars": n | None, "pre": 8, "from": seconds, "acts": {"<act>": bars | None, ...}}, ...];
    the last song's bars may be None: whatever fits max_len, in whole 2-bar phrases. Each song's window is chosen
    around its strongest drop (song_window), or starts on the bar nearest "from" (e.g. where a Latin or pop song is
    already driving: a drop finder looks for a build, and the build is the soft part). Where the next song's tempo is within mix["max_ramp"] of this one's, the outgoing
    song speeds up or slows down over its last mix["ramp"] bars to the next song's tempo (time-stretched, pitch
    kept), then the two play beat on beat for mix["bars"] bars centred on the handover; mix() crossfades them there
    and swaps the bass on the handover's downbeat. Further apart (or mix false), a clean cut on the downbeat.
    The result is music.json with the beat grid of the joined windows (real output times, so the tempo changes
    where it should) plus "sections" (where each song sits, with its source->output beat map) and "acts" (bars of
    the joined grid), and the mix to render."""
    import librosa
    cfg = dict(MIX, **(mix or {})) if mix is not False else dict(MIX, bars=0)
    h = int(cfg["bars"]) // 2                              # overlap bars either side of the handover
    ms = []
    for spec in songs:
        m = analyse(spec["song"], start=0.0, max_len=1e9)
        y, sr = librosa.load(spec["song"], sr=22050, mono=True)
        ms.append((m, drops(y, sr, np.array(m["bars"]))))
    modes = []
    for k in range(len(ms) - 1):
        ta, tb = ms[k][0]["tempo"], ms[k + 1][0]["tempo"]
        modes.append("mix" if h and abs(tb / ta - 1) <= cfg["max_ramp"] else "cut")
    modes.append("end")
    parts, used = [], 0.0
    for k, (spec, (m, drop)) in enumerate(zip(songs, ms)):
        bars = np.array(m["bars"])
        n = spec.get("bars")
        if n is None:                                      # the last song fills the reel
            n = int((max_len - used) // (4 * m["beat"]))
            n -= n % 2                                     # whole 2-bar phrases; the stinger covers the end
        tail = max(h, 1) if modes[k] == "mix" else 1
        if spec.get("from") is not None:                   # the editor's choice: from the bar nearest this time
            s = int(np.argmin(np.abs(bars - float(spec["from"]))))
            s = min(s, max(len(bars) - 1 - tail - n, 0))
            e = s + n
        else:
            s, e = song_window(m, drop, n, spec.get("pre", 8), tail=tail)
        parts.append(dict(spec=spec, m=m, s=s, e=e, out=modes[k]))
        used += bars[e] - bars[s]                          # (a ramp changes this a little; the last song absorbs it)
    for k, p in enumerate(parts):                          # overlap per handover: as much as both songs have room for
        p["h_out"] = 0
        if p["out"] == "mix":
            q = parts[k + 1]
            p["h_out"] = min(h, q["s"], (p["e"] - p["s"]) // 2)
            if not p["h_out"]:
                p["out"] = "cut"
    beats, energy, accent, sections, acts, t = [], [], [], [], [], preroll
    for k, p in enumerate(parts):
        m, s, e = p["m"], p["s"], p["e"]
        bt, ph = np.array(m["beats"]), m["bar_phase"]
        b0, b1 = ph + 4 * s, ph + 4 * e                    # beat indices of the window's first and end bars
        h_in = parts[k - 1]["h_out"] if k else 0
        lo, hi = b0 - 4 * h_in, b1 + 4 * p["h_out"]        # the beats this song is heard for
        dur = np.diff(bt[lo:hi + 1])                       # output length of each beat: its own, unless mixing out
        if p["out"] == "mix":
            q = parts[k + 1]
            qb, qph = np.array(q["m"]["beats"]), q["m"]["bar_phase"]
            c = qph + 4 * q["s"]
            ov = 8 * p["h_out"]                            # beats heard together: on the next song's own beats
            dur[-ov:] = np.diff(qb[c - ov // 2:c + ov // 2 + 1])
            nr = min(4 * int(cfg["ramp"]), len(dur) - ov - 4 * h_in)
            if nr > 0:
                dur[-ov - nr:-ov] = ramp(dur[-ov - nr:-ov], q["m"]["beat"], nr)
        out_t = t - dur[:4 * h_in].sum() + np.concatenate([[0.0], np.cumsum(dur)])
        bmap = np.column_stack([bt[lo:hi + 1], out_t])
        win = slice(4 * h_in, 4 * h_in + (b1 - b0) + 1)
        wt = out_t[win]
        beats += list(wt[:-1])
        energy += list(np.array(m["beat_energy"] + [m["beat_energy"][-1]])[b0:b1])
        accent += list(np.array(m["beat_accent"])[b0:b1])
        if k == 0:                                         # cuts land ahead of the beat: the first song starts early
            bmap = np.vstack([[bt[lo] - preroll, out_t[0] - preroll], bmap])
        if p["out"] == "end":
            bmap = np.vstack([bmap, [bt[hi] + 6.0, out_t[-1] + 6.0]])   # runs on under the stinger
        sections.append(dict(song=m["song"], tempo=round(m["tempo"], 2), src_start=float(bt[b0]), src_end=float(bt[b1]),
                             at=float(t), bars=e - s, out=p["out"], bar=float(4 * m["beat"]),
                             map=bmap.round(5).tolist(),
                             fade_in=[float(out_t[0]), float(t), float(out_t[8 * h_in])] if h_in else None,
                             fade_out=[float(out_t[-1 - 8 * p["h_out"]]), float(wt[-1]), float(out_t[-1])]
                             if p["h_out"] else None))
        bar0 = sum(q["e"] - q["s"] for q in parts[:k])
        a = bar0
        for name, nbars in (p["spec"].get("acts") or {}).items():
            b = bar0 + (e - s) if nbars is None else a + nbars
            if acts and acts[-1][0] == name and acts[-1][2] == a:
                acts[-1][2] = b                            # the same act carries on into this song (two sets, one act)
            else:
                acts.append([name, a, b])
            a = b
        t = float(wt[-1])
    beats = np.array(beats)
    return dict(song=None, tempo=float(60 / np.median(np.diff(beats))), beat=float(np.median(np.diff(beats))),
                beats=beats.tolist(), bars=beats[::4].tolist(), beat_energy=energy, beat_accent=accent,
                start=preroll, end=float(t), length=float(t - preroll), bar_phase=0, sections=sections,
                acts=acts), parts


def stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch (pitch kept) with ffmpeg's Rubber Band filter, tuned for drums: rate > 1 plays faster."""
    import subprocess
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-", "-af",
                        f"rubberband=tempo={rate:.6f}:transients=crisp:detector=percussive:window=short",
                        "-f", "f32le", "-"], input=np.ascontiguousarray(y, np.float32).tobytes(),
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2)


def warp(y: np.ndarray, sr: int, bmap: np.ndarray, x: float = 0.01) -> np.ndarray:
    """Audio y (starting at source time bmap[0][0]) laid out on the output times of bmap (source beat -> output
    time pairs): runs of beats at one rate are stretched together, with x-second crossfades where the rate changes."""
    src, dst = bmap[:, 0] - bmap[0, 0], bmap[:, 1] - bmap[0, 1]
    rate = np.diff(src) / np.diff(dst)
    out = np.zeros((int(round(dst[-1] * sr)) + 1, 2), np.float32)
    i = 0
    while i < len(rate):
        j = i + 1
        while j < len(rate) and abs(rate[j] - rate[i]) < 1e-4:
            j += 1
        r, edge_a, edge_b = rate[i], i > 0, j < len(rate)
        pad = 0.5 if abs(r - 1) > 1e-4 else 0.0
        a0 = max(src[i] - (x if edge_a else 0) * r - pad, 0.0)
        a1 = src[j] + (x if edge_b else 0) * r + pad
        seg = y[int(a0 * sr):int(a1 * sr)]
        if pad:
            seg = stretch(seg, sr, r)
        k0 = int(round((src[i] - a0) / r * sr - (x * sr if edge_a else 0)))
        o0 = int(round(dst[i] * sr - (x * sr if edge_a else 0)))
        o1 = min(int(round(dst[j] * sr + (x * sr if edge_b else 0))), len(out))
        seg = seg[k0:k0 + o1 - o0]
        g = np.ones(len(seg), np.float32)
        nx = int(2 * x * sr)
        if edge_a:
            g[:nx] = np.linspace(0, 1, nx)[:len(g)]
        if edge_b and len(g) >= nx:
            g[-nx:] *= np.linspace(1, 0, nx)
        out[o0:o0 + len(seg)] += seg * g[:, None]
        i = j
    return out


def mix(sections: list[dict], out: str, sr: int = 48000, split: float = MIX["split"]) -> str:
    """Render the joined songs to one WAV from join()'s sections: each song laid on its beat map (time-stretched
    where it ramps into the next), level-matched, and handed over the DJ way where join() overlaps them -- highs
    crossfaded (equal power) across the overlap, the bass (below `split` Hz) swapped in the half beat before the
    handover's downbeat so two kicks never stack. Where they don't overlap, a cut on the downbeat (10 ms fades)."""
    import librosa
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, split, "lowpass", fs=sr, output="sos")
    total = sections[-1]["map"][-1][1]
    buf = np.zeros((int(total * sr) + sr, 2), dtype=np.float32)
    levels, pieces = [], []
    for sec in sections:
        bmap = np.array(sec["map"])
        a = max(bmap[0, 0], 0.0)
        y, _ = librosa.load(sec["song"], sr=sr, mono=False, offset=a, duration=bmap[-1, 0] - a + 0.6)
        y = np.atleast_2d(y)
        y = (np.vstack([y, y]) if y.shape[0] == 1 else y[:2]).T.astype(np.float32)
        w0, w1 = int((sec["src_start"] - a) * sr), int((sec["src_end"] - a) * sr)
        levels.append(float(np.sqrt((y[w0:w1] ** 2).mean())) + 1e-9)
        if bmap[0, 0] < 0:                                 # starts before the file does: silence first
            y = np.vstack([np.zeros((int(-bmap[0, 0] * sr), 2), np.float32), y])
        pieces.append(warp(y, sr, bmap))
    ref = float(np.exp(np.mean(np.log(levels))))
    for k, (sec, y) in enumerate(zip(sections, pieces)):
        y = y * float(np.clip(ref / levels[k], 0.5, 2.0))
        t0 = sec["map"][0][1]
        t = t0 + np.arange(len(y)) / sr
        hi_g, lo_g = np.ones(len(y), np.float32), np.ones(len(y), np.float32)
        beat = sec["bar"] / 4
        if sec.get("fade_in"):
            f0, sw, f1 = sec["fade_in"]                    # overlap start, handover downbeat, overlap end
            hi_g *= np.sin(np.pi / 2 * np.clip((t - f0) / (f1 - f0), 0, 1))
            lo_g *= np.sin(np.pi / 2 * np.clip((t - sw + beat / 2) / (beat / 2), 0, 1))
        elif k:
            hi_g *= np.clip((t - t0) / 0.01, 0, 1); lo_g *= np.clip((t - t0) / 0.01, 0, 1)
        if sec.get("fade_out"):
            f0, sw, f1 = sec["fade_out"]                   # overlap start, handover downbeat, overlap end
            hi_g *= np.cos(np.pi / 2 * np.clip((t - f0) / (f1 - f0), 0, 1))
            lo_g *= np.cos(np.pi / 2 * np.clip((t - sw + beat / 2) / (beat / 2), 0, 1))
        elif sec["out"] == "cut":
            te = t0 + len(y) / sr
            hi_g *= np.clip((te - t) / 0.01, 0, 1); lo_g *= np.clip((te - t) / 0.01, 0, 1)
        if sec.get("fade_in") or sec.get("fade_out"):
            low = sosfiltfilt(sos, y, axis=0).astype(np.float32)
            y = (y - low) * hi_g[:, None] + low * lo_g[:, None]
        else:
            y = y * hi_g[:, None]
        at = int(round(t0 * sr))
        buf[at:at + len(y)] += y[: len(buf) - at]
    peak = float(np.abs(buf).max()) or 1.0
    sf.write(out, buf[: int(total * sr)] / max(peak, 1.0), sr, subtype="PCM_24")
    return out


def run(project, stinger_len: float = 0.0, overlap: float = 1.2) -> dict:
    cfg = project.config
    if cfg.get("songs"):                                   # several songs, one after another (join)
        length = float(cfg.get("length", 180))
        max_music = length - (stinger_len - overlap if stinger_len else 0.0)
        m, _ = join(cfg["songs"], max_music, mix=cfg.get("mix"))
        m["song"] = mix(m["sections"], project.path(project.music_file.replace(".json", "_mix.wav")),
                        split=float((cfg.get("mix") or {}).get("split", MIX["split"])))
        project.dump(project.music_file, m)
        for sec in m["sections"]:
            print(f"[reel] {os.path.basename(sec['song'])}: {sec['tempo']:.1f} BPM, {sec['bars']} bars from "
                  f"{sec['src_start']:.2f}s at {sec['at']:.2f}s, then {sec['out']}"
                  + (f" ({(sec['map'][-1][1] - sec['map'][0][1]):.1f}s heard)" if sec.get("map") else ""))
        print(f"[reel] music {m['length']:.1f}s, acts {m['acts']}")
        return m
    if not cfg.get("song"):
        raise SystemExit("reel.json has no song (run `focalcut reel init <folder> --song SONG`)")
    length = float(cfg.get("length", 180))
    max_music = length - (stinger_len - overlap if stinger_len else 0.0)   # the stinger overlaps the music's end
    start = cfg.get("start", "auto")
    if isinstance(start, str) and start != "auto":
        start = float(start)
    m = analyse(cfg["song"], start=start, start_hint=cfg.get("start_hint"), max_len=max_music,
                phrase=cfg.get("phrase_end", True))   # false: the last whole bar that fits (a reel that must
                                                        # run its full length), not the last phrase
    if cfg.get("to_end"):
        m = to_end(m, max_music)
    project.dump(project.music_file, m)
    print(f"[reel] {m['tempo']:.1f} BPM; music {m['start']:.2f}s -> {m['end']:.2f}s ({m['length']:.1f}s)")
    return m
