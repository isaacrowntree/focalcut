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


def window(bars: np.ndarray, start_idx: int, max_len: float) -> int:
    """Index of the bar that ends the window: an 8-bar phrase boundary within max_len if one is close, else the last
    bar that fits."""
    start = bars[start_idx]
    cands = [i for i in range(start_idx + 1, len(bars)) if bars[i] - start <= max_len]
    if not cands:
        return len(bars) - 1
    phr = [i for i in cands if (i - start_idx) % 8 == 0]
    return phr[-1] if phr and bars[phr[-1]] - start > max_len - 8 else cands[-1]


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


def analyse(song: str, start="auto", start_hint=None, max_len: float = 180.0) -> dict:
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
    e_idx = window(bars, s_idx, max_len)
    return dict(song=song, tempo=tempo, beat=float(period), beats=bt.tolist(), bars=bars.tolist(), beat_energy=beat_energy,
                start=float(bars[s_idx]), end=float(bars[e_idx]), length=float(bars[e_idx] - bars[s_idx]),
                bar_phase=phase, beat_accent=accent.tolist())


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


def join(songs: list[dict], max_len: float, preroll: float = 0.5, xfade_tol: float = 0.03) -> tuple[dict, list]:
    """One reel's music from several songs, each used as it is, in order (no time-stretching, no remixing).

    songs: [{"song": path, "bars": n | None, "pre": 8, "acts": {"<act>": bars | None, ...}}, ...]; the last song's
    bars may be None: whatever fits max_len, in whole 2-bar phrases. Each song's window is chosen around its
    strongest drop (song_window). Two songs within xfade_tol of each other's tempo crossfade over one bar centred on
    the cut, beat on beat (the way a DJ mixes); further apart, a clean cut on the downbeat. The result is music.json
    with the beat grid of the joined windows (beats are real times, so the tempo may change at a song boundary) plus
    "sections" (where each song sits) and "acts" (bars of the joined grid), and the mix to render."""
    import librosa
    parts, used = [], 0.0
    for spec in songs:
        m = analyse(spec["song"], start=0.0, max_len=1e9)
        y, sr = librosa.load(spec["song"], sr=22050, mono=True)
        bars = np.array(m["bars"])
        n = spec.get("bars")
        if n is None:                                      # the last song fills the reel
            per_bar = 4 * m["beat"]
            n = int((max_len - used) // per_bar)
            n -= n % 2                                     # whole 2-bar phrases; the stinger covers the end
        s, e = song_window(m, drops(y, sr, bars), n, spec.get("pre", 8))
        used += bars[e] - bars[s]
        parts.append(dict(spec=spec, m=m, s=s, e=e))
    beats, energy, accent, sections, acts, t = [], [], [], [], [], preroll
    for k, p in enumerate(parts):
        m, s, e = p["m"], p["s"], p["e"]
        bt, ph = np.array(m["beats"]), m["bar_phase"]
        b0, b1 = ph + 4 * s, ph + 4 * e                    # beat indices of the window's first and end bars
        off = t - bt[b0]
        beats += list(bt[b0:b1] + off)
        energy += list(np.array(m["beat_energy"] + [m["beat_energy"][-1]])[b0:b1])
        accent += list(np.array(m["beat_accent"])[b0:b1])
        nxt = parts[k + 1]["m"] if k + 1 < len(parts) else None
        mode = ("xfade" if nxt and abs(nxt["tempo"] / m["tempo"] - 1) <= xfade_tol else "cut") if nxt else "end"
        sections.append(dict(song=m["song"], tempo=round(m["tempo"], 2), src_start=float(bt[b0]), src_end=float(bt[b1]),
                             at=float(t), bars=e - s, out=mode, bar=float(4 * m["beat"])))
        bar0 = sum(q["e"] - q["s"] for q in parts[:k])
        spec_acts = p["spec"].get("acts") or {}
        a = bar0
        for name, nbars in spec_acts.items():
            b = bar0 + (e - s) if nbars is None else a + nbars
            acts.append([name, a, b]); a = b
        t += bt[b1] - bt[b0]
    beats = np.array(beats)
    return dict(song=None, tempo=float(60 / np.median(np.diff(beats))), beat=float(np.median(np.diff(beats))),
                beats=beats.tolist(), bars=beats[::4].tolist(), beat_energy=energy, beat_accent=accent,
                start=preroll, end=float(t), length=float(t - preroll), bar_phase=0, sections=sections,
                acts=acts), parts


def mix(sections: list[dict], out: str, preroll: float = 0.5, tail: float = 6.0, sr: int = 48000) -> str:
    """Render the joined songs to one WAV: each song's window at its place, crossfaded (equal power, one bar centred
    on the cut) into the next where the tempos match, cut on the downbeat (with a 10 ms fade) where they don't. The
    first song starts `preroll` early (cuts land ahead of the beat), the last runs `tail` past its window for the
    stinger to fade it under."""
    import librosa
    import soundfile as sf
    total = sections[-1]["at"] + (sections[-1]["src_end"] - sections[-1]["src_start"]) + tail
    buf = np.zeros((int(total * sr) + sr, 2), dtype=np.float32)
    for k, sec in enumerate(sections):
        prev = sections[k - 1] if k else None
        half_in = sec["bar"] / 2 if prev and prev["out"] == "xfade" else 0.0
        half_out = sec["bar"] / 2 if sec["out"] == "xfade" else 0.0
        lead = preroll if k == 0 else half_in
        trail = tail if sec["out"] == "end" else half_out
        a = max(sec["src_start"] - lead, 0.0)
        y, _ = librosa.load(sec["song"], sr=sr, mono=False, offset=a, duration=sec["src_end"] + trail - a)
        y = np.atleast_2d(y)
        y = (np.vstack([y, y]) if y.shape[0] == 1 else y[:2]).T.astype(np.float32)
        n = len(y)
        g = np.ones(n, dtype=np.float32)
        fade = lambda d: max(int(d * sr), int(0.01 * sr))
        fi = fade(2 * half_in) if half_in else (fade(0.01) if k else 0)
        fo = fade(2 * half_out) if half_out else (fade(0.01) if sec["out"] == "cut" else 0)
        if fi:
            g[:fi] = np.sin(np.linspace(0, np.pi / 2, fi)) if half_in else np.linspace(0, 1, fi)
        if fo and sec["out"] != "end":
            g[n - fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) if half_out else np.linspace(1, 0, fo)
        at = int((sec["at"] - (sec["src_start"] - a)) * sr)
        buf[at:at + n] += y * g[:, None]
    peak = float(np.abs(buf).max()) or 1.0
    sf.write(out, buf[: int(total * sr)] / max(peak, 1.0), sr, subtype="PCM_24")
    return out


def run(project, stinger_len: float = 0.0, overlap: float = 1.2) -> dict:
    cfg = project.config
    if cfg.get("songs"):                                   # several songs, one after another (join)
        length = float(cfg.get("length", 180))
        max_music = length - (stinger_len - overlap if stinger_len else 0.0)
        m, _ = join(cfg["songs"], max_music)
        m["song"] = mix(m["sections"], project.path("music_mix.wav"))
        project.dump("music.json", m)
        for sec in m["sections"]:
            print(f"[reel] {os.path.basename(sec['song'])}: {sec['tempo']:.1f} BPM, {sec['bars']} bars from "
                  f"{sec['src_start']:.2f}s at {sec['at']:.2f}s, then {sec['out']}")
        print(f"[reel] music {m['length']:.1f}s, acts {m['acts']}")
        return m
    if not cfg.get("song"):
        raise SystemExit("reel.json has no song (run `focalcut reel init <folder> --song SONG`)")
    length = float(cfg.get("length", 180))
    max_music = length - (stinger_len - overlap if stinger_len else 0.0)   # the stinger overlaps the music's end
    start = cfg.get("start", "auto")
    if isinstance(start, str) and start != "auto":
        start = float(start)
    m = analyse(cfg["song"], start=start, start_hint=cfg.get("start_hint"), max_len=max_music)
    project.dump("music.json", m)
    print(f"[reel] {m['tempo']:.1f} BPM; music {m['start']:.2f}s -> {m['end']:.2f}s ({m['length']:.1f}s)")
    return m
