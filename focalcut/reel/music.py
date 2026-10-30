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


def run(project, stinger_len: float = 0.0, overlap: float = 1.2) -> dict:
    cfg = project.config
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
