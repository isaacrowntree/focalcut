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

What the music is doing, beat by beat (describe(); the planner and the light show read it through feel()):
    beat_kick      the low end (below 150 Hz: the kick and the bass, sustained or struck) against the song's loud level
    beat_drive     the rhythm section's drive, 0..1: mostly beat_kick, a little of the percussion, steadied over 5 beats
    beat_section   intro / groove / break / build / drop / outro: runs of driving and calm beats, named by where they
                   sit (a break is a calm run between driving ones; a drop the driving run a break or build leads to)
    beat_sustain   1 where nothing new sounds on the beat: a note, chord or voice held over from the beat before, or
                   its tail ringing out
    sustains       [[beat index of the hit, beats held after it], ...]: the music lands and holds
beat_accent says how hard a beat's onset is against its neighbours, whatever makes it (a kick, a horn stab, a sung
syllable, a sweep); beat_energy is loudness, and stays high through a vocal breakdown. Neither says whether the
rhythm section is playing, so a hit (feel()["hit"]) is an accent on a beat the music drives: a flash, a blink or a
glitch belongs on those and nowhere else. A music.json written before these fields existed reads as driving
throughout with no sustains: everything behaves as it did.
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


STEADY = 0.7              # steady_grid: this share of the tracked beats on one grid makes the song's tempo steady


def steady_grid(bt: np.ndarray, max_jitter: float = 0.03, share: float = STEADY):
    """(grid, period) when the tracked beats sit on one steady grid but for stretches where the tracker lost it (an
    intro with no drums, a breakdown: it wanders, adds or drops a beat, and a straight line through the beat numbers
    no longer fits), else None. The period and phase are the ones most of the beats agree on, whatever their
    numbers; the grid then runs through the whole song, the lost stretches included -- a produced track's tempo
    does not change because the drums stopped."""
    bt = np.asarray(bt, float)
    if len(bt) < 16:
        return None
    p0 = float(np.median(np.diff(bt)))
    span = float(bt[-1] - bt[0])

    def best(ps):
        z = np.exp(2j * np.pi * (bt[None, :] - bt[0]) / ps[:, None]).mean(1)
        k = int(np.argmax(np.abs(z)))
        return float(ps[k]), float(np.angle(z[k]) / (2 * np.pi) * ps[k])
    step = p0 * p0 / (8 * span)                              # an eighth of a beat's slip over the song, per step
    p, ph = best(np.arange(p0 * 0.96, p0 * 1.04, step))
    p, ph = best(np.linspace(p - step, p + step, 41))
    k = np.round((bt - bt[0] - ph) / p)
    r = bt - bt[0] - ph - k * p
    ok = np.abs(r) < max_jitter
    if ok.mean() < share:
        return None
    p, c = np.polyfit(k[ok], bt[ok], 1)                      # exact, from the beats on the grid
    k0, k1 = np.ceil(-c / p), np.floor((bt[-1] + p / 2 - c) / p)          # from the song's first moment on
    return c + p * np.arange(k0, k1 + 1), float(p)


def smooth_beats(bt: np.ndarray, half: int = 8, max_jitter: float = 0.03) -> np.ndarray:
    """Tracked beats with the tracker's ~23 ms steps taken out: each beat moves onto the straight line through its
    neighbours (half either side) where they are steady; through a tempo change or a slip they stay as tracked."""
    bt = np.asarray(bt, float)
    out = bt.copy()
    for i in range(len(bt)):
        a, b = max(0, i - half), min(len(bt), i + half + 1)
        if b - a < 5:
            continue
        k = np.arange(a, b)
        p, c = np.polyfit(k, bt[a:b], 1)
        if np.std(bt[a:b] - (c + p * k)) < max_jitter / 2:
            out[i] = c + p * i
    return out


def fit_beats(bt: np.ndarray, max_jitter: float = 0.03) -> tuple[np.ndarray, float]:
    """Steady tempo: a straight line through the beats gives the exact period and phase. The tracker's own intervals
    are quantised to its ~23 ms hop -- enough to drift about a second off the beat over three minutes. Where the
    line does not fit because the tracker lost the beat for a stretch, the grid most of the beats agree on is used
    (steady_grid: it may hold a beat more or fewer than were tracked). If the tempo really drifts (a live band, a
    DJ mix), the tracked beats are kept, smoothed, with their median interval -- a nominal period only: the planner
    cuts on the beats themselves (plan.clock)."""
    k = np.arange(len(bt))
    period, phase0 = np.polyfit(k, bt, 1)
    if np.std(bt - (phase0 + period * k)) < max_jitter:
        return phase0 + period * k, float(period)
    grid = steady_grid(bt, max_jitter)
    if grid is not None:
        return grid
    bt = smooth_beats(bt, max_jitter=max_jitter)
    return bt, float(np.median(np.diff(bt)))


def accents(onset: np.ndarray, onset_t: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """How hard each beat hits: the peak onset strength right at the beat against the median of the 8 beats either
    side -- horn stabs and breaks, where a flash on the cut feels earned."""
    hit = np.array([onset[(onset_t >= b - 0.04) & (onset_t <= b + 0.04)].max(initial=0) for b in beats])
    local = np.array([np.median(hit[max(0, i - 8):i + 9]) for i in range(len(hit))])
    return (hit / (local + 1e-6)).round(3)


KICK_HZ = 150.0           # the low end: kick and bass
DRIVE = (0.35, 0.5)       # sections: a calm run starts when the drive falls under the first, a driving one over the second
RUN = 4                   # ... and a run shorter than this many beats belongs to the one around it
HELD = 0.5                # a beat whose percussive onsets are under this share of the song's typical beat is held
SUSTAIN_MAX = 8           # a sustain counts this many held beats at most (past it the music has simply stopped)
BUILD = 0.7               # a calm run's last beats with percussion this strong (of the song's loud level): a build
HIT_DRIVE = 0.4           # feel: an accent counts as a hit where the drive is at least this
BLINK = (0.3, 0.6)        # feel: the per-beat blink fades in between these drives
CALM = ("intro", "break", "build")       # sections with no rhythm section under them


def _runs(flags):
    """[(first, end, value), ...] of a sequence's runs of equal values."""
    out, a = [], 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[a]:
            out.append((a, i, flags[a]))
            a = i
    return out


def sections(drive, perc=None) -> list:
    """A label per beat from the drive (0..1 per beat): intro / groove / break / build / drop / outro.

    Beats are driving or calm (DRIVE, with hysteresis; runs under RUN beats join their surroundings). A calm run is
    the intro before the first driving one, the outro after the last, a break between two; its last beats are a
    build when the percussion is strong there (perc, per beat: a snare roll with no kick under it). A driving run is
    a drop when a break or build of two bars or more leads to it, or an intro does and it is as strong as the song
    gets; otherwise a groove. Inside a drop, two bars or more well under its level are a groove again -- an outro
    when they run to the end."""
    d = np.asarray(drive, float)
    n = len(d)
    if not n:
        return []
    on, state = np.zeros(n, bool), d[0] >= DRIVE[1]
    for i in range(n):
        state = d[i] >= (DRIVE[0] if state else DRIVE[1])
        on[i] = state
    for _ in range(4):                                       # short runs join what is around them
        runs = _runs(on)
        short = [(a, b) for a, b, _ in runs if b - a < RUN and len(runs) > 1]
        if not short:
            break
        a, b = min(short, key=lambda r: r[1] - r[0])
        on[a:b] = not on[a]
    runs = _runs(on)
    level = [float(np.median(d[a:b])) if v else 0.0 for a, b, v in runs]
    top = max(level) if level else 0.0
    out = [""] * n
    for k, (a, b, v) in enumerate(runs):
        if not v:
            name = "intro" if k == 0 else "outro" if k == len(runs) - 1 else "break"
            out[a:b] = [name] * (b - a)
            if perc is not None and k < len(runs) - 1:       # a build: strong percussion right up to the entry
                j = b
                while j > a and b - j < 16 and perc[j - 1] >= BUILD:
                    j -= 1
                if b - j >= RUN:
                    out[j:b] = ["build"] * (b - j)
            continue
        led = k > 0 and runs[k - 1][1] - runs[k - 1][0] >= 8
        drop = led and (k > 1 or level[k] >= 0.9 * top) and level[k] >= 0.75 * top
        out[a:b] = ["drop" if drop else "groove"] * (b - a)
        if drop:                                             # the drop easing off: a groove again, or the outro
            ref = float(np.percentile(d[a:b], 75))
            for x, y, low in _runs(list(d[a:b] < 0.8 * ref)):
                if low and y - x >= 8 and x > 0:
                    out[a + x:a + y] = ["outro" if a + y == n else "groove"] * (y - x)
            for x, y, name in _runs(out[a:b]):               # (a beat or two back at its level between them: not
                if name == "drop" and y - x < 8 and x > 0:   #  a drop of its own)
                    out[a + x:a + y] = [out[a + x - 1]] * (y - x)
    return out


def sustains(held) -> list:
    """[[hit, n], ...]: each run of held beats (beat_sustain) with the beat that sounded before it -- the music lands
    on beat `hit` and holds for the n beats after (SUSTAIN_MAX at most)."""
    return [[a - 1, min(b - a, SUSTAIN_MAX)] for a, b, v in _runs([bool(x) for x in held]) if v and a > 0]


def features(y: np.ndarray, sr: int, bt: np.ndarray, lo: float, hi: float) -> dict:
    """The per-beat description of audio y (beat times bt in y's own seconds; the reel plays lo..hi, and levels are
    measured against that stretch): beat_kick, beat_drive, beat_section, beat_sustain, sustains (see the module's
    notes). Beats outside y read as the nearest one inside it, nothing held."""
    import librosa
    bt = np.asarray(bt, float)
    n = len(bt)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    _, P = librosa.decompose.hpss(S, margin=2.0)
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low, low_p, perc = S[f < KICK_HZ].sum(0), P[f < KICK_HZ].sum(0), P.sum(0)
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(P), sr=sr)
    period = float(np.median(np.diff(bt))) if n > 1 else 0.5
    ends = np.append(bt[1:], bt[-1] + period) if n else bt
    ia, ib = np.searchsorted(t, bt - 0.02), np.searchsorted(t, ends - 0.07)   # frames of each beat, clear of the next
    ja, jb = np.searchsorted(t, bt - 0.04), np.searchsorted(t, ends - 0.04)   # (its onset sits just ahead of it)
    have = (bt >= 0) & (ends <= len(y) / sr) & (ib > ia)
    win = have & (bt >= lo - 1e-3) & (bt < hi)
    if not win.any():
        win = have
    if not win.any():
        return dict(beat_kick=[1.0] * n, beat_drive=[1.0] * n, beat_section=["groove"] * n, beat_sustain=[0] * n,
                    beat_texture=[1.0] * n, sustains=[])
    per = lambda x, a, b: np.array([x[i:j].mean() if h and j > i else 0.0 for i, j, h in zip(a, b, have)])
    norm = lambda v: v / (np.percentile(v[win], 90) + 1e-9)
    kick = np.clip(np.maximum(norm(per(low, ia, ib)), norm(per(low_p, ia, ib))), 0, 1.5)
    pc = np.clip(norm(per(perc, ia, ib)), 0, 1.5)
    mix = np.clip(0.8 * kick + 0.2 * pc, 0, 1)
    drive = np.array([np.median(mix[max(i - 2, 0):i + 3]) for i in range(n)])
    act = per(onset, ja, jb)
    held = have & (act < HELD * (np.median(act[win]) + 1e-9))
    tex = np.clip(0.6 * pc + 0.4 * kick, 0, 1)               # texture: how much of the beat is struck -- drums,
                                                             # stabs, plucks against the song's loud level, and
                                                             # the kick under them (pads and held voices: soft)
    tex = np.array([np.median(tex[max(i - 1, 0):i + 2]) for i in range(n)])
    wi = np.where(win)[0]
    a, b = wi[0], wi[-1] + 1
    sec = ["groove"] * n
    sec[a:b] = sections(drive[a:b], pc[a:b])
    hv = np.where(have)[0]
    for i in np.where(~have)[0]:                             # past the audio: as the nearest beat that was heard
        j = hv[np.argmin(np.abs(hv - i))]
        kick[i], drive[i], sec[i], tex[i] = kick[j], drive[j], sec[j], tex[j]
    return dict(beat_kick=kick.astype(float).round(3).tolist(), beat_drive=drive.astype(float).round(3).tolist(),
                beat_section=sec, beat_texture=tex.astype(float).round(3).tolist(),
                beat_sustain=held.astype(int).tolist(), sustains=sustains(held))


def describe(m: dict, pad: float = 8.0) -> dict:
    """music.json m with what the music is doing on each beat (features), measured over the stretch the reel plays
    (and pad seconds either side: an hour-long mix is not read whole)."""
    import librosa
    a = max(m["start"] - pad, 0.0)
    y, sr = librosa.load(m["song"], sr=22050, mono=True, offset=a, duration=m["end"] + pad - a)
    return dict(m, **features(y, sr, np.array(m["beats"]) - a, m["start"] - a, m["end"] - a))


STOP = dict(fall=0.6, deep=0.45, floor=0.3, driving=0.7, kick=0.35, back=1.8, max=4)   # movement(): see there
ENTRY = dict(rise=1.5, level=0.7, kick=0.5)
MOVES = ("stop", "close", "rise", "burst", "pulse", "glide")


def movement(level, kick=None, calm=None, held=None, section=None) -> dict:
    """The song's movement, beat by beat -- what the picture's own movement is matched to (plan.follow_motion):

      stop    everything pauses or closes down: the level falls away suddenly from a driving passage (to STOP["fall"]
              of the four beats before it, or the kick drops out from under them), for at most STOP["max"] beats or
              until the music sounds again. The picture comes to rest with it: a settle lands here
      close   the same fall inside a calm passage, or a beat the music holds (a sustain): a phrase resolving, the
              breath before a drop. A settle lands here too, a little less insistently
      burst   the music comes in: the level jumps (ENTRY) onto a loud or kicked beat after a stop, a close or a calm
              passage -- a drop, an entry. A movement takes off here
      rise    the beats leading into a burst that build towards it
      pulse   a driving groove
      glide   calm: an intro, a breakdown

    level: per-beat loudness, 1 about the song's loud level (feel() norms beat_energy); kick, calm, held, section:
    as feel() has them (optional: level alone finds the stops). Returns dict(move=[label per beat],
    stops=[(beat, end, weight)]: the picture at rest from `beat` until `end` (weight 1 a stop, 0.6 a close, 0.5 a
    sustain), bursts=[beat, ...]). Only these structural moments are matched: motion rhythm and music rhythm are
    not one to one (M2Beats, IJCAI 2024), and a cut synced to every beat reads as a gimmick."""
    e = np.asarray(level, float)
    n = len(e)
    k = np.asarray(kick, float) if kick is not None else None
    calm = np.asarray(calm, bool) if calm is not None else e < 0.5
    held = np.asarray(held, bool) if held is not None else np.zeros(n, bool)
    move = ["glide" if c else "pulse" for c in calm]
    stops, bursts = [], []
    i = 4
    while i < n:
        pe = float(np.median(e[i - 4:i]))
        pk = float(np.median(k[i - 4:i])) if k is not None else 0.0
        fell = pe >= STOP["floor"] and e[i] <= STOP["fall"] * pe
        cut = k is not None and pk >= 0.5 and k[i] <= STOP["kick"] * pk and e[i] <= 0.8 * pe
        if not (fell or cut):
            i += 1
            continue
        j = i + 1
        while j < n and j - i < STOP["max"] and e[j] <= max(STOP["fall"] * pe, e[i] * 1.2) and e[j] < STOP["back"] * e[j - 1]:
            j += 1
        if j - i == 1 and e[i] > STOP["deep"] * pe:          # one soft beat in a groove (a salsa bar's quiet 4) is
            i += 1                                           # the groove itself, not a stop
            continue
        hard = pe >= STOP["driving"] or pk >= 0.5
        if stops and stops[-1][1] == i:                      # it goes on falling: the same stop
            a0, _, w0 = stops.pop()
            i, hard = a0, hard or w0 == 1.0
        stops.append((i, j, 1.0 if hard else 0.6))
        move[i:j] = ["stop" if hard else "close"] * (j - i)
        i = j
    taken = {b for a, b2, _ in stops for b in range(a, b2)}
    for a, b, v in _runs([bool(x) for x in held]):           # a sustain: the music lands on the beat before and holds
        if v and a > 0 and not any(x in taken for x in range(a - 1, b + 1)):
            stops.append((a, min(b, a + SUSTAIN_MAX), 0.5))
            move[a:b] = ["close"] * (b - a)
    stops.sort()
    for i in range(2, n):
        before = float(np.median(e[max(i - 2, 0):i]))
        loud = e[i] >= ENTRY["level"] or (k is not None and k[i] >= ENTRY["kick"])
        quiet = move[i - 1] in ("stop", "close", "glide", "rise")   # (a groove's own loud beat is no entry)
        if loud and quiet and e[i] >= ENTRY["rise"] * before and (not bursts or i - bursts[-1] > 2):
            bursts.append(i)
            move[i] = "burst"
            for j in range(i - 1, max(i - 5, -1), -1):       # the beats that build into it
                if move[j] != "glide" or (section is not None and section[j] != "build" and e[j] <= e[max(j - 1, 0)]):
                    break
                move[j] = "rise"
    if section is not None:
        move = ["rise" if s == "build" and m == "glide" else m for s, m in zip(section, move)]
    return dict(move=move, stops=stops, bursts=bursts)


def feel(music: dict, whole: bool = False) -> dict:
    """What the music is doing on each beat, as arrays from the reel's first beat (whole: from the grid's first):

      accent   how hard the beat's onset is against its neighbours (beat_accent; 1 without)
      drive    the rhythm section's drive, 0..1 (1 without: an old music.json drives throughout)
      section  intro / groove / break / build / drop / outro
      calm     no rhythm section under the beat (an intro, a break, a build, or the drive under HIT_DRIVE)
      hit      the accent where the music really hits -- 0 on a calm beat: what a flash, a dip, a glitch or a band
               insert goes by
      blink    0..1: how much of the per-beat blink and punch the beat carries (none through a vocal break)
      held     nothing new sounds on the beat (a sustain)
      sustain  [(hit, end), ...] in beats: the music lands on beat `hit` and holds until beat `end` sounds
      entries  beats where the rhythm section comes in after two bars or more without it (a drop, the first groove)
      level    loudness, 1 about the reel's loud level (beat_energy)
      texture  0 soft and flowing (pads, held voices) .. 1 hard and struck (drums, stabs): beat_texture, else from
               the drive and the kick
      move     the song's movement on the beat: stop / close / rise / burst / pulse / glide (movement())
      stops    [(beat, end, weight), ...]: the music pauses or closes down from `beat` until `end`
      bursts   beats where the music comes in
      holds    [(hit, end), ...]: the sustains and the stops together, as sustain has them -- a shot holds from the
               last beat that sounded (`hit`) until the music sounds again (`end`): no cut between, so the movement
               is seen landing and at rest, and the cut comes with the music
      known    the analysis had these fields (False: the fallbacks above)

    No audio library needed: the planner and the render read music.json with this."""
    bt = np.array(music.get("beats") or [], float)
    n = len(bt)
    i0 = 0 if whole or not n else int(np.argmin(np.abs(bt - music["start"])))

    def arr(key, fill, kind=float):
        v = list(music.get(key) or [])
        v = (v + [v[-1] if v else fill] * n)[:n] if v else [fill] * n
        return np.array(v, kind)[i0:]
    accent, drive, kick = arr("beat_accent", 1.0), arr("beat_drive", 1.0), arr("beat_kick", 1.0)
    section = list(arr("beat_section", "groove", object))
    held = arr("beat_sustain", 0).astype(bool)
    calm = np.array([s in CALM for s in section], bool) | (drive < HIT_DRIVE)
    x = np.clip((drive - BLINK[0]) / (BLINK[1] - BLINK[0]), 0, 1)
    blink = np.where(calm, 0.0, x * x * (3 - 2 * x))
    driving = [not c for c in calm]
    entries = [a for k, (a, b, v) in enumerate(_runs(driving)) if v and k and a - _runs(driving)[k - 1][0] >= 8] \
        if n > i0 else []
    known = bool(music.get("beat_drive"))
    en = arr("beat_energy", 1.0)
    j1 = len(en) if whole or not n else max(int(np.searchsorted(bt, music.get("end", bt[-1]) + 1e-6)) - i0, 1)
    level = en / (np.percentile(en[:j1], 90) + 1e-9) if len(en) else en
    texture = arr("beat_texture", 1.0) if music.get("beat_texture") else np.clip(0.6 * drive + 0.4 * kick, 0, 1)
    mv = movement(level, kick if known else None, calm if known else None, held, section if known else None) \
        if music.get("beat_energy") else dict(move=["pulse"] * len(level), stops=[], bursts=[])
    sus = [(a, a + k + 1) for a, k in sustains(held)]
    holds = sorted(set(sus) | {(a - 1, e) for a, e, _ in mv["stops"] if a >= 1})
    holds = [h for h in holds if not any(o != h and o[0] <= h[0] and h[1] <= o[1] for o in holds)]
    return dict(accent=accent, drive=drive, kick=kick, section=section, calm=calm, held=held, holds=holds,
                level=level, texture=texture, move=mv["move"], stops=mv["stops"], bursts=mv["bursts"],
                hit=np.where(calm, 0.0, accent), blink=blink,
                sustain=sus, entries=entries,
                known=known)


def analyse(song: str, start="auto", start_hint=None, max_len: float = 180.0, phrase: bool = True,
            feel: bool = True) -> dict:
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
    tracked = bt
    bt, period = fit_beats(bt)
    tempo = 60 / period
    if len(bt) != len(tracked) or np.max(np.abs(bt - tracked)) > 0.035:   # a grid of its own (steady_grid): its phase
        beats = np.clip(librosa.time_to_frames(np.clip(bt, 0, None), sr=sr), 0, len(low) - 1)
        strength = np.array([low[max(f - 1, 0):f + 2].max() for f in beats])
        phase = int(np.argmax([strength[p::4].mean() for p in range(4)]))
    hint0 = start if isinstance(start, (int, float)) else start_hint
    if hint0 is not None and float(hint0) < bt[0] - period / 2:   # asked to start before the first beat found (a
        k = int((bt[0] - max(float(hint0), 0.0)) / period + 1e-6)   # quiet intro with no drums): the grid carries
        bt = np.concatenate([bt[0] - period * np.arange(k, 0, -1), bt])   # back to it at the song's period
        phase = (phase + k) % 4
    bars = bt[phase::4]
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    beat_energy = [float(rms[(rms_t >= a) & (rms_t < b)].mean()) for a, b in zip(bt[:-1], bt[1:])]
    accent = accents(onset, librosa.frames_to_time(np.arange(len(onset)), sr=sr), bt)
    hint = start if isinstance(start, (int, float)) else start_hint
    s_idx = int(np.argmin(np.abs(bars - float(hint)))) if hint is not None else auto_start(bars, rms, rms_t)
    e_idx = window(bars, s_idx, max_len, phrase)
    m = dict(song=song, tempo=tempo, beat=float(period), beats=bt.tolist(), bars=bars.tolist(), beat_energy=beat_energy,
             start=float(bars[s_idx]), end=float(bars[e_idx]), length=float(bars[e_idx] - bars[s_idx]),
             bar_phase=phase, beat_accent=accent.tolist())
    if feel:                                               # what the music does on each beat, over the reel's stretch
        a = max(m["start"] - 8.0, 0.0)
        i, j = int(a * sr), int((m["end"] + 8.0) * sr)
        m.update(features(y[i:j], sr, bt - a, m["start"] - a, m["end"] - a))
    return m


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
    m = dict(m, beats=bt, bars=bars, beat_energy=energy, beat_accent=accent, end=float(end),
             length=float(end - m["start"]))
    return describe(m) if m.get("beat_drive") else m      # (the stretch the reel plays changed: measured over it)


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
    extra = {k: [] for k in ("beat_kick", "beat_drive", "beat_section", "beat_sustain", "beat_texture")}
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
        for key in extra:                                  # each song's own description, where it plays
            extra[key] += list(m[key][b0:b1])
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
                acts=acts, **extra, sustains=sustains(extra["beat_sustain"])), parts


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
    if cfg.get("arrange"):                                 # the song re-arranged to the film's shape first (arrange)
        cfg = dict(cfg, song=arranged(project, cfg["song"], cfg["arrange"]))
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


# -- re-arranging a song to a film -----------------------------------------------------------------------------------
# A stock track rarely has the film's shape: too short, its quiet break in the wrong place. arrange() finds a path
# through the song's own bars -- playing on, or jumping at a bar line to another bar that sounds the same (a repeat or
# a cut) -- that runs a given number of bars, starts on its first and ends on its last, and puts its quiet bars where
# the film wants them (reel.json "arrange": {"bars": n, "quiet": [first_bar, last_bar]} in the film's bars) and
# nowhere else. Dynamic programming over (bar played, film bar): each jump costs, a short loop costs more, a quiet bar
# out of place costs a lot, a loud bar inside the quiet window some.

JUMP_SIM = 0.975         # two bars this alike (chroma + timbre, over the bar and its neighbours) can be cut between
QUIET_DB = -4.5          # a bar this far under the song's loud level is a quiet one


def bar_profile(song: str, sr: int = 22050) -> dict:
    """Per bar of the song (its own grid, carried back to 0 over a drumless intro): start time, level (dB under the
    loudest bars) and a feature for comparing bars."""
    import librosa
    m = analyse(song, start=0.0, max_len=1e9, feel=False)
    y, _ = librosa.load(song, sr=sr, mono=True)
    bars = np.asarray(m["bars"], float)
    bars = bars[bars < len(y) / sr - 0.5]
    beats = np.asarray(m["beats"], float)
    fr = librosa.time_to_frames(beats, sr=sr)
    ch = librosa.util.sync(librosa.feature.chroma_cqt(y=y, sr=sr), fr, aggregate=np.median)
    mf = librosa.util.sync(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), fr)[1:]
    f = np.vstack([ch / (np.linalg.norm(ch, axis=0) + 1e-9), mf / (np.linalg.norm(mf, axis=0) + 1e-9)]) / np.sqrt(2)
    bi = [int(np.argmin(np.abs(beats - b))) for b in bars]
    feat = [f[:, min(i, f.shape[1] - 4):min(i, f.shape[1] - 4) + 4].ravel() for i in bi]
    rms = np.array([np.sqrt(np.mean(y[int(a * sr):int(b * sr)] ** 2) + 1e-12)
                    for a, b in zip(bars, list(bars[1:]) + [len(y) / sr], strict=True)])
    db = 20 * np.log10(rms / np.percentile(rms, 90))
    return dict(song=song, bars=bars.tolist(), db=db.tolist(), feat=np.array(feat), beat=m["beat"],
                end=len(y) / sr)


def jumps(feat: np.ndarray, sim: float = JUMP_SIM, ctx: int = 2, gap: int = 4) -> dict:
    """{bar i: [bar j, ...]}: after bar i-1 the song may go on with bar j instead of bar i -- the two alike over ctx
    bars either side (so the cut is unheard)."""
    n = len(feat)
    fn = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    S = fn @ fn.T
    out = {}
    for i in range(ctx, n - ctx):
        for j in range(ctx, n - ctx):
            if abs(i - j) >= gap and np.mean([S[i + k, j + k] for k in range(-ctx, ctx + 1)]) >= sim:
                out.setdefault(i, []).append(j)
    return out


def arrange(db, J, n_bars, quiet=None, intro=0, outro=8, jump_cost=1.0, back_cost=2.0, loop_cost=3.0,
            repeat_cost=4.0, min_run=8, run_cost=6.0, stray_cost=25.0, quiet_db=QUIET_DB, beam=4000):
    """The path (a list of song bars, one per film bar) -- see the note above; None when no path runs n_bars (a
    song's repeats come in whole phrases: try n_bars +- 8). A beam search over whole paths, so that it can count how
    often each bar has played: a bar's third play costs repeat_cost (and each after it twice that), a jump back
    back_cost (more for a loop under 16 bars), any jump jump_cost (plus run_cost out of a stretch under min_run
    bars); a quiet bar out of the quiet window stray_cost, a loud one inside it 1.5. intro: the song's opening bars, never jumped into; outro: its last bars, played through."""
    n = len(db)
    loud = np.asarray(db) > quiet_db
    q0, q1 = quiet if quiet else (-1, -1)

    def place(b, t):
        if b < intro:
            return 0.0
        if q0 <= t <= q1:
            return 0.0 if not loud[b] else 1.5
        return 0.0 if loud[b] else stray_cost

    def succ(b):
        out = [(b + 1, 0.0)] if b + 1 < n else []
        for j in J.get(b + 1, []):
            if intro <= j < n - outro:
                back = b + 1 > j
                out.append((j, jump_cost + (back_cost + (loop_cost if b + 1 - j < 16 else 0.0) if back else 0)))
        return out
    nexts = [succ(b) for b in range(n)]
    reach = np.zeros((n_bars, n), bool)                # reach[k, b]: from bar b, the song's last bar in exactly k
    reach[0, n - 1] = True                             # more bars
    for k in range(1, n_bars):
        for b in range(n):
            reach[k, b] = any(reach[k - 1, j] for j, _ in nexts[b])
    if not reach[n_bars - 1, 0]:
        return None
    states = [(place(0, 0), [0], np.bincount([0], minlength=n))]
    for t in range(1, n_bars):
        left = n_bars - t
        grown = {}
        for c0, path, cnt in states:
            run = 1                                    # bars played since the last jump
            while run < len(path) and path[-run - 1] == path[-run] - 1:
                run += 1
            for j, jc in nexts[path[-1]]:
                if not reach[left - 1, j]:             # (could not end on the last bar in time from there)
                    continue
                if j != path[-1] + 1 and run < min_run:   # (a stretch this short between cuts sounds chopped)
                    jc += run_cost
                k = cnt[j]
                c = c0 + jc + place(j, t) + (repeat_cost * (2 ** (k - 2)) if k >= 2 else 0.0)
                key = (j, cnt.tobytes())
                if key not in grown or c < grown[key][0]:
                    c2 = cnt.copy()
                    c2[j] += 1
                    grown[key] = (c, path + [j], c2)
        states = sorted(grown.values(), key=lambda x: x[0])[:beam]
    done = [s for s in states if s[1][-1] == n - 1]
    return min(done, key=lambda x: x[0])[1] if done else None


def render_path(song: str, bars: list, path: list, out: str, sr: int = 48000, x: float = 0.04) -> list:
    """The song played along path (song bars), each cut made at the bar line with a short equal-power crossfade and
    the joining bar nudged (up to 30 ms) to where its waveform best meets the bar it follows. Returns the cuts
    [(film time, from song time, to song time)]."""
    import librosa
    import soundfile as sf
    y, _ = librosa.load(song, sr=sr, mono=False)
    y = y if y.ndim == 2 else np.stack([y, y])
    mono = y.mean(0)
    ends = list(bars[1:]) + [y.shape[1] / sr]
    runs, k = [], 0                                   # contiguous stretches of song bars
    while k < len(path):
        j = k
        while j + 1 < len(path) and path[j + 1] == path[j] + 1:
            j += 1
        runs.append([bars[path[k]], ends[path[j]]])
        k = j + 1
    n, w = int(0.5 * sr), int(0.03 * sr)
    for a, b in zip(runs, runs[1:], strict=False):   # the incoming stretch nudged to meet the outgoing one
        ref = mono[int(a[1] * sr):int(a[1] * sr) + n]
        best = max(range(-w, w + 1, 8), key=lambda d: float(ref @ mono[int(b[0] * sr) + d:int(b[0] * sr) + d + n])
                   / (np.linalg.norm(mono[int(b[0] * sr) + d:int(b[0] * sr) + d + n]) + 1e-9))
        b[0] += best / sr
    xn = int(x * sr)
    win = np.sin(np.linspace(0, np.pi / 2, xn))
    out_y, cuts, at, prev_end = None, [], 0.0, 0.0
    for a, b in runs:
        seg = y[:, int(a * sr):min(int(b * sr) + xn, y.shape[1])]
        if out_y is None:
            out_y = seg
        else:
            cuts.append((round(at, 2), round(prev_end, 2), round(a, 2)))
            out_y = np.concatenate([out_y[:, :-xn], out_y[:, -xn:] * win[::-1] + seg[:, :xn] * win, seg[:, xn:]], 1)
        at, prev_end = out_y.shape[1] / sr - xn / sr, b
    sf.write(out, out_y.T, sr)
    return cuts


def arranged(project, song: str, a: dict) -> str:
    """reel.json "arrange": {"bars": 141, "quiet": [46, 57], "intro": 11, "outro": 8} -- the song re-arranged to
    that many bars with its quiet bars inside the window (film bars), written to <reel>/music_arranged.wav (made
    again only when the song or the settings change). Without "bars" the song's own length is kept."""
    import hashlib
    import json
    src = song if os.path.isabs(song) else os.path.join(project.folder, song)
    out = project.path("music_arranged.wav")
    sig = hashlib.md5(json.dumps([src, os.path.getmtime(src), a]).encode()).hexdigest()[:12]
    meta = project.path("music_arranged.json")
    if os.path.exists(out) and os.path.exists(meta) and json.load(open(meta)).get("sig") == sig:
        return out
    P = bar_profile(src)
    J = jumps(P["feat"], sim=float(a.get("sim", JUMP_SIM)))
    want = int(a.get("bars", len(P["db"])))
    path = None
    for d in sorted(range(-16, 17), key=abs):              # (repeats come in whole phrases: the nearest length)
        path = arrange(P["db"], J, want + d, quiet=a.get("quiet"), intro=int(a.get("intro", 0)),
                       outro=int(a.get("outro", 8)), min_run=int(a.get("min_run", 8)))
        if path:
            break
    if path is None:
        raise SystemExit(f"arrange: no path through {os.path.basename(src)} near {want} bars")
    cuts = render_path(src, P["bars"], path, out)
    runs = [(k, b) for k, b in enumerate(path) if k == 0 or b != path[k - 1] + 1]
    json.dump(dict(sig=sig, bars=len(path), path=path, cuts=cuts), open(meta, "w"))
    print(f"[reel] arrange: {os.path.basename(src)} -> {len(path)} bars, {len(cuts)} cuts at "
          + ", ".join(f"{c[0]:.1f}s" for c in cuts) + f" (stretches from song bars {[b for _, b in runs]})", flush=True)
    return out
