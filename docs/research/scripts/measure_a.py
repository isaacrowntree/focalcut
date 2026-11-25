"""Stage A (focalcut venv): shots (R.analyse), whole-film camera motion at 10 fps, music grid + energy.
usage: measure_a.py <video> <out.json>"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.expanduser("~/focalcut"))
from focalcut.reel import moves, music  # noqa: E402
from focalcut.reel import reference as R  # noqa: E402

video, out = sys.argv[1], sys.argv[2]
ref = R.analyse(video)
dur, fps = ref["duration"], ref["fps"]

# -- camera motion, whole film, 10 fps ------------------------------------------------------------------------
g = moves.grey(video, 0.0, dur, gpu=False)
mv = moves.frames_moves(g)                       # (n, 4) dx dy rot logzoom per 0.1 s
# mean abs frame difference between neighbouring grey frames (for dissolve / fade finding)
diff = np.zeros(len(g))
for i in range(1, len(g)):
    diff[i] = np.abs(g[i].astype(np.int16) - g[i - 1].astype(np.int16)).mean()
luma = g.reshape(len(g), -1).mean(1)
del g

# -- music -----------------------------------------------------------------------------------------------------
import librosa  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    wav = os.path.join(td, "a.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "22050", wav], check=True)
    m = music.analyse(wav, start=0.0, max_len=dur, feel=False)
    y, sr = librosa.load(wav, sr=22050, mono=True)
hop = 512
onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
ot = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop)
rms = librosa.feature.rms(y=y, hop_length=hop)[0]
peaks = librosa.onset.onset_detect(onset_envelope=onset, sr=sr, hop_length=hop, units="time", backtrack=False)
pk_strength = onset[np.clip(librosa.time_to_frames(peaks, sr=sr, hop_length=hop), 0, len(onset) - 1)]
_, tracked = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, units="time", tightness=120)
tg = librosa.feature.tempogram(onset_envelope=onset, sr=sr, hop_length=hop)
# beat-grid confidence: onset strength on grid beats against half-way between them
bt = np.array(m["beats"])
on_at = lambda t: np.array([onset[(ot >= x - 0.05) & (ot <= x + 0.05)].max(initial=0) for x in t])
conf = float(on_at(bt).mean() / (on_at(bt[:-1] + np.diff(bt) / 2).mean() + 1e-9)) if len(bt) > 2 else 0.0

json.dump(dict(
    ref=ref,
    motion=dict(rate=moves.RATE, mv=np.round(mv, 5).tolist(), diff=np.round(diff, 2).tolist(),
                luma=np.round(luma, 1).tolist()),
    music=dict(tempo=m["tempo"], beat=m["beat"], beats=m["beats"], bars=m["bars"], bar_phase=m["bar_phase"],
               grid_conf=conf, tracked=np.round(tracked, 3).tolist(),
               onset_peaks=np.round(peaks, 3).tolist(), onset_peak_strength=np.round(pk_strength, 3).tolist(),
               env_rate=sr / hop, rms=np.round(rms, 5).tolist(), onset=np.round(onset, 3).tolist()),
), open(out, "w"))
print(os.path.basename(video), "shots", len(ref["shots"]), "tempo", round(m["tempo"], 1), "conf", round(conf, 2))
