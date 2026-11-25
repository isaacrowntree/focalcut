"""Stage B (ComfyUI venv, FOCALCUT_DISABLE_CUDNN=1, PYTHONPATH=~/focalcut): SigLIP 2 on 2 fps frames of each film.
usage: measure_b.py <outdir> <video>...   -> <outdir>/<slug>.b.npz  (t, emb (float16), scale probs)"""
import os
import subprocess
import sys

import numpy as np

from focalcut.reel import reference as R

GENERIC = {
    "aerial": ["an aerial drone photo of a landscape from high above", "a drone view looking down from the sky",
               "an aerial view of a city, coast or mountains"],
    "wide": ["a wide shot of a landscape taken from the ground", "a wide shot of a crowd of people",
             "a wide shot of a room or a venue with people small in the frame"],
    "medium": ["a medium shot of a person from the waist up", "a photo of a few people standing together"],
    "close": ["a close-up portrait of a person's face", "a close-up of hands or a small object",
              "a close-up of two people smiling"],
}
NAMES = ["aerial", "wide", "medium", "close"]
RATE, W = 2, 512

outdir, vids = sys.argv[1], sys.argv[2:]
E = R.Embedder()
prompts, owner = [], []
for n in NAMES:
    for p in R.SCALES[n] + GENERIC[n]:
        prompts.append(p)
        owner.append(n)
T = E.texts(prompts)
for v in vids:
    slug = os.path.splitext(os.path.basename(v))[0]
    dst = os.path.join(outdir, slug + ".b.npz")
    if os.path.exists(dst):
        continue
    d, sw, sh, _ = R.probe(v)
    h = W * sh // sw // 2 * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", v, "-an", "-vf", f"fps={RATE},scale={W}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8)[: len(raw) // (W * h * 3) * W * h * 3].reshape(-1, h, W, 3)
    emb = E.images(list(fr), batch=32)
    sims = emb @ T.T
    cls = np.stack([sims[:, [i for i, o in enumerate(owner) if o == n]].mean(1) for n in NAMES], 1)
    t = (np.arange(len(fr)) + 0.5) / RATE          # ffmpeg fps filter: frame k covers [k/RATE, (k+1)/RATE)
    np.savez_compressed(dst, t=t, emb=emb.astype(np.float16), cls=cls.astype(np.float32))
    print(slug, len(fr), "frames", dict(zip(NAMES, np.bincount(cls.argmax(1), minlength=4).tolist())), flush=True)
