"""Baptism films: what each shot shows (SigLIP 2 zero-shot over the stored 2 fps embeddings). ComfyUI venv."""
import json, sys, numpy as np
from focalcut.reel import reference as R
C = {
 "scene":   ["a beach and the sea with no people", "an aerial view of a beach and coastline", "waves and sky, a landscape"],
 "gather":  ["a crowd of people standing on a beach watching", "people arriving and walking on a beach", "people chatting and waiting on the sand"],
 "pray":    ["people praying with heads bowed", "a pastor speaking to a person standing in the water before a baptism", "people standing in the water holding hands, talking"],
 "walk_in": ["people walking into the sea", "a person wading out into the water"],
 "dunk":    ["a person being baptized, leaned backwards under the water", "a person being dunked under water by two people", "a big splash as someone goes under the water"],
 "rise":    ["a person coming up out of the water soaking wet, smiling", "a wet person wiping water from their face and laughing", "a person raising their arms in celebration in the water"],
 "hug":     ["two wet people hugging in the water", "people embracing each other on a beach"],
 "cheer":   ["a crowd clapping and cheering", "people on a beach applauding with raised hands", "people smiling and filming with phones"],
 "portrait":["a close-up portrait of a person smiling at the camera", "a group of friends posing for a photo"],
}
names=list(C); E=R.Embedder(); T={n:E.texts(p) for n,p in C.items()}
ref=json.load(open("out/references.json")); out={}
for f in ref["films"]:
    if not (f["slug"].startswith("baptism") or f["slug"].startswith("malabar")): continue
    b=np.load(f"out/{f['slug']}.b.npz"); t=b["t"]; e=b["emb"].astype(np.float32)
    S=np.stack([(e@T[n].T).max(1) for n in names],1)
    lab=[]
    for s in f["shots"]:
        i=np.where((t>=s["t0"])&(t<s["t1"]))[0]
        if not len(i): i=[int(np.argmin(np.abs(t-(s["t0"]+s["t1"])/2)))]
        lab.append(names[int(S[i].mean(0).argmax())])
    out[f["slug"]]=lab
    print(f["slug"], " ".join(lab), flush=True)
json.dump(out,open("out/content.json","w"))
