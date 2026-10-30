"""Units: one per clip in the curated export folder, mapped back to its source camera file and time range, so the
reel cuts from the originals (no logo intro, no stinger, full quality).

    export/<category>/<name>.mp4
      segments/<clip>-NN_<subject>.mp4   time range from library.json clips[].segments
      broll/<name>.mp4                   time range from library.json broll[] ("name" may join files with "+")
      anything else/<clip>.mp4           the whole clip
"""
from __future__ import annotations

import json
import os
import subprocess

from .project import Project


def probe(path: str):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height:stream_side_data=rotation:format=duration",
                                            "-of", "json", path]))
    s = j["streams"][0]
    rot = abs(int(next((d.get("rotation", 0) for d in s.get("side_data_list", [])), 0)))
    w, h = (s["height"], s["width"]) if rot in (90, 270) else (s["width"], s["height"])
    return float(j["format"]["duration"]), w, h


def build(project: Project) -> list[dict]:
    lib = json.load(open(project.library)) if os.path.exists(project.library) else {"clips": [], "broll": []}
    segs = {}
    for c in lib.get("clips", []):
        stem = c["name"].split(".")[0]
        for k, ((a, b), lab) in enumerate(zip(c["segments"], c["labels"]), 1):
            segs[f"{stem}-{k:02d}_{lab['subject']}"] = (stem, a, b, lab.get("desc", ""))
    broll = {os.path.splitext(os.path.basename(b["file"]))[0]: b for b in lib.get("broll", [])}
    rotate, spans = project.config.get("rotate", {}), project.config.get("spans", {})
    units = []
    for cat in sorted(os.listdir(project.export_dir)):
        d = os.path.join(project.export_dir, cat)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".mp4"):
                continue
            name = f[:-4]
            parts, desc, tag = [], "", ""
            if name in segs:                               # a focal-subject segment
                clip, a, b, desc = segs[name]
                parts.append((clip, a, b))
            elif name in broll:                            # a B-roll piece: possibly several files joined
                b = broll[name]
                tag = name.split("_", 1)[1] if "_" in name else cat
                clips = [n.split(".")[0] for n in b.get("name", name.split("_")[0]).split("+")]
                span = b.get("span") or [0, None]
                parts += [(c, span[0], span[1]) if len(clips) == 1 else (c, 0, None) for c in clips]
                desc = b.get("desc", "")
            else:                                          # a whole clip
                parts.append((name.split("_")[0] if name not in project.config.get("sources", {}) else name, 0, None))
            for i, (clip, a, b) in enumerate(parts):
                ex0 = float(a or 0)                        # where the export file starts in the source (for picks)
                src = project.source_for(clip)
                dur, w, h = probe(os.path.join(project.folder, src))
                if clip in spans:
                    a, b = spans[clip]
                rot = rotate.get(clip)
                if rot and ("transpose=1" in rot or "transpose=2" in rot):
                    w, h = h, w
                units.append(dict(id=name if len(parts) == 1 else f"{name}#{i + 1}", category=cat,
                                  role=project.role(cat), tag=tag or cat, clip=clip, score=project.score_name(src),
                                  source=src, t0=float(a), t1=float(b if b is not None else dur), rotate=rot,
                                  orient="landscape" if w >= h else "portrait", desc=desc, export_t0=ex0))
    return units


def run(project: Project) -> list[dict]:
    units = build(project)
    project.dump("units.json", units)
    by = {}
    for u in units:
        by[(u["category"], u["orient"])] = by.get((u["category"], u["orient"]), 0) + 1
    print(f"[reel] {len(units)} units:", {f"{c}/{o}": n for (c, o), n in sorted(by.items())})
    return units
