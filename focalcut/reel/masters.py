"""Editing masters: whole clips as the film shows them -- the camera's colour (its LUT and finish), steadied where
the camera shook -- at a size and bitrate to hand on, so another editor recuts from footage that already looks like
the film, with all its handles.

    out/premiere/footage/<camera>/<clip>.mp4      (+ out/premiere/footage/masters.json: per unit, its file and
                                                   whether it was steadied, and the in-point offset)

reel.json "masters": {"cameras": {"drone": "DJI_*", "a": "C2*", "b": ["C0*"]}, "only_used": ["b"],
"bitrate": "30M"} (cameras default to the project's own reel.json "cameras"; patterns a string or a list) -- which units go (every clip of a camera, or only those the cut uses), each graded and, where
its measured shake is over the plan's shake limit (or it is listed in "stab_strong"), steadied over its whole
length by render.steady_crop. Encoded with NVENC (H.264, 8-bit 4:2:0) at the project's resolution, the camera's own
sound kept. Made again only when its recipe changes.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from . import hw, render

DEFAULT = dict(cameras={"all": "*"}, only_used=[], bitrate="30M")
VERSION = 3
MAX_SHIFT = 0.05         # the most vid.stab moves the frame (of its width): its zoom stays near 1.1x at most
MAX_ANGLE = 0.03         # radians it may turn it
STAB_SMOOTH = 25         # vid.stab's smoothing: frames either side the camera path is averaged over (0.5 s at 50p)


def settings(project):
    cams = project.config.get("cameras")
    return {**DEFAULT, **({"cameras": cams} if cams else {}), **(project.config.get("masters") or {})}


def camera_of(u, cams):
    return next((k for k, pats in cams.items()
                 if any(fnmatch.fnmatch(u["clip"], p) for p in ([pats] if isinstance(pats, str) else pats))), None)


def used_units(project, cut):
    plan = project.load(f"plan_{cut}.json")
    ids = {s["unit"]["id"] for s in plan["shots"]}
    ids |= {p["unit"]["id"] for s in plan["shots"] if s.get("split") for p in s["split"]["panels"]}
    return ids


def needs_steady(project, u):
    """Steadied when listed in "stab_strong" or its measured jitter is over the plan's shake limit."""
    from . import plan as P
    from . import shake
    if any(fnmatch.fnmatch(u["clip"], q) for q in project.config.get("stab_strong") or []):
        return True
    j = shake.jitter(project, u, u["t0"], u["t1"])
    return bool(j is not None and j > P.config(project, "landscape")["shake"][0])


def master(project, u, out, steady, bitrate, qsv=True):
    """One unit, whole: graded, steadied if asked, at the project's landscape size; the camera's sound kept."""
    W, H = render.size(project, "landscape")
    dur = u["t1"] - u["t0"]
    src, t = project.decode_from(u["source"], u["t0"])
    exe, dec, P, pre = hw.ffmpeg(src, qsv)
    f = [u["rotate"]] if u.get("rotate") else []
    lut = render.lut_filter(project, u, P, at=u["t0"] + dur / 2)
    if lut:
        f.append(lut)
    if steady:                                              # vid.stab over the whole clip: the shake measured
        trf = out + ".trf"                                  # (pass 1), the camera path smoothed over about half
        q = lambda x: P(x).replace(chr(92), "/")            # a second and the frame zoomed just enough (pass 2)
        r = subprocess.run([exe, "-v", "error", "-y", *dec, "-ss", f"{t:.3f}", "-t", f"{dur:.3f}", "-i", P(src),
                            "-vf", pre + ",".join([*f, f"vidstabdetect=result='{q(trf)}':shakiness=6:accuracy=12"]),
                            "-f", "null", "-"], capture_output=True, text=True)
        if r.returncode and dec:
            return master(project, u, out, steady, bitrate, qsv=False)
        if r.returncode or not os.path.exists(trf):
            steady = False                                   # (couldn't measure it: as shot)
        else:
            sw, _sh = _dims(src)                             # (the correction capped -- a shift of MAX_SHIFT of the
            f.append(f"vidstabtransform=input='{q(trf)}':smoothing={STAB_SMOOTH}:optzoom=1:zoomspeed=0.2:"   # width,
                     f"maxshift={int(sw * MAX_SHIFT)}:maxangle={MAX_ANGLE}:"   # MAX_ANGLE turned -- so the zoom that
                     f"interpol=bicubic,unsharp=5:5:0.4:3:3:0")                # hides it stays small; a walk sways)
    _rate, after = render.finish_filters(render.cam_finish(project, u))
    fr = _fps(src)                                          # (a constant frame rate: a camera's dropped frames
    f += [f"fps={fr}", f"scale={W}:{H}:flags=lanczos", "setsar=1", "format=yuv420p", *after]   # filled, so an
                                                            # editor counting frames lands where the film did)
    enc = (["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-b:v", bitrate, "-maxrate",
            f"{int(float(bitrate.rstrip('M')) * 1.4)}M", "-bufsize", f"{int(float(bitrate.rstrip('M')) * 2)}M",
            "-profile:v", "high"] if hw.nvenc() else ["-c:v", "libx264", "-preset", "medium", "-b:v", bitrate])
    tmp = out + ".part.mp4"
    cmd = [exe, "-v", "error", "-y", *dec, "-ss", f"{t:.3f}", "-t", f"{dur:.3f}", "-i", P(src),
           "-map", "0:v:0", "-map", "0:a:0?", "-vf", pre + ",".join(f), *enc, "-pix_fmt", "yuv420p",
           *render.hw.COLOUR, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", P(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode and dec:                                 # (the iGPU out of sessions: the CPU reads it)
        return master(project, u, out, steady, bitrate, qsv=False)
    if r.returncode:
        raise RuntimeError(f"master {u['id']}: {r.stderr[-400:]}")
    os.replace(tmp, out)
    if os.path.exists(out + ".trf"):
        os.remove(out + ".trf")
    return steady


def _fps(path):
    """The source's nominal frame rate (r_frame_rate), as ffmpeg writes it (e.g. 25/1)."""
    return subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=r_frame_rate", "-of", "csv=p=0", path]).decode().strip()


def _dims(path):
    w, h = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2]
    return int(w), int(h)


def package_dir(project):
    """reel.json "premiere_dir": where the hand-on package goes (a drive with room), else out/premiere."""
    d = project.config.get("premiere_dir")
    return d if d else project.path("out", "premiere")


def run(project, cut="landscape", jobs=4):
    """Every master the settings ask for; returns {unit id: {file, steady, t0}} (also footage/masters.json)."""
    cfg = settings(project)
    root = os.path.join(package_dir(project), "footage")
    used = used_units(project, cut)
    todo = []
    for u in project.load("units.json"):
        cam = camera_of(u, cfg["cameras"])
        if cam is None or (cam in cfg["only_used"] and u["id"] not in used):
            continue
        todo.append((cam, u))
    index_f = os.path.join(root, "masters.json")
    index = json.load(open(index_f)) if os.path.exists(index_f) else {}

    def one(item):
        cam, u = item
        steady = needs_steady(project, u)
        os.makedirs(os.path.join(root, cam), exist_ok=True)
        out = os.path.join(root, cam, u["id"].replace("#", "-") + ".mp4")
        sig = hashlib.md5(json.dumps([VERSION if steady else 2, u["source"], u["t0"], u["t1"], steady, cfg["bitrate"],
                                      render.cam_finish(project, u), project.config.get("lut"),
                                      project.config.get("lut_segments"),
                                      project.config.get("stab_strong_settings")], sort_keys=True,
                                     default=str).encode()).hexdigest()[:12]
        if os.path.exists(out) and (index.get(u["id"]) or {}).get("sig") == sig:
            return u["id"], index[u["id"]]
        st = master(project, u, out, steady, cfg["bitrate"])
        print(f"[reel] master {cam}/{os.path.basename(out)}{' (steadied)' if st else ''}", flush=True)
        return u["id"], dict(file=os.path.relpath(out, package_dir(project)), steady=st, t0=u["t0"],
                             camera=cam, sig=sig)
    with ThreadPoolExecutor(jobs) as ex:
        for uid, rec in ex.map(one, todo):
            index[uid] = rec
    with open(index_f, "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"[reel] masters: {len(index)} clips in {root}", flush=True)
    return index
