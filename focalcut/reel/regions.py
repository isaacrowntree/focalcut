"""The cameras matched on the same things: reel.json "look_method": "regions".

reference.looks() matches a camera to the hero camera by whole-frame colour statistics, which framing fools: a
drone looks down at sand and teal water, the hero across at faces and a pale sea. This hands the job to
color-grade-ai's match_camera.py (https://github.com/isaacrowntree/color-grade-ai, MIT), which finds the same
regions in both cameras' frames with SAM 3 (sand, grass, rock, black shirts, skin ...), pairs frames by wall clock
(reel.json "clock") and fits a tone curve, white balance, saturation and flare per camera, with a trim per clip
for a camera in "match_per_clip". The hero camera stays as shot.

  look_method     "regions"
  match_camera    path of match_camera.py (default: $FOCALCUT_MATCH_CAMERA, else ~/color-grade-ai/match_camera.py)
  look_flat       ["drone"]: cameras shooting a log / flat profile (their tone curve is fitted freely)
  look_classes    {"drone": ["sand", "grass", "foliage", "rock", "black", "white"]}: the regions a camera is
                  matched on (default all; leave out what it sees differently -- a drone's sea from above)

It needs torch and transformers with SAM 3, as match_camera.py does. The cubes go in looks/ as <camera>_regions.cube
(<camera>_regions_<clip>.cube per clip), a before/after page in review/colour/.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import os

DEFAULT = "~/color-grade-ai/match_camera.py"


def load(path=None):
    """The match_camera module, from its file."""
    path = os.path.expanduser(path or os.environ.get("FOCALCUT_MATCH_CAMERA") or DEFAULT)
    if not os.path.exists(path):
        raise FileNotFoundError(f'"look_method": "regions" needs color-grade-ai\'s match_camera.py (not at {path}): '
                                'set reel.json "match_camera" to its path')
    spec = importlib.util.spec_from_file_location("match_camera", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def job(project, units, cams, hero):
    """match_camera's job for this project: the hero camera as the reference, every other camera a source, each
    unit's clip read from its proxy (reel.json "proxies") when it has one."""
    cfg = project.config
    files = {}
    for u in units:
        files.setdefault(u["clip"], os.path.join(project.folder, u["source"]))

    def cam(name):
        mine = sorted(f for c, f in files.items() if any(fnmatch.fnmatch(c, p) for p in cams[name]))
        return dict(name=name, files=mine, clock=dict(cfg.get("clock") or {}))
    sources = []
    for name in cams:
        if name == hero:
            continue
        c = cam(name)
        if not c["files"]:
            continue
        per = name in (cfg.get("match_per_clip") or ())
        c.update(flat=name in (cfg.get("look_flat") or ()), per_clip=per,
                 cube=f"{name}_regions_{{clip}}.cube" if per else f"{name}_regions.cube")
        if (cfg.get("look_classes") or {}).get(name):
            c["classes"] = list(cfg["look_classes"][name])
        sources.append(c)
    proxies = {os.path.splitext(os.path.basename(src))[0]: os.path.join(project.folder, px[0])
               for src, px in (cfg.get("proxies") or {}).items()}
    spans = {c: [v[0] or 0.0, v[1] if v[1] is not None else 1e9] for c, v in (cfg.get("spans") or {}).items()}
    return dict(title=f"{project.name}: cameras matched to {hero}", work=project.path("work", "regions"),
                out=project.path("looks"), page=project.path("review", "colour"), analyse=proxies, spans=spans,
                reference=cam(hero), sources=sources)


def luts(report, cams, hero, reel_dir, look_dir):
    """match_camera's report as reference.looks() returns it: {camera or "camera:clip": {match, cube, how, check}}."""
    out = {hero: dict(match=list(cams[hero]), cube=None, how="as shot (the hero camera)")}
    for name, rep in report["sources"].items():
        cubes = {}
        for clip, r in rep["clips"].items():
            rel = os.path.relpath(os.path.join(look_dir, r["cube"]), reel_dir)
            chk = r.get("check") or {}
            check = dict(before=chk["before"]["all"], after=chk["after"]["all"]) if chk.get("after") else None
            cubes.setdefault(rel, []).append(check)
            if len({c["cube"] for c in rep["clips"].values()}) > 1:          # one per clip
                out[f"{name}:{clip}"] = dict(match=[clip], cube=rel, how=f"{clip} matched to {hero} by region",
                                             check=check, flag=r.get("flag") or [])
        if len(cubes) == 1:                                                  # the camera's one cube
            rel, checks = next(iter(cubes.items()))
            checks = [c for c in checks if c]
            out[name] = dict(match=list(cams[name]), cube=rel, how=f"matched to {hero} by region")
            if checks:
                out[name]["check"] = dict(before=round(sum(c["before"] for c in checks) / len(checks), 2),
                                          after=round(sum(c["after"] for c in checks) / len(checks), 2))
    return out


def looks(project, units, cams, hero, log=print):
    """Run the match and return the LUTs as reference.looks() does (reel.json "lut" is made from them)."""
    mc = load(project.config.get("match_camera"))
    report = mc.run(job(project, units, cams, hero), root=project.reel_dir, log=log)
    return luts(report, cams, hero, project.reel_dir, project.path("looks"))
