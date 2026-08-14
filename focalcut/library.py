"""A library is a folder of clips plus a manifest of where the cuts should be.

The accurate pipeline is coarse-to-fine with a vision judge (see README):

  1. build_sheets()  — one contact sheet per clip (coarse, every ~8s). GPU-free.
  2. a vision model (Claude, fanned out one agent per sheet) reads each sheet and
     returns the focal-couple cut points  ->  write_manifest()
  3. window_sheets() around each cut (fine 2s, then ultra 0.5s) let the judge verify
     each cut is real and pin the boundary; a skeptical second read of the
     single-segment clips catches missed cuts.

This module owns the deterministic parts (sheet building, manifest read/write,
segment derivation). The judgement itself is model-driven and lives in the workflow.
"""
from __future__ import annotations

import json
import os

from . import video as V
from .contactsheet import build_sheet

LIBDIR = "focalcut"          # all library artifacts live in <folder>/focalcut/
MANIFEST = "library.json"
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".mkv")


def lib_dir(folder: str) -> str:
    """The subfolder that holds this folder's library (manifest + sheets)."""
    d = os.path.join(folder, LIBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def list_clips(folder: str, portrait_only: bool = False) -> list[dict]:
    """Scan a folder; return clip records with metadata. Optionally portrait-only."""
    out = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(VIDEO_EXTS):
            continue
        info = V.probe(os.path.join(folder, fn))
        if portrait_only and info.height <= info.width:
            continue
        out.append({"name": fn, "duration": round(info.duration, 2),
                    "width": info.width, "height": info.height, "fps": info.fps})
    return out


def build_sheets(folder: str, out_dir: str, interval: float = 8.0, cols: int = 6,
                 thumb_w: int = 240, portrait_only: bool = False,
                 on_progress=None) -> list[dict]:
    """Build one coarse contact sheet per clip. Returns [{name, sheet, duration}, ...]."""
    os.makedirs(out_dir, exist_ok=True)
    clips = list_clips(folder, portrait_only=portrait_only)
    jobs = []
    for i, c in enumerate(clips):
        stem = os.path.splitext(c["name"])[0]
        sheet = os.path.join(out_dir, stem + ".jpg")
        try:
            build_sheet(os.path.join(folder, c["name"]), interval=interval, cols=cols,
                        thumb_w=thumb_w, out_path=sheet)
            jobs.append({"name": c["name"], "sheet": sheet, "duration": c["duration"]})
        except Exception as e:
            jobs.append({"name": c["name"], "sheet": None, "duration": c["duration"],
                         "error": str(e)[:120]})
        if on_progress:
            on_progress(i + 1, len(clips), jobs[-1])
    return jobs


def window_sheet(folder: str, clip: str, cut: float, out_path: str,
                 half_width: float = 9.0, interval: float = 2.0, thumb_w: int = 200) -> str:
    """Build a fine/ultra contact sheet centred on a cut (for verify + boundary pinning)."""
    info = V.probe(os.path.join(folder, clip))
    start = max(0.0, cut - half_width)
    dur = min(2 * half_width, info.duration - start)
    cols = max(1, round(dur / interval))
    return build_sheet(os.path.join(folder, clip), interval=interval, cols=cols,
                       thumb_w=thumb_w, out_path=out_path, start=start, duration=dur)


def derive_segments(duration: float, cuts: list[float]) -> list[list[float]]:
    b = [0.0, *sorted(cuts), duration]
    return [[round(b[i], 2), round(b[i + 1], 2)] for i in range(len(b) - 1)]


def load(folder: str) -> dict:
    p = os.path.join(lib_dir(folder), MANIFEST)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


def save(folder: str, clips: list[dict], meta: dict) -> str:
    multi = sum(1 for c in clips if c.get("cuts"))
    payload = {"folder": folder,
               "method": "contact-sheet coarse->fine->ultra, vision judge",
               **meta,
               "clips_count": len(clips), "multi_couple": multi,
               "single_segment": len(clips) - multi,
               "total_cuts": sum(len(c.get("cuts", [])) for c in clips),
               "clips": clips}
    p = os.path.join(lib_dir(folder), MANIFEST)
    with open(p, "w") as fh:
        json.dump(payload, fh, indent=2)
    return p
