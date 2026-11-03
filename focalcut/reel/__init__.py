"""Recap reels: beat-synced highlight reels cut from a curated footage folder.

    units -> score -> shake -> focus -> subject -> semantic -> moments -> faces -> people -> role -> music -> plan
          -> render -> xml -> review

  score     poses, people, camera / subject motion, sharpness and exposure (RT-DETR + ViTPose, 5-10 fps)
  shake     camera jitter per source: shaky stretches are avoided, mildly shaky ones stabilised at render
  focus     per-person sharpness on keyframes: the fallback subject picker
  subject   who each shot is about: SAM 3 person masks scored by BiRefNet saliency and Depth Anything V2 nearness
  semantic  what the subject's crop shows (SigLIP 2 zero-shot): hips / butt working, an expressive face
  moments   ranked moves per unit, with how well each frames after reframing the 4K source
  people    everyone in the footage by face (YuNet + SFace), grouped across clips, so the plan can show every person
  role      who leads and who follows (SigLIP 2 per person): a couple shares the portrait frame, the follow featured
  plan      the story in acts, locked to the beat grid; effects and transitions chosen per cut (picks.json: the
            editor's bans and preferences, by export file and time)
  review    a local HTML page per reel: a frame per shot, its export file and time range, a copyable picks line;
            and one source browser for the project (browse.py): every clip on a wide timeline, in-out selections
            (`focalcut reel serve` lets the pages save picks.json themselves)

Each step reads and writes JSON / npz under <folder>/focalcut/reel/, so any step can be re-run on its own, and the
model passes skip sources they have already done. Everything except `units`, `xml` and `review` needs the `reel` extra
(torch, transformers, timm, librosa, opencv; `plan` needs opencv when its flow cache is incomplete); `subject` and the
SAM 3 effects need a CUDA GPU.
"""
from __future__ import annotations

import json
import subprocess

from .project import Project

STEPS = ["units", "score", "shake", "focus", "subject", "semantic", "moments", "faces", "people", "role", "music", "plan", "render",
         "xml", "review"]
FORMATS = ["landscape", "portrait"]


def stinger_length(project: Project, formats) -> float:
    """Length of the stinger (the formats share one timing; the longest wins)."""
    best = 0.0
    for fmt in formats:
        s = project.stinger(fmt)
        if s:
            d = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                    "-of", "json", s]))
            best = max(best, float(d["format"]["duration"]))
    return best


def run(folder: str, steps=None, formats=None, workers: int = 2, draft: bool = False, cut: str | None = None) -> None:
    """cut: one of reel.json's "cuts" (another reel from the same footage) in place of the main reel; the analysis
    steps are the project's own, so a cut runs from music on."""
    project = Project(folder, cut)
    steps = steps or STEPS
    formats = formats or project.formats or FORMATS
    sl = stinger_length(project, formats)
    import importlib
    for step in STEPS:
        if step not in steps:
            continue
        mod = importlib.import_module(f".{'xml_export' if step == 'xml' else step}", __name__)
        if step == "score":
            mod.run(project, workers=workers)
        elif step == "music":
            mod.run(project, stinger_len=sl)
        elif step == "plan":
            for f in formats:
                mod.run(project, f, stinger_len=sl)
        elif step == "render":
            for f in formats:
                mod.run(project, f, draft=draft, cut=project.cut_name(f))
        elif step in ("xml", "review"):
            for f in formats:
                mod.run(project, f, cut=project.cut_name(f))
            if step == "review" and not cut:
                from . import browse
                browse.run(project)
        elif step in ("shake", "focus"):
            mod.run(project, workers=max(workers, 1) + 1)
        else:                                       # units, subject, semantic, moments, faces, people, role
            mod.run(project)
