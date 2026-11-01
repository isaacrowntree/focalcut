"""A recap-reel project: one footage folder, its curated export, and the reel's config.

    <folder>/export/<category>/*.mp4     the curated clips (category folder names decide each clip's role)
    <folder>/focalcut/library.json       segment and B-roll time ranges (from `focalcut library` / `export`)
    <folder>/focalcut/reel/reel.json     this reel's settings (written by `focalcut reel init`)
    <folder>/focalcut/reel/...           every artifact: scores/, thumbs/, units.json, moments.json, music.json,
                                         plan_<fmt>.json, work/, out/

reel.json keys (all optional except song):
    song               path to the track
    songs              several tracks instead of song, one per act group: [{"song": path, "bars": n | null,
                                         "pre": 8, "acts": {"<act>": bars | null}}, ...] (music.join)
    mix                how songs hand over: {"bars": 4, "ramp": 8, "max_ramp": 0.15, "split": 150} (overlap bars,
                                         tempo-ramp bars, largest tempo gap to ramp, bass split Hz); false: cut
    start              "auto" (first bar after the intro) or seconds
    start_hint         seconds: pick the bar nearest this instead of detecting the intro
    length             total reel length in seconds, stinger included (default 180: Instagram's Reels-tab limit)
    stingers           {"landscape": path, "portrait": path}
    name               output file prefix (default: the folder name)
    roles              {"<category>": "dance" | "show" | "transition"}; defaults below
    sources            {"<clip>": "<file in folder>"}  e.g. a stabilised or joined replacement
    rotate             {"<clip>": "<ffmpeg filter>"}   e.g. "transpose=2" for a clip shot sideways
    spans              {"<clip>": [t0, t1]}            usable part of a clip for the reels (t1 may be null: to the end)
    proxies            {"<source>": ["<file>", [[at, shift], ...]]}  a smaller file to decode a source from (e.g. the
                                         H.264 export of a DNxHD master, starting on the same frame); from source
                                         time `at` on, the proxy's time is source time + shift (a dropped join)
    picks              {"<export file name>": {"ban": ["0:00-0:20" | "all"], "prefer": ["0:41-0:46"], "add": ["1:12.5"],
                                         "more": 2}}  the editor's
                                         calls, times in that export file's own timeline; merged over picks.json
                                         in this folder (the review step's page gives a copyable line per shot)
    hips               "movement": the semantic step scores hips on dancing movement, no from-behind / close-up
                                         prompts (an audience, not performers); false: left out
    logo               image file (PNG with transparency) laid front and centre over the opening shots
    plan               {"landscape": {...}, "portrait": {...}}  overrides for plan.CFG
"""
from __future__ import annotations

import json
import os
import re

DEFAULT_ROLES = {"segments": "dance", "broll": "transition"}      # anything else is a "show"
VIDEO_EXT = (".mp4", ".mov", ".mxf", ".m4v", ".mkv")


class Project:
    def __init__(self, folder: str):
        self.folder = os.path.abspath(folder)
        self.focalcut_dir = os.path.join(self.folder, "focalcut")
        self.reel_dir = os.path.join(self.focalcut_dir, "reel")
        self.export_dir = os.path.join(self.folder, "export")
        self.library = os.path.join(self.focalcut_dir, "library.json")
        self.config_path = os.path.join(self.reel_dir, "reel.json")
        self.config = json.load(open(self.config_path)) if os.path.exists(self.config_path) else {}

    # paths -------------------------------------------------------------------------------------
    def path(self, *parts: str) -> str:
        return os.path.join(self.reel_dir, *parts)

    @property
    def name(self) -> str:
        return self.config.get("name") or re.sub(r"\W+", "", os.path.basename(self.folder)) or "reel"

    def role(self, category: str) -> str:
        return {**DEFAULT_ROLES, **self.config.get("roles", {})}.get(category, "show")

    def source_for(self, clip: str) -> str:
        """The file a clip is cut from: a configured replacement, else <clip>.<ext> found in the folder."""
        if clip in self.config.get("sources", {}):
            return self.config["sources"][clip]
        for f in sorted(os.listdir(self.folder)):
            stem, ext = os.path.splitext(f)
            if stem == clip and ext.lower() in VIDEO_EXT:
                return f
        return clip + ".MP4"

    @staticmethod
    def score_name(source: str) -> str:
        """scores/<name>.npz for a source file (stabilised / joined versions share the camera clip's name)."""
        return os.path.splitext(os.path.basename(source))[0].replace("_stabilized", "").replace("_joined", "")

    def decode_from(self, source: str, t: float) -> tuple[str, float]:
        """File and time to decode a source's time t from: its proxy when one is configured."""
        px = (self.config.get("proxies") or {}).get(source)
        if not px:
            return os.path.join(self.folder, source), t
        f, shifts = px[0], px[1] if len(px) > 1 else []
        return os.path.join(self.folder, f), t + sum(d for at, d in shifts if t >= at)

    def source_time(self, source: str, t: float) -> float:
        """The inverse: a decoded file's time back in the source's own time."""
        px = (self.config.get("proxies") or {}).get(source)
        shifts = px[1] if px and len(px) > 1 else []
        return t - sum(d for at, d in shifts if t + d >= at)

    def picks(self) -> dict:
        """The editor's picks: focalcut/reel/picks.json, with reel.json's "picks" merged over it."""
        f = self.path("picks.json")
        out = json.load(open(f)) if os.path.exists(f) else {}
        return {**out, **(self.config.get("picks") or {})}

    def logo(self) -> str | None:
        """The event's logo as an absolute path (relative paths are inside the folder), or None."""
        f = self.config.get("logo")
        return (f if os.path.isabs(f) else os.path.join(self.folder, f)) if f else None

    def out_stem(self, cut: str, fmt: str) -> str:
        """Output file stem for a cut: <name>_recap_<fmt> for the reels, <name>_<cut> for any other cut."""
        return f"{self.name}_recap_{fmt}" if cut == fmt else f"{self.name}_{cut}"

    def stinger(self, fmt: str) -> str | None:
        return (self.config.get("stingers") or {}).get(fmt)

    def save_config(self) -> str:
        os.makedirs(self.reel_dir, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=1)
        return self.config_path

    # json helpers ------------------------------------------------------------------------------
    def load(self, name: str):
        return json.load(open(self.path(name)))

    def dump(self, name: str, obj) -> str:
        os.makedirs(self.reel_dir, exist_ok=True)
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=1)
        return self.path(name)
