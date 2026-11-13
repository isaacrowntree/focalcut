"""A recap-reel project: one footage folder, its curated export, and the reel's config.

    <folder>/export/<category>/*.mp4     the curated clips (category folder names decide each clip's role)
    <folder>/focalcut/library.json       segment and B-roll time ranges (from `focalcut library` / `export`)
    <folder>/focalcut/reel/reel.json     this reel's settings (written by `focalcut reel init`)
    <folder>/focalcut/reel/...           every artifact: scores/, thumbs/, units.json, moments.json, music.json,
                                         plan_<fmt>.json, work/, out/

reel.json keys (all optional except song):
    song               path to the track
    songs              several tracks instead of song, one per act group: [{"song": path, "bars": n | null,
                                         "pre": 8, "from": seconds, "acts": {"<act>": bars | null}}, ...]
                                         (music.join; "from": the window starts on the bar nearest)
                                         song sets (plan): "footage": [category/tag, "!not"] the only footage
                                         the song's beats use; "title": image over its first shot of
                                         "title_match" footage (the artist's name); "feature": share of the
                                         song's shots kept for "feature_match" footage (default title_match)
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
    reject             {"<export file name>": ["0:41-0:46", ...]}  stretches a review turned down (a back, a crowd,
                                         blur): banned for the planner on top of the editor's picks, without
                                         touching them, and never taken as the editor's taste (taste.py)
    picks_only         ["<category>", ...]  export categories used only where the editor added a pick (e.g. a dance
                                         class to dip into); left out of "use everything"
    close_only         {"match": ["day"], "face": 0.2}  export files whose category or tag contains a match keep only
                                         close shots (a face at least `face` of the frame height; people step),
                                         e.g. daytime footage where the venue still looked empty
    people             {"min_clear": 2}  clear sightings (2 a second) that make someone a person the reel must show; 1
                                         also finds people the camera only passed (people step)
    hips               "movement": the semantic step scores hips on dancing movement, no from-behind / close-up
                                         prompts (an audience, not performers); false: left out
    resolution         the size the reel renders at: "1080p" (default: 1920x1080 landscape, 1080x1920 portrait),
                                         "1440p", "4k" (3840x2160 / 2160x3840), or [width, height] of the landscape
                                         frame; {"landscape": "4k", "portrait": "1080p"} sets it per format, and a
                                         cut can carry its own. Shots, effects, the overlay, the stinger and the
                                         XML's sequence all follow it (render.py)
    stabilise          true: shaky stretches the planner keeps play lightly steadied (render.steady_crop: the
                                         crop rides the camera's jitter, at most 1.2% of the frame, only
                                         where that measurably steadies the shot); "all": sources
                                         already stabilised ("_stabilized") too, where they still shake. Off by default: the
                                         planner plays the steadiest stretch of a clip it chose instead
                                         (plan.steady_shots), and the editor's picks play as shot
    stab_analysis      "half": measure a stabilised shot's camera shake on a half-size picture (render.stab): about
                                         7x less CPU, the heaviest thing a render does; the stabilised picture
                                         then sits within a pixel or two of the full-size analysis's. Default: full
    logo               image file (PNG with transparency) laid front and centre over the opening shots
    act_logos          {"<act>": "<image>"}  a title over an act's first shot, e.g. the headliner's name on "climax";
                                         {"<act>": {"image": ..., "match": ["<category or tag>"]}}: on that footage
    plan               {"landscape": {...}, "portrait": {...}}  overrides for plan.CFG (e.g. "climax_from": ["night"])
    cuts               {"<cut>": {"formats": ["portrait"], "footage": ["@portrait", "band"], ...}}  other reels from the
                                         same footage and analysis (`reel run --cut <cut>`): "footage" keeps the
                                         footage that fits (plan.fits: category/tag substrings, "!x" rules x out,
                                         "@portrait" / "@landscape" the way it was shot), "ban" export files, unit
                                         ids or camera clips kept out whatever they match, "formats" the formats it is
                                         made in, and any other key replaces the project's for this cut ("plan" is
                                         merged per format; a song of its own -- "song", "start_hint", ... --
                                         makes its own music_<cut>.json: run its music step). Out as
                                         <name>_<cut>_<fmt>; the editor's picks of other footage are left out of it.
                                         A cut that says how it ends (plan.<fmt> "ending" or "end_with") doesn't
                                         take the project's "close" as well
    mode               "picks": the reel is cut from the editor's picks (plan.pick_story) -- every pick once, whole,
                                         in the story's order, trimmed only when they overrun the reel (the least
                                         starred first), B-roll the editor left alone filling what they leave,
                                         the same picks in every format. A cut may set its own
    profiles           {"<name>": {...}}  named rule sets, any keys a cut takes (and "use" of other profiles), e.g.
                                         "no-night": {"footage": ["!night"]}, "short": {"length": 90}
    use                ["<profile>", ...]  profiles applied in order, then this level's own keys: at the top level
                                         for the main reel, inside a cut for that cut. "footage" and "ban" add up
                                         across profiles, "plan" merges per format, any other key: the last wins
"""
from __future__ import annotations

import copy
import json
import os
import re
import time

DEFAULT_ROLES = {"segments": "dance", "broll": "transition"}      # anything else is a "show"
VIDEO_EXT = (".mp4", ".mov", ".mxf", ".m4v", ".mkv")
MUSIC_KEYS = ("song", "songs", "start", "start_hint", "length", "mix", "to_end", "phrase_end")   # a cut with any of these: own music


ADDS_UP = ("footage", "ban")     # rule lists that add up across profiles instead of replacing


def merge(base: dict, over: dict) -> dict:
    """over laid on base: "footage" and "ban" add up, "plan" merges per format, any other key replaces base's."""
    out = copy.deepcopy(base)
    for k, v in copy.deepcopy(over).items():
        if k in ADDS_UP and out.get(k):
            out[k] = out[k] + [r for r in v if r not in out[k]]
        elif k == "plan":
            plan = out.get("plan") or {}
            out[k] = {**plan, **{fmt: {**plan.get(fmt, {}), **o} for fmt, o in v.items()}}
        else:
            out[k] = v
    return out


def with_profiles(level: dict, profiles: dict, seen: tuple = ()) -> dict:
    """A config level (reel.json, a cut, a profile) with its "use" profiles applied in order, then its own keys."""
    out = {}
    for name in level.get("use") or []:
        if name in seen:
            raise ValueError(f"reel.json profile {name!r} uses itself ({' -> '.join(seen + (name,))})")
        if name not in profiles:
            raise KeyError(f"reel.json has no profile {name!r} (profiles: {sorted(profiles)})")
        out = merge(out, with_profiles(profiles[name], profiles, seen + (name,)))
    return merge(out, {k: v for k, v in level.items() if k != "use"})


RESOLUTIONS = {"1080p": 1080, "hd": 1080, "fhd": 1080, "1440p": 1440, "qhd": 1440, "2k": 1440,
               "2160p": 2160, "4k": 2160, "uhd": 2160}


def frame_size(resolution, fmt: str) -> tuple[int, int]:
    """(width, height) of a format's frame for a reel.json "resolution": a name ("1080p", "1440p", "4k"), "<n>p",
    [width, height] (either way round: landscape takes the long side across, portrait upright), or a
    {format: resolution} of those. Nothing: 1080p. Both sides come out even (4:2:0 video)."""
    r = resolution.get(fmt) if isinstance(resolution, dict) else resolution
    if r is None or r is True or r == "":
        short, long = 1080, 1920
    elif isinstance(r, (list, tuple)):
        if len(r) != 2 or not all(isinstance(v, (int, float)) and v >= 16 for v in r):
            raise ValueError(f'reel.json "resolution": {r!r} is not [width, height]')
        short, long = sorted(int(v) for v in r)
    else:
        name = str(r).strip().lower()
        short = RESOLUTIONS.get(name) or (int(name[:-1]) if name.endswith("p") and name[:-1].isdigit() else
                                          int(name) if name.isdigit() else None)
        if not short or short < 16:
            raise ValueError(f'reel.json "resolution": {r!r} is not one of {sorted(RESOLUTIONS)}, "<n>p" or '
                             f'[width, height]')
        long = round(short * 16 / 9)
    short, long = short // 2 * 2, long // 2 * 2
    return (short, long) if fmt == "portrait" else (long, short)


class Project:
    def __init__(self, folder: str, cut: str | None = None):
        self.folder = os.path.abspath(folder)
        self.focalcut_dir = os.path.join(self.folder, "focalcut")
        self.reel_dir = os.path.join(self.focalcut_dir, "reel")
        self.export_dir = os.path.join(self.folder, "export")
        self.library = os.path.join(self.focalcut_dir, "library.json")
        self.config_path = os.path.join(self.reel_dir, "reel.json")
        config = json.load(open(self.config_path)) if os.path.exists(self.config_path) else {}
        self.profiles, self.uses = config.pop("profiles", None) or {}, bool(config.get("use"))
        profiles = self.profiles
        self.config = with_profiles(config, profiles)
        self.cut, self.footage, self.formats, self.own_music, self.ban = cut, None, None, False, set()
        self.own_plan = config.get("plan") or {}                 # the "plan" this reel itself wrote (no profile's,
        if cut:                                                  # another reel from the same footage (reel.json "cuts")
            c = (self.config.get("cuts") or {}).get(cut)
            if c is None:
                raise KeyError(f"reel.json has no cut {cut!r} (cuts: {sorted(self.config.get('cuts') or {})})")
            c = with_profiles(c, profiles)
            self.own_plan = ((self.config.get("cuts") or {}).get(cut) or {}).get("plan") or {}   # nor, for a cut,
            self.footage, self.formats = c.pop("footage", None), c.pop("formats", None)            # the project's)
            self.ban = set(c.pop("ban", ()))
            self.own_music = any(k in c for k in MUSIC_KEYS)
            self.config = merge({k: v for k, v in self.config.items() if k != "cuts"}, c)
            for fmt, own in (c.get("plan") or {}).items():   # a cut that says how it ends ends that way: the
                if any(k in own for k in ("ending", "end_with")) and "close" not in own:   # project's "close"
                    self.config["plan"][fmt].pop("close", None)                             # isn't its own

    # paths -------------------------------------------------------------------------------------
    def path(self, *parts: str) -> str:
        return os.path.join(self.reel_dir, *parts)

    @property
    def name(self) -> str:
        return self.config.get("name") or re.sub(r"\W+", "", os.path.basename(self.folder)) or "reel"

    def cut_name(self, fmt: str) -> str:
        """The name a format's plan, work and output go by: the format itself, or <cut>_<fmt> for a cut."""
        return f"{self.cut}_{fmt}" if self.cut else fmt

    @property
    def music_file(self) -> str:
        """music.json, or music_<cut>.json for a cut with its own song (any of MUSIC_KEYS)."""
        return f"music_{self.cut}.json" if self.own_music else "music.json"

    def in_cut(self, unit: dict) -> bool:
        """Is this footage (a unit) in the cut (its "ban" first, then its "footage")? Everything is, outside one."""
        from .plan import fits
        if self.ban & {unit.get("id"), unit.get("clip"), unit.get("id", "").split("#")[0]}:
            return False
        return not self.footage or fits(unit, self.footage)

    def size(self, fmt: str) -> tuple[int, int]:
        """(width, height) the reel renders at in a format: reel.json "resolution" (a cut's own wins), 1080p unset."""
        return frame_size(self.config.get("resolution"), fmt)

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

    @property
    def reel_id(self) -> str:
        """This reel's name in picks.json's per-selection "only" lists: the cut, or "main" for the project's reels."""
        return self.cut or "main"

    def picks(self, rejects=True, reel=True) -> dict:
        """The editor's picks: focalcut/reel/picks.json, with reel.json's "picks" merged over it, and its "reject"
        stretches added to their files' bans (rejects=False: the editor's own calls only). A clip's "only" maps a
        selection ("add" range, as written) to the reels it is for (["main"], ["short"], ...); one not listed is for
        every reel. reel=True keeps this reel's selections only; reel=False leaves them all (the source page). A clip
        whose picks are all for other reels gives this one nothing (banned whole), unless the reel's settings name it."""
        out = {**self._picks_file(), **(self.config.get("picks") or {})}
        if reel:
            raw = out
            out = {k: only_reel(v, self.reel_id) if isinstance(v, dict) and not k.startswith("_") else v
                   for k, v in out.items()}
            named = self._named()
            for k, e in out.items():                     # every pick of a clip for other reels: this reel takes
                if isinstance(e, dict) and not k.startswith("_") and (raw[k] or {}).get("add") and \
                        not e.get("add") and k not in named:      # nothing of it (not the planner's choice either)
                    out[k] = {**e, "ban": [*e.get("ban", []), "all"]}
        for k, spans in (self.config.get("reject") or {}).items() if rejects else ():
            e = dict(out.get(k) or {})
            e["ban"] = [*e.get("ban", []), *spans]
            out[k] = e
        if self.footage:                                         # a cut: only picks of its own footage
            ids = self._cut_ids()
            out = {k: v for k, v in out.items() if k.startswith("_") or k in ids}
        if not any(isinstance(e, dict) and e.get("open") for e in out.values()):
            for clip, v in (self.config.get("open") or {}).items():   # reel.json "open": {clip: in-out} -- the
                out[clip] = {**out.get(clip, {}), "open": v}           # opening, unless a pick is marked as it
        return out

    def _named(self) -> set:
        """Clips the reel's settings name on purpose (an opening, ending, close, open_with / end_with pick): never
        banned for having their picks in other reels only."""
        out = set((self.config.get("open") or {}).keys())
        for p in (self.config.get("plan") or {}).values():
            for key in ("ending", "close"):
                if isinstance(p.get(key), list) and p[key]:
                    out.add(p[key][0])
            for key in ("open_with", "end_with"):
                out.update(x[0] for x in p.get(key) or ())
        return out

    def _cut_ids(self) -> set:
        """Export file names and unit ids of the cut's footage."""
        if getattr(self, "_ids", None) is None:
            recs = self.load("moments.json") if os.path.exists(self.path("moments.json")) else []
            self._ids = {x for r in recs if self.in_cut(r["unit"]) for x in (r["id"], r["unit"]["id"])}
        return self._ids

    def _picks_file(self) -> dict:
        """picks.json, read again only when it changes (the planner asks for it on every crop). The review server
        saves it by swapping in a new copy, so a read that lands mid-swap (gone, or half there) waits and tries
        again rather than taking the picks as empty."""
        f = self.path("picks.json")
        for _ in range(40):
            try:
                st = os.stat(f)
                key = (st.st_mtime_ns, st.st_size)
                if getattr(self, "_picks_cache", (None,))[0] != key:
                    with open(f) as fh:
                        self._picks_cache = (key, json.load(fh))
                return copy.deepcopy(self._picks_cache[1])
            except FileNotFoundError:
                if not os.path.exists(f + ".bak"):       # never saved: no picks
                    return {}
            except ValueError:
                pass
            time.sleep(0.05)
        raise RuntimeError(f"{f}: unreadable (still being written?)")

    def logo(self, image: str | None = None) -> str | None:
        """The event's logo (or another image, e.g. an act's title) as an absolute path (relative paths are inside the
        folder), or None."""
        f = image or self.config.get("logo")
        return (f if os.path.isabs(f) else os.path.join(self.folder, f)) if f else None

    def out_stem(self, cut: str, fmt: str) -> str:
        """Output file stem for a cut: <name>_recap_<fmt> for the reels, <name>_<cut> for any other cut."""
        return f"{self.name}_recap_{fmt}" if cut == fmt else f"{self.name}_{cut}"

    def stinger(self, fmt: str) -> str | None:
        return (self.config.get("stingers") or {}).get(fmt)

    def save_config(self) -> str:
        os.makedirs(self.reel_dir, exist_ok=True)
        if self.cut:
            raise RuntimeError("save_config on a cut would write its merged settings over reel.json")
        if self.uses:
            raise RuntimeError("save_config would write the profiles reel.json uses into it")
        with open(self.config_path, "w") as f:
            json.dump({**self.config, **({"profiles": self.profiles} if self.profiles else {})}, f, indent=1)
        return self.config_path

    # json helpers ------------------------------------------------------------------------------
    def load(self, name: str):
        return json.load(open(self.path(name)))

    def dump(self, name: str, obj) -> str:
        os.makedirs(self.reel_dir, exist_ok=True)
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=1)
        return self.path(name)


def only_reel(entry: dict, reel: str) -> dict:
    """A clip's picks for one reel. Its "only" gives values to some reels: keyed by the "add" range as written, or
    "<field>:<value>" for "ban", "prefer", "open" and "end" ("more:" for "more"); a value not listed is in every reel."""
    only = entry.get("only")
    if not only:
        return entry
    keyed = {}
    for k, v in only.items():
        if isinstance(v, list):
            f, _, val = k.partition(":") if k.split(":", 1)[0] in TAGGED else ("add", "", k)
            keyed[(f, "*" if f == "more" else val.strip() if val.strip() == "all" else _span_key(val))] = v

    def ours(f, v):
        key = (f, "*" if f == "more" else "all" if str(v).strip() == "all" else _span_key(v))
        return reel in keyed.get(key, [reel])
    e = {k: v for k, v in entry.items() if k != "only"}
    for f in ("add", "ban", "prefer"):
        if isinstance(e.get(f), list):
            e[f] = [v for v in e[f] if ours(f, v)]
            if not e[f]:
                del e[f]
    for f in ("open", "end"):                       # the opening / ending a reel was given
        if isinstance(e.get(f), str) and not (ours(f, e[f]) and ours("add", e[f])):   # (and goes with its pick)
            del e[f]
    if e.get("more") and not ours("more", None):
        del e["more"]
    return e


TAGGED = ("ban", "prefer", "open", "end", "more")


def _span_key(v) -> tuple | None:
    """A range ("1:12.40-1:15.10", or seconds) as a rounded (in, out), so differently written ones match."""
    def secs(x):
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(str(x).strip().split(":"))))
    try:
        s = str(v).strip()
        i = s.find("-", 1)
        a, b = (secs(s[:i]), secs(s[i + 1:])) if i > 0 else (secs(s), secs(s))
        return round(a, 2), round(b, 2)
    except ValueError:
        return None

