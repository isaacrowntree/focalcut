"""The source browser: one page for the whole project, whatever the format -- every export clip in a list, one at a
time in a big player over a wide, zoomable timeline (filmstrip, the moments the planner found, where the landscape and
portrait reels use it), where the editor marks as many in-out selections per clip as they like (Clips); the cut shot
by shot with those picks in it (Cut & picks); and a reel's portrait cut shot by shot with its crop over the picture,
to pan or to set on a dancer (Frame portrait). The page's HTML, CSS and JavaScript are files in web/ (webui.py).

    review/sources.html  (+ review/sources/<unit>.jpg: a filmstrip per clip, one frame a second)
    focalcut/reel/previews/<unit>.mp4: what the page plays for exports bigger than 1080p (4K H.264 seeks too slowly
    in a browser to scrub, and vertical 4K is past most hardware decoders): 1280 px on the long side, a keyframe every
    0.2 s, same timeline. reel.json "previews": false plays the exports themselves.

Selections are picks "add" ranges ("1:12.40-1:15.10", export time): the planner puts each in the reel as one shot, from its
in-point, for its length. One selection can open the reel (the clip's "open": its in-out, under the event's logo) and
one can end it ("end"), set with the pick editor's Opening and Ending buttons. Served by `focalcut reel serve`, the page saves picks.json itself on every change; opened as a file, it
keeps changes in the browser and offers a download.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from . import hw, webui

TILE = (160, 90)          # filmstrip frame size
MAX_TILES = 400           # a longer clip gets a frame every dur / MAX_TILES seconds
PREVIEW_EDGE = 1280       # long side of the page's preview copies
PREVIEW_OVER = 1920       # exports bigger than this on either side get one


def duration(f):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def created(f):
    """The camera's recording start (its creation_time tag) as a naive datetime, or None."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time", "-of", "csv=p=0", f],
                         capture_output=True, text=True).stdout.strip()
    try:
        return dt.datetime.fromisoformat(out.replace("Z", "")) if out else None
    except ValueError:
        return None


def size(f):
    """(width, height) as displayed, rotation tag applied."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation", "-of", "json", f],
                         capture_output=True, text=True).stdout
    try:
        st = json.loads(out)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return TILE
    w, h = st.get("width", TILE[0]), st.get("height", TILE[1])
    rot = next((abs(int(d.get("rotation", 0))) for d in st.get("side_data_list", []) if "rotation" in d), 0)
    return (h, w) if rot % 180 == 90 else (w, h)


def tile(w, h):
    """A clip's filmstrip frame: TILE's height at the clip's own shape (a vertical clip gets narrow frames, so the
    timeline shows more of them), never wider than TILE."""
    return TILE if w / h >= TILE[0] / TILE[1] - 0.02 else (max(2, round(TILE[1] * w / h / 2) * 2), TILE[1])


def filmstrip(video, jpg, dur, size=TILE):
    """One row of frames across the clip, from keyframes only (the exports have one every 0.5 s), so it is quick.
    Each frame keeps its shape inside the tile, as the player shows it."""
    step = max(1.0, dur / MAX_TILES)
    n = max(1, math.ceil(dur / step))
    if not os.path.exists(jpg):
        fit = (f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,"
               f"pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-skip_frame", "nokey", "-i", video, "-vf",
                        f"fps=1/{step},{fit},tile={n}x1", "-frames:v", "1", "-q:v", "5", jpg],
                       check=False)
    return step, n


def preview(video, out, wh):
    """The page's copy of a big export (see the module notes); decoded and scaled on the GPU when it can be."""
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(video):
        return True
    os.makedirs(os.path.dirname(out), exist_ok=True)
    scale = (f"-2:{PREVIEW_EDGE}" if wh[1] > wh[0] else f"{PREVIEW_EDGE}:-2")
    enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "28"] if hw.nvenc()
           else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"])
    tail = ["-an", *enc, "-g", "10", "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out + ".tmp.mp4"]
    for dec, vf in ((["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], f"scale_cuda={scale}:format=yuv420p"),
                    ([], f"scale={scale}")):
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", *dec, "-i", video, "-vf", vf, *tail], capture_output=True)
        if r.returncode == 0:
            os.replace(out + ".tmp.mp4", out)
            return True
    if os.path.exists(out + ".tmp.mp4"):
        os.remove(out + ".tmp.mp4")
    return False


def clips(project, page_dir, plans, reels=None):
    """Every unit with an export file: its video and filmstrip (relative to the page), clock time, the moments found
    in it and the shots each reel takes from it, all in the export's own timeline. reels: {reel: {fmt: plan}} for
    every reel the picks serve (the project's own as "main", each cut by name): their shots as R[reel] = {L, P}."""
    units = project.load("units.json")
    moments = {r["unit"]["id"]: r for r in project.load("moments.json")} if os.path.exists(
        project.path("moments.json")) else {}
    clock_shift = float(project.config.get("clock_offset_h", 0.0))
    strips = os.path.join(page_dir, "sources")
    os.makedirs(strips, exist_ok=True)
    found = [(u, os.path.join(project.export_dir, u.get("category", ""), u["id"] + ".mp4")) for u in units]
    found = [(u, f) for u, f in found if "#" not in u["id"] and os.path.exists(f)]
    sizes = {u["id"]: size(f) for u, f in found}
    plays = {u["id"]: f for u, f in found}
    big = [(u, f) for u, f in found if max(sizes[u["id"]]) > PREVIEW_OVER]
    if big and project.config.get("previews", True) is not False:
        print(f"[reel] sources: preview copies of {len(big)} clips bigger than 1080p (once)", flush=True)
        prev = {u["id"]: project.path("previews", u["id"] + ".mp4") for u, _ in big}
        with ThreadPoolExecutor(3) as ex:
            ok = list(ex.map(lambda uf: preview(uf[1], prev[uf[0]["id"]], sizes[uf[0]["id"]]), big))
        plays.update({u["id"]: prev[u["id"]] for (u, _), good in zip(big, ok) if good})
    out = []
    for u, f in found:
        x0 = u.get("export_t0", 0.0)
        d = duration(f)
        w, h = sizes[u["id"]]
        tw, th = tile(w, h)
        name = u["id"] + ("" if (tw, th) == TILE else f"_{tw}x{th}")
        step, n = filmstrip(f, os.path.join(strips, name + ".jpg"), d, (tw, th))
        strip = f"sources/{quote(name)}.jpg"
        src = os.path.join(project.folder, u["source"])
        c = created(src) if os.path.exists(src) else None
        clock = (c + dt.timedelta(hours=clock_shift, seconds=u["t0"])).strftime("%H:%M") if c else ""
        ms = moments.get(u["id"], {}).get("moments", [])
        uses = _uses(plans, u["id"], x0)
        R = {r: dict(zip("LP", (lambda m: (m.get("landscape", []), m.get("portrait", [])))(_uses(ps, u["id"], x0))))
             for r, ps in (reels or {}).items()}
        out.append(dict(id=u["id"], cat=u.get("category", ""), tag=u.get("tag", ""), clock=clock,
                        sort=(c + dt.timedelta(seconds=u["t0"])).isoformat() if c else u["id"], dur=round(d, 2),
                        video=quote(os.path.relpath(plays[u["id"]], page_dir).replace(os.sep, "/")), ar=round(w / h, 4),
                        strip=strip, tile=[tw, th], step=step, n=n,
                        moments=[[round(m["t"] - x0, 2), m["kind"]] for m in ms if m["t"] - x0 >= 0],
                        L=uses.get("landscape", []), P=uses.get("portrait", []), R=R))
    out.sort(key=lambda c: c["sort"])
    return out


def _uses(plans, uid, x0):
    """{fmt: [[in, out, shot number], ...]}: the shots each plan takes from this unit, in its export's timeline. A
    portrait shot carries its crop too ([in, out, number, crop]: the page draws the frame from it at once, and can
    with no server to ask)."""
    return {fmt: [[round(s["src_in"] - x0, 2), round(s["src_in"] - x0 + s["length"], 2), i + 1,
                   *([_crop(s)] if fmt == "portrait" else [])]
                  for i, s in enumerate(p["shots"]) if s["unit"]["id"] == uid] for fmt, p in plans.items()}


def _crop(shot):
    """A planned shot's crop as the page draws it: its size, zoom and path ([[seconds from the shot's in, x, y]]);
    None when the picture fills the frame as it is."""
    c = shot.get("crop")
    if not c or not c.get("path"):
        return None
    out = dict(w=c["w"], h=c["h"], z=c.get("z", 1.0), path=[[t, round(x, 3), round(y, 3)] for t, x, y in c["path"]])
    if c.get("follow"):
        out["follow"] = c["follow"]
    return out


JS = ("picks.js", "app.js", "sources.js", "cut.js", "frame.js", "main.js")


def page(clips, picks=None, key="sources", title="Sources", root="", nav="", reels=None):
    """root: the project folder relative to the page ("../../../"), so its API calls work under any server prefix.
    reels: [{"id", "label"}] of the reels picks can be given to (the page's reel switch; none: no switch)."""
    data = dict(clips=clips, picks=picks or {}, key=key, tile=TILE, root=root, reels=reels or [])
    return webui.document(f"{title} — sources", "sources.html", data, css=("ui.css", "sources.css"), js=JS,
                          subs={"TITLE": html.escape(title), "NAV": nav})


def cut_label(cut: str) -> str:
    """A review page's name as people read it: "short_portrait" -> "Short portrait"."""
    return cut.replace("_", " ").capitalize()


def menu(project, current: str) -> str:
    """The pages' shared menu: this project's sources page, its reel reviews and the portrait framing of each reel
    (relative links, so it works opened as files too) and, when served, every project's index ("All projects", shown
    by the page once the server answers)."""
    rev = project.path("review")
    cuts = [c for c in ("landscape", "portrait") if os.path.exists(project.path(f"plan_{c}.json"))]
    cuts += sorted(f[:-5] for f in (os.listdir(rev) if os.path.isdir(rev) else [])
                   if f.endswith(".html") and f[:-5] not in cuts and f != "sources.html")
    frames = [("main", "Main")] if os.path.exists(project.path("plan_portrait.json")) else []
    from .project import Project
    frames += [(c, c.capitalize()) for c in (Project(project.folder).config.get("cuts") or {})
               if os.path.exists(project.path(f"plan_{c}_portrait.json"))]
    cur = lambda k: " class=cur" if k == current else ""
    out = f'<nav class=pages><a href="sources.html"{cur("sources")}>Sources</a>'
    if cuts:
        out += "<i>Review</i>" + "".join(f'<a href="{quote(c)}.html"{cur(c)}>{html.escape(cut_label(c))}</a>'
                                         for c in cuts)
    if frames:
        out += "<i>Frame portrait</i>" + "".join(
            f'<a href="sources.html#frame={quote(r)}" title="the {html.escape(label)} portrait cut shot by shot: drag '
            f'to pan the crop, click a dancer to follow them">{html.escape(label)}</a>' for r, label in frames)
    return out + '<a href="/" id=allp hidden data-served>All projects</a></nav>'


def title_of(project) -> str:
    """The project as people call it: its folder name ("Electric Nation 2026"; reel.json's "name" is for files)."""
    return os.path.basename(os.path.normpath(project.folder))


def api_root(project) -> str:
    """The project folder relative to its review pages, where `focalcut reel serve` answers api/... for it."""
    return os.path.relpath(project.folder, project.path("review")).replace(os.sep, "/") + "/"


def reels(project, plans) -> list[dict]:
    """The reels one set of picks serves: the project's own ("main", from these plans) and each of reel.json's cuts
    that has been planned, labelled with their length ("Main 2:46", "Short 1:30")."""
    def label(name, ps):
        n = max((p.get("music_length") or p.get("video_length") or 0) for p in ps.values()) if ps else 0
        m, sec = divmod(int(round(n)), 60)
        return name.capitalize() + (f" {m}:{sec:02d}" if n else "")
    out = [{"id": "main", "plans": plans, "label": label("main", plans)}]
    for cut in (project.config.get("cuts") or {}):
        ps = {f: project.load(f"plan_{cut}_{f}.json") for f in ("landscape", "portrait")
              if os.path.exists(project.path(f"plan_{cut}_{f}.json"))}
        if ps:
            out.append({"id": cut, "plans": ps, "label": label(cut, ps)})
    return out


def run(project, plans=None) -> str:
    page_dir = project.path("review")
    os.makedirs(page_dir, exist_ok=True)
    if plans is None:
        plans = {f: project.load(f"plan_{f}.json") for f in ("landscape", "portrait")
                 if os.path.exists(project.path(f"plan_{f}.json"))}
    rs = reels(project, plans)
    cs = clips(project, page_dir, plans, {r["id"]: r["plans"] for r in rs})
    path = os.path.join(page_dir, "sources.html")
    with open(path, "w") as f:
        f.write(page(cs, project.picks(reel=False), key=project.name, title=title_of(project), root=api_root(project),
                     nav=menu(project, "sources"),
                     reels=[{"id": r["id"], "label": r["label"]} for r in rs] if len(rs) > 1 else None))
    print(f"[reel] sources: {path}  ({len(cs)} clips; `focalcut reel serve {project.folder}` saves picks itself)")
    return path
