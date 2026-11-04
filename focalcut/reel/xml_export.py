"""Premiere timeline (FCP7 XML) for each reel, so it can be re-edited from the original camera files.

    out/<name>_recap_<fmt>.xml   (+ out/<fmt>_media/ for rendered shots)

V1  every shot. Uncropped plain shots reference the original camera file scaled to fit, with the source's own
    handles for trimming and slipping. Reframed shots (most of them: the crop follows the subject), shots with
    effect modules or a transition, and rotated clips use their rendered 1080p files (from render's manifest),
    copied to out/<fmt>_media/.
V2  (disabled) the original source under each rendered shot, lined up, for redoing it by hand.
A1  the song from the reel's start point.   A2  the stinger's sound.   The stinger video closes V1.
Animated position keyframes are deliberately not used: they don't survive FCP XML import reliably.
"""
from __future__ import annotations

import json
import os
import subprocess
from urllib.parse import quote
from xml.sax.saxutils import escape

from . import hw

FPS = 50
SIZE = {"landscape": (1920, 1080), "portrait": (1080, 1920)}


def pathurl(path: str) -> str:
    """/mnt/k/x y.MP4 -> file://localhost/K:/x%20y.MP4 (Premiere on Windows via WSL); other paths as file URLs."""
    path = os.path.abspath(path)
    if path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
        return "file://localhost/" + quote(f"{path[5].upper()}:{path[6:]}", safe="/:")
    return "file://localhost" + quote(path, safe="/:")


def probe(path):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                            "stream=codec_type,width,height:stream_side_data=rotation:format=duration",
                                            "-of", "json", path]))
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = any(s["codec_type"] == "audio" for s in j["streams"])
    w = h = 0
    if v:
        rot = abs(int(next((d.get("rotation", 0) for d in v.get("side_data_list", [])), 0)))
        w, h = (v["height"], v["width"]) if rot in (90, 270) else (v["width"], v["height"])
    return dict(frames=round(float(j["format"]["duration"]) * FPS), w=w, h=h, video=v is not None, audio=a)


class Files:
    def __init__(self):
        self.ids, self.info = {}, {}

    def ref(self, path):
        """<file> element: full definition the first time, then just the id."""
        if path in self.ids:
            return f'<file id="{self.ids[path]}"/>'
        fid = self.ids[path] = f"file-{len(self.ids) + 1}"
        p = self.info[path] = probe(path)
        media = ""
        if p["video"]:
            media += (f'<video><samplecharacteristics><width>{p["w"]}</width><height>{p["h"]}</height>'
                      f'</samplecharacteristics></video>')
        if p["audio"]:
            media += ('<audio><samplecharacteristics><samplerate>48000</samplerate><depth>16</depth>'
                      '</samplecharacteristics><channelcount>2</channelcount></audio>')
        return (f'<file id="{fid}"><name>{escape(os.path.basename(path))}</name><pathurl>{pathurl(path)}</pathurl>'
                f'{rate()}<duration>{p["frames"]}</duration><media>{media}</media></file>')

    def fit(self, path, fmt):
        """Basic Motion scale (percent) that fits a source into the sequence frame."""
        p = self.info.get(path) or probe(path)
        W, H = SIZE[fmt]
        return round(100 * min(W / max(p["w"], 1), H / max(p["h"], 1)), 3)


def rate():
    return f"<rate><timebase>{FPS}</timebase><ntsc>FALSE</ntsc></rate>"


def scale_filter(pct):
    return ('<filter><effect><name>Basic Motion</name><effectid>basic</effectid><effectcategory>motion</effectcategory>'
            '<effecttype>motion</effecttype><mediatype>video</mediatype><parameter authoringApp="PremierePro">'
            f'<parameterid>scale</parameterid><name>Scale</name><valuemin>0</valuemin><valuemax>1000</valuemax>'
            f'<value>{pct:g}</value></parameter></effect></filter>')


def clip(cid, files, path, start, end, src_in, name=None, fmt=None, enabled=True, kind="video"):
    f = files.ref(path)
    flt = scale_filter(files.fit(path, fmt)) if fmt and kind == "video" else ""
    return (f'<clipitem id="{cid}"><name>{escape(name or os.path.basename(path))}</name>'
            f'<enabled>{"TRUE" if enabled else "FALSE"}</enabled>{rate()}<start>{start}</start><end>{end}</end>'
            f'<in>{src_in}</in><out>{src_in + end - start}</out>{f}{flt}</clipitem>')


def run(project, fmt, cut=None) -> str:
    cut = cut or fmt
    plan = project.load(f"plan_{cut}.json")
    fmt = plan.get("format", fmt)
    song = plan.get("song") or project.load(project.music_file)["song"]
    manifest = project.load(os.path.join("out", f"{cut}_shots.json"))      # final file per shot, from render
    W, H = SIZE[fmt]
    media = project.path("out", f"{cut}_media"); os.makedirs(media, exist_ok=True)
    files = Files()
    v1, v2, n, rec = [], [], 0, 0
    for i, s in enumerate(plan["shots"]):
        u = s["unit"]
        start = round(s["rec_in"] * FPS)
        end = start + s["frames"]                  # same beat-exact cut points as the render
        src = os.path.join(project.folder, u["source"])
        base = os.path.basename(manifest[i])
        rendered = (bool(s.get("fx")) or bool(s.get("logo")) or bool(s.get("stab")) or bool(u.get("rotate")) or bool(s.get("crop")) or "_o" in base
                    or "_i" in base)
        label = f'{i + 1:03d} {u["id"]} {s["kind"]}' + (f' [{"+".join(s["fx"])}]' if s.get("fx") else "")
        if rendered:
            keep = os.path.join(media, f'{i + 1:03d}_{u["id"].replace("#", "-")}_{s["kind"]}'
                                       f'{"_" + "+".join(s["fx"]) if s.get("fx") else ""}'
                                       f'{"_" + s["transition"] if s.get("transition") else ""}.mp4')
            hw.retag(project.path(manifest[i]), keep)
            n += 1; v1.append(clip(f"v1-{n}", files, keep, start, end, 0, label))
            n += 1; v2.append(clip(f"v2-{n}", files, src, start, end, round(s["src_in"] * FPS), f"{label} (source)",
                                   fmt=fmt, enabled=False))
        else:
            n += 1; v1.append(clip(f"v1-{n}", files, src, start, end, round(s["src_in"] * FPS), label, fmt=fmt))
        rec = end
    total = rec
    a2 = ""
    sting = project.stinger(fmt) if plan.get("stinger") else None
    if sting:                                      # stinger closes V1; its sound on A2
        sframes = round(plan["stinger"] * FPS)
        n += 1; v1.append(clip(f"v1-{n}", files, sting, rec, rec + sframes, 0, "Stinger", fmt=fmt))
        a2 = clip("a2-1", files, sting, rec, rec + sframes, 0, "Stinger sound", kind="audio")
        total = rec + sframes
    a1 = clip("a1-1", files, song, 0, round(plan["music_length"] * FPS), round(plan["music_start"] * FPS),
              f'{os.path.basename(song)} (from {plan["music_start"]:.2f}s)', kind="audio")
    fmt_xml = (f'<format><samplecharacteristics>{rate()}<width>{W}</width><height>{H}</height>'
               f'<pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance>'
               f'</samplecharacteristics></format>')
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
           f'<sequence id="seq-{cut}"><name>{escape(project.out_stem(cut, fmt).replace("_", " "))}</name><duration>{total}</duration>'
           f'{rate()}<media><video>{fmt_xml}<track>{"".join(v1)}</track><track>{"".join(v2)}</track></video>'
           f'<audio><track>{a1}</track><track>{a2}</track></audio></media></sequence></xmeml>\n')
    out = project.path("out", f"{project.out_stem(cut, fmt)}.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"[reel] {fmt}: {len(v1)} clips on V1, {len(v2)} sources on V2 (disabled) -> {out}")
    return out
