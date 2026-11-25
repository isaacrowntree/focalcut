"""Premiere timeline (FCP7 XML) for each reel, so it can be re-edited from the original camera files.

    out/<name>_recap_<fmt>.xml   (+ out/<fmt>_media/ for rendered shots)

V1  every shot. Uncropped plain shots reference the original camera file scaled to fit, with the source's own
    handles for trimming and slipping. Reframed shots (most of them: the crop follows the subject), shots with
    effect modules or a transition, and rotated clips use their rendered files (from render's manifest, at the
    reel's own size: reel.json "resolution"),
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

import numpy as np

from . import hw

FPS = 50
SIZE = {"landscape": (1920, 1080), "portrait": (1080, 1920)}   # the default; a project's own: render.size


def pathurl(path: str) -> str:
    """/mnt/k/x y.MP4 -> file://localhost/K:/x%20y.MP4 (Premiere on Windows via WSL); other paths as file URLs."""
    path = os.path.abspath(path)
    if path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
        return "file://localhost/" + quote(f"{path[5].upper()}:{path[6:]}", safe="/:")
    return "file://localhost" + quote(path, safe="/:")


def probe(path):
    if path.lower().endswith(".png"):                    # a still (a caption): its size, as long as it is wanted
        from PIL import Image
        with Image.open(path) as im:
            return dict(frames=FPS * 3600, w=im.width, h=im.height, video=True, audio=False)
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                            "stream=codec_type,width,height,r_frame_rate,bits_per_sample,"
                                            "bits_per_raw_sample:stream_side_data=rotation:format=duration",
                                            "-of", "json", path]))
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = any(s["codec_type"] == "audio" for s in j["streams"])
    au = next((s for s in j["streams"] if s["codec_type"] == "audio"), {})
    depth = int(au.get("bits_per_sample") or au.get("bits_per_raw_sample") or 16) or 16
    w = h = 0
    if v:
        rot = abs(int(next((d.get("rotation", 0) for d in v.get("side_data_list", [])), 0)))
        w, h = (v["height"], v["width"]) if rot in (90, 270) else (v["width"], v["height"])
    fps = FPS
    if v and v.get("r_frame_rate"):
        n_, d_ = (v["r_frame_rate"].split("/") + ["1"])[:2]
        fps = round(float(n_) / max(float(d_), 1.0)) or FPS
    return dict(frames=round(float(j["format"]["duration"]) * FPS), w=w, h=h, video=v is not None, audio=a,
                fps=fps, own=round(float(j["format"]["duration"]) * fps), depth=depth)


class Files:
    def __init__(self, sizes=None):
        self.ids, self.info, self.sizes = {}, {}, sizes or SIZE

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
            media += ('<audio><samplecharacteristics><samplerate>48000</samplerate>'
                      f'<depth>{p.get("depth", 16)}</depth></samplecharacteristics><channelcount>2</channelcount>'
                      '</audio>')
        fr = p.get("fps", FPS)                    # (the file at its own rate and length, as Premiere writes it --
        return (f'<file id="{fid}"><name>{escape(os.path.basename(path))}</name><pathurl>{pathurl(path)}</pathurl>'
                f'{rate(fr)}<duration>{p.get("own", p["frames"])}</duration><media>{media}</media></file>')   # the
                                                  # clips stay in the sequence's 50ths)

    def fit(self, path, fmt):
        """Basic Motion scale (percent) that fits a source into the sequence frame."""
        p = self.info.get(path) or probe(path)
        W, H = self.sizes[fmt]
        return round(100 * min(W / max(p["w"], 1), H / max(p["h"], 1)), 3)


def rate(fps=FPS):
    return f"<rate><timebase>{fps}</timebase><ntsc>FALSE</ntsc></rate>"


def scale_filter(pct):
    return ('<filter><effect><name>Basic Motion</name><effectid>basic</effectid><effectcategory>motion</effectcategory>'
            '<effecttype>motion</effecttype><mediatype>video</mediatype><parameter authoringApp="PremierePro">'
            f'<parameterid>scale</parameterid><name>Scale</name><valuemin>0</valuemin><valuemax>1000</valuemax>'
            f'<value>{pct:g}</value></parameter></effect></filter>')


def levels_filter(points):
    """Premiere's Audio Levels with keyframes: points [(clip frame, gain as a linear factor)]."""
    kf = "".join(f"<keyframe><when>{int(w)}</when><value>{v:.4f}</value></keyframe>" for w, v in points)
    return ('<filter><effect><name>Audio Levels</name><effectid>audiolevels</effectid><effectcategory>audiolevels'
            '</effectcategory><effecttype>audiolevels</effecttype><mediatype>audio</mediatype>'
            '<parameter authoringApp="PremierePro">'
            '<parameterid>level</parameterid><name>Level</name><valuemin>0</valuemin><valuemax>3.98109</valuemax>'
            f'<value>{points[0][1]:.4f}</value>{kf}</parameter></effect></filter>')


def song_levels(project, plan, gain_db=0.0, step=0.1, tol=0.35):
    """The song's level over the film as the render made it -- reel.json "music_shape" (render.shape_envelope: down
    under the talking, the build through the event), dipped under each cheer (render.NAT_DUCK), and the final mix's
    gain -- as keyframes [(film seconds, linear factor)], thinned to where the curve turns (within `tol` dB)."""
    import numpy as np

    from . import render
    t = np.arange(0.0, plan["video_length"] + step, step)
    db = np.zeros_like(t)
    if project.config.get("music_shape"):
        db = render.shape_envelope(plan, project.config["music_shape"], t)[0].astype(float)
    duck = 20 * np.log10(render.NAT_DUCK)
    for n in plan.get("nat") or []:                       # (eased over a tenth of a second either side)
        a, b = n["at"], n["at"] + n["length"]
        w = np.clip(np.minimum((t - (a - 0.1)) / 0.1, ((b + 0.1) - t) / 0.1), 0, 1)
        db = db + duck * w
    db = db + gain_db
    keep = [0]
    for i in range(1, len(t) - 1):                        # a point wherever a straight line from the last kept
        j = keep[-1]                                      # one would miss the curve by more than tol
        line = db[j] + (db[i + 1] - db[j]) * (t[j:i + 2] - t[j]) / max(t[i + 1] - t[j], 1e-9)
        if np.abs(line - db[j:i + 2]).max() > tol:
            keep.append(i)
    keep.append(len(t) - 1)
    return [(float(t[i]), float(10 ** (db[i] / 20))) for i in keep]


def clip(cid, files, path, start, end, src_in, name=None, fmt=None, enabled=True, kind="video", extra="",
         attrs="", inner=""):
    f = files.ref(path)
    flt = (scale_filter(files.fit(path, fmt)) if fmt and kind == "video" else "") + extra
    return (f'<clipitem id="{cid}"{attrs}><name>{escape(name or os.path.basename(path))}</name>'
            f'<enabled>{"TRUE" if enabled else "FALSE"}</enabled>{rate()}<start>{start}</start><end>{end}</end>'
            f'<in>{src_in}</in><out>{src_in + end - start}</out>{f}{inner}{flt}</clipitem>')


def readable(xml):
    """The xmeml indented, one element a line (text-only elements kept on theirs), its DOCTYPE kept."""
    import xml.etree.ElementTree as ET
    body = xml[xml.index("<xmeml"):]
    tree = ET.ElementTree(ET.fromstring(body))
    ET.indent(tree, space="\t")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'
            + ET.tostring(tree.getroot(), encoding="unicode") + "\n")


def stereo(cid, files, path, start, end, src_in, name, extra="", lane=1):
    """A stereo clip as Premiere writes one: a linked pair, channel 1 and channel 2 of the file, for the two
    exploded tracks of a stereo track (lane: which logical audio track, 1-based). Returns (left, right)."""
    ids = (f"{cid}-L", f"{cid}-R")
    links = "".join(f"<link><linkclipref>{ids[k]}</linkclipref><mediatype>audio</mediatype>"
                    f"<trackindex>{2 * (lane - 1) + k + 1}</trackindex><clipindex>1</clipindex>"
                    f"<groupindex>1</groupindex></link>" for k in (0, 1))
    return tuple(clip(ids[k], files, path, start, end, src_in, name, kind="audio", extra=extra,
                      attrs=' premiereChannelType="stereo"',
                      inner=f"<sourcetrack><mediatype>audio</mediatype><trackindex>{k + 1}</trackindex>"
                            f"</sourcetrack>{links}") for k in (0, 1))


def stereo_tracks(pairs):
    """Two exploded tracks (Premiere: one stereo track) from a list of (left, right) clip pairs."""
    out = []
    for k in (0, 1):
        out.append(f'<track currentExplodedTrackIndex="{k}" totalExplodedTrackCount="2" premiereTrackType="Stereo">'
                   + "".join(p[k] for p in pairs)
                   + f"<enabled>TRUE</enabled><locked>FALSE</locked><outputchannelindex>{k + 1}"
                     "</outputchannelindex></track>")
    return "".join(out)


AUDIO_HEAD = ("<numOutputChannels>2</numOutputChannels><format><samplecharacteristics><depth>16</depth>"
              "<samplerate>48000</samplerate></samplecharacteristics></format><outputs>"
              + "".join(f"<group><index>{k}</index><numchannels>1</numchannels><downmix>0</downmix><channel><index>"
                        f"{k}</index></channel></group>" for k in (1, 2)) + "</outputs>")
LEVEL_MAX = 3.98109      # Premiere's Audio Levels ceiling (+12 dB, linear)


def dissolve(start, end, alignment):
    """A Cross Dissolve transition item (a caption fading in from nothing, or out to it)."""
    return (f"<transitionitem>{rate()}<start>{start}</start><end>{end}</end><alignment>{alignment}</alignment>"
            "<effect><name>Cross Dissolve</name><effectid>Cross Dissolve</effectid><effectcategory>Dissolve"
            "</effectcategory><effecttype>transition</effecttype><mediatype>video</mediatype><wipecode>0</wipecode>"
            "<wipeaccuracy>100</wipeaccuracy><startratio>0</startratio><endratio>1</endratio><reverse>FALSE"
            "</reverse></effect></transitionitem>")


def run(project, fmt, cut=None) -> str:
    cut = cut or fmt
    plan = project.load(f"plan_{cut}.json")
    fmt = plan.get("format", fmt)
    song = plan.get("song") or project.load(project.music_file)["song"]
    manifest = project.load(os.path.join("out", f"{cut}_shots.json"))      # final file per shot, from render
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    from .render import size
    W, H = size(project, fmt)                      # the sequence is the reel's size, as the rendered shots are
    media = project.path("out", f"{cut}_media"); os.makedirs(media, exist_ok=True)
    files = Files({fmt: (W, H)})
    copies = []
    v1, v2, n, rec, written = [], [], 0, 0, set()
    for i, s in enumerate(plan["shots"]):
        u = s["unit"]
        start = round(s["rec_in"] * FPS)
        end = start + s["frames"]                  # same beat-exact cut points as the render
        src = os.path.join(project.folder, u["source"])
        base = os.path.basename(manifest[i])
        rendered = (any(s.get(k) for k in ("title", "split", "stutter", "card", "photo", "photos")) or bool(s.get("fx")) or bool(s.get("logo")) or
                    bool(project.config.get("lut")) or bool(s.get("stab")) or bool(u.get("rotate")) or bool(s.get("crop")) or "_o" in base
                    or "_i" in base)
        label = f'{i + 1:03d} {u["id"]} {s["kind"]}' + (f' [{"+".join(s["fx"])}]' if s.get("fx") else "")
        if rendered:
            keep = os.path.join(media, f'{i + 1:03d}_{u["id"].replace("#", "-")}_{s["kind"]}'
                                       f'{"_" + "+".join(s["fx"]) if s.get("fx") else ""}'
                                       f'{"_" + s["transition"] if s.get("transition") else ""}.mp4')
            copies.append((project.path(manifest[i]), keep))
            written.add(os.path.basename(keep))
            n += 1; v1.append(partial(clip, f"v1-{n}", files, keep, start, end, 0, label))
            if not (s.get("photo") or s.get("photos")):    # (a still made into a scene: no camera file under it)
                n += 1; v2.append(partial(clip, f"v2-{n}", files, src, start, end, round(s["src_in"] * FPS),
                                       f"{label} (source)", fmt=fmt, enabled=False))
        else:
            n += 1; v1.append(partial(clip, f"v1-{n}", files, src, start, end, round(s["src_in"] * FPS), label, fmt=fmt))
        rec = end
    total = rec
    with ThreadPoolExecutor(8) as ex:              # the rendered shots, copied (re-tagged) beside the XML: many small
        list(ex.map(lambda c: hw.retag(*c), copies))   # files, so several at once
    v1, v2 = [f() for f in v1], [f() for f in v2]  # (their <file> entries are probed once the copies exist)
    for f in os.listdir(media):                    # an earlier cut's shots go: the folder is this cut's media alone
        if f.endswith(".mp4") and f not in written:
            os.remove(os.path.join(media, f))
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
    if project.config.get("xml_package"):                 # reel.json "xml_package": the timeline to hand on, too
        package(project, fmt, cut)
    return out


def _lufs(path, a=None, d=None):
    """Integrated loudness of a file (or a stretch of it)."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", *(["-ss", f"{a:.3f}", "-t", f"{d:.3f}"] if a is not None else []),
           "-i", path, "-af", "ebur128", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    return float(err.rsplit("I:", 1)[1].split("LUFS")[0])


def package(project, fmt, cut=None) -> str:
    """A Premiere-ready timeline to hand on (reel.json "xml_package"), all in one folder (reel.json "premiere_dir",
    else out/premiere) to upload as it is: <name>.xml, README.txt and
        V1  plain shots from footage/ (masters.run: whole clips colour-corrected, steadied where they shook) at
            their in-points, with handles; shots with built effects (speed ramps, fitted picks, spatial photos and
            the photo grid, split screens, titles, the end card) baked, copied into effects/
        V2  the captions, transparent PNGs (captions/), each for its time -- never baked into the shots
        A1  the song alone, from its start point (audio/): swap it for another master of the song
        A2  the film's own sound alone (audio/our_sound.wav): the cheers at each rise, the place's ambience, the
            close's own sound -- levelled as in the film
    Premiere asks once for a missing file and finds the rest."""
    import shutil
    import textwrap

    from . import cards, render
    from .render import size
    cut = cut or fmt
    plan = project.load(f"plan_{cut}.json")
    fmt = plan.get("format", fmt)
    W, H = size(project, fmt)
    manifest = project.load(os.path.join("out", f"{cut}_shots.json"))
    from .masters import package_dir
    root = package_dir(project)
    for d in ("audio", "captions"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    files = Files({fmt: (W, H)})
    mfile = os.path.join(root, "footage", "masters.json")  # (masters.run: the graded, steadied whole clips)
    masters = json.load(open(mfile)) if os.path.exists(mfile) else {}
    os.makedirs(os.path.join(root, "effects"), exist_ok=True)
    v1, used, rec = [], [], 0
    for i, s in enumerate(plan["shots"]):
        start = round(s["rec_in"] * FPS)
        end = start + s["frames"]
        u = s["unit"]
        plain = (not any(s.get(k) for k in ("fx", "split", "photos", "photo", "card", "title", "transition", "reverse",
                                            "blend", "stutter", "logo", "fit"))
                 and (s.get("speed") or 1.0) == 1.0 and u["id"] in masters)
        if plain:                                          # a plain shot: the master, at its in-point, with handles
            m = masters[u["id"]]
            f = os.path.join(root, m["file"])
            v1.append(clip(f"v1-{i + 1}", files, f, start, end, round((s["src_in"] - m["t0"]) * FPS),
                           f'{i + 1:03d} {u["id"]} @ {s["src_in"]:.1f}s', fmt=fmt))
            used.append((f, "steadied" if m.get("steady") else "graded"))
        else:                                              # an effect shot: as rendered, baked (copied here)
            what = ("photo grid" if s.get("photos") else "end card" if s.get("card") else
                    "speaker grid" if s.get("split") and len(s["split"]["panels"]) > 4 else
                    "side by side" if s.get("split") else
                    f'{u["id"]} @ {s.get("src_in", 0):.1f}s' + (f' {"+".join(s["fx"])}' if s.get("fx") else "")
                    + (" title" if s.get("title") else ""))
            safe = "".join(ch if ch.isalnum() or ch in "-_. @+" else "_" for ch in what).replace(" ", "_")
            f = os.path.join(root, "effects", f"{i + 1:03d}_{safe}.mp4")
            srcf = project.path(manifest[i])
            if not os.path.exists(f) or os.path.getsize(f) != os.path.getsize(srcf):
                shutil.copyfile(srcf, f)
            v1.append(clip(f"v1-{i + 1}", files, f, start, end, 0, f"{i + 1:03d} {what} (baked)"))
            used.append((f, "baked"))
        rec = end
    total = rec
    cuts = [round(sh["rec_in"] * FPS) for sh in plan["shots"]]
    v2 = []
    fade = round(0.6 * FPS)                                # (the film fades each caption over 0.6 s)
    for k, c in enumerate(project.config.get("captions") or []):
        img = cards.caption(project, c, (W, H))
        dst = os.path.join(root, "captions", f"caption_{k + 1:02d}.png")
        shutil.copyfile(img, dst)
        a, b = round(float(c["at"]) * FPS), round(float(c["to"]) * FPS)
        b = min([x for x in cuts if b - 5 <= x < b] or [b])   # (ending a hair past a cut: on the cut)
        a = max([x for x in cuts if a < x <= a + 5] or [a])   # (starting a hair before one: on it)
        f_ = min(fade, (b - a) // 3)
        v2 += [dissolve(a, a + f_, "start-black"), clip(f"v2-{k + 1}", files, dst, a, b, 0,
                                                       f'caption {k + 1}: {c["text"][:40]}'),
               dissolve(b - f_, b, "end-black")]
        used.append((dst, f"caption {k + 1}"))
    song = plan.get("song") or project.load(project.music_file)["song"]
    song_dst = os.path.join(root, "audio", os.path.basename(song))
    if not os.path.exists(song_dst) or os.path.getsize(song_dst) != os.path.getsize(song):
        shutil.copyfile(song, song_dst)
    used.append((song_dst, "the song (A1)"))
    render.PROJECT[0] = project
    gain_film = 0.0                                        # the gain the film's whole mix took to its loudness
    mix = project.path("work", cut, "mix.wav")
    if os.path.exists(mix):
        try:
            gain_film = render.I - _lufs(mix)
        except (IndexError, ValueError):
            pass
    m0 = round(plan["music_start"] * FPS)                  # A1: the song, its level keyframed as the film mixed it
    m1 = m0 + round(plan["music_length"] * FPS)
    pts = song_levels(project, plan, gain_film)            # (keyframe times: the clip's own media frames)
    kf = [(min(m0 + round(t * FPS), m1), min(v, LEVEL_MAX)) for t, v in pts]
    a1 = [stereo("a1-1", files, song_dst, 0, m1 - m0, m0, f'{os.path.basename(song)} (from {plan["music_start"]:.2f}s)',
                 extra=levels_filter(kf), lane=1)]
    a2, capped = [], []                                    # A2: each cheer, the camera's own sound at the rise
    for k, n in enumerate(plan.get("nat") or []):
        uid = os.path.splitext(os.path.basename(n["source"]))[0]
        m = masters.get(uid)
        if not m:
            continue
        f = os.path.join(root, m["file"])
        st, ln = round(n["at"] * FPS), round(n["length"] * FPS)
        src_in = round((n["src_t"] - m["t0"]) * FPS)
        ln = min(ln, probe(f)["frames"] - src_in)         # (never past the end of its file)
        g = 10 ** ((render.NAT_LUFS - _lufs(f, n["src_t"] - m["t0"], ln / FPS) + gain_film) / 20)
        if g > LEVEL_MAX:
            capped.append((uid, 20 * np.log10(g / LEVEL_MAX)))
            g = LEVEL_MAX
        fade_out = min(round(1.2 * FPS), ln // 2)
        a2.append(stereo(f"a2-{k + 1}", files, f, st, st + ln, src_in, f"{uid} sound at the rise", lane=2,
                         extra=levels_filter([(src_in, 0.0), (src_in + round(0.15 * FPS), g),
                                              (src_in + ln - fade_out, g), (src_in + ln, 0.0)])))
        used.append((f, "cheer sound (A2)"))
    a3 = []                                                # A3: the place's ambience, baked as the film has it
    amb = project.config.get("ambience")
    if amb:
        amb_f = os.path.join(root, "audio", "ambience.wav")
        silent = os.path.join(root, "audio", ".silence.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t",
                        f"{plan['music_start'] + plan['music_length'] + 1:.3f}", silent], check=True)
        tmp = amb_f + ".raw.wav"
        render.mix_audio(silent, dict(plan, nat=[], stinger_at=plan["video_length"]), None, tmp)   # (the bed alone)
        os.remove(silent)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", f"volume={gain_film:.2f}dB",
                        "-c:a", "pcm_s24le", amb_f], check=True)
        os.remove(tmp)
        a3.append(stereo("a3-1", files, amb_f, 0, total, 0,
                         f'ambience ({os.path.splitext(amb["source"])[0]} {amb["from"]:g}-{amb["to"]:g} s, looped)',
                         lane=3))
        used.append((amb_f, "the ambience bed (A3)"))
    fmt_xml = (f'<format><samplecharacteristics>{rate()}<width>{W}</width><height>{H}</height>'
               f'<pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance>'
               f'</samplecharacteristics></format>')
    name = project.out_stem(cut, fmt).replace("_", " ")
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
           f'<sequence id="seq-{cut}"><name>{escape(name)}</name><duration>{total}</duration>'
           f'{rate()}<media><video>{fmt_xml}<track>{"".join(v1)}</track><track>{"".join(v2)}</track></video>'
           f'<audio>{AUDIO_HEAD}{stereo_tracks(a1)}{stereo_tracks(a2)}{stereo_tracks(a3)}</audio></media>'
           f'</sequence></xmeml>\n')
    out = os.path.join(root, f"{project.out_stem(cut, fmt)}.xml")
    with open(out, "w", newline="\r\n") as f:              # (indented like Premiere's own exports, Windows
        f.write(readable(xml))                             # line endings: it opens in any text editor)
    seen = {}
    for f, what in used:
        seen.setdefault(f, what)
    gb = sum(os.path.getsize(f) for f in seen) / 1e9
    song_note = project.config.get("song_credit") or ("the song's title, artist and licence are not recorded here "
                                                      "-- ask the producer before publishing")
    cams = sorted({(m.get("camera"), p_["fps"]) for m in masters.values()
                   for p_ in [probe(os.path.join(root, m["file"]))]})
    P = lambda t, ind=0: textwrap.wrap(t, 96, initial_indent=" " * ind, subsequent_indent=" " * ind)
    T = lambda tag, t: textwrap.wrap(t, 96, initial_indent=f"  {tag}  ", subsequent_indent=" " * 6)
    title = f"PREMIERE PACKAGE: {name.upper()}"
    lines = [title, "=" * len(title), "",
             f"Sequence   {W} x {H}, {FPS} fps, {int(total / FPS // 60)}:{total / FPS % 60:04.1f}",
             *textwrap.wrap(f"Song       {os.path.basename(song)} ({song_note})", 96,
                            subsequent_indent=" " * 11),
             "Cameras    " + ", ".join(f"{c} {f} fps" for c, f in cams), "",
             "OPENING IT", "----------",
             *P("Download the whole folder with Google Drive for desktop. If you download from the website "
                "instead, Drive splits it into several drive-download-*.zip files: extract every one of them "
                "into the same folder.", 2), "",
             *P("Then in Premiere: File > Import, and choose the .xml file.", 2), "",
             "THE TIMELINE", "------------",
             *T("V1", "The cut. Plain shots use the clips in footage\\ (whole clips, colour-corrected, and "
                      "steadied where the camera shook), with handles to trim. They are full frame; the film "
                      "punches in slightly on some, so reframe to taste. Shots with built effects are baked "
                      "into effects\\: \"rise\" is a baptism with its speed ramp, \"fit\" is a shot slowed to "
                      "fit its slot, plus the titles, speaker grid, side-by-sides, spatial photo grid and end "
                      "card."), "",
             *T("V2", "The captions (captions\\), transparent PNGs with a 0.6 s cross dissolve in and out. "
                      "Edit or delete freely. The finished films have no captions."), "",
             *T("A1", "The song on its own (audio\\), its volume keyframed as the film mixed it: down under "
                      "the talking, building through the baptisms, dipping under each cheer. Replace Footage "
                      "with another master of the same song and it stays in sync. The film also softens the "
                      "music under the talking (a 2.5 kHz low-pass): add a Lowpass there to match."), "",
             *T("A2", "The cameras' own sound at each baptism (the cheers), faded in and out at the film's "
                      "level." + "".join(f" {u} needs about {d:.0f} dB more than Premiere's +12 dB keyframe "
                                         "limit: add it with Audio Gain." for u, d in capped)), "",
             *T("A3", "The beach ambience (audio\\ambience.wav): water lapping, looped and levelled as in "
                      "the film."), "",
             "OTHER FOLDERS", "-------------",
             *P("film\\      the finished films: the 4K master and a 1080p 30 fps copy for projectors.", 2),
             *P("footage\\   also holds every other clip of the shoot, graded the same way, for recutting.", 2),
             "",
             "FILES THE TIMELINE USES", "-----------------------",
             f"  {len(seen)} files, {gb:.1f} GB"]
    rel = {f: os.path.relpath(f, root).replace(os.sep, "\\") for f in seen}
    wide = max(len(r.split("\\")[-1]) for r in rel.values())
    group = None
    for f, what in sorted(seen.items(), key=lambda x: rel[x[0]]):
        g = rel[f].split("\\")[0] if rel[f].count("\\") < 2 else "\\".join(rel[f].split("\\")[:2])
        if g != group:
            lines += ["", f"  {g}\\"]
            group = g
        nm = rel[f].split("\\")[-1]
        lines.append(f"    {nm:<{wide}}  {os.path.getsize(f) / 1e6:7.1f} MB  {what}".rstrip())
    film = project.path("out", f"{project.out_stem(cut, fmt)}.mp4")   # the finished film (and its copies
    if os.path.exists(film):                                          # made from it since) beside the timeline
        os.makedirs(os.path.join(root, "film"), exist_ok=True)
        stem = os.path.splitext(film)[0]
        for f in [film] + [os.path.join(os.path.dirname(film), x) for x in os.listdir(os.path.dirname(film))
                           if x.startswith(os.path.basename(stem) + "_") and x.endswith(".mp4")
                           and os.path.getmtime(os.path.join(os.path.dirname(film), x)) >= os.path.getmtime(film)]:
            dst = os.path.join(root, "film", os.path.basename(f))
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(f):
                shutil.copyfile(f, dst)
    with open(os.path.join(root, "README.txt"), "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[reel] premiere package: {out} ({len(v1)} shots, {len(v2)} captions; README.txt)")
    return out
