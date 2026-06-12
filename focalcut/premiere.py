"""Export a Premiere-ready FCP7 (xmeml v5) selects timeline from a focalcut library.

Each couple segment becomes a linked video+audio clipitem laid end-to-end. Mirrors
ButterCut's fcp7.rb where it matters:
  - sequence + clipitem in/out are in the SEQUENCE timebase (not the file's) — avoids
    Premiere rendering blank frames at clip boundaries with mixed rates;
  - portrait sequence with square pixels, anamorphic FALSE;
  - rotation handled by a Basic Motion filter (FCP7 rotation sign is negated vs the
    metadata convention) so landscape-stream + rotation clips display upright;
  - WSL /mnt/<d>/... -> <D>:/... file URLs for Premiere on Windows.
"""
from __future__ import annotations
import hashlib, json, os, subprocess
from xml.sax.saxutils import escape


def _probe(path: str) -> dict:
    o = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_streams", "-show_format", path], capture_output=True, text=True).stdout
    d = json.loads(o)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if "tags" in v and "rotate" in v["tags"]:
        rot = int(v["tags"]["rotate"])
    num, den = (v.get("r_frame_rate", "50/1").split("/") + ["1"])[:2]
    return {"w": int(v["width"]), "h": int(v["height"]), "rot": rot,
            "fps_num": int(num), "fps_den": int(den or 1),
            "duration": float(d["format"]["duration"]),
            "audio_rate": int(a["sample_rate"]) if a else 48000, "has_audio": a is not None}


def _file_url(path: str, windows: bool) -> str:
    p = os.path.abspath(path)
    if windows and p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        win = f"{p[5].upper()}:{p[6:]}"
        return "file://localhost/" + win.replace(" ", "%20")
    from urllib.request import pathname2url
    return "file://localhost" + pathname2url(p)


def _motion_filter(rot: int, seq_w: int, seq_h: int, clip_w: int, clip_h: int) -> str:
    """Basic Motion (scale + rotation) — replicates ButterCut build_motion_filter."""
    if rot == 90:
        eff_w, eff_h, fcp_rot = clip_h, clip_w, -90
    elif rot in (270, -90):
        eff_w, eff_h, fcp_rot = clip_h, clip_w, 90
    else:
        eff_w, eff_h, fcp_rot = clip_w, clip_h, 0
    seq_portrait = seq_h > seq_w
    clip_landscape = eff_w > eff_h
    if seq_portrait and clip_landscape:
        scale = seq_h / eff_h * 100
    else:
        scale = min(seq_w / eff_w, seq_h / eff_h) * 100
    rot_xml = ""
    if fcp_rot != 0:
        rot_xml = (f"<parameter><parameterid>rotation</parameterid><name>Rotation</name>"
                   f"<valuemin>-8640</valuemin><valuemax>8640</valuemax><value>{fcp_rot}</value></parameter>")
    return (f"<filter><effect><name>Basic Motion</name><effectid>basic</effectid>"
            f"<effectcategory>motion</effectcategory><effecttype>motion</effecttype>"
            f"<mediatype>video</mediatype><pproBypass>false</pproBypass>"
            f"<parameter><parameterid>scale</parameterid><name>Scale</name>"
            f"<valuemin>0</valuemin><valuemax>1000</valuemax><value>{round(scale,2)}</value></parameter>"
            f"{rot_xml}"
            f"<parameter><parameterid>center</parameterid><name>Center</name>"
            f"<value><horiz>0</horiz><vert>0</vert></value></parameter></effect></filter>")


def export_selects(folder: str, clips: list[dict], out_path: str,
                   sequence_name: str, windows_paths: bool = True) -> str:
    """clips: library records (already filtered to couple clips). Lays every segment
    end-to-end as a linked V/A clipitem. Sequence matches the (display) portrait media."""
    # probe assets (cache by name)
    assets = {}
    for c in clips:
        p = os.path.join(folder, c["name"])
        assets[c["name"]] = _probe(p)

    # sequence rate/size from the first clip's display orientation
    first = assets[clips[0]["name"]]
    tb = round(first["fps_num"] / first["fps_den"])
    ntsc = "FALSE" if first["fps_den"] == 1 else "TRUE"
    dw, dh = first["w"], first["h"]
    if abs(first["rot"]) % 180 == 90:
        dw, dh = dh, dw                      # display dims
    seq_w, seq_h = dw, dh

    vitems, aitems, links = [], [], []
    tl = 0
    idx = 0
    for c in clips:
        a = assets[c["name"]]
        p = os.path.join(folder, c["name"])
        url = _file_url(p, windows_paths)
        fid = "file-" + hashlib.md5(os.path.abspath(p).encode()).hexdigest()[:16]
        total_frames = round(a["duration"] * tb)
        base = os.path.splitext(c["name"])[0]
        fname = escape(c["name"])
        file_el = (f'<file id="{fid}"><name>{fname}</name><pathurl>{escape(url)}</pathurl>'
                   f'<rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>'
                   f'<duration>{total_frames}</duration>'
                   f'<timecode><rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>'
                   f'<frame>0</frame><displayformat>NDF</displayformat></timecode>'
                   f'<media><video><samplecharacteristics>'
                   f'<rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>'
                   f'<width>{a["w"]}</width><height>{a["h"]}</height>'
                   f'<anamorphic>FALSE</anamorphic><pixelaspectratio>square</pixelaspectratio>'
                   f'<fielddominance>none</fielddominance></samplecharacteristics></video>'
                   f'<audio><samplecharacteristics><samplerate>{a["audio_rate"]}</samplerate>'
                   f'<sampledepth>16</sampledepth></samplecharacteristics></audio></media></file>')
        mfilter = _motion_filter(a["rot"], seq_w, seq_h, a["w"], a["h"]) if a["rot"] != 0 else ""

        for seg in c["segments"]:
            idx += 1
            s_in = round(seg[0] * tb)
            s_out = round(seg[1] * tb)
            dur = s_out - s_in
            t_start, t_end = tl, tl + dur
            tl = t_end
            vid, aid = f"clipitem-video-{idx}", f"clipitem-audio-{idx}"
            link = (f'<link><linkclipref>{vid}</linkclipref><mediatype>video</mediatype>'
                    f'<trackindex>1</trackindex><clipindex>{idx}</clipindex></link>'
                    f'<link><linkclipref>{aid}</linkclipref><mediatype>audio</mediatype>'
                    f'<trackindex>1</trackindex><clipindex>{idx}</clipindex><groupindex>1</groupindex></link>')
            vitems.append(
                f'<clipitem id="{vid}"><name>{escape(base)}</name><enabled>TRUE</enabled>'
                f'<duration>{dur}</duration><start>{t_start}</start><end>{t_end}</end>'
                f'<in>{s_in}</in><out>{s_out}</out>{file_el}'
                f'<sourcetrack><mediatype>video</mediatype><trackindex>1</trackindex></sourcetrack>'
                f'{mfilter}{link}</clipitem>')
            if a["has_audio"]:
                aitems.append(
                    f'<clipitem id="{aid}"><name>{escape(base)}</name><enabled>TRUE</enabled>'
                    f'<duration>{dur}</duration><start>{t_start}</start><end>{t_end}</end>'
                    f'<in>{s_in}</in><out>{s_out}</out><file id="{fid}"/>'
                    f'<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>'
                    f'<channelcount>2</channelcount>{link}</clipitem>')

    seq_dur = tl
    rate = f'<rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>'
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="5">\n'
        f'<sequence id="sequence-1"><name>{escape(sequence_name)}</name>'
        f'<duration>{seq_dur}</duration>{rate}<in>0</in><out>{seq_dur}</out>'
        f'<timecode>{rate}<frame>0</frame><displayformat>NDF</displayformat></timecode>'
        f'<media><video><format><samplecharacteristics>{rate}'
        f'<width>{seq_w}</width><height>{seq_h}</height><anamorphic>FALSE</anamorphic>'
        f'<pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance>'
        f'</samplecharacteristics></format><track>{"".join(vitems)}</track></video>'
        f'<audio><numOutputChannels>2</numOutputChannels>'
        f'<format><samplecharacteristics><samplerate>{first["audio_rate"]}</samplerate>'
        f'<sampledepth>16</sampledepth></samplecharacteristics></format>'
        f'<track>{"".join(aitems)}</track></audio></media></sequence>\n</xmeml>\n')
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path
