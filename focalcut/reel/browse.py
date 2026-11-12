"""The source browser: one page for the whole project, whatever the format -- every export clip in a list, one at a
time in a big player over a wide, zoomable timeline (filmstrip, the moments the planner found, where the landscape and
portrait reels use it), where the editor marks as many in-out selections per clip as they like, and a final view of
every selection that plays them back to back.

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

from . import hw

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
    """{fmt: [[in, out, shot number], ...]}: the shots each plan takes from this unit, in its export's timeline."""
    return {fmt: [[round(s["src_in"] - x0, 2), round(s["src_in"] - x0 + s["length"], 2), i + 1]
                  for i, s in enumerate(p["shots"]) if s["unit"]["id"] == uid] for fmt, p in plans.items()}


TAGS_JS = r"""// per-reel picks: one picks.json serves the project's reel ("main") and its cuts. A clip's "only" gives a value to
// some reels: its key is the "add" range as written, or "<field>:<value>" (ban, prefer, open, end; "more:"); a value
// not listed is in every reel. R is the reel a page edits for ("all": every reel).
const RS=D.reels||[],RID=RS.map(r=>r.id),near2=(u,v)=>Math.abs(u-v)<0.006;
const TF=/^(ban|prefer|open|end|more):(.*)$/;
function valEq(f,x,v){if(f=="more")return true;if(x=="all"||v=="all")return x==v;const[a,b]=span(x),[c,d]=span(v);return near2(a,c)&&near2(b,d)}
function tagParts(k){const m=TF.exec(k);return m?[m[1],m[2]]:["add",k]}
function tagsOf(id,f,v){const o=(P[id]||{}).only;if(!o)return null;
 for(const k in o){const[kf,kv]=tagParts(k);if(kf==f&&valEq(f,kv,v))return Array.isArray(o[k])?o[k]:null}return null}
function setTags(id,f,v,list){const e=ent(id),o={};
 for(const k in e.only||{}){const[kf,kv]=tagParts(k);if(!(kf==f&&valEq(f,kv,v)))o[k]=e.only[k]}
 if(list&&RID.length>1&&!RID.every(r=>list.includes(r)))o[f=="add"?v:f+":"+(f=="more"?"":v)]=RID.filter(r=>list.includes(r));
 if(Object.keys(o).length)e.only=o;else delete e.only}
function isIn(id,f,v,R){if(R=="all")return true;const t=tagsOf(id,f,v);return!t||t.includes(R)}
function vals(id,f,R){return((P[id]||{})[f]||[]).filter(x=>isIn(id,f,x,R))}
// add a value for reel R (already there for other reels: R joins them); take it out of R (gone when no reel is left)
function addVal(id,f,v,R){const e=ent(id),L=e[f]=e[f]||[],x=L.find(y=>valEq(f,y,v));
 if(x===undefined){L.push(v);setTags(id,f,v,R=="all"?null:[R]);return true}
 const t=tagsOf(id,f,x);if(R=="all"){setTags(id,f,x,null);return!!t}
 if(t&&!t.includes(R)){setTags(id,f,x,[...t,R]);return true}return false}
function delVal(id,f,v,R){const e=P[id];if(!e||!e[f])return;const k=e[f].findIndex(y=>valEq(f,y,v));if(k<0)return;
 const x=e[f][k],t=(tagsOf(id,f,x)||RID).filter(r=>r!=R);
 if(R=="all"||!t.length){e[f].splice(k,1);setTags(id,f,x,null)}else setTags(id,f,x,t)}
// "more" (more shots from the clip) and "open" / "end" (the pick a reel opens / ends on: one value per clip)
function moreIn(id,R){const e=P[id]||{};return e.more&&isIn(id,"more","",R)?e.more:0}
function addMore(id,d,R){const e=ent(id),had=e.more||0,t=tagsOf(id,"more","");e.more=Math.max(0,had+d);
 if(!had)setTags(id,"more","",R=="all"?null:[R]);else if(d>0&&t&&R!="all"&&!t.includes(R))setTags(id,"more","",[...t,R])}
function roleIn(id,f,R){const e=P[id]||{};return typeof e[f]=="string"&&isIn(id,f,e[f],R)?e[f]:null}
function clearRole(id,f,R){const e=P[id];if(!e||typeof e[f]!="string")return;const t=(tagsOf(id,f,e[f])||RID).filter(r=>r!=R);
 if(R=="all"||!t.length){setTags(id,f,e[f],null);delete e[f]}else setTags(id,f,e[f],t)}
// a clip's tags whose value is gone go too
function pruneTags(e){if(!e.only)return;for(const k in e.only){const[f,v]=tagParts(k);
 const live=f=="more"?!!e.more:(f=="open"||f=="end")?typeof e[f]=="string"&&valEq(f,e[f],v):(e[f]||[]).some(x=>valEq(f,x,v));
 if(!live)delete e.only[k]}if(!Object.keys(e.only).length)delete e.only}
"""


def page(clips, picks=None, key="sources", title="Sources", root="", nav="", reels=None):
    """root: the project folder relative to the page ("../../../"), so its API calls work under any server prefix.
    reels: [{"id", "label"}] of the reels picks can be given to (the page's reel switch; none: no switch)."""
    data = json.dumps(dict(clips=clips, picks=picks or {}, key=key, tile=TILE, root=root,
                           reels=reels or [])).replace("</", "<\\/")
    return PAGE.replace("%DATA%", data).replace("%TAGS%", TAGS_JS).replace("%TITLE%", html.escape(title)).replace("%NAV%", nav).replace("%MENU%", MENU_CSS)


MENU_CSS = """.menu{display:inline-flex;gap:12px;flex-wrap:wrap;font-size:13px;margin-right:8px}
.menu a{color:var(--mut);text-decoration:none;padding:2px 0;border-bottom:2px solid transparent}
.menu a:hover{color:var(--fg)}.menu a.cur{color:var(--fg);font-weight:600;border-bottom-color:currentColor}
.menu a[hidden]{display:none}"""


def menu(project, current: str) -> str:
    """The pages' shared menu: this project's sources page and reel reviews (relative links, so it works opened as
    files too) and, when served, every project's index ("All projects", shown by the page once the server answers)."""
    rev = project.path("review")
    cuts = [c for c in ("landscape", "portrait") if os.path.exists(project.path(f"plan_{c}.json"))]
    cuts += sorted(f[:-5] for f in (os.listdir(rev) if os.path.isdir(rev) else [])
                   if f.endswith(".html") and f[:-5] not in cuts and f != "sources.html")
    links = [("sources", "Sources")] + [(c, c.capitalize()) for c in cuts]
    frames = [("main", "Frame portrait")] if os.path.exists(project.path("plan_portrait.json")) else []
    from .project import Project
    frames += [(c, f"Frame {c} portrait") for c in (Project(project.folder).config.get("cuts") or {})
               if os.path.exists(project.path(f"plan_{c}_portrait.json"))]
    return ('<nav class=menu>' + "".join(f'<a href="{quote(k)}.html"{" class=cur" if k == current else ""}>'
                                         f'{html.escape(label)}</a>' for k, label in links)
            + "".join(f'<a href="sources.html#frame={quote(r)}" title="the portrait cut shot by shot, its crop over the '
                      f'picture: drag to pan, G then click to follow a dancer">{html.escape(label)}</a>'
                      for r, label in frames)
            + '<a href="/" id=allp hidden>All projects</a></nav>')


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


PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>%TITLE%</title>
<style>
:root{--bg:#0e0f12;--panel:#16181d;--card:#1d2027;--line:#2c303a;--fg:#e8e8ea;--mut:#8b8f99;--sel:#3ddc84;
--io:#ffd23f;--land:#4cc9f0;--port:#f72585;--mom:#7b8cff;--ban:#ff5c5c;--ph:#fff}
@media (prefers-color-scheme:light){:root:not([data-theme=dark]){--bg:#f3f3f5;--panel:#fff;--card:#f7f7f9;
--line:#d9dbe0;--fg:#15161a;--mut:#5d6270;--sel:#139a52;--io:#b88a00;--land:#087ea4;--port:#c2185b;--mom:#4b5bd6;
--ban:#d32f2f;--ph:#000}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif;display:grid;
grid-template-rows:auto 1fr;overflow:hidden}
button,select{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:4px 10px;cursor:pointer}button:hover{border-color:var(--mut)}button.on{background:var(--fg);color:var(--bg)}
button.go{background:var(--sel);color:#000;border-color:var(--sel);font-weight:600}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;font-size:11px;
font-family:ui-monospace,monospace}
header{display:flex;gap:10px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);
background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0 8px 0 0}.pj{display:inline-flex;gap:4px}.pj button{padding:2px 8px;font-size:12px}%MENU%#saved{color:var(--mut);font-size:12px}#saved.ok{color:var(--sel)}
#saved.err{color:var(--ban)}.sp{flex:1}
#views{display:flex;gap:4px}#counts{color:var(--mut);font-size:12px}#counts b{color:var(--fg)}
#counts .y b{color:var(--sel)}#counts .f b{color:var(--io)}#counts .x b{color:var(--ban)}
#clipsview{display:grid;grid-template-columns:290px 1fr;min-height:0}
@media (max-width:860px){body{overflow:auto;height:auto}#clipsview{grid-template-columns:1fr}
#side{max-height:38vh;border-right:0;border-bottom:1px solid var(--line)}#main{overflow:visible}}
#side{border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
#filters{position:sticky;top:0;background:var(--panel);padding:8px;display:flex;gap:4px;flex-wrap:wrap;
border-bottom:1px solid var(--line);z-index:2}#filters button{padding:2px 8px;font-size:12px}
.ci{display:grid;grid-template-columns:96px 1fr;gap:8px;padding:6px 8px;border-bottom:1px solid var(--line);
cursor:pointer;align-items:center}.ci:hover{background:var(--card)}.ci.cur{background:var(--card);
box-shadow:inset 3px 0 var(--io)}
.ci .th{width:96px;height:54px;background-size:auto 54px;border-radius:4px;background-color:#000;position:relative}
.ci .th b{position:absolute;right:3px;bottom:3px;background:var(--sel);color:#000;border-radius:8px;padding:0 6px;
font-size:11px}.ci .nm{font-weight:600;word-break:break-all;font-size:12px}
.ci .meta{color:var(--mut);font-size:11px}.ci.seen .th::after{content:"✓";position:absolute;left:3px;top:2px;color:#fff;font-size:11px;text-shadow:0 0 3px #000}
.chip{display:inline-block;padding:0 6px;border-radius:8px;font-size:10px;background:var(--line);color:var(--fg)}
#main{display:flex;flex-direction:column;min-height:0;overflow:auto;padding:10px 14px;gap:8px}
#vwrap{background:#000;border-radius:8px;position:relative;height:min(50vh,560px);flex:none;overflow:hidden}
#vid,#still{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
#still{background-repeat:no-repeat;display:none;image-rendering:auto}
#still.on{display:block}
#tc{position:absolute;left:8px;top:8px;background:#000a;color:#fff;padding:2px 8px;border-radius:4px;
font:600 14px ui-monospace,monospace}
#cliphead{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}#cliphead h2{font-size:16px;margin:0}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
#io{font:12px ui-monospace,monospace;color:var(--io);min-width:16em}
.tl{position:relative;user-select:none;touch-action:none;border-radius:6px;background:var(--card);
border:1px solid var(--line)}
#ov{height:30px;cursor:pointer;overflow:hidden}#ov canvas{position:absolute;inset:0;width:100%;height:100%;opacity:.55}
#tl{cursor:crosshair}
#ruler{height:20px;position:relative;border-bottom:1px solid var(--line);cursor:pointer}
#strip{height:80px;position:relative;cursor:pointer}#strip canvas{position:absolute;inset:0;width:100%;height:100%}
.lane{position:relative;border-top:1px solid var(--line)}.lane .lab{position:absolute;left:4px;top:50%;
transform:translateY(-50%);font-size:10px;color:var(--mut);pointer-events:none;z-index:1;text-transform:uppercase;
letter-spacing:.05em}
#lsel{height:34px;cursor:copy}#lland,#lport{height:18px;cursor:cell}
.tl i{position:absolute;display:block;font-style:normal}
.tk{top:12px;bottom:0;width:1px;background:var(--line)}.tk.big{top:6px;background:var(--mut)}
.lb{top:1px;font-size:10px;color:var(--mut);transform:translateX(3px);white-space:nowrap}
.mom{top:0;bottom:0;width:2px;background:var(--mom);opacity:.85;pointer-events:none}
.blk{top:3px;bottom:3px;border-radius:4px;background:color-mix(in srgb,var(--sel) 45%,transparent);
border:2px solid var(--sel);cursor:grab;font-size:11px;color:var(--fg);padding-left:4px;overflow:hidden;
white-space:nowrap;font-weight:600}.blk.act{background:color-mix(in srgb,var(--sel) 75%,transparent);color:#000}
.stars{white-space:nowrap;margin-right:6px}.stars b{cursor:pointer;color:var(--io);font-size:15px;padding:0 1px}
.stars b:hover{transform:scale(1.2);display:inline-block}.stars sub{font-size:9px;color:var(--mut);font-weight:400;margin-left:1px}
.blk.away{opacity:.35;border-style:dashed;background:transparent}
#reelsw{display:inline-flex;gap:2px;align-items:center;padding-left:8px;border-left:1px solid var(--line)}#reelsw:empty{display:none}
#reelsw button{padding:2px 8px;font-size:12px}#reelsw b{font-size:11px;color:var(--mut);font-weight:500;margin-right:4px}
.rk{font-size:11px;color:var(--mut);margin-right:8px;white-space:nowrap;cursor:pointer}.rk input{vertical-align:-2px}
#preels button.on{background:var(--sel);color:#000;border-color:var(--sel)}
.blk .h{position:absolute;top:0;bottom:0;width:5px;cursor:ew-resize;background:var(--sel);opacity:0}.blk .h.l{left:0}.blk .h.r{right:0}
#lsel .blk:hover .h{opacity:.9}
.pend{top:3px;bottom:3px;border:2px dashed var(--io);background:color-mix(in srgb,var(--io) 22%,transparent);
border-radius:4px;pointer-events:none}
.pt{top:2px;bottom:2px;width:3px;background:var(--sel);border-radius:2px}
.ban{top:0;bottom:0;background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--ban) 40%,transparent) 0 6px,
transparent 6px 12px);pointer-events:none}
.use{top:2px;bottom:2px;border-radius:3px;opacity:.9;font-size:9px;color:#000;overflow:hidden;padding-left:2px}
#lland .use{background:var(--land)}#lport .use{background:var(--port)}.use{cursor:copy}.use:hover{outline:2px solid var(--fg)}
.use.out{opacity:.25;text-decoration:line-through}.use.out .ux{display:none!important}
.use .ux{position:absolute;right:0;top:0;bottom:0;width:13px;line-height:14px;text-align:center;font-size:10px;font-weight:700;
background:#0006;color:#fff;cursor:pointer;display:none}.use:hover .ux{display:block}.use .ux:hover{background:var(--ban,#c0392b)}
.rub{top:0;bottom:0;background:color-mix(in srgb,#c0392b 30%,transparent);border:1px solid #c0392b;pointer-events:none;z-index:2}
.empty{left:90px;top:50%;transform:translateY(-50%);color:var(--mut);font-size:11px;pointer-events:none}
.ph{top:0;bottom:0;width:2px;background:var(--ph);pointer-events:none;z-index:3;margin-left:-1px}
.inm,.outm{top:0;bottom:0;width:2px;background:var(--io);pointer-events:none;z-index:2}
.win{top:0;bottom:0;border:2px solid var(--io);border-radius:4px;pointer-events:none}
#hover{position:fixed;display:none;pointer-events:none;z-index:9;background:#000;border:1px solid var(--line);
border-radius:6px;padding:3px;color:#fff;font:11px ui-monospace,monospace;text-align:center}
#hover div{width:192px;height:108px;background-repeat:no-repeat;border-radius:3px}
#mysel table,#allsel table{border-collapse:collapse;width:100%}
td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums}
th{color:var(--mut);font-weight:500;font-size:11px}tr.act td{background:var(--card)}
#keys{color:var(--mut);font-size:12px;line-height:1.9}
#selview{display:none;grid-template-columns:minmax(0,1fr);overflow:auto;padding:12px 16px;gap:12px;align-content:start}
#selhead{display:flex;gap:10px;align-items:center;flex-wrap:wrap}#selhead h2{margin:0;font-size:16px}
#ped{display:none;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
#pvwrap{background:#000;border-radius:8px;position:relative;height:min(42vh,460px);overflow:hidden}
#pv,#pstill{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}#pstill{display:none;background-repeat:no-repeat}
#pstill.on{display:block}.grp{display:inline-flex;gap:2px;align-items:center;padding-left:8px;border-left:1px solid var(--line);
color:var(--mut);font-size:11px}.grp button{padding:3px 7px;font-size:12px}
#pruler{height:18px;position:relative;border-bottom:1px solid var(--line)}#pruler .lb{color:var(--fg);font-weight:600}
#pstrip{height:64px;position:relative;cursor:crosshair}#pstrip canvas{position:absolute;inset:0;width:100%;height:100%}
#pstrip .blk{top:2px;bottom:2px;background:transparent;border-width:3px;color:#fff;text-shadow:0 0 3px #000;cursor:grab}
#pstrip .blk .h{width:12px;background:var(--sel);opacity:.0}#pstrip .blk .h:hover{opacity:.6}
.dim{top:0;bottom:0;background:#000b;pointer-events:none}.oth{top:40%;bottom:40%;background:var(--sel);opacity:.5;pointer-events:none}
#seq i.new{background:var(--mut)}#seq i.cut{background:var(--land);opacity:.45}#seq i.cut.p{background:var(--port)}
#pfmts{display:inline-flex;gap:2px}.card [data-del]{opacity:0}.card:hover [data-del],.card [data-del]:focus{opacity:1}.card .tag{font-size:10px;border-radius:6px;padding:0 5px;background:var(--sel);color:#000}
.card.cut .tag{background:var(--line);color:var(--fg)}.card{cursor:pointer}#pvtc{position:absolute;left:8px;top:8px;background:#000a;color:#fff;
padding:2px 8px;border-radius:4px;font:600 13px ui-monospace,monospace}
#seq{display:flex;height:26px;border-radius:6px;overflow:hidden;border:1px solid var(--line)}
#seq i{display:block;height:100%;border-right:1px solid var(--bg);background:var(--sel);opacity:.75;cursor:pointer}
#seq i:hover,#seq i.act{opacity:1;background:var(--io)}
#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.card.act{outline:2px solid var(--io)}
.card .ck{position:absolute;right:4px;top:4px;width:18px;height:18px;margin:0;cursor:pointer;opacity:0;accent-color:var(--io)}
.card:hover .ck,.card .ck:focus,.card .ck:checked,#cards.picking .ck{opacity:1}.card.chk{outline:2px solid var(--io);background:var(--sel2,var(--panel))}
.card.chk .th{opacity:.7}#seq i.chk{opacity:1;background:var(--io)}
#bulk{display:none;gap:8px;align-items:center;width:100%;padding:6px 8px;border:1px solid var(--io);border-radius:8px}#bulk.on{display:flex}
.card .th{aspect-ratio:16/9;background-color:#000;background-repeat:no-repeat;cursor:pointer;position:relative}
.card .th span{position:absolute;left:4px;top:4px;background:#000a;color:#fff;border-radius:4px;padding:0 5px;
font-size:11px}.card .bd{padding:6px 8px;display:flex;justify-content:space-between;gap:6px;align-items:center}
.card .bd div{font-size:12px}.card .bd small{color:var(--mut)}
.catgrp{grid-column:1/-1;font-weight:600;color:var(--mut);margin-top:6px}
.fr{position:absolute;inset:0;pointer-events:none;z-index:2}.fr.on{pointer-events:auto;cursor:crosshair}
.fr i{position:absolute;pointer-events:none}.fbox{border:2px solid var(--io);border-radius:2px}
.fbox.act{box-shadow:0 0 0 9999px #000a}.fbox.der{border:2px dashed var(--port)}
.fdot{width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;border:2px solid #000;background:var(--io)}
.frl{position:absolute;left:8px;bottom:8px;background:#000c;color:#fff;padding:3px 8px;border-radius:4px;
font:12px ui-monospace,monospace;pointer-events:none}.frl b{color:var(--io)}
.fpin{width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;border:3px solid var(--port);box-shadow:0 0 0 2px #000a}
.fr.follow{cursor:copy}button.fol.on{background:var(--port);color:#fff;border-color:var(--port)}
.fk.fl{background:var(--port)}
.fk{top:1px;bottom:1px;width:6px;margin-left:-3px;background:var(--io);border-radius:2px;pointer-events:none;z-index:2}
</style>
<header><h1>%TITLE%</h1>%NAV%
<div id=views><button id=v1 class=on title="every clip, one at a time">Clips</button><button id=v2 title="the cut, shot by shot, with your picks in it">Cut &amp; picks</button></div><span id=reelsw title="which reel you're picking for: its shots show in the lanes; picks for other reels fade; new picks, bans, the opening and the ending are for this reel only, and removing one takes it out of this reel only (All: every reel)"></span><span id=counts></span>
<span class=sp></span><span id=saved></span>
<span class=pj><button id=copy title="copy picks.json to the clipboard">Copy</button><button id=dl title="download picks.json">Download</button></span>
</header>
<section id=clipsview><aside id=side><div id=filters></div><div id=list></div></aside>
<div id=main>
<div id=cliphead><h2 id=cname></h2><span id=cmeta style="color:var(--mut)"></span><span class=sp></span>
<button id=prevc title="previous clip (↑)">↑ Prev clip</button><button id=nextc title="next clip (↓)">Next clip ↓</button></div>
<div id=vwrap><video id=vid preload=auto playsinline muted></video><div id=still></div><span id=tc></span><div id=fr class=fr></div></div>
<div class=row><button id=play>▶ Play</button><button id=fb title="one frame back (←)">◀ frame</button>
<button id=ff title="one frame on (→)">frame ▶</button>
<select id=rate><option value=0.25>¼×</option><option value=0.5>½×</option><option value=1 selected>1×</option>
<option value=2>2×</option><option value=4>4×</option></select>
<button id=sound title="sound on / off">🔇</button>
<span style="width:10px"></span>
<button id=bin title="mark in (I)">In <kbd>I</kbd></button><button id=bout title="mark out (O)">Out <kbd>O</kbd></button>
<button id=badd class=go title="add the in–out as a selection (Enter)">+ Add selection <kbd>Enter</kbd></button>
<span id=io></span><span class=sp></span>
<button id=zfit title="whole clip (0)">Fit <kbd>0</kbd></button><button id=zin>Zoom +</button><button id=zout>Zoom −</button>
<button id=banc title="never use this clip">Ban clip</button></div>
<div class=row><button id=bframe class=frt title="show the portrait reel's crop here; drag on the picture to set a focal point (F)">▯ Portrait frame <kbd>F</kbd></button>
<button id=bfollow class=fol title="then click a dancer: the portrait crop follows that person through the shot (G)">👤 Follow person <kbd>G</kbd></button>
<button id=fclr title="remove the focal points and the followed person around the playhead (this selection, reel shot or 3 s)">Clear focus here</button>
<span id=fhelp style="color:var(--mut);font-size:12px" hidden>dashed pink: what the planner works out · yellow: what the portrait reel shows — drag to set a focal point at the playhead; set more at other times and the crop glides between them</span></div>
<div id=ov class=tl title="the whole clip; the yellow box is what the timeline below shows — click or drag to move"><canvas></canvas></div>
<div id=tl class=tl>
<div id=ruler title="click or drag to scrub"></div>
<div id=strip title="click or drag to scrub; scroll to zoom, shift+scroll to pan"><canvas></canvas></div>
<div id=lsel class=lane title="drag here to make a selection; drag a selection to move it, either end to make it longer or shorter"><span class=lab>selections</span></div>
<div id=lland class=lane><span class=lab>landscape</span></div><div id=lport class=lane><span class=lab>portrait</span></div>
</div>
<div id=mysel></div>
<div id=keys><kbd>Space</kbd>/<kbd>K</kbd> play · <kbd>J</kbd>/<kbd>L</kbd> shuttle back / on (again: faster) ·
<kbd>←</kbd><kbd>→</kbd> frame (<kbd>Shift</kbd> 1 s) · <kbd>I</kbd> in · <kbd>O</kbd> out · <kbd>Enter</kbd> add
selection · <kbd>X</kbd> clear in–out · <kbd>Backspace</kbd> delete selection at playhead · <kbd>[</kbd><kbd>]</kbd>
previous / next mark · <kbd>\</kbd> loop selection · <kbd>↑</kbd><kbd>↓</kbd> previous / next clip ·
<kbd>=</kbd><kbd>−</kbd> zoom · <kbd>0</kbd> fit · <kbd>B</kbd> ban in–out · <kbd>1</kbd>–<kbd>5</kbd> star the selection at the playhead · <kbd>F</kbd> portrait frame · <kbd>G</kbd> follow a person (then click them) · <kbd>Tab</kbd> Cut &amp; picks<br>
Or drag in the <b style="color:var(--sel)">selections</b> lane to mark a range directly; drag a block to move it,
its edges to trim. <span style="color:var(--land)">■</span> landscape and <span style="color:var(--port)">■</span>
portrait show what the current reels use; <span style="color:var(--mom)">|</span> moments the planner found.
Each selection goes in the reel starting at its in-point.</div>
</div></section>
<section id=selview>
<div id=selhead><h2 id=seltitle></h2><span id=seltot style="color:var(--mut)"></span><span class=sp></span>
<span id=pfmts><button data-f=L title="the landscape reel's shots, in order, with your picks">Landscape cut</button><button data-f=P title="the portrait reel's shots, in order, with your picks">Portrait cut</button><button data-f=picks title="only the picks you made">Picks only</button></span>
<button id=playall class=go>▶ Play all</button><button id=stopall>■ Stop</button>
<div id=bulk><b id=bulkn></b><span class=sp></span><button id=bulkall title="select every shot the planner chose (not your picks) (A)">Select all planner shots</button>
<button id=bulkdrop class=go title="drop every selected shot: picks are removed, the planner's shots banned (Delete)">Drop selected</button>
<button id=bulkclr title="clear the selection (Esc)">Clear</button></div></div>
<div id=ped><div id=pvwrap><video id=pv playsinline muted preload=auto></video><div id=pstill></div><span id=pvtc></span><div id=pfr class=fr></div></div>
<div class=row style="margin:8px 0"><button id=pprev title="previous pick (←)">◀ Prev</button>
<button id=pplay class=go title="play this pick (Space)">▶ Play pick</button><button id=ploop title="loop this pick (\)">↻ Loop</button>
<button id=pnext title="next pick (→)">Next ▶</button><b id=plen style="font:600 14px ui-monospace,monospace;margin:0 8px"></b>
<span class=grp>In <button data-n="in:-0.5" title="lengthen at the start (Shift: 5 s)">−½s</button><button data-n="in:-0.02">−f</button>
<button data-n="in:0.02">+f</button><button data-n="in:0.5" title="trim the start">+½s</button><button id=pin title="in at the playhead (I)">⇤ I</button></span>
<span class=grp>Out <button data-n="out:-0.5" title="trim the end">−½s</button><button data-n="out:-0.02">−f</button>
<button data-n="out:0.02">+f</button><button data-n="out:0.5" title="lengthen at the end (Shift: 5 s)">+½s</button><button id=pout title="out at the playhead (O)">O ⇥</button></span>
<span class=grp>Slip <button data-n="slip:-0.5" title="move the whole pick earlier (,)">◀ ½s</button><button data-n="slip:-0.02">◀ f</button>
<button data-n="slip:0.02">f ▶</button><button data-n="slip:0.5" title="move the whole pick later (.)">½s ▶</button></span>
<button id=pframe class=frt title="portrait crop; drag on the picture to set a focal point (F)">▯ Portrait <kbd>F</kbd></button>
<button id=pfollow class=fol title="then click a dancer: the portrait crop follows that person through the pick (G)">👤 Follow</button>
<button id=pfclr title="remove this pick's focal points and followed person">Clear focus</button>
<span class=grp><button id=pfirst title="open the reel on this pick, under the event's logo (one opening per reel)">★ Opening</button><button id=plast title="end the reel on this pick (one ending per reel)">⚑ Ending</button></span><span class=grp id=pstars></span><span class=grp id=preels></span><span class=sp></span><button id=popen title="open this clip in Clips">Open in Clips</button><button id=pdel title="remove (Backspace)">✕ Remove</button></div>
<div id=pt class=tl><div id=pruler></div><div id=pstrip title="drag the edges to trim or lengthen, the block to slip it, anywhere else to scrub; scroll to zoom"><canvas></canvas></div></div>
<p style="color:var(--mut);font-size:12px;margin:6px 0 0">Drag the pick's edges to lengthen or trim, drag the block to slip it, click
elsewhere to scrub · tick cards (<kbd>X</kbd>, shift-click for a range, <kbd>A</kbd> all the planner's) then <kbd>Delete</kbd> drops them all · <kbd>Space</kbd> play · <kbd>←</kbd><kbd>→</kbd> previous / next pick · <kbd>I</kbd>/<kbd>O</kbd> in / out
at the playhead · <kbd>,</kbd><kbd>.</kbd> slip a frame (<kbd>Shift</kbd> ½ s) · <kbd>\</kbd> loop · <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo · <kbd>Tab</kbd> Clips</p></div>
<div id=seq title="every selection in order, widths by length — click to play from there"></div>
<div id=cards></div>
</section>
<div id=hover><div></div><span></span></div>
<script>
const D=%DATA%,TW=D.tile[0],TH=D.tile[1],FR=1/50;
const $=id=>document.getElementById(id);
const fmt=x=>{x=Math.max(0,x);const m=Math.floor(x/60);return m+":"+(x-60*m).toFixed(2).padStart(5,"0")};
const short=x=>{x=Math.max(0,x);const m=Math.floor(x/60);return m+":"+(x-60*m).toFixed(1).padStart(4,"0")};
const esc=x=>String(x).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const secs=v=>{if(typeof v==="number")return v;const p=String(v).trim().split(":");
 return p.reduce((a,x)=>a*60+(+x),0)};
const span=v=>{if(Array.isArray(v))return v.map(secs);const s=String(v),i=s.indexOf("-",1);
 return i>0?[secs(s.slice(0,i)),secs(s.slice(i+1))]:[secs(s),secs(s)]};
const isRange=v=>{const[a,b]=span(v);return b-a>0.05};
// --- picks: saved to picks.json by `focalcut reel serve`, else kept in this browser ---
const LS="sources:"+D.key;
const ls={get(k){try{return JSON.parse(localStorage.getItem(LS+k))}catch(e){return null}},
 set(k,v){try{localStorage.setItem(LS+k,JSON.stringify(v))}catch(e){}}};
let P=JSON.parse(JSON.stringify(D.picks)),server=false,timer=null;
const ent=id=>(P[id]&&typeof P[id]==="object"&&!Array.isArray(P[id]))?P[id]:(P[id]={});
%TAGS%
function tidy(){for(const k in P){const e=P[k];if(k[0]=="_"||!e||typeof e!=="object"||Array.isArray(e))continue;
 pruneTags(e);if(e.stars){const ss=(e.add||[]).map(span);for(const q in e.stars){const[u,v]=span(q);
  if(!ss.some(([a,b])=>near2(a,u)&&near2(b,v)))delete e.stars[q]}if(!Object.keys(e.stars).length)delete e.stars}
 for(const f of["ban","prefer","add","portrait","follow"])if(e[f]&&!e[f].length)delete e[f];if(e.more===0)delete e.more;
 if(!Object.keys(e).length)delete P[k]}}
function status(t,c){const s=$("saved");s.textContent=t;s.className=c||""}
const hist=[];let lastSnap=JSON.stringify(P);
function save(){tidy();{const now_=JSON.stringify(P);if(now_!==lastSnap){hist.push(lastSnap);if(hist.length>200)hist.shift();lastSnap=now_}}if(server){status("saving…");clearTimeout(timer);timer=setTimeout(push,250)}
 else{ls.set("picks",P);status("kept in this browser — run `focalcut reel serve` to save to picks.json","err")}
 renderAll()}
async function push(){try{const r=await fetch(D.root+"api/picks",{method:"PUT",headers:{"Content-Type":"application/json"},
 body:JSON.stringify(P,null,1)});if(!r.ok)throw new Error(r.status);status("saved to picks.json ✓","ok");
 if(staleCrop){const id=staleCrop;staleCrop=null;for(const k of[...crops.keys()])if(k.startsWith(id+"|"))crops.delete(k);lastCrop.delete(id);draw();pdraw()}}
 catch(e){status("could not save: "+e.message,"err")}}
async function boot(){try{const r=await fetch(D.root+"api/picks",{cache:"no-store"});if(!r.ok)throw 0;P=await r.json();
 server=true;lastSnap=JSON.stringify(P);($("allp")||{}).hidden=false;status("picks.json — saves as you go","ok")}catch(e){const l=ls.get("picks");if(l)P=l;
 status("opened as a file: changes stay in this browser","err")}
 const last=ls.get("clip");go(Math.max(0,D.clips.findIndex(c=>c.id==last)),true);framing()}
// sources.html#frame=<reel>: straight to framing that reel's portrait cut -- Cut & picks, the portrait cut's shots in
// order, the crop over the picture: drag to pan it, G then a click to follow a dancer, ←/→ from shot to shot
function framing(){const m=/frame=([\w-]+)/.exec(location.hash);if(!m)return;
 if(RID.includes(m[1])&&reel!==m[1]){reel=m[1];ls.set("reel",reel);useReel();place.clear()}
 pfmt="P";ls.set("pfmt","P");if(!frameOn)toggleFrame();view(2);pick(0,false);
 status(`framing the ${reel=="all"?"":rlab(reel)+" "}portrait cut — drag the picture to pan · G then click a dancer to follow · ←/→ shot to shot · Clear focus to undo`,"ok")}
addEventListener("hashchange",framing);
// selections of a clip: [[a,b],...] sorted; written back as "m:ss.ss-m:ss.ss" (single times stay as they were)
function sels(id){return((P[id]||{}).add||[]).filter(isRange).map(span).sort((x,y)=>x[0]-y[0])}
function points(id){return((P[id]||{}).add||[]).filter(v=>!isRange(v)).map(v=>span(v)[0])}
function setSels(id,list,quiet){const e=ent(id);const pts=(e.add||[]).filter(v=>!isRange(v));
 list=list.map(([a,b])=>[Math.max(0,Math.min(a,b)),Math.max(a,b)]).filter(([a,b])=>b-a>=0.1);
 // each keeps its reels (the old selection it overlaps most: moved or trimmed); a new one is for the reel being picked
 // for; one taken away while picking for a reel leaves that reel only, and stays in the others
 const old=sels(id).map(([a,b])=>[a,b,onlyOf(id,a,b),starsOf(id,a,b)]);
 const items=list.map(([a,b])=>{let best=null,ov=0;old.forEach(([u,v,o])=>{const x=Math.min(b,v)-Math.max(a,u);if(x>ov){ov=x;best=o}});
  let st=0,ov2=0;old.forEach(([u,v,,n])=>{const x=Math.min(b,v)-Math.max(a,u);if(x>ov2){ov2=x;st=n}});
  return{a,b,t:ov>0?best:(reel=="all"?null:[reel]),st}});
 if(reel!="all")old.forEach(([u,v,o])=>{if(list.some(([a,b])=>Math.min(b,v)-Math.max(a,u)>0))return;
  const t=(o||RID).filter(r=>r!=reel);if(t.length&&RID.length>1)items.push({a:u,b:v,t,st:starsOf(id,u,v)})});
 items.sort((x,y)=>x.a-y.a);
 if(e.only)for(const k in e.only)if(tagParts(k)[0]=="add"&&isRange(k))delete e.only[k];
 e.add=[...pts,...items.map(x=>fmt(x.a)+"-"+fmt(x.b))];items.forEach(x=>setTags(id,"add",fmt(x.a)+"-"+fmt(x.b),x.t));
 const so={};items.forEach(x=>{if(x.st)so[fmt(x.a)+"-"+fmt(x.b)]=x.st});if(Object.keys(so).length)e.stars=so;else delete e.stars;
 if(!quiet)save()}
function banned(id){return vals(id,"ban",reel)}
// reels: one picks.json serves the project's reel ("main") and its cuts (reel.json "cuts"); a clip's "only" maps a
// selection to the reels it is for, and one not listed is for every reel
const rlab=r=>(RS.find(x=>x.id==r)||{label:r}).label;
let reel=ls.get("reel")||"all";if(!RID.includes(reel))reel="all";
const rkey=(a,b)=>fmt(a)+"-"+fmt(b);
const onlyOf=(id,a,b)=>tagsOf(id,"add",rkey(a,b));
const inReel=(id,a,b,r=reel)=>isIn(id,"add",rkey(a,b),r);
function setIn(id,a,b,r,on){let v=onlyOf(id,a,b)||RID.slice();
 v=on?[...new Set([...v,r])]:v.filter(x=>x!=r);if(!v.length){status("a pick is in one reel at least — ✕ removes it","err");return renderAll()}
 // (its opening / ending role follows it out of a reel)
 if(!on)for(const f of["open","end"]){const w=roleIn(id,f,r);if(w&&same(w,a,b))clearRole(id,f,r)}
 setTags(id,"add",rkey(a,b),v);save()}
// stars: how much the editor wants a pick (1-5; a clip's "stars": {range: n}, every reel): the planner keeps and
// lengthens the most starred first and drops the least starred first (unrated counts as 3)
function starsOf(id,a,b){const o=(P[id]||{}).stars;if(!o)return 0;for(const k in o){const[u,v]=span(k);if(near2(u,a)&&near2(v,b))return o[k]}return 0}
function setStars(id,a,b,n){const e=ent(id),o={};for(const k in e.stars||{}){const[u,v]=span(k);if(!(near2(u,a)&&near2(v,b)))o[k]=e.stars[k]}
 if(n)o[rkey(a,b)]=n;if(Object.keys(o).length)e.stars=o;else delete e.stars;save();
 status(n?`${"★".repeat(n)} ${fmt(a)}–${fmt(b)}`:"stars cleared","ok")}
const starBar=(id,a,b,attr)=>{const n=starsOf(id,a,b);return`<span class=stars title="how much you want it (1–5; click again to clear)">`+
 [1,2,3,4,5].map(i=>`<b data-star=${i} ${attr} title="${i} star${i>1?"s":""} — key ${i}">${i<=n?"★":"☆"}<sub>${i}</sub></b>`).join("")+`</span>`};
function useReel(){const r=reel=="all"?"main":reel;D.clips.forEach(c=>{const u=(c.R||{})[r];if(u){c.L=u.L;c.P=u.P}});
 $("reelsw").innerHTML=RS.length>1?"<b>Reel</b>"+[["all","All"],...RS.map(x=>[x.id,x.label])].map(([k,l])=>
  `<button data-reel="${esc(k)}" class="${k==reel?"on":""}">${esc(l)}</button>`).join(""):""}
$("reelsw").onclick=ev=>{const b=ev.target.closest("[data-reel]");if(!b)return;reel=b.dataset.reel;ls.set("reel",reel);useReel();place.clear();
 status(reel=="all"?"all reels: picks and bans go in every reel":`${rlab(reel)}: picks, bans, opening and ending are for this reel only`,"ok");renderAll()}
useReel();
// one pick opens the reel and one ends it: the clip's "open" / "end" is that pick's in-out (it stays a selection too)
const same=(v,a,b)=>{if(!v)return false;const[u,w]=span(v);return Math.abs(u-a)<0.01&&Math.abs(w-b)<0.01};
function roleOf(id,a,b){return same(roleIn(id,"open",reel),a,b)?"open":same(roleIn(id,"end",reel),a,b)?"end":null}
// (per reel: picking for one reel sets or clears that reel's; a clip holds one opening and one ending)
function setRole(x,f){const on=same(roleIn(x.c.id,f,reel),x.a,x.b);
 for(const k in P){const e=P[k];if(k[0]!="_"&&e&&typeof e=="object"&&!Array.isArray(e)&&roleIn(k,f,reel))clearRole(k,f,reel)}
 if(!on){const e=ent(x.c.id);if(typeof e[f]=="string"){setTags(x.c.id,f,e[f],null);delete e[f]}
  e[f]=fmt(x.a)+"-"+fmt(x.b);setTags(x.c.id,f,e[f],reel=="all"?null:[reel]);
  if(x.k<0){const ss=sels(x.c.id);ss.push([x.a,x.b]);return setSels(x.c.id,ss)}}
 save();status(on?`no ${f=="open"?"opening":"ending"} pick: the planner chooses`:`${x.c.id} ${fmt(x.a)} ${f=="open"?"opens":"ends"} the reel`,"ok")}
// --- clip list ---
let cur=0,cat="all",onlySel=false,rub=null;const seen=new Set(ls.get("seen")||[]);
const C=()=>D.clips[cur];
const cats=["all",...new Set(D.clips.map(c=>c.cat))];
$("filters").innerHTML=cats.map(c=>`<button data-c="${esc(c)}" class="${c=="all"?"on":""}">${esc(c)}</button>`).join("")+
 `<button id=onlysel title="only clips with selections">● selected</button>`;
$("filters").onclick=ev=>{const b=ev.target.closest("button");if(!b)return;if(b.id=="onlysel"){onlySel=!onlySel;
 b.classList.toggle("on",onlySel)}else{cat=b.dataset.c;$("filters").querySelectorAll("[data-c]").forEach(x=>
 x.classList.toggle("on",x===b))}list()};
const visible=()=>D.clips.map((c,i)=>i).filter(i=>(cat=="all"||D.clips[i].cat==cat)&&(!onlySel||sels(D.clips[i].id).length));
// a clip's frame at t as a background, h tall; in a box W wide, a narrower (vertical) frame sits in the middle, its
// neighbours in the strip clipped off
const tileOf=c=>c.tile||[TW,TH];
const thumbAt=(c,t,h,W)=>{const[tw,th]=tileOf(c),k=Math.min(c.n-1,Math.max(0,Math.floor(t/c.step))),s=h/th,pad=W?Math.max(0,(W-tw*s)/2):0;
 return`background-image:url('${c.strip}');background-size:${c.n*tw*s}px ${h}px;background-position:-${k*tw*s}px 0`+
  (pad?`;box-sizing:border-box;padding-left:${pad}px;padding-right:${pad}px;background-origin:content-box;background-clip:content-box`:"")};
const mine=id=>sels(id).filter(([a,b])=>inReel(id,a,b));
function list(){$("list").innerHTML=visible().map(i=>{const c=D.clips[i],n=mine(c.id).length,
 tot=sels(c.id).reduce((a,[x,y])=>a+y-x,0),bn=banned(c.id).includes("all");
 return`<div class="ci${i==cur?" cur":""}${seen.has(c.id)?" seen":""}" data-i=${i} style="${bn?"opacity:.4":""}">
 <div class=th style="${thumbAt(c,c.dur*0.3,54,96)}">${n?`<b>${n}</b>`:""}</div><div><div class=nm>${esc(c.id)}</div>
 <div class=meta><span class=chip>${esc(c.cat)}</span> ${esc(c.clock)} · ${short(c.dur)}${n?` · <span style="color:var(--sel)">● ${short(tot)}</span>`:""}</div>
 <div class=meta>${c.L.length?`<span style="color:var(--land)">L${c.L.length}</span> `:""}${c.P.length?`<span style="color:var(--port)">P${c.P.length}</span>`:""}${bn?" banned":""}</div></div></div>`}).join("");
 const el=$("list").querySelector(".cur");if(el)el.scrollIntoView({block:"nearest"})}
$("list").onclick=ev=>{const d=ev.target.closest(".ci");if(d)go(+d.dataset.i)};
// --- player + timeline ---
const vid=$("vid"),tl=$("tl"),ov=$("ov"),hov=$("hover");
let va=0,vb=1,inPt=null,outPt=null,act=-1,loop=false,shuttle=0,rew=null,drag=null,img=new Image();
// scrubbing: one seek in flight per video, the next goes to wherever the pointer is by then (a new currentTime on every
// mouse move restarts the seek, so the picture would never catch up until the mouse stops)
function goTo(v,t){if(v.seeking){v._want=t;return}v._want=null;v.currentTime=t}
for(const v of[$("vid"),$("pv")])v.addEventListener("seeked",()=>{if(v._want!=null){const t=v._want;v._want=null;v.currentTime=t}});
const now=()=>vid._want??(vid.currentTime||0),dur=()=>C().dur||vid.duration||1;
function go(i,first){if(i<0||i>=D.clips.length)return;cur=i;const c=C();seen.add(c.id);ls.set("seen",[...seen]);
 ls.set("clip",c.id);inPt=outPt=null;act=-1;loop=false;stopShuttle();vid.pause();
 vid.src=c.video;va=0;vb=c.dur;img=new Image();img.onload=paint;img.src=c.strip;
 $("cname").textContent=c.id;$("cmeta").innerHTML=`<span class=chip>${esc(c.cat)}</span> ${esc(c.tag)} · ${esc(c.clock)}
  · ${short(c.dur)} · clip ${i+1} of ${D.clips.length}`;
 vid.addEventListener("loadedmetadata",()=>{vid.currentTime=0;renderAll()},{once:true});renderAll()}
function seek(t){if(rew)stopShuttle();goTo(vid,Math.min(Math.max(0,t),dur()-0.02));follow();draw()}
function follow(){const t=now(),w=vb-va;if(t<va||t>vb){va=Math.max(0,Math.min(t-w*0.1,dur()-w));vb=va+w;paint()}}
const X=(t,a=va,b=vb)=>(100*(t-a)/(b-a)).toFixed(3)+"%";
function wspan(a,b,A=va,B=vb){const x0=Math.max(a,A),x1=Math.min(b,B);return`left:${X(x0,A,B)};width:${(100*(x1-x0)/(B-A)).toFixed(3)}%`}
function film(canvas,A,B,h,c=C(),im=img){const r=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1,img=im;
 canvas.width=Math.max(1,r.width*dpr);canvas.height=h*dpr;const g=canvas.getContext("2d");
 g.fillStyle="#000";g.fillRect(0,0,canvas.width,canvas.height);if(!img.complete||!img.naturalWidth)return;
 const[TW_,TH_]=tileOf(c),tw=h*TW_/TH_*dpr,W=canvas.width;
 for(let x=0;x<W;x+=tw){const t=A+(x+tw/2)/W*(B-A);if(t>c.dur)break;
  const k=Math.min(c.n-1,Math.max(0,Math.floor(t/c.step)));g.drawImage(img,k*TW_,0,TW_,TH_,x,0,tw,h*dpr)}}
function paint(){film($("strip").querySelector("canvas"),va,vb,80);film(ov.querySelector("canvas"),0,dur(),30);draw()}
function draw(){const c=C();if(!c)return;const W=vb-va,t=now();
 // ruler
 const step=[0.1,0.2,0.5,1,2,5,10,15,30,60].find(s=>W/s<=60)||120,lab=[0.5,1,2,5,10,15,30,60,120].find(s=>W/s<=10)||300;
 let h="";for(let x=Math.ceil(va/step)*step;x<=vb;x+=step){const big=Math.abs(x/lab-Math.round(x/lab))<1e-6;
  h+=`<i class="tk${big?" big":""}" style="left:${X(x)}"></i>`;if(big)h+=`<i class=lb style="left:${X(x)}">${W<15?fmt(x):short(x)}</i>`}
 $("ruler").innerHTML=h;
 // strip overlays: moments, bans, in/out, playhead
 const S=$("strip");S.querySelectorAll("i").forEach(e=>e.remove());let s="";
 c.moments.forEach(([m,k])=>{if(m>=va&&m<=vb)s+=`<i class=mom style="left:${X(m)}" title="${esc(k)} ${short(m)}"></i>`});
 banned(c.id).forEach(v=>{const[a,b]=v=="all"?[0,c.dur]:span(v);if(b>va&&a<vb)s+=`<i class=ban style="${wspan(a,b)}" title="banned ${esc(v)}"></i>`});
 S.insertAdjacentHTML("beforeend",s);
 // selections lane
 const L=$("lsel");L.querySelectorAll("i").forEach(e=>e.remove());s="";
 sels(c.id).forEach(([a,b],k)=>{const o=onlyOf(c.id,a,b);if(b>va&&a<vb)s+=`<i class="blk${k==act?" act":""}${inReel(c.id,a,b)?"":" away"}" data-k=${k} style="${wspan(a,b)}"
  title="${fmt(a)} – ${fmt(b)} (${(b-a).toFixed(2)} s)${RS.length>1?" · "+(o?o.map(rlab).join(" + ")+" only":"every reel"):""}"><i class="h l"></i>${k+1} · ${(b-a).toFixed(1)}s${starsOf(c.id,a,b)?" "+"★".repeat(starsOf(c.id,a,b)):""}<i class="h r"></i></i>`});
 points(c.id).forEach(p=>{if(p>=va&&p<=vb)s+=`<i class=pt style="left:${X(p)}" title="marked moment ${fmt(p)}"></i>`});
 const io=pend();if(io)s+=`<i class=pend style="${wspan(...io)}"></i>`;
 if(!sels(c.id).length&&!io)s+=`<i class=empty>drag here to pick a range · or I / O then Enter · or click a reel shot below to take it</i>`;
 L.insertAdjacentHTML("beforeend",s);
 for(const[id,arr]of[["lland",c.L],["lport",c.P]]){const e=$(id);e.querySelectorAll("i").forEach(x=>x.remove());
  const bans=banned(c.id).map(v=>v=="all"?[-1,1e9]:span(v)),out=(a,b)=>bans.some(([u,v])=>(a+b)/2>=u&&(a+b)/2<=v);
  e.insertAdjacentHTML("beforeend",arr.filter(([a,b])=>b>va&&a<vb).map(([a,b,n])=>`<i class="use${out(a,b)?" out":""}" data-a=${a} data-b=${b} style="${wspan(a,b)}" title="reel shot ${n}: ${short(a)}–${short(b)} — click to pick it, ✕ to take it out; drag along the lanes to take out every shot in a range">${n}<b class=ux title="take this shot out of the cut (bans its stretch)">✕</b></i>`).join(""))}
 if(rub&&rub.moved){const[a,b]=[Math.min(rub.t0,rub.t1),Math.max(rub.t0,rub.t1)];
  for(const id of["lland","lport"])$(id).insertAdjacentHTML("beforeend",`<i class=rub style="${wspan(a,b)}"></i>`)}
 fkeys(c.id).forEach(([k])=>{if(k>=va&&k<=vb)$("lport").insertAdjacentHTML("beforeend",`<i class=fk style="left:${X(k)}" title="focal point ${fmt(k)}"></i>`)});
 folkeys(c.id).forEach(([k])=>{if(k>=va&&k<=vb)$("lport").insertAdjacentHTML("beforeend",`<i class="fk fl" style="left:${X(k)}" title="following a person from ${fmt(k)}"></i>`)});
 // markers spanning the timeline
 tl.querySelectorAll(":scope>i").forEach(e=>e.remove());let m="";
 if(inPt!==null&&inPt>=va&&inPt<=vb)m+=`<i class=inm style="left:${X(inPt)}"></i>`;
 if(outPt!==null&&outPt>=va&&outPt<=vb)m+=`<i class=outm style="left:${X(outPt)}"></i>`;
 if(t>=va&&t<=vb)m+=`<i class=ph style="left:${X(t)}"></i>`;tl.insertAdjacentHTML("beforeend",m);
 // overview
 ov.querySelectorAll("i").forEach(e=>e.remove());let o="";const D0=dur();
 sels(c.id).forEach(([a,b])=>o+=`<i class=blk style="${wspan(a,b,0,D0)};top:4px;bottom:4px;pointer-events:none"></i>`);
 o+=`<i class=win style="${wspan(va,vb,0,D0)}"></i><i class=ph style="left:${X(t,0,D0)}"></i>`;ov.insertAdjacentHTML("beforeend",o);
 $("tc").textContent=fmt(t)+"  /  "+short(D0);
 const st=$("still"),noVid=vid.error||vid.readyState<2&&!vid.seeking||!vid.videoWidth;st.classList.toggle("on",!!noVid);
 if(noVid){const r=$("vwrap").getBoundingClientRect(),[tw,th]=tileOf(c),h=Math.min(r.height,r.width*th/tw);
  st.setAttribute("style",thumbAt(c,t,h)+`;width:${h*tw/th}px;height:${h}px;left:${(r.width-h*tw/th)/2}px;top:${(r.height-h)/2}px`)}
 $("io").textContent=(inPt===null&&outPt===null)?"":"in "+(inPt===null?"—":fmt(inPt))+"  out "+(outPt===null?"—":fmt(outPt))+
  (io?"  ("+(io[1]-io[0]).toFixed(2)+" s)":"");
 $("play").textContent=vid.paused&&!rew?"▶ Play":"❚❚ Pause";
 $("banc").textContent=banned(c.id).includes("all")?"Unban clip":"Ban clip";
 frameDraw($("fr"),$("vwrap"),vid,c,...rangeFor(c,t),t)}
function pend(){if(inPt===null&&outPt===null)return null;const a=inPt??now(),b=outPt??now();
 return Math.abs(b-a)<0.05?null:[Math.min(a,b),Math.max(a,b)]}
function mysel(){const c=C(),ss=sels(c.id);
 $("mysel").innerHTML=ss.length?`<table><tr><th>#</th><th>in</th><th>out</th><th>length</th><th></th></tr>`+ss.map(([a,b],k)=>
  `<tr class="${k==act?"act":""}"><td>${k+1}</td><td>${fmt(a)}</td><td>${fmt(b)}</td><td>${(b-a).toFixed(2)} s</td>
  <td>${starBar(c.id,a,b,`data-k=${k}`)}${RS.length>1?RS.map(r=>`<label class=rk><input type=checkbox data-r="${esc(r.id)}" data-k=${k}${inReel(c.id,a,b,r.id)?" checked":""}> ${esc(r.label)}</label>`).join(""):""}<button data-p=${k}>▶ play</button> <button data-l=${k}>↻ loop</button> <button data-d=${k}>✕</button></td></tr>`).join("")+
  `</table>`:`<p style="color:var(--mut);margin:0">No selections in this clip yet: <kbd>I</kbd> at the start, <kbd>O</kbd>
  at the end, <kbd>Enter</kbd> — or drag in the selections lane.</p>`}
$("mysel").onchange=ev=>{const i=ev.target;if(!i.dataset.r)return;const ss=sels(C().id),x=ss[+i.dataset.k];
 if(x)setIn(C().id,x[0],x[1],i.dataset.r,i.checked)};
$("mysel").onclick=ev=>{const st=ev.target.closest("[data-star]");if(st){const ss=sels(C().id),x=ss[+st.dataset.k];
  if(x){const n=+st.dataset.star;setStars(C().id,x[0],x[1],starsOf(C().id,x[0],x[1])==n?0:n)}return}
 const b=ev.target.closest("button");if(!b)return;const ss=sels(C().id);
 if(b.dataset.d!==undefined){ss.splice(+b.dataset.d,1);act=-1;setSels(C().id,ss)}
 else{const k=+(b.dataset.p??b.dataset.l);act=k;loop=b.dataset.l!==undefined;seek(ss[k][0]);vid.play();renderAll()}};
function renderAll(){list();draw();mysel();selview();
 counts()}
// what's in play: the planner's shots per reel, and the editor's own calls
function counts(){const n=f=>D.clips.reduce((a,c)=>a+f(c),0),L=n(c=>c.L.length),Pn=n(c=>c.P.length),
  my=n(c=>mine(c.id).length+points(c.id).length),fk=n(c=>fkeys(c.id).length),
  bc=n(c=>banned(c.id).includes("all")?1:0),br=n(c=>banned(c.id).filter(v=>v!="all").length),
  more=n(c=>moreIn(c.id,reel)?1:0),unseen=D.clips.length-D.clips.filter(c=>seen.has(c.id)).length;
 const it=(v,t,cls="")=>`<span class="${cls}" title="${t}"><b>${v}</b> ${t.split(":")[0]}</span>`;
 $("counts").innerHTML=[it(L,"landscape shots: in the current landscape cut"),it(Pn,"portrait shots: in the current portrait cut"),
  it(my,"your picks"+(reel=="all"?"":" for "+rlab(reel))+": selections and marked moments the next re-plan must use","y"),
  fk?it(fk,"focus points: where the portrait crop goes","f"):"",
  bc+br?it(bc+br,"bans: "+bc+" whole clips, "+br+" stretches","x"):"",more?it(more,"clips asked for more"):"",
  it(unseen,"clips not looked at yet")].filter(Boolean).join(" · ")}
// timeline mouse: scrub on ruler/strip/overview; make, move and trim selections in the selections lane
function tAt(el,ev,A=va,B=vb){const r=el.getBoundingClientRect();return A+Math.min(Math.max((ev.clientX-r.left)/r.width,0),1)*(B-A)}
function showHover(ev,t,el){const c=C();hov.style.display="block";hov.querySelector("div").setAttribute("style",thumbAt(c,t,108,192));
 hov.querySelector("span").textContent=fmt(t);const r=el.getBoundingClientRect();
 hov.style.left=Math.min(innerWidth-205,Math.max(4,ev.clientX-100))+"px";hov.style.top=(r.top-136)+"px"}
for(const el of[$("ruler"),$("strip")]){
 el.addEventListener("pointerdown",ev=>{pause();el.setPointerCapture(ev.pointerId);drag={k:"scrub",el};seek(tAt(el,ev))});
 el.addEventListener("pointermove",ev=>{const t=tAt(el,ev);showHover(ev,t,tl);if(drag&&drag.el===el)seek(t)})}
ov.addEventListener("pointerdown",ev=>{ov.setPointerCapture(ev.pointerId);drag={k:"ov",el:ov};movewin(tAt(ov,ev,0,dur()))});
ov.addEventListener("pointermove",ev=>{const t=tAt(ov,ev,0,dur());showHover(ev,t,ov);if(drag&&drag.k=="ov")movewin(t)});
function movewin(t){const w=vb-va;va=Math.max(0,Math.min(t-w/2,dur()-w));vb=va+w;if(now()<va||now()>vb)seek(t);paint()}
// drags in the lanes snap to the playhead (where it was when the drag began) within 10 px
const near=(el,t,p)=>p!=null&&Math.abs(t-p)*el.getBoundingClientRect().width/(vb-va)<10;
const snapT=(el,ev,p=now())=>{const t=tAt(el,ev);return near(el,t,p)?p:t};
const L=$("lsel");
// a selection's edge within 8 px of the pointer, inside or just outside the block (the nearer one wins)
function edgeAt(ev){const r=L.getBoundingClientRect(),px=(r.right-r.left)/(vb-va),x=ev.clientX-r.left;let best=null,bd=8;
 sels(C().id).forEach(([a,b],k)=>{for(const[t,side]of[[a,"trimL"],[b,"trimR"]]){const d=Math.abs((t-va)*px-x);
  if(d<=bd){bd=d;best={k,side}}}});return best}
L.addEventListener("pointerdown",ev=>{pause();L.setPointerCapture(ev.pointerId);const ph=now(),t=snapT(L,ev,ph),b=ev.target.closest(".blk"),
  e=edgeAt(ev);
 if(e||b){const k=e?e.k:+b.dataset.k,ss=sels(C().id);act=k;
  drag={k:e?e.side:"move",i:k,t0:tAt(L,ev),orig:ss[k].slice(),ss,moved:false,ph}}
 else drag={k:"new",a:t,b:t,ph};draw()});
L.addEventListener("pointermove",ev=>{showHover(ev,tAt(L,ev),tl);
 L.style.cursor=drag?(drag.k.startsWith("trim")?"ew-resize":""):edgeAt(ev)?"ew-resize":"";if(!drag||drag.el)return;const t=snapT(L,ev,drag.ph);
 if(drag.k=="new"){drag.b=t;inPt=Math.min(drag.a,t);outPt=Math.max(drag.a,t);seek(t)}
 else{const o=drag.orig,d=t-drag.t0;if(Math.abs(d)>0.03)drag.moved=true;let[a,b]=o;
  if(drag.k=="move"){const w=b-a;a=Math.max(0,Math.min(o[0]+tAt(L,ev)-drag.t0,dur()-w));
   if(near(L,a,drag.ph))a=drag.ph;else if(near(L,a+w,drag.ph))a=drag.ph-w;b=a+w}
  else if(drag.k=="trimL"){a=Math.max(0,Math.min(t,b-0.1));seek(a)}else{b=Math.min(dur(),Math.max(t,a+0.1));seek(b)}
  drag.ss[drag.i]=[a,b];const blk=L.querySelector(`.blk[data-k="${drag.i}"]`);
  if(blk){blk.setAttribute("style",wspan(a,b));blk.childNodes[1].textContent=`${drag.i+1} · ${(b-a).toFixed(1)}s`}}});
L.addEventListener("pointerup",ev=>{if(!drag||drag.el)return;const c=C();
 if(drag.k=="new"){if(Math.abs(drag.b-drag.a)>=0.2){const ss=sels(c.id);ss.push([Math.min(drag.a,drag.b),Math.max(drag.a,drag.b)]);
   inPt=outPt=null;drag=null;setSels(c.id,ss);act=sels(c.id).findIndex(([a])=>Math.abs(a-Math.min(...ss[ss.length-1]))<0.01);renderAll()}
  else{inPt=outPt=null;drag=null;seek(tAt(L,ev))}return}
 const d=drag;drag=null;if(d.moved)setSels(c.id,d.ss);else seek(d.orig[0]);mysel()});
addEventListener("pointerup",()=>{if(drag&&drag.el){drag=null;draw()}});
// the reel lanes: ✕ on a shot takes it out of the cut; drag along either lane to take out every landscape and portrait
// shot in the range (bans their stretches; one undo step; the ends snap to the playhead); a plain click makes it a pick
function banShots(c,list){let n=0;list.forEach(([a,b])=>{if(addVal(c.id,"ban",fmt(a)+"-"+fmt(b),reel))n++});if(n)save();return n}
for(const id of["lland","lport"]){const el=$(id);
 el.addEventListener("pointerdown",ev=>{if(ev.button||ev.target.closest(".ux"))return;el.setPointerCapture(ev.pointerId);
  rub={t0:snapT(el,ev),t1:snapT(el,ev),x:ev.clientX,moved:false}});
 el.addEventListener("pointermove",ev=>{showHover(ev,tAt(el,ev),tl);if(!rub)return;rub.t1=snapT(el,ev);
  if(Math.abs(ev.clientX-rub.x)>4)rub.moved=true;if(rub.moved)draw()});
 el.addEventListener("pointerup",()=>{const r=rub;if(!r)return;if(!r.moved){rub=null;return}
  rub={moved:false,done:true};const c=C(),[a,b]=[Math.min(r.t0,r.t1),Math.max(r.t0,r.t1)];
  const hit=[...c.L,...c.P].filter(([x,y])=>(x+y)/2>=a&&(x+y)/2<=b);
  const n=banShots(c,hit);status(n?`took ${n} shot${n==1?"":"s"} out of the cut — Ctrl+Z undoes`:"no reel shots in that range",n?"ok":"");draw()})}
for(const id of["lland","lport"])$(id).addEventListener("click",ev=>{if(rub&&rub.done){rub=null;return}
 const x=ev.target.closest(".ux");if(x){const u=x.closest(".use"),n=banShots(C(),[[+u.dataset.a,+u.dataset.b]]);
  status(n?"took the shot out of the cut — Ctrl+Z undoes":"already out","ok");return}
 const u=ev.target.closest(".use");if(!u)return;
 const a=+u.dataset.a,b=+u.dataset.b,c=C(),ss=sels(c.id);if(!ss.some(([x,y])=>Math.abs(x-a)<0.05&&Math.abs(y-b)<0.05))ss.push([a,b]);
 setSels(c.id,ss);act=sels(c.id).findIndex(([x])=>Math.abs(x-a)<0.01);seek(a);renderAll()});
vid.addEventListener("error",draw);vid.addEventListener("loadeddata",draw);vid.addEventListener("seeking",draw);
for(const el of[tl,ov])el.addEventListener("pointerleave",()=>hov.style.display="none");
tl.addEventListener("wheel",ev=>{ev.preventDefault();const r=tl.getBoundingClientRect(),f=(ev.clientX-r.left)/r.width;
 if(ev.shiftKey||Math.abs(ev.deltaX)>Math.abs(ev.deltaY)){const w=vb-va,d=(ev.deltaX||ev.deltaY)/600*w;
  va=Math.max(0,Math.min(va+d,dur()-w));vb=va+w}else zoom(ev.deltaY>0?1.25:0.8,va+f*(vb-va));paint()},{passive:false});
function zoom(k,at){const D0=dur(),w=Math.min(Math.max((vb-va)*k,0.5),D0);at=at??now();const f=(at-va)/(vb-va||1);
 va=Math.max(0,Math.min(at-f*w,D0-w));vb=va+w;paint()}
// playback
function stopShuttle(){if(rew){clearInterval(rew.id);rew=null}shuttle=0;vid.playbackRate=+$("rate").value}
function pause(){vid.pause();stopShuttle();draw()}
function playL(){if(rew)stopShuttle();shuttle=vid.paused?1:Math.min((shuttle||1)*2,8);
 vid.playbackRate=+$("rate").value*shuttle;vid.play()}
function playJ(){vid.pause();shuttle=0;const n=rew?Math.min(rew.n*2,8):1;if(rew)clearInterval(rew.id);
 rew={n,id:setInterval(()=>{if(now()<=0)return stopShuttle();goTo(vid,Math.max(0,now()-0.04*n));follow();draw()},40)};draw()}
const step=d=>{pause();seek(now()+d)};
vid.addEventListener("play",()=>{const f=()=>{const ss=sels(C().id);
  if(loop&&act>=0&&ss[act]&&now()>=ss[act][1])vid.currentTime=ss[act][0];
  else if(!loop&&act>=0&&ss[act]&&playOne&&now()>=ss[act][1]){vid.pause();playOne=false}
  follow();draw();if(!vid.paused)requestAnimationFrame(f)};f()});
let playOne=false;vid.addEventListener("pause",draw);vid.addEventListener("seeked",draw);
$("play").onclick=()=>vid.paused&&!rew?playL():pause();
$("fb").onclick=()=>step(-FR);$("ff").onclick=()=>step(FR);$("rate").onchange=()=>vid.playbackRate=+$("rate").value;
$("sound").onclick=()=>{vid.muted=!vid.muted;$("pv").muted=vid.muted;$("sound").textContent=vid.muted?"🔇":"🔊"};
$("zfit").onclick=()=>{va=0;vb=dur();paint()};$("zin").onclick=()=>zoom(0.6);$("zout").onclick=()=>zoom(1.6);
$("prevc").onclick=()=>nav(-1);$("nextc").onclick=()=>nav(1);
function nav(d){const v=visible(),k=v.indexOf(cur);const j=k<0?v[0]:v[Math.min(Math.max(k+d,0),v.length-1)];if(j!==undefined&&j!==cur)go(j)}
function addSel(){const io=pend();if(!io)return status("mark an in (I) and an out (O) first","err");const c=C(),ss=sels(c.id);
 ss.push(io);inPt=outPt=null;setSels(c.id,ss);act=sels(c.id).findIndex(([a])=>Math.abs(a-io[0])<0.01);renderAll()}
function delHere(){const c=C(),ss=sels(c.id),t=now();let k=act>=0&&ss[act]&&t>=ss[act][0]-0.05&&t<=ss[act][1]+0.05?act:
 ss.findIndex(([a,b])=>t>=a-0.05&&t<=b+0.05);if(k<0)return;ss.splice(k,1);act=-1;setSels(c.id,ss)}
function banIO(){const io=pend();if(!io)return status("mark an in–out to ban first","err");
 addVal(C().id,"ban",short(io[0])+"-"+short(io[1]),reel);inPt=outPt=null;save()}
$("banc").onclick=()=>{const id=C().id;if(banned(id).includes("all"))delVal(id,"ban","all",reel);else addVal(id,"ban","all",reel);save()};
$("bin").onclick=()=>{inPt=now();draw()};$("bout").onclick=()=>{outPt=now();draw()};$("badd").onclick=addSel;
function marks(){const c=C();return[...sels(c.id).flat(),...points(c.id),...c.moments.map(m=>m[0])].sort((a,b)=>a-b)}
function jump(d){const t=now(),p=marks();const x=d>0?p.find(v=>v>t+0.03):p.reverse().find(v=>v<t-0.03);if(x!==undefined)seek(x)}
// --- picks, in the order they play in the cut ---
// Each pick is matched to the reel shot that uses it (landscape first); picks the planner has not placed yet come
// after, in the order they were filmed. Tweaks keep a pick's place until the next re-plan.
const pv=$("pv");let order=[],pk=null,mode=null,pa=0,pb=1,pimg=new Image(),pdrag=null;const place=new Map();
const pkey=(id,a)=>id+"|"+a.toFixed(2);
let pfmt=ls.get("pfmt")||"L";
function shotOf(c,a,b){for(const[fmt,arr]of(pfmt=="P"?[["P",c.P],["L",c.L]]:[["L",c.L],["P",c.P]])){let best=null,ov=0;
  arr.forEach(([x,y,n])=>{const o=Math.min(b,y)-Math.max(a,x);if(o>ov+1e-6){ov=o;best=n}});
  if(best!==null&&ov>=Math.min(0.3,(b-a)/2))return{fmt,n:best}}return null}
// the chosen cut's shots come in too (k -1: not a pick yet): editing one makes it a pick, removing one bans its stretch
function allSels(){const out=[];D.clips.forEach((c,i)=>sels(c.id).forEach(([a,b],k)=>{if(!inReel(c.id,a,b))return;const key=pfmt+"|"+pkey(c.id,a);
  if(!place.has(key))place.set(key,shotOf(c,a,b));out.push({c,i,k,a,b,shot:place.get(key)})}));
 if(pfmt!="picks"){const have=new Set(out.filter(x=>x.shot&&x.shot.fmt==pfmt).map(x=>x.shot.n));
  D.clips.forEach((c,i)=>{const bans=banned(c.id).map(v=>v=="all"?[-1,1e9]:span(v));
   c[pfmt].forEach(([a,b,n])=>{if(!have.has(n)&&!bans.some(([u,v])=>(a+b)/2>=u&&(a+b)/2<=v))
    out.push({c,i,k:-1,a,b,shot:{fmt:pfmt,n},cut:true})})})}
 const rank=x=>x.shot?(x.shot.fmt==(pfmt=="P"?"P":"L")?0:1e4)+x.shot.n:1e9+x.i*1e3+x.a;return out.sort((x,y)=>rank(x)-rank(y))}
$("pfmts").onclick=ev=>{const b=ev.target.closest("button");if(!b)return;pfmt=b.dataset.f;ls.set("pfmt",pfmt);pk=null;pv.pause();selview()};
// several shots at once: tick a card's box (shift-click for a range), then drop them all in one go (one undo step)
const chosen=new Set();let lastCk=-1;
const ckey=x=>pfmt+"|"+x.c.id+"|"+x.a.toFixed(2)+"|"+x.b.toFixed(2);
function tick(j,range){const x=order[j];if(!x)return;const on=!chosen.has(ckey(x));
 const[lo,hi]=range&&lastCk>=0?[Math.min(lastCk,j),Math.max(lastCk,j)]:[j,j];
 for(let k=lo;k<=hi;k++){const y=order[k];if(y)on?chosen.add(ckey(y)):chosen.delete(ckey(y))}lastCk=j;selview()}
function dropMany(xs){const gone=new Map();
 xs.forEach(x=>{if(x.k>=0){if(!gone.has(x.c.id))gone.set(x.c.id,[]);gone.get(x.c.id).push([x.a,x.b])}
  else addVal(x.c.id,"ban",fmt(x.a)+"-"+fmt(x.b),reel)});
 gone.forEach((ab,id)=>{for(const[a,b]of ab)for(const f of["open","end"])if(same(roleIn(id,f,reel),a,b))clearRole(id,f,reel);
  setSels(id,sels(id).filter(([a,b])=>!ab.some(([u,v])=>Math.abs(a-u)<0.002&&Math.abs(b-v)<0.002)),true)});
 save()}
$("bulkall").onclick=()=>{order.forEach(x=>{if(x.cut)chosen.add(ckey(x))});selview()};
$("bulkclr").onclick=()=>{chosen.clear();lastCk=-1;selview()};
$("bulkdrop").onclick=()=>{const xs=order.filter(x=>chosen.has(ckey(x)));if(!xs.length)return;
 if(pk&&xs.some(x=>x.c.id==pk.id&&Math.abs(x.a-pk.a)<0.002)){pk=null;pv.pause()}
 chosen.clear();lastCk=-1;dropMany(xs);status(`dropped ${xs.length} shot${xs.length==1?"":"s"} — Ctrl+Z undoes`,"ok")};
const pj=()=>pk?order.findIndex(x=>x.c.id==pk.id&&Math.abs(x.a-pk.a)<0.002):-1;
const PX=()=>order[pj()];
function selview(){order=allSels();
 const live=new Set(order.map(ckey));[...chosen].forEach(k=>{if(!live.has(k))chosen.delete(k)});
 $("bulk").classList.toggle("on",chosen.size>0);$("cards").classList.toggle("picking",chosen.size>0);
 $("bulkn").textContent=chosen.size+" selected";const tot=order.reduce((s,x)=>s+x.b-x.a,0),j=pj(),placed=order.filter(x=>x.shot).length;
 const np=order.filter(x=>!x.cut).length,nc=order.length-np;
 $("pfmts").querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.f==pfmt));
 $("seltitle").textContent=(pfmt=="picks"?"":`${nc+order.filter(x=>!x.cut&&x.shot&&x.shot.fmt==pfmt).length} shots in the ${pfmt=="P"?"portrait":"landscape"} cut · `)+
  np+" pick"+(np==1?"":"s");
 $("seltot").textContent=short(tot)+" of footage · in cut order"+(order.length>placed?` · ${order.length-placed} not in the cut yet (I place them on the next re-plan)`:"")+
  (nc?" · trim, slip or drag a shot to make it a pick; ✕ bans it":"");
 $("seq").innerHTML=order.map((x,k)=>`<i data-j=${k} class="${k==j?"act":""}${chosen.has(ckey(x))?" chk":""}${x.shot?"":" new"}${x.cut?" cut"+(pfmt=="P"?" p":""):""}" style="flex:${(x.b-x.a).toFixed(2)}" title="${esc(x.c.id)} ${fmt(x.a)}–${fmt(x.b)}"></i>`).join("");
 let h="",sep=false;order.forEach((x,k)=>{if(!x.shot&&!sep){sep=true;h+=`<div class=catgrp>Not in the cut yet — placed on the next re-plan</div>`}
  const ck=chosen.has(ckey(x));
  h+=`<div class="card${k==j?" act":""}${x.cut?" cut":""}${ck?" chk":""}" data-j=${k}><div class=th data-j=${k}></div>
  <div class=bd><div><b>${x.shot?(x.shot.fmt=="L"?"Shot ":"Portrait shot ")+x.shot.n:"New"}</b> ${x.cut?"":"<span class=tag>pick</span> "}${!x.cut&&starsOf(x.c.id,x.a,x.b)?`<span style="color:var(--io)">${"★".repeat(starsOf(x.c.id,x.a,x.b))}</span> `:""}${!x.cut&&reel=="all"&&onlyOf(x.c.id,x.a,x.b)?`<span class=tag>${esc(onlyOf(x.c.id,x.a,x.b).map(rlab).join(" + "))} only</span> `:""}${{open:"<span class=tag>★ opening</span> ",end:"<span class=tag>⚑ ending</span> "}[roleOf(x.c.id,x.a,x.b)]||""}· ${esc(x.c.id)}<br>
  <small>${fmt(x.a)} – ${fmt(x.b)} · <b>${(x.b-x.a).toFixed(2)} s</b> · ${esc(x.c.clock)}</small></div>
  <div><button data-del=${k} title="${x.cut?"take this shot out of the cut (bans its stretch)":"remove this pick"}">✕</button></div></div></div>`});
 $("cards").innerHTML=h||`<p style="color:var(--mut)">No picks yet — choose footage in Sources.</p>`;
 const ths=$("cards").querySelectorAll(".th"),tw0=(ths[0]&&ths[0].clientWidth)||200;
 ths.forEach(el=>{const x=order[+el.dataset.j],hh=tw0*TH/TW;
  el.setAttribute("style",thumbAt(x.c,x.a,hh,tw0));
  el.innerHTML=`<span>${k1(el)}</span><input type=checkbox class=ck data-ck=${el.dataset.j} title="select (shift-click: a range)"${chosen.has(ckey(x))?" checked":""}>`});
 function k1(el){const x=order[+el.dataset.j];return (+el.dataset.j+1)+" · "+(x.b-x.a).toFixed(1)+"s"}
 $("ped").style.display=PX()?"block":"none";const px=PX();
 $("pstars").innerHTML=px&&px.k>=0?starBar(px.c.id,px.a,px.b,""):"";
 $("preels").innerHTML=px&&px.k>=0&&RS.length>1?"In "+RS.map(r=>`<button data-reel="${esc(r.id)}" class="${inReel(px.c.id,px.a,px.b,r.id)?"on":""}" title="put this pick in / take it out of the ${esc(r.label)} reel">${esc(r.label)}</button>`).join(""):"";
 if(px)pdraw()}
$("cards").onclick=ev=>{const c=ev.target.closest("[data-ck]");
 if(c){ev.stopPropagation();ev.preventDefault();tick(+c.dataset.ck,ev.shiftKey);return}
 const cd=ev.target.closest(".card");if(cd&&(ev.shiftKey||ev.ctrlKey||ev.metaKey)&&!ev.target.closest("[data-del]")){tick(+cd.dataset.j,ev.shiftKey);return}
 const b=ev.target.closest("[data-del],.card");if(!b)return;
 const x=order[+b.dataset.j??+b.dataset.del];
 if(b.dataset.del!==undefined){ev.stopPropagation();const y=order[+b.dataset.del];if(pk&&pk.id==y.c.id&&Math.abs(pk.a-y.a)<0.002){pk=null;pv.pause()}
  drop(y)}
 else pick(+b.dataset.j,false)};
$("seq").onclick=ev=>{const i=ev.target.closest("i");if(i)pick(+i.dataset.j,true)};
function pick(j,play){const x=order[j];if(!x)return stopAll();pk={id:x.c.id,a:x.a};
 const ctx=Math.max(2,(x.b-x.a)*0.75);pa=Math.max(0,x.a-ctx);pb=Math.min(x.c.dur,x.b+ctx);
 pimg=new Image();pimg.onload=pdraw;pimg.src=x.c.strip;
 const start=()=>{pv.currentTime=x.a;if(play)pv.play();pdraw()};
 if(pv.dataset.c!==x.c.id){pv.dataset.c=x.c.id;pv.src=x.c.video;pv.addEventListener("loadedmetadata",start,{once:true})}else start();
 selview();const card=$("cards").querySelector(`.card[data-j="${j}"]`);if(card)card.scrollIntoView({block:"nearest"})}
// change the current pick's in / out (lengthen, trim, slip); it keeps its place in the cut
function edit(a,b){const x=PX();if(!x)return;const L=x.c.dur;if(b-a<0.1)return;
 if(a<0){b-=a;a=0}if(b>L){a-=b-L;b=L}a=Math.max(0,a);
 const e0=P[x.c.id]||{};for(const f of["open","end"])if(same(e0[f],x.a,x.b)){const t=tagsOf(x.c.id,f,e0[f]);   // (its role goes with it)
  setTags(x.c.id,f,e0[f],null);e0[f]=fmt(a)+"-"+fmt(b);setTags(x.c.id,f,e0[f],t)}
 const ss=sels(x.c.id);if(x.k>=0)ss[x.k]=[a,b];else ss.push([a,b]);place.set(pfmt+"|"+pkey(x.c.id,a),x.shot);pk={id:x.c.id,a};
 if(a<pa||b>pb){const ctx=Math.max(2,(b-a)*0.75);pa=Math.max(0,a-ctx);pb=Math.min(L,b+ctx)}
 setSels(x.c.id,ss)}
const pnow=()=>pv._want??(pv.currentTime||0);
function pdraw(){const x=PX();if(!x)return;const T=$("pt"),W=pb-pa,t=pnow();
 film($("pstrip").querySelector("canvas"),pa,pb,64,x.c,pimg);
 T.querySelectorAll(":scope>i,#pstrip>i").forEach(e=>e.remove());let a=x.a,b=x.b;if(pdrag&&pdrag.cur)[a,b]=pdrag.cur;
 let h=`<i class=dim style="${wspan(pa,a,pa,pb)}"></i><i class=dim style="${wspan(b,pb,pa,pb)}"></i>`;
 sels(x.c.id).forEach(([u,v],k)=>{if(k!=x.k&&v>pa&&u<pb)h+=`<i class=oth style="${wspan(u,v,pa,pb)}" title="another pick"></i>`});
 h+=`<i class="blk act" data-k=0 style="${wspan(a,b,pa,pb)}"><i class="h l"></i>${(b-a).toFixed(2)} s<i class="h r"></i></i>`;
 fkeys(x.c.id).forEach(([k])=>{if(k>=pa&&k<=pb)h+=`<i class=fk style="left:${X(k,pa,pb)}" title="focal point ${fmt(k)}"></i>`});
 if(t>=pa&&t<=pb)h+=`<i class=ph style="left:${X(t,pa,pb)}"></i>`;
 $("pstrip").insertAdjacentHTML("beforeend",h);
 const step=[0.1,0.2,0.5,1,2,5,10].find(s=>W/s<=40)||30;let r="";
 for(let v=Math.ceil(pa/step)*step;v<=pb;v+=step)r+=`<i class="tk" style="left:${X(v,pa,pb)}"></i>`;
 $("pruler").innerHTML=r+`<i class=lb style="left:${X(a,pa,pb)}">in ${fmt(a)}</i><i class=lb style="left:${X(b,pa,pb)}">out ${fmt(b)}</i>`;
 $("plen").textContent=(b-a).toFixed(2)+" s";
 const ro=roleOf(x.c.id,x.a,x.b);$("pfirst").classList.toggle("on",ro=="open");$("plast").classList.toggle("on",ro=="end");
 $("pvtc").textContent=(pj()+1)+"/"+order.length+" · "+x.c.id+" · "+fmt(t);
 $("pplay").textContent=pv.paused?"▶ Play pick":"❚❚ Pause";$("ploop").classList.toggle("on",mode=="loop");
 const st=$("pstill"),no=pv.error||pv.readyState<2&&!pv.seeking||!pv.videoWidth;st.classList.toggle("on",!!no);
 if(no){const R=$("pvwrap").getBoundingClientRect(),[tw,th]=tileOf(x.c),hh=Math.min(R.height,R.width*th/tw);
  st.setAttribute("style",thumbAt(x.c,t,hh)+`;width:${hh*tw/th}px;height:${hh}px;left:${(R.width-hh*tw/th)/2}px;top:${(R.height-hh)/2}px`)}
 frameDraw($("pfr"),$("pvwrap"),pv,x.c,x.a,x.b,t)}
const PT=$("pstrip");
PT.addEventListener("pointerdown",ev=>{const x=PX();if(!x)return;pv.pause();PT.setPointerCapture(ev.pointerId);
 const t=tAt(PT,ev,pa,pb),blk=ev.target.closest(".blk");
 pdrag=blk?{k:ev.target.classList.contains("l")?"l":ev.target.classList.contains("r")?"r":"slip",t0:t,o:[x.a,x.b],cur:[x.a,x.b]}:{k:"scrub"};
 if(!blk)goTo(pv,t);pdraw()});
PT.addEventListener("pointermove",ev=>{const x=PX();if(!x)return;const t=tAt(PT,ev,pa,pb);
 hov.style.display="block";hov.querySelector("div").setAttribute("style",thumbAt(x.c,t,108,192));hov.querySelector("span").textContent=fmt(t);
 const R=PT.getBoundingClientRect();hov.style.left=Math.min(innerWidth-205,Math.max(4,ev.clientX-100))+"px";hov.style.top=(R.top-136)+"px";
 if(!pdrag)return;if(pdrag.k=="scrub"){goTo(pv,t);pdraw();return}
 let[a,b]=pdrag.o;const d=t-pdrag.t0;
 if(pdrag.k=="slip"){const w=b-a;a=Math.max(0,Math.min(a+d,x.c.dur-w));b=a+w;goTo(pv,a)}
 else if(pdrag.k=="l"){a=Math.max(0,Math.min(t,b-0.1));goTo(pv,a)}else{b=Math.min(x.c.dur,Math.max(t,a+0.1));goTo(pv,b)}
 pdrag.cur=[a,b];pdraw()});
PT.addEventListener("pointerup",()=>{const d=pdrag;pdrag=null;if(d&&d.cur&&(d.cur[0]!==d.o[0]||d.cur[1]!==d.o[1]))edit(...d.cur);else pdraw()});
PT.addEventListener("pointerleave",()=>hov.style.display="none");
PT.addEventListener("wheel",ev=>{ev.preventDefault();const x=PX();if(!x)return;const R=PT.getBoundingClientRect(),f=(ev.clientX-R.left)/R.width,
 at=pa+f*(pb-pa),w=Math.min(Math.max((pb-pa)*(ev.deltaY>0?1.25:0.8),1),x.c.dur);pa=Math.max(0,Math.min(at-f*w,x.c.dur-w));pb=pa+w;pdraw()},{passive:false});
pv.addEventListener("timeupdate",()=>{const x=PX();if(!x){return}
 if(!pv.paused&&pnow()>=x.b-0.02){if(mode=="loop")pv.currentTime=x.a;else if(mode=="all"){const j=pj()+1;order[j]?pick(j,true):stopAll()}else pv.pause()}
 pdraw()});
for(const e of["seeked","pause","play","loadeddata","error"])pv.addEventListener(e,pdraw);
pv.addEventListener("play",()=>{const f=()=>{pdraw();const x=PX();if(x&&mode=="loop"&&pnow()>=x.b-0.02)pv.currentTime=x.a;if(!pv.paused)requestAnimationFrame(f)};f()});
function playPick(){const x=PX();if(!x)return order.length&&pick(0,true);if(!pv.paused)return pv.pause();
 if(mode!="all"&&mode!="loop")mode="one";if(pnow()<x.a-0.05||pnow()>=x.b-0.05)pv.currentTime=x.a;pv.play()}
function stopAll(){pv.pause();mode=null;pdraw()}
function nudge(which,d){const x=PX();if(!x)return;let[a,b]=[x.a,x.b];
 if(which=="in")a=Math.min(a+d,b-0.1);else if(which=="out")b=Math.max(b+d,a+0.1);else{a+=d;b+=d}
 edit(a,b);pv.currentTime=which=="out"?b:a}
$("ped").onclick=ev=>{const st=ev.target.closest("[data-star]");if(st){const x=PX();if(x&&x.k>=0){const n=+st.dataset.star;
  setStars(x.c.id,x.a,x.b,starsOf(x.c.id,x.a,x.b)==n?0:n)}return}
 const b=ev.target.closest("button");if(!b)return;const x=PX();
 if(b.dataset.n){const[w,d]=b.dataset.n.split(":");nudge(w,+d*(ev.shiftKey?10:1))}
 else if(b.id=="pplay")playPick();else if(b.id=="ploop"){mode=mode=="loop"?"one":"loop";playPick()}
 else if(b.id=="pprev")pick(Math.max(0,pj()-1),!pv.paused);else if(b.id=="pnext")pick(Math.min(order.length-1,pj()+1),!pv.paused)
 else if(b.id=="pin")edit(Math.min(pnow(),x.b-0.1),x.b);else if(b.id=="pout")edit(x.a,Math.max(pnow(),x.a+0.1))
 else if(b.id=="popen"){view(1);go(x.i);act=x.k;vid.addEventListener("loadedmetadata",()=>seek(x.a),{once:true})}
 else if(b.dataset.reel){if(x.k>=0){const r=b.dataset.reel,on=!inReel(x.c.id,x.a,x.b,r);
  if(!on&&r==reel&&(onlyOf(x.c.id,x.a,x.b)||RID).length>1){pk=null;pv.pause()}setIn(x.c.id,x.a,x.b,r,on)}}
 else if(b.id=="pfirst")setRole(x,"open");else if(b.id=="plast")setRole(x,"end")
 else if(b.id=="pdel"){const j=pj();pk=null;pv.pause();drop(x);if(order[j])pick(j,false)}};
// a pick goes; a shot of the cut is banned, so the next re-plan puts something else there
function drop(x){if(x.k>=0){for(const f of["open","end"])if(same(roleIn(x.c.id,f,reel),x.a,x.b))clearRole(x.c.id,f,reel);
  const ss=sels(x.c.id);ss.splice(x.k,1);setSels(x.c.id,ss)}
 else{addVal(x.c.id,"ban",fmt(x.a)+"-"+fmt(x.b),reel);save()}}
$("playall").onclick=()=>{mode="all";const j=Math.max(0,pj());pick(j,true)};
$("stopall").onclick=stopAll;
// --- portrait framing: the crop the planner derives (served by /api/crop) and the editor's focal points, picks
// "portrait": [["m:ss.ss", x, y], ...] (fractions of the frame); the reel's crop centre follows them, its size as derived
let frameOn=!!ls.get("frame");const crops=new Map();
const fkeys=id=>((P[id]||{}).portrait||[]).map(([t,x,y])=>[secs(t),+x,+y]).sort((p,q)=>p[0]-q[0]);
// the person to follow: picks "follow": [["m:ss.ss", x, y], ...] where the editor clicked them; the server's crop
// (/api/crop) then tracks their face through the shot, so the page asks for it again once the click is saved
const folkeys=id=>((P[id]||{}).follow||[]).map(([t,x,y])=>[secs(t),+x,+y]).sort((p,q)=>p[0]-q[0]);
let followOn=false,staleCrop=null;
function setFollow(id,t,x,y,a,b){const e=ent(id),ks=folkeys(id).filter(k=>k[0]<a-0.5||k[0]>b+0.5);
 if(x!=null)ks.push([t,x,y]);if(ks.length)e.follow=ks.sort((p,q)=>p[0]-q[0]).map(([t,x,y])=>[fmt(t),+x.toFixed(4),+y.toFixed(4)]);
 else delete e.follow;staleCrop=id;save()}
function followMode(on){followOn=on;if(on&&!frameOn)toggleFrame();document.querySelectorAll(".fol").forEach(b=>b.classList.toggle("on",on));
 document.querySelectorAll(".fr").forEach(f=>f.classList.toggle("follow",on));if(on)status("click the dancer to follow","ok")}
const putKeys=(id,ks)=>{ent(id).portrait=ks.sort((p,q)=>p[0]-q[0]).map(([t,x,y])=>[fmt(t),+x.toFixed(4),+y.toFixed(4)])};
const clamp=(v,lo,hi)=>Math.min(Math.max(v,lo),hi);
const lerp=(ts,vs,t)=>{if(t<=ts[0])return vs[0];for(let i=1;i<ts.length;i++)if(t<=ts[i]){const f=(t-ts[i-1])/(ts[i]-ts[i-1]||1);
 return vs[i-1]+f*(vs[i]-vs[i-1])}return vs[vs.length-1]};
// one request at a time (each is worked out by the server): a drag asks for a new stretch on every move, so only the
// latest one waiting is fetched next, and the clip's last crop stands in meanwhile. Returns [crop, its a] or a state.
let cropBusy=false,cropNext=null;const lastCrop=new Map();
function fetchCrop(c,a,b,k){cropBusy=true;
 fetch(`${D.root}api/crop?id=${encodeURIComponent(c.id)}&a=${a.toFixed(3)}&b=${b.toFixed(3)}&fmt=portrait`)
  .then(r=>r.ok?r.json():Promise.reject()).then(j=>{crops.set(k,[j,a]);lastCrop.set(c.id,[j,a])}).catch(()=>crops.set(k,false))
  .then(()=>{cropBusy=false;const n=cropNext;cropNext=null;if(n&&!crops.has(n[3]))fetchCrop(...n);draw();pdraw()})}
function derived(c,a,b){const k=c.id+"|"+a.toFixed(2)+"|"+b.toFixed(2);if(crops.has(k))return crops.get(k)??lastCrop.get(c.id);
 if(!server){crops.set(k,false);return false}
 if(cropBusy)cropNext=[c,a,b,k];else{crops.set(k,undefined);fetchCrop(c,a,b,k)}return lastCrop.get(c.id)}
// where the frame is worked out from: the selection or portrait reel shot under the playhead, else the 3 s around it
function rangeFor(c,t){const s=sels(c.id).find(([a,b])=>t>=a-0.02&&t<=b+0.02)||c.P.find(([a,b])=>t>=a&&t<=b);
 if(s)return[s[0],s[1]];const a=Math.max(0,Math.min(Math.floor(t/3)*3,c.dur-3));return[a,Math.min(a+3,c.dur)]}
function frameAt(c,a,b,t){const g=derived(c,a,b);if(g===undefined)return"wait";if(g===false)return"off";
 const[cr,a0]=g;if(cr===null)return"fills";
 const ts=cr.path.map(q=>a0+q[0]),d={x:lerp(ts,cr.path.map(q=>q[1]),t),y:lerp(ts,cr.path.map(q=>q[2]),t)};
 const ks=fkeys(c.id).filter(k=>k[0]>=a-0.5&&k[0]<=b+0.5);let o=null;
 if(ks.length){const kt=ks.map(k=>k[0]);o={x:clamp(lerp(kt,ks.map(k=>k[1]),t),cr.w/2,1-cr.w/2),
  y:clamp(lerp(kt,ks.map(k=>k[2]),t),cr.h/2,1-cr.h/2)}}
 return{d,o,w:cr.w,h:cr.h,z:cr.z,fol:cr.follow}}
function vrect(wrap,v){const R=wrap.getBoundingClientRect(),ar=v.videoWidth&&v.videoHeight?v.videoWidth/v.videoHeight:(C()?.ar||TW/TH);
 const w=Math.min(R.width,R.height*ar),h=w/ar;return{l:(R.width-w)/2,t:(R.height-h)/2,w,h}}
function frameDraw(el,wrap,v,c,a,b,t){el.classList.toggle("on",frameOn);$("fhelp").hidden=!frameOn;document.querySelectorAll(".frt").forEach(x=>x.classList.toggle("on",frameOn));
 if(!frameOn){el.innerHTML="";return}const F=frameAt(c,a,b,t);
 if(typeof F=="string"){el.innerHTML=`<span class=frl>${{fills:"this clip fills the portrait frame as it is — no crop to place",
  off:"the portrait frame needs the page served by <code>focalcut reel serve</code>",wait:"working out the portrait crop…"}[F]}</span>`;return}
 const r=vrect(wrap,v),box=(p,cls)=>`<i class="fbox ${cls}" style="left:${r.l+(p.x-F.w/2)*r.w}px;top:${r.t+(p.y-F.h/2)*r.h}px;width:${F.w*r.w}px;height:${F.h*r.h}px"></i>`;
 const A=F.o||F.d,dx=F.o?F.o.x-F.d.x:0,pin=folkeys(c.id).find(k=>Math.abs(k[0]-t)<=0.6);
 el.innerHTML=box(A,"act")+(F.o?box(F.d,"der"):"")+`<i class=fdot style="left:${r.l+A.x*r.w}px;top:${r.t+A.y*r.h}px"></i>`+
  (pin?`<i class=fpin style="left:${r.l+pin[1]*r.w}px;top:${r.t+pin[2]*r.h}px" title="the person you chose to follow"></i>`:"")+
  `<span class=frl>${F.fol=="face"?"<b>following the person you clicked</b> · ":F.fol=="point"?"<b>no face found there: held on your click</b> · ":""}planner x ${F.d.x.toFixed(2)} y ${F.d.y.toFixed(2)}${F.z>1.001?" · zoom "+F.z.toFixed(2):""}`+
  (F.o?` → <b>your focus x ${F.o.x.toFixed(2)} y ${F.o.y.toFixed(2)} (${dx>=0?"+":""}${dx.toFixed(2)})</b>`:" · drag to set a focal point")+`</span>`}
function toggleFrame(){frameOn=!frameOn;ls.set("frame",frameOn);draw();pdraw()}
// drag on the picture: a focal point at the playhead (one within ¼ s of it moves instead)
function frameDrag(el,wrap,v,ctx,redraw,stop){let x=null;
 const put=ev=>{const r=vrect(wrap,v),R=wrap.getBoundingClientRect();
  const ks=fkeys(x.c.id).filter(k=>Math.abs(k[0]-x.t)>0.25);
  ks.push([x.t,clamp((ev.clientX-R.left-r.l)/r.w,0,1),clamp((ev.clientY-R.top-r.t)/r.h,0,1)]);putKeys(x.c.id,ks);redraw()};
 el.addEventListener("pointerdown",ev=>{if(!frameOn)return;x=ctx();if(!x)return;stop();
  if(followOn){const r=vrect(wrap,v),R=wrap.getBoundingClientRect(),[a,b]=x.r||rangeFor(x.c,x.t);
   setFollow(x.c.id,x.t,clamp((ev.clientX-R.left-r.l)/r.w,0,1),clamp((ev.clientY-R.top-r.t)/r.h,0,1),a,b);
   x=null;followMode(false);status("following that person — the crop updates in a moment","ok");return}
  el.setPointerCapture(ev.pointerId);put(ev)});
 el.addEventListener("pointermove",ev=>{if(x)put(ev)});
 el.addEventListener("pointerup",()=>{if(x){x=null;save()}})}
frameDrag($("fr"),$("vwrap"),vid,()=>({c:C(),t:now()}),draw,pause);
frameDrag($("pfr"),$("pvwrap"),pv,()=>PX()&&{c:PX().c,t:pnow(),r:[PX().a,PX().b]},pdraw,()=>pv.pause());
function clearFocus(c,a,b){const ks=fkeys(c.id),kept=ks.filter(k=>k[0]<a-0.5||k[0]>b+0.5),fl=folkeys(c.id).some(k=>k[0]>=a-0.5&&k[0]<=b+0.5);
 if(kept.length==ks.length&&!fl)return status("no focal points here");putKeys(c.id,kept);if(fl)setFollow(c.id,0,null,null,a,b);else save()}
$("bfollow").onclick=$("pfollow").onclick=()=>followMode(!followOn);
$("bframe").onclick=$("pframe").onclick=toggleFrame;
$("fclr").onclick=()=>clearFocus(C(),...rangeFor(C(),now()));
$("pfclr").onclick=()=>{const x=PX();if(x)clearFocus(x.c,x.a,x.b)};
// views
function view(n){$("clipsview").style.display=n==1?"grid":"none";$("selview").style.display=n==2?"grid":"none";
 $("v1").classList.toggle("on",n==1);$("v2").classList.toggle("on",n==2);if(n==2){pause();selview()}else{pv.pause();paint()}}
$("v1").onclick=()=>view(1);$("v2").onclick=()=>view(2);
$("copy").onclick=()=>{navigator.clipboard&&navigator.clipboard.writeText(JSON.stringify(P,null,1));$("copy").textContent="Copied ✓";
 setTimeout(()=>$("copy").textContent="Copy picks",1500)};
$("dl").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(P,null,1)],
 {type:"application/json"}));a.download="picks.json";a.click()};
addEventListener("resize",()=>paint());
function undo(){if(!hist.length)return status("nothing to undo");P=JSON.parse(hist.pop());lastSnap=JSON.stringify(P);
 tidy();if(server){clearTimeout(timer);timer=setTimeout(push,250)}else ls.set("picks",P);status("undone");renderAll()}
addEventListener("keydown",ev=>{if((ev.ctrlKey||ev.metaKey)&&ev.key.toLowerCase()=="z"){ev.preventDefault();undo();return}},true);
addEventListener("keydown",ev=>{if(ev.target.tagName=="SELECT"||ev.target.tagName=="INPUT"||ev.ctrlKey||ev.metaKey||ev.altKey)return;
 const k=ev.key,K=k.toLowerCase();let done=true;
 if(k=="Tab")view($("selview").style.display=="grid"?1:2);
 else if(K=="f")toggleFrame();else if(K=="g")followMode(!followOn);else if(k=="Escape"&&followOn)followMode(false);
 else if($("selview").style.display=="grid"){const x=PX();
  if(chosen.size&&(k=="Backspace"||k=="Delete"))$("bulkdrop").click();
  else if(k=="Escape"&&chosen.size)$("bulkclr").click();
  else if(K=="a"&&!ev.shiftKey)$("bulkall").click();
  else if(K=="x"&&x)tick(pj(),false);
  else if(k==" ")playPick();else if(k=="Escape")stopAll();
  else if(k=="ArrowRight"||k=="ArrowDown")pick(Math.min(order.length-1,pj()+1),!pv.paused);
  else if(k=="ArrowLeft"||k=="ArrowUp")pick(Math.max(0,pj()-1),!pv.paused);
  else if(!x)done=false;
  else if(K=="i")edit(Math.min(pnow(),x.b-0.1),x.b);else if(K=="o")edit(x.a,Math.max(pnow(),x.a+0.1));
  else if(k==","||k=="<")nudge("slip",ev.shiftKey?-0.5:-0.02);else if(k=="."||k==">")nudge("slip",ev.shiftKey?0.5:0.02);
  else if(k=="\\"){mode=mode=="loop"?"one":"loop";playPick()}
  else if(k=="Backspace"||k=="Delete")$("pdel").click();
  else if(/^[1-5]$/.test(k)&&x.k>=0){const n=+k;setStars(x.c.id,x.a,x.b,starsOf(x.c.id,x.a,x.b)==n?0:n)}else done=false}
 else if(/^[1-5]$/.test(k)){const ss=sels(C().id),t=now();let j=act>=0&&ss[act]&&t>=ss[act][0]-0.05&&t<=ss[act][1]+0.05?act:ss.findIndex(([a,b])=>t>=a-0.05&&t<=b+0.05);
  if(j<0)j=act;if(j>=0&&ss[j]){const n=+k;setStars(C().id,ss[j][0],ss[j][1],starsOf(C().id,ss[j][0],ss[j][1])==n?0:n)}else status("put the playhead on a selection to star it","err")}
 else if(k==" "||K=="k")vid.paused&&!rew?(playOne=false,playL()):pause();
 else if(K=="l")playL();else if(K=="j")playJ();
 else if(k=="ArrowLeft"||k==",")step(ev.shiftKey?-1:-FR);else if(k=="ArrowRight"||k==".")step(ev.shiftKey?1:FR);
 else if(k=="ArrowUp"||k=="PageUp")nav(-1);else if(k=="ArrowDown"||k=="PageDown")nav(1);
 else if(K=="i"){inPt=now();if(outPt!==null&&outPt<=inPt)outPt=null;draw()}
 else if(K=="o"){outPt=now();if(inPt!==null&&inPt>=outPt)inPt=null;draw()}
 else if(k=="Enter"||K=="a")addSel();else if(K=="x"){inPt=outPt=null;draw()}
 else if(k=="Backspace"||k=="Delete")delHere();else if(k=="[")jump(-1);else if(k=="]")jump(1);
 else if(k=="\\"){const ss=sels(C().id);if(act<0)act=ss.findIndex(([a,b])=>now()>=a&&now()<=b);if(act>=0){loop=!loop;seek(ss[act][0]);vid.play()}}
 else if(K=="b")banIO();else if(k=="="||k=="+")zoom(0.6);else if(k=="-"||k=="_")zoom(1.6);
 else if(k=="0"){va=0;vb=dur();paint()}else if(k=="Home")seek(0);else if(k=="End")seek(dur());
 else done=false;if(done){ev.preventDefault();ev.stopPropagation()}},true);
boot();
</script>"""
