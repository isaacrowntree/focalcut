"""The source browser: one page for the whole project, whatever the format -- every export clip in a list, one at a
time in a big player over a wide, zoomable timeline (filmstrip, the moments the planner found, where the landscape and
portrait reels use it), where the editor marks as many in-out selections per clip as they like, and a final view of
every selection that plays them back to back.

    review/sources.html  (+ review/sources/<unit>.jpg: a filmstrip per clip, one frame a second)

Selections are picks "add" ranges ("1:12.40-1:15.10", export time): the planner puts each in the reel as one shot, from its
in-point, for its length. Served by `focalcut reel serve`, the page saves picks.json itself on every change; opened as a file, it
keeps changes in the browser and offers a download.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
from urllib.parse import quote

TILE = (160, 90)          # filmstrip frame size
MAX_TILES = 400           # a longer clip gets a frame every dur / MAX_TILES seconds


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


def filmstrip(video, jpg, dur):
    """One row of frames across the clip, from keyframes only (the exports have one every 0.5 s), so it is quick."""
    step = max(1.0, dur / MAX_TILES)
    n = max(1, math.ceil(dur / step))
    if not os.path.exists(jpg):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-skip_frame", "nokey", "-i", video, "-vf",
                        f"fps=1/{step},scale={TILE[0]}:{TILE[1]},tile={n}x1", "-frames:v", "1", "-q:v", "5", jpg],
                       check=False)
    return step, n


def clips(project, page_dir, plans):
    """Every unit with an export file: its video and filmstrip (relative to the page), clock time, the moments found
    in it and the shots each reel takes from it, all in the export's own timeline."""
    units = project.load("units.json")
    moments = {r["unit"]["id"]: r for r in project.load("moments.json")} if os.path.exists(
        project.path("moments.json")) else {}
    clock_shift = float(project.config.get("clock_offset_h", 0.0))
    strips = os.path.join(page_dir, "sources")
    os.makedirs(strips, exist_ok=True)
    out = []
    for u in units:
        f = os.path.join(project.export_dir, u.get("category", ""), u["id"] + ".mp4")
        if "#" in u["id"] or not os.path.exists(f):
            continue
        x0 = u.get("export_t0", 0.0)
        d = duration(f)
        step, n = filmstrip(f, os.path.join(strips, u["id"] + ".jpg"), d)
        src = os.path.join(project.folder, u["source"])
        c = created(src) if os.path.exists(src) else None
        clock = (c + dt.timedelta(hours=clock_shift, seconds=u["t0"])).strftime("%H:%M") if c else ""
        ms = moments.get(u["id"], {}).get("moments", [])
        uses = {fmt: [[round(s["src_in"] - x0, 2), round(s["src_in"] - x0 + s["length"], 2), i + 1]
                      for i, s in enumerate(p["shots"]) if s["unit"]["id"] == u["id"]] for fmt, p in plans.items()}
        out.append(dict(id=u["id"], cat=u.get("category", ""), tag=u.get("tag", ""), clock=clock,
                        sort=(c + dt.timedelta(seconds=u["t0"])).isoformat() if c else u["id"], dur=round(d, 2),
                        video=quote(os.path.relpath(f, page_dir).replace(os.sep, "/")),
                        strip=f"sources/{quote(u['id'])}.jpg", step=step, n=n,
                        moments=[[round(m["t"] - x0, 2), m["kind"]] for m in ms if m["t"] - x0 >= 0],
                        L=uses.get("landscape", []), P=uses.get("portrait", [])))
    out.sort(key=lambda c: c["sort"])
    return out


def page(clips, picks=None, key="sources", title="Sources", root=""):
    """root: the project folder relative to the page ("../../../"), so its API calls work under any server prefix."""
    data = json.dumps(dict(clips=clips, picks=picks or {}, key=key, tile=TILE, root=root)).replace("</", "<\\/")
    return PAGE.replace("%DATA%", data).replace("%TITLE%", title)


def api_root(project) -> str:
    """The project folder relative to its review pages, where `focalcut reel serve` answers api/... for it."""
    return os.path.relpath(project.folder, project.path("review")).replace(os.sep, "/") + "/"


def run(project, plans=None) -> str:
    page_dir = project.path("review")
    os.makedirs(page_dir, exist_ok=True)
    if plans is None:
        plans = {f: project.load(f"plan_{f}.json") for f in ("landscape", "portrait")
                 if os.path.exists(project.path(f"plan_{f}.json"))}
    cs = clips(project, page_dir, plans)
    path = os.path.join(page_dir, "sources.html")
    with open(path, "w") as f:
        f.write(page(cs, project.picks(), key=project.name, title=f"{project.name} — sources", root=api_root(project)))
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
header h1{font-size:15px;margin:0 8px 0 0}#saved{color:var(--mut);font-size:12px}#saved.ok{color:var(--sel)}
#saved.err{color:var(--ban)}.sp{flex:1}
#views{display:flex;gap:4px}#counts{color:var(--mut);font-size:12px}#counts b{color:var(--fg)}
#counts .y b{color:var(--sel)}#counts .f b{color:var(--io)}#counts .x b{color:var(--ban)}
#clipsview{display:grid;grid-template-columns:290px 1fr;min-height:0}
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
#lsel{height:34px;cursor:copy}#lland,#lport{height:14px}
.tl i{position:absolute;display:block;font-style:normal}
.tk{top:12px;bottom:0;width:1px;background:var(--line)}.tk.big{top:6px;background:var(--mut)}
.lb{top:1px;font-size:10px;color:var(--mut);transform:translateX(3px);white-space:nowrap}
.mom{top:0;bottom:0;width:2px;background:var(--mom);opacity:.85;pointer-events:none}
.blk{top:3px;bottom:3px;border-radius:4px;background:color-mix(in srgb,var(--sel) 45%,transparent);
border:2px solid var(--sel);cursor:grab;font-size:11px;color:var(--fg);padding-left:4px;overflow:hidden;
white-space:nowrap;font-weight:600}.blk.act{background:color-mix(in srgb,var(--sel) 75%,transparent);color:#000}
.blk .h{position:absolute;top:0;bottom:0;width:7px;cursor:ew-resize}.blk .h.l{left:-2px}.blk .h.r{right:-2px}
.pend{top:3px;bottom:3px;border:2px dashed var(--io);background:color-mix(in srgb,var(--io) 22%,transparent);
border-radius:4px;pointer-events:none}
.pt{top:2px;bottom:2px;width:3px;background:var(--sel);border-radius:2px}
.ban{top:0;bottom:0;background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--ban) 40%,transparent) 0 6px,
transparent 6px 12px);pointer-events:none}
.use{top:2px;bottom:2px;border-radius:3px;opacity:.9;font-size:9px;color:#000;overflow:hidden;padding-left:2px}
#lland .use{background:var(--land)}#lport .use{background:var(--port)}.use{cursor:copy}.use:hover{outline:2px solid var(--fg)}
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
#pfmts{display:inline-flex;gap:2px}.card.cut{opacity:.8}.card .tag{font-size:10px;border-radius:6px;padding:0 5px;background:var(--sel);color:#000}
.card.cut .tag{background:var(--line);color:var(--fg)}.card{cursor:pointer}#pvtc{position:absolute;left:8px;top:8px;background:#000a;color:#fff;
padding:2px 8px;border-radius:4px;font:600 13px ui-monospace,monospace}
#seq{display:flex;height:26px;border-radius:6px;overflow:hidden;border:1px solid var(--line)}
#seq i{display:block;height:100%;border-right:1px solid var(--bg);background:var(--sel);opacity:.75;cursor:pointer}
#seq i:hover,#seq i.act{opacity:1;background:var(--io)}
#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.card.act{outline:2px solid var(--io)}
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
.fk{top:1px;bottom:1px;width:6px;margin-left:-3px;background:var(--io);border-radius:2px;pointer-events:none;z-index:2}
</style>
<header><h1>%TITLE%</h1>
<div id=views><button id=v1 class=on>Sources</button><button id=v2 title="the cut, shot by shot, with your picks in it">Cut &amp; picks</button></div><span id=counts></span>
<span class=sp></span><span id=saved></span>
<button id=copy title="copy picks.json to the clipboard">Copy picks</button><button id=dl>Download picks.json</button>
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
<button id=fclr title="remove the focal points around the playhead (this selection, reel shot or 3 s)">Clear focus here</button>
<span style="color:var(--mut);font-size:12px">dashed pink: what the planner works out · yellow: what the portrait reel shows — drag to set a focal point at the playhead; set more at other times and the crop glides between them</span></div>
<div id=ov class=tl title="the whole clip; the yellow box is what the timeline below shows — click or drag to move"><canvas></canvas></div>
<div id=tl class=tl>
<div id=ruler title="click or drag to scrub"></div>
<div id=strip title="click or drag to scrub; scroll to zoom, shift+scroll to pan"><canvas></canvas></div>
<div id=lsel class=lane title="drag here to make a selection; drag a selection to move it, its edges to trim"><span class=lab>selections</span></div>
<div id=lland class=lane><span class=lab>landscape</span></div><div id=lport class=lane><span class=lab>portrait</span></div>
</div>
<div id=mysel></div>
<div id=keys><kbd>Space</kbd>/<kbd>K</kbd> play · <kbd>J</kbd>/<kbd>L</kbd> shuttle back / on (again: faster) ·
<kbd>←</kbd><kbd>→</kbd> frame (<kbd>Shift</kbd> 1 s) · <kbd>I</kbd> in · <kbd>O</kbd> out · <kbd>Enter</kbd> add
selection · <kbd>X</kbd> clear in–out · <kbd>Backspace</kbd> delete selection at playhead · <kbd>[</kbd><kbd>]</kbd>
previous / next mark · <kbd>\</kbd> loop selection · <kbd>↑</kbd><kbd>↓</kbd> previous / next clip ·
<kbd>=</kbd><kbd>−</kbd> zoom · <kbd>0</kbd> fit · <kbd>B</kbd> ban in–out · <kbd>F</kbd> portrait frame · <kbd>Tab</kbd> Picks view<br>
Or drag in the <b style="color:var(--sel)">selections</b> lane to mark a range directly; drag a block to move it,
its edges to trim. <span style="color:var(--land)">■</span> landscape and <span style="color:var(--port)">■</span>
portrait show what the current reels use; <span style="color:var(--mom)">|</span> moments the planner found.
Each selection goes in the reel starting at its in-point.</div>
</div></section>
<section id=selview>
<div id=selhead><h2 id=seltitle></h2><span id=seltot style="color:var(--mut)"></span><span class=sp></span>
<span id=pfmts><button data-f=L title="the landscape reel's shots, in order, with your picks">Landscape cut</button><button data-f=P title="the portrait reel's shots, in order, with your picks">Portrait cut</button><button data-f=picks title="only the picks you made">Picks only</button></span>
<button id=playall class=go>▶ Play all</button><button id=stopall>■ Stop</button></div>
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
<button id=pfclr title="remove this pick's focal points">Clear focus</button><span class=sp></span><button id=popen title="open in Sources">Open in Sources</button><button id=pdel title="remove (Backspace)">✕ Remove</button></div>
<div id=pt class=tl><div id=pruler></div><div id=pstrip title="drag the edges to trim or lengthen, the block to slip it, anywhere else to scrub; scroll to zoom"><canvas></canvas></div></div>
<p style="color:var(--mut);font-size:12px;margin:6px 0 0">Drag the pick's edges to lengthen or trim, drag the block to slip it, click
elsewhere to scrub · <kbd>Space</kbd> play · <kbd>←</kbd><kbd>→</kbd> previous / next pick · <kbd>I</kbd>/<kbd>O</kbd> in / out
at the playhead · <kbd>,</kbd><kbd>.</kbd> slip a frame (<kbd>Shift</kbd> ½ s) · <kbd>\</kbd> loop · <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo · <kbd>Tab</kbd> Sources</p></div>
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
function tidy(){for(const k in P){const e=P[k];if(k[0]=="_"||!e||typeof e!=="object"||Array.isArray(e))continue;
 for(const f of["ban","prefer","add","portrait"])if(e[f]&&!e[f].length)delete e[f];if(e.more===0)delete e.more;
 if(!Object.keys(e).length)delete P[k]}}
function status(t,c){const s=$("saved");s.textContent=t;s.className=c||""}
const hist=[];let lastSnap=JSON.stringify(P);
function save(){tidy();{const now_=JSON.stringify(P);if(now_!==lastSnap){hist.push(lastSnap);if(hist.length>200)hist.shift();lastSnap=now_}}if(server){status("saving…");clearTimeout(timer);timer=setTimeout(push,250)}
 else{ls.set("picks",P);status("kept in this browser — run `focalcut reel serve` to save to picks.json","err")}
 renderAll()}
async function push(){try{const r=await fetch(D.root+"api/picks",{method:"PUT",headers:{"Content-Type":"application/json"},
 body:JSON.stringify(P,null,1)});if(!r.ok)throw new Error(r.status);status("saved to picks.json ✓","ok")}
 catch(e){status("could not save: "+e.message,"err")}}
async function boot(){try{const r=await fetch(D.root+"api/picks",{cache:"no-store"});if(!r.ok)throw 0;P=await r.json();
 server=true;lastSnap=JSON.stringify(P);status("picks.json — saves as you go","ok")}catch(e){const l=ls.get("picks");if(l)P=l;
 status("opened as a file: changes stay in this browser","err")}
 const last=ls.get("clip");go(Math.max(0,D.clips.findIndex(c=>c.id==last)),true)}
// selections of a clip: [[a,b],...] sorted; written back as "m:ss.ss-m:ss.ss" (single times stay as they were)
function sels(id){return((P[id]||{}).add||[]).filter(isRange).map(span).sort((x,y)=>x[0]-y[0])}
function points(id){return((P[id]||{}).add||[]).filter(v=>!isRange(v)).map(v=>span(v)[0])}
function setSels(id,list){const e=ent(id);const pts=(e.add||[]).filter(v=>!isRange(v));
 list=list.map(([a,b])=>[Math.max(0,Math.min(a,b)),Math.max(a,b)]).filter(([a,b])=>b-a>=0.1).sort((x,y)=>x[0]-y[0]);
 e.add=[...pts,...list.map(([a,b])=>fmt(a)+"-"+fmt(b))];save()}
function banned(id){return((P[id]||{}).ban||[])}
// --- clip list ---
let cur=0,cat="all",onlySel=false;const seen=new Set(ls.get("seen")||[]);
const C=()=>D.clips[cur];
const cats=["all",...new Set(D.clips.map(c=>c.cat))];
$("filters").innerHTML=cats.map(c=>`<button data-c="${esc(c)}" class="${c=="all"?"on":""}">${esc(c)}</button>`).join("")+
 `<button id=onlysel title="only clips with selections">● selected</button>`;
$("filters").onclick=ev=>{const b=ev.target.closest("button");if(!b)return;if(b.id=="onlysel"){onlySel=!onlySel;
 b.classList.toggle("on",onlySel)}else{cat=b.dataset.c;$("filters").querySelectorAll("[data-c]").forEach(x=>
 x.classList.toggle("on",x===b))}list()};
const visible=()=>D.clips.map((c,i)=>i).filter(i=>(cat=="all"||D.clips[i].cat==cat)&&(!onlySel||sels(D.clips[i].id).length));
const thumbAt=(c,t,h)=>{const k=Math.min(c.n-1,Math.max(0,Math.floor(t/c.step))),s=h/TH;
 return`background-image:url('${c.strip}');background-size:${c.n*TW*s}px ${h}px;background-position:-${k*TW*s}px 0`};
function list(){$("list").innerHTML=visible().map(i=>{const c=D.clips[i],n=sels(c.id).length,
 tot=sels(c.id).reduce((a,[x,y])=>a+y-x,0),bn=banned(c.id).includes("all");
 return`<div class="ci${i==cur?" cur":""}${seen.has(c.id)?" seen":""}" data-i=${i} style="${bn?"opacity:.4":""}">
 <div class=th style="${thumbAt(c,c.dur*0.3,54)}">${n?`<b>${n}</b>`:""}</div><div><div class=nm>${esc(c.id)}</div>
 <div class=meta><span class=chip>${esc(c.cat)}</span> ${esc(c.clock)} · ${short(c.dur)}${n?` · <span style="color:var(--sel)">● ${short(tot)}</span>`:""}</div>
 <div class=meta>${c.L.length?`<span style="color:var(--land)">L${c.L.length}</span> `:""}${c.P.length?`<span style="color:var(--port)">P${c.P.length}</span>`:""}${bn?" banned":""}</div></div></div>`}).join("");
 const el=$("list").querySelector(".cur");if(el)el.scrollIntoView({block:"nearest"})}
$("list").onclick=ev=>{const d=ev.target.closest(".ci");if(d)go(+d.dataset.i)};
// --- player + timeline ---
const vid=$("vid"),tl=$("tl"),ov=$("ov"),hov=$("hover");
let va=0,vb=1,inPt=null,outPt=null,act=-1,loop=false,shuttle=0,rew=null,drag=null,img=new Image();
const now=()=>vid.currentTime||0,dur=()=>C().dur||vid.duration||1;
function go(i,first){if(i<0||i>=D.clips.length)return;cur=i;const c=C();seen.add(c.id);ls.set("seen",[...seen]);
 ls.set("clip",c.id);inPt=outPt=null;act=-1;loop=false;stopShuttle();vid.pause();
 vid.src=c.video;va=0;vb=c.dur;img=new Image();img.onload=paint;img.src=c.strip;
 $("cname").textContent=c.id;$("cmeta").innerHTML=`<span class=chip>${esc(c.cat)}</span> ${esc(c.tag)} · ${esc(c.clock)}
  · ${short(c.dur)} · clip ${i+1} of ${D.clips.length}`;
 vid.addEventListener("loadedmetadata",()=>{vid.currentTime=0;renderAll()},{once:true});renderAll()}
function seek(t){if(rew)stopShuttle();vid.currentTime=Math.min(Math.max(0,t),dur()-0.02);follow();draw()}
function follow(){const t=now(),w=vb-va;if(t<va||t>vb){va=Math.max(0,Math.min(t-w*0.1,dur()-w));vb=va+w;paint()}}
const X=(t,a=va,b=vb)=>(100*(t-a)/(b-a)).toFixed(3)+"%";
function wspan(a,b,A=va,B=vb){const x0=Math.max(a,A),x1=Math.min(b,B);return`left:${X(x0,A,B)};width:${(100*(x1-x0)/(B-A)).toFixed(3)}%`}
function film(canvas,A,B,h,c=C(),im=img){const r=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1,img=im;
 canvas.width=Math.max(1,r.width*dpr);canvas.height=h*dpr;const g=canvas.getContext("2d");
 g.fillStyle="#000";g.fillRect(0,0,canvas.width,canvas.height);if(!img.complete||!img.naturalWidth)return;
 const tw=h*TW/TH*dpr,W=canvas.width;
 for(let x=0;x<W;x+=tw){const t=A+(x+tw/2)/W*(B-A);if(t>c.dur)break;
  const k=Math.min(c.n-1,Math.max(0,Math.floor(t/c.step)));g.drawImage(img,k*TW,0,TW,TH,x,0,tw,h*dpr)}}
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
 sels(c.id).forEach(([a,b],k)=>{if(b>va&&a<vb)s+=`<i class="blk${k==act?" act":""}" data-k=${k} style="${wspan(a,b)}"
  title="${fmt(a)} – ${fmt(b)} (${(b-a).toFixed(2)} s)"><i class="h l"></i>${k+1} · ${(b-a).toFixed(1)}s<i class="h r"></i></i>`});
 points(c.id).forEach(p=>{if(p>=va&&p<=vb)s+=`<i class=pt style="left:${X(p)}" title="marked moment ${fmt(p)}"></i>`});
 const io=pend();if(io)s+=`<i class=pend style="${wspan(...io)}"></i>`;
 if(!sels(c.id).length&&!io)s+=`<i class=empty>drag here to pick a range · or I / O then Enter · or click a reel shot below to take it</i>`;
 L.insertAdjacentHTML("beforeend",s);
 for(const[id,arr]of[["lland",c.L],["lport",c.P]]){const e=$(id);e.querySelectorAll("i").forEach(x=>x.remove());
  e.insertAdjacentHTML("beforeend",arr.filter(([a,b])=>b>va&&a<vb).map(([a,b,n])=>`<i class=use data-a=${a} data-b=${b} style="${wspan(a,b)}" title="reel shot ${n}: ${short(a)}–${short(b)} — click to pick it">${n}</i>`).join(""))}
 fkeys(c.id).forEach(([k])=>{if(k>=va&&k<=vb)$("lport").insertAdjacentHTML("beforeend",`<i class=fk style="left:${X(k)}" title="focal point ${fmt(k)}"></i>`)});
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
 const st=$("still"),noVid=vid.error||vid.readyState<2||vid.seeking;st.classList.toggle("on",!!noVid);
 if(noVid){const r=$("vwrap").getBoundingClientRect(),h=Math.min(r.height,r.width*TH/TW);
  st.setAttribute("style",thumbAt(c,t,h)+`;width:${h*TW/TH}px;height:${h}px;left:${(r.width-h*TW/TH)/2}px;top:${(r.height-h)/2}px`)}
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
  <td><button data-p=${k}>▶ play</button> <button data-l=${k}>↻ loop</button> <button data-d=${k}>✕</button></td></tr>`).join("")+
  `</table>`:`<p style="color:var(--mut);margin:0">No selections in this clip yet: <kbd>I</kbd> at the start, <kbd>O</kbd>
  at the end, <kbd>Enter</kbd> — or drag in the selections lane.</p>`}
$("mysel").onclick=ev=>{const b=ev.target.closest("button");if(!b)return;const ss=sels(C().id);
 if(b.dataset.d!==undefined){ss.splice(+b.dataset.d,1);act=-1;setSels(C().id,ss)}
 else{const k=+(b.dataset.p??b.dataset.l);act=k;loop=b.dataset.l!==undefined;seek(ss[k][0]);vid.play();renderAll()}};
function renderAll(){list();draw();mysel();selview();
 counts()}
// what's in play: the planner's shots per reel, and the editor's own calls
function counts(){const n=f=>D.clips.reduce((a,c)=>a+f(c),0),L=n(c=>c.L.length),Pn=n(c=>c.P.length),
  my=n(c=>sels(c.id).length+points(c.id).length),fk=n(c=>fkeys(c.id).length),
  bc=n(c=>banned(c.id).includes("all")?1:0),br=n(c=>banned(c.id).filter(v=>v!="all").length),
  more=n(c=>(P[c.id]||{}).more?1:0),unseen=D.clips.length-D.clips.filter(c=>seen.has(c.id)).length;
 const it=(v,t,cls="")=>`<span class="${cls}" title="${t}"><b>${v}</b> ${t.split(":")[0]}</span>`;
 $("counts").innerHTML=[it(L,"landscape shots: in the current landscape cut"),it(Pn,"portrait shots: in the current portrait cut"),
  it(my,"your picks: selections and marked moments the next re-plan must use","y"),
  fk?it(fk,"focus points: where the portrait crop goes","f"):"",
  bc+br?it(bc+br,"bans: "+bc+" whole clips, "+br+" stretches","x"):"",more?it(more,"clips asked for more"):"",
  it(unseen,"clips not looked at yet")].filter(Boolean).join(" · ")}
// timeline mouse: scrub on ruler/strip/overview; make, move and trim selections in the selections lane
function tAt(el,ev,A=va,B=vb){const r=el.getBoundingClientRect();return A+Math.min(Math.max((ev.clientX-r.left)/r.width,0),1)*(B-A)}
function showHover(ev,t,el){const c=C();hov.style.display="block";hov.querySelector("div").setAttribute("style",thumbAt(c,t,108));
 hov.querySelector("span").textContent=fmt(t);const r=el.getBoundingClientRect();
 hov.style.left=Math.min(innerWidth-205,Math.max(4,ev.clientX-100))+"px";hov.style.top=(r.top-136)+"px"}
for(const el of[$("ruler"),$("strip")]){
 el.addEventListener("pointerdown",ev=>{pause();el.setPointerCapture(ev.pointerId);drag={k:"scrub",el};seek(tAt(el,ev))});
 el.addEventListener("pointermove",ev=>{const t=tAt(el,ev);showHover(ev,t,tl);if(drag&&drag.el===el)seek(t)})}
ov.addEventListener("pointerdown",ev=>{ov.setPointerCapture(ev.pointerId);drag={k:"ov",el:ov};movewin(tAt(ov,ev,0,dur()))});
ov.addEventListener("pointermove",ev=>{const t=tAt(ov,ev,0,dur());showHover(ev,t,ov);if(drag&&drag.k=="ov")movewin(t)});
function movewin(t){const w=vb-va;va=Math.max(0,Math.min(t-w/2,dur()-w));vb=va+w;if(now()<va||now()>vb)seek(t);paint()}
const L=$("lsel");
L.addEventListener("pointerdown",ev=>{pause();L.setPointerCapture(ev.pointerId);const t=tAt(L,ev),b=ev.target.closest(".blk");
 if(b){const k=+b.dataset.k,ss=sels(C().id);act=k;
  drag={k:ev.target.classList.contains("l")?"trimL":ev.target.classList.contains("r")?"trimR":"move",i:k,t0:t,
   orig:ss[k].slice(),ss,moved:false}}
 else drag={k:"new",a:t,b:t};draw()});
L.addEventListener("pointermove",ev=>{const t=tAt(L,ev);showHover(ev,t,tl);if(!drag||drag.el)return;
 if(drag.k=="new"){drag.b=t;inPt=Math.min(drag.a,t);outPt=Math.max(drag.a,t);seek(t)}
 else{const o=drag.orig,d=t-drag.t0;if(Math.abs(d)>0.03)drag.moved=true;let[a,b]=o;
  if(drag.k=="move"){const w=b-a;a=Math.max(0,Math.min(a+d,dur()-w));b=a+w}
  else if(drag.k=="trimL"){a=Math.min(t,b-0.1);seek(a)}else{b=Math.max(t,a+0.1);seek(b)}
  drag.ss[drag.i]=[a,b];const blk=L.querySelector(`.blk[data-k="${drag.i}"]`);if(blk)blk.setAttribute("style",wspan(a,b))}});
L.addEventListener("pointerup",ev=>{if(!drag||drag.el)return;const c=C();
 if(drag.k=="new"){if(Math.abs(drag.b-drag.a)>=0.2){const ss=sels(c.id);ss.push([Math.min(drag.a,drag.b),Math.max(drag.a,drag.b)]);
   inPt=outPt=null;drag=null;setSels(c.id,ss);act=sels(c.id).findIndex(([a])=>Math.abs(a-Math.min(...ss[ss.length-1]))<0.01);renderAll()}
  else{inPt=outPt=null;drag=null;seek(tAt(L,ev))}return}
 const d=drag;drag=null;if(d.moved)setSels(c.id,d.ss);else seek(d.orig[0]);mysel()});
addEventListener("pointerup",()=>{if(drag&&drag.el){drag=null;draw()}});
for(const id of["lland","lport"])$(id).addEventListener("click",ev=>{const u=ev.target.closest(".use");if(!u)return;
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
 rew={n,id:setInterval(()=>{if(now()<=0)return stopShuttle();vid.currentTime=Math.max(0,now()-0.04*n);follow();draw()},40)};draw()}
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
function banIO(){const io=pend();if(!io)return status("mark an in–out to ban first","err");const e=ent(C().id);
 e.ban=e.ban||[];e.ban.push(short(io[0])+"-"+short(io[1]));inPt=outPt=null;save()}
$("banc").onclick=()=>{const e=ent(C().id);e.ban=e.ban||[];const k=e.ban.indexOf("all");
 if(k>=0)e.ban.splice(k,1);else e.ban.push("all");save()};
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
function allSels(){const out=[];D.clips.forEach((c,i)=>sels(c.id).forEach(([a,b],k)=>{const key=pfmt+"|"+pkey(c.id,a);
  if(!place.has(key))place.set(key,shotOf(c,a,b));out.push({c,i,k,a,b,shot:place.get(key)})}));
 if(pfmt!="picks"){const have=new Set(out.filter(x=>x.shot&&x.shot.fmt==pfmt).map(x=>x.shot.n));
  D.clips.forEach((c,i)=>{const bans=banned(c.id).map(v=>v=="all"?[-1,1e9]:span(v));
   c[pfmt].forEach(([a,b,n])=>{if(!have.has(n)&&!bans.some(([u,v])=>(a+b)/2>=u&&(a+b)/2<=v))
    out.push({c,i,k:-1,a,b,shot:{fmt:pfmt,n},cut:true})})})}
 const rank=x=>x.shot?(x.shot.fmt==(pfmt=="P"?"P":"L")?0:1e4)+x.shot.n:1e9+x.i*1e3+x.a;return out.sort((x,y)=>rank(x)-rank(y))}
$("pfmts").onclick=ev=>{const b=ev.target.closest("button");if(!b)return;pfmt=b.dataset.f;ls.set("pfmt",pfmt);pk=null;pv.pause();selview()};
const pj=()=>pk?order.findIndex(x=>x.c.id==pk.id&&Math.abs(x.a-pk.a)<0.002):-1;
const PX=()=>order[pj()];
function selview(){order=allSels();const tot=order.reduce((s,x)=>s+x.b-x.a,0),j=pj(),placed=order.filter(x=>x.shot).length;
 const np=order.filter(x=>!x.cut).length,nc=order.length-np;
 $("pfmts").querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.f==pfmt));
 $("seltitle").textContent=(pfmt=="picks"?"":`${nc+order.filter(x=>!x.cut&&x.shot&&x.shot.fmt==pfmt).length} shots in the ${pfmt=="P"?"portrait":"landscape"} cut · `)+
  np+" pick"+(np==1?"":"s");
 $("seltot").textContent=short(tot)+" of footage · in cut order"+(order.length>placed?` · ${order.length-placed} not in the cut yet (I place them on the next re-plan)`:"")+
  (nc?" · trim, slip or drag a shot to make it a pick; ✕ bans it":"");
 $("seq").innerHTML=order.map((x,k)=>`<i data-j=${k} class="${k==j?"act":""}${x.shot?"":" new"}${x.cut?" cut"+(pfmt=="P"?" p":""):""}" style="flex:${(x.b-x.a).toFixed(2)}" title="${esc(x.c.id)} ${fmt(x.a)}–${fmt(x.b)}"></i>`).join("");
 let h="",sep=false;order.forEach((x,k)=>{if(!x.shot&&!sep){sep=true;h+=`<div class=catgrp>Not in the cut yet — placed on the next re-plan</div>`}
  h+=`<div class="card${k==j?" act":""}${x.cut?" cut":""}" data-j=${k}><div class=th data-j=${k}></div>
  <div class=bd><div><b>${x.shot?(x.shot.fmt=="L"?"Shot ":"Portrait shot ")+x.shot.n:"New"}</b> <span class=tag>${x.cut?"cut":"pick"}</span> · ${esc(x.c.id)}<br>
  <small>${fmt(x.a)} – ${fmt(x.b)} · <b>${(x.b-x.a).toFixed(2)} s</b> · ${esc(x.c.clock)}</small></div>
  <div><button data-del=${k} title="${x.cut?"take this shot out of the cut (bans its stretch)":"remove this pick"}">✕</button></div></div></div>`});
 $("cards").innerHTML=h||`<p style="color:var(--mut)">No picks yet — choose footage in Sources.</p>`;
 $("cards").querySelectorAll(".th").forEach(el=>{const x=order[+el.dataset.j],w=el.clientWidth||200,hh=w*TH/TW;
  el.setAttribute("style",thumbAt(x.c,x.a,hh));el.innerHTML=`<span>${k1(el)}</span>`});
 function k1(el){const x=order[+el.dataset.j];return (+el.dataset.j+1)+" · "+(x.b-x.a).toFixed(1)+"s"}
 $("ped").style.display=PX()?"block":"none";if(PX())pdraw()}
$("cards").onclick=ev=>{const b=ev.target.closest("[data-del],.card");if(!b)return;
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
 const ss=sels(x.c.id);if(x.k>=0)ss[x.k]=[a,b];else ss.push([a,b]);place.set(pfmt+"|"+pkey(x.c.id,a),x.shot);pk={id:x.c.id,a};
 if(a<pa||b>pb){const ctx=Math.max(2,(b-a)*0.75);pa=Math.max(0,a-ctx);pb=Math.min(L,b+ctx)}
 setSels(x.c.id,ss)}
const pnow=()=>pv.currentTime||0;
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
 $("pvtc").textContent=(pj()+1)+"/"+order.length+" · "+x.c.id+" · "+fmt(t);
 $("pplay").textContent=pv.paused?"▶ Play pick":"❚❚ Pause";$("ploop").classList.toggle("on",mode=="loop");
 const st=$("pstill"),no=pv.error||pv.readyState<2||pv.seeking;st.classList.toggle("on",!!no);
 if(no){const R=$("pvwrap").getBoundingClientRect(),hh=Math.min(R.height,R.width*TH/TW);
  st.setAttribute("style",thumbAt(x.c,t,hh)+`;width:${hh*TW/TH}px;height:${hh}px;left:${(R.width-hh*TW/TH)/2}px;top:${(R.height-hh)/2}px`)}
 frameDraw($("pfr"),$("pvwrap"),pv,x.c,x.a,x.b,t)}
const PT=$("pstrip");
PT.addEventListener("pointerdown",ev=>{const x=PX();if(!x)return;pv.pause();PT.setPointerCapture(ev.pointerId);
 const t=tAt(PT,ev,pa,pb),blk=ev.target.closest(".blk");
 pdrag=blk?{k:ev.target.classList.contains("l")?"l":ev.target.classList.contains("r")?"r":"slip",t0:t,o:[x.a,x.b],cur:[x.a,x.b]}:{k:"scrub"};
 if(!blk)pv.currentTime=t;pdraw()});
PT.addEventListener("pointermove",ev=>{const x=PX();if(!x)return;const t=tAt(PT,ev,pa,pb);
 hov.style.display="block";hov.querySelector("div").setAttribute("style",thumbAt(x.c,t,108));hov.querySelector("span").textContent=fmt(t);
 const R=PT.getBoundingClientRect();hov.style.left=Math.min(innerWidth-205,Math.max(4,ev.clientX-100))+"px";hov.style.top=(R.top-136)+"px";
 if(!pdrag)return;if(pdrag.k=="scrub"){pv.currentTime=t;pdraw();return}
 let[a,b]=pdrag.o;const d=t-pdrag.t0;
 if(pdrag.k=="slip"){const w=b-a;a=Math.max(0,Math.min(a+d,x.c.dur-w));b=a+w;pv.currentTime=a}
 else if(pdrag.k=="l"){a=Math.max(0,Math.min(t,b-0.1));pv.currentTime=a}else{b=Math.min(x.c.dur,Math.max(t,a+0.1));pv.currentTime=b}
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
$("ped").onclick=ev=>{const b=ev.target.closest("button");if(!b)return;const x=PX();
 if(b.dataset.n){const[w,d]=b.dataset.n.split(":");nudge(w,+d*(ev.shiftKey?10:1))}
 else if(b.id=="pplay")playPick();else if(b.id=="ploop"){mode=mode=="loop"?"one":"loop";playPick()}
 else if(b.id=="pprev")pick(Math.max(0,pj()-1),!pv.paused);else if(b.id=="pnext")pick(Math.min(order.length-1,pj()+1),!pv.paused)
 else if(b.id=="pin")edit(Math.min(pnow(),x.b-0.1),x.b);else if(b.id=="pout")edit(x.a,Math.max(pnow(),x.a+0.1))
 else if(b.id=="popen"){view(1);go(x.i);act=x.k;vid.addEventListener("loadedmetadata",()=>seek(x.a),{once:true})}
 else if(b.id=="pdel"){const j=pj();pk=null;pv.pause();drop(x);if(order[j])pick(j,false)}};
// a pick goes; a shot of the cut is banned, so the next re-plan puts something else there
function drop(x){if(x.k>=0){const ss=sels(x.c.id);ss.splice(x.k,1);setSels(x.c.id,ss)}
 else{const e=ent(x.c.id);e.ban=e.ban||[];e.ban.push(fmt(x.a)+"-"+fmt(x.b));save()}}
$("playall").onclick=()=>{mode="all";const j=Math.max(0,pj());pick(j,true)};
$("stopall").onclick=stopAll;
// --- portrait framing: the crop the planner derives (served by /api/crop) and the editor's focal points, picks
// "portrait": [["m:ss.ss", x, y], ...] (fractions of the frame); the reel's crop centre follows them, its size as derived
let frameOn=!!ls.get("frame");const crops=new Map();
const fkeys=id=>((P[id]||{}).portrait||[]).map(([t,x,y])=>[secs(t),+x,+y]).sort((p,q)=>p[0]-q[0]);
const putKeys=(id,ks)=>{ent(id).portrait=ks.sort((p,q)=>p[0]-q[0]).map(([t,x,y])=>[fmt(t),+x.toFixed(4),+y.toFixed(4)])};
const clamp=(v,lo,hi)=>Math.min(Math.max(v,lo),hi);
const lerp=(ts,vs,t)=>{if(t<=ts[0])return vs[0];for(let i=1;i<ts.length;i++)if(t<=ts[i]){const f=(t-ts[i-1])/(ts[i]-ts[i-1]||1);
 return vs[i-1]+f*(vs[i]-vs[i-1])}return vs[vs.length-1]};
function derived(c,a,b){const k=c.id+"|"+a.toFixed(2)+"|"+b.toFixed(2);if(crops.has(k))return crops.get(k);
 if(!server){crops.set(k,false);return false}crops.set(k,undefined);
 fetch(`${D.root}api/crop?id=${encodeURIComponent(c.id)}&a=${a.toFixed(3)}&b=${b.toFixed(3)}&fmt=portrait`)
  .then(r=>r.ok?r.json():Promise.reject()).then(j=>crops.set(k,j)).catch(()=>crops.set(k,false)).then(()=>{draw();pdraw()});return undefined}
// where the frame is worked out from: the selection or portrait reel shot under the playhead, else the 3 s around it
function rangeFor(c,t){const s=sels(c.id).find(([a,b])=>t>=a-0.02&&t<=b+0.02)||c.P.find(([a,b])=>t>=a&&t<=b);
 if(s)return[s[0],s[1]];const a=Math.max(0,Math.min(Math.floor(t/3)*3,c.dur-3));return[a,Math.min(a+3,c.dur)]}
function frameAt(c,a,b,t){const cr=derived(c,a,b);if(cr===undefined)return"wait";if(cr===false)return"off";if(cr===null)return"fills";
 const ts=cr.path.map(q=>a+q[0]),d={x:lerp(ts,cr.path.map(q=>q[1]),t),y:lerp(ts,cr.path.map(q=>q[2]),t)};
 const ks=fkeys(c.id).filter(k=>k[0]>=a-0.5&&k[0]<=b+0.5);let o=null;
 if(ks.length){const kt=ks.map(k=>k[0]);o={x:clamp(lerp(kt,ks.map(k=>k[1]),t),cr.w/2,1-cr.w/2),
  y:clamp(lerp(kt,ks.map(k=>k[2]),t),cr.h/2,1-cr.h/2)}}
 return{d,o,w:cr.w,h:cr.h,z:cr.z}}
function vrect(wrap,v){const R=wrap.getBoundingClientRect(),ar=v.videoWidth&&v.videoHeight?v.videoWidth/v.videoHeight:TW/TH;
 const w=Math.min(R.width,R.height*ar),h=w/ar;return{l:(R.width-w)/2,t:(R.height-h)/2,w,h}}
function frameDraw(el,wrap,v,c,a,b,t){el.classList.toggle("on",frameOn);document.querySelectorAll(".frt").forEach(x=>x.classList.toggle("on",frameOn));
 if(!frameOn){el.innerHTML="";return}const F=frameAt(c,a,b,t);
 if(typeof F=="string"){el.innerHTML=`<span class=frl>${{fills:"this clip fills the portrait frame as it is — no crop to place",
  off:"the portrait frame needs the page served by <code>focalcut reel serve</code>",wait:"working out the portrait crop…"}[F]}</span>`;return}
 const r=vrect(wrap,v),box=(p,cls)=>`<i class="fbox ${cls}" style="left:${r.l+(p.x-F.w/2)*r.w}px;top:${r.t+(p.y-F.h/2)*r.h}px;width:${F.w*r.w}px;height:${F.h*r.h}px"></i>`;
 const A=F.o||F.d,dx=F.o?F.o.x-F.d.x:0;
 el.innerHTML=box(A,"act")+(F.o?box(F.d,"der"):"")+`<i class=fdot style="left:${r.l+A.x*r.w}px;top:${r.t+A.y*r.h}px"></i>`+
  `<span class=frl>planner x ${F.d.x.toFixed(2)} y ${F.d.y.toFixed(2)}${F.z>1.001?" · zoom "+F.z.toFixed(2):""}`+
  (F.o?` → <b>your focus x ${F.o.x.toFixed(2)} y ${F.o.y.toFixed(2)} (${dx>=0?"+":""}${dx.toFixed(2)})</b>`:" · drag to set a focal point")+`</span>`}
function toggleFrame(){frameOn=!frameOn;ls.set("frame",frameOn);draw();pdraw()}
// drag on the picture: a focal point at the playhead (one within ¼ s of it moves instead)
function frameDrag(el,wrap,v,ctx,redraw,stop){let x=null;
 const put=ev=>{const r=vrect(wrap,v),R=wrap.getBoundingClientRect();
  const ks=fkeys(x.c.id).filter(k=>Math.abs(k[0]-x.t)>0.25);
  ks.push([x.t,clamp((ev.clientX-R.left-r.l)/r.w,0,1),clamp((ev.clientY-R.top-r.t)/r.h,0,1)]);putKeys(x.c.id,ks);redraw()};
 el.addEventListener("pointerdown",ev=>{if(!frameOn)return;x=ctx();if(!x)return;stop();el.setPointerCapture(ev.pointerId);put(ev)});
 el.addEventListener("pointermove",ev=>{if(x)put(ev)});
 el.addEventListener("pointerup",()=>{if(x){x=null;save()}})}
frameDrag($("fr"),$("vwrap"),vid,()=>({c:C(),t:now()}),draw,pause);
frameDrag($("pfr"),$("pvwrap"),pv,()=>PX()&&{c:PX().c,t:pnow()},pdraw,()=>pv.pause());
function clearFocus(c,a,b){const ks=fkeys(c.id),kept=ks.filter(k=>k[0]<a-0.5||k[0]>b+0.5);
 if(kept.length==ks.length)return status("no focal points here");putKeys(c.id,kept);save()}
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
 else if(K=="f")toggleFrame();
 else if($("selview").style.display=="grid"){const x=PX();
  if(k==" ")playPick();else if(k=="Escape")stopAll();
  else if(k=="ArrowRight"||k=="ArrowDown")pick(Math.min(order.length-1,pj()+1),!pv.paused);
  else if(k=="ArrowLeft"||k=="ArrowUp")pick(Math.max(0,pj()-1),!pv.paused);
  else if(!x)done=false;
  else if(K=="i")edit(Math.min(pnow(),x.b-0.1),x.b);else if(K=="o")edit(x.a,Math.max(pnow(),x.a+0.1));
  else if(k==","||k=="<")nudge("slip",ev.shiftKey?-0.5:-0.02);else if(k=="."||k==">")nudge("slip",ev.shiftKey?0.5:0.02);
  else if(k=="\\"){mode=mode=="loop"?"one":"loop";playPick()}
  else if(k=="Backspace"||k=="Delete")$("pdel").click();else done=false}
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
