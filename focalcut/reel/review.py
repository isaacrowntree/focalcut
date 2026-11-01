"""A local review page per reel: every shot with a frame, its export file and the time range it uses in that file --
exactly what picks take, so a bad shot can be banned or a better moment suggested. Each shot opens a scrubber on
the export it comes from, with an overview strip and a zoomed strip around the playhead (drag either to scrub, scroll
to zoom) and editor keys -- J/K/L shuttle, frame steps, I/O, M markers, [ ] to hop between marks: mark extra shots
anywhere, swap the shot to another moment or range, ban it, or ask for more from that video. The page writes the
picks.json to save (it remembers unsaved changes in the browser).

    review/<fmt>.html  (+ review/<fmt>/NNN.jpg)

Frames come from the rendered reel when it exists, else from the shot's source.
"""
from __future__ import annotations

import html
import json
import os
import subprocess
from urllib.parse import quote

from . import browse


def mmss(x):
    return f"{int(x // 60)}:{x % 60:04.1f}"


def pick_line(s):
    """The shot as a picks entry: its export file and the range it plays, in that file's own timeline."""
    u = s["unit"]
    x0 = u.get("export_t0", 0.0)
    a = s["src_in"] - x0
    return f'"{u["id"]}": {{"ban": ["{mmss(a)}-{mmss(a + s["length"])}"]}}'


def page(plan, fmt, thumbs, sources=None, picks=None, key="review", root="", nav=""):
    """The review page's HTML; thumbs[i] is shot i's image path relative to the page (or None). sources: per unit id,
    {"video": its export file relative to the page, "moments": [[t, kind, rank], ...]} in the export's own timeline;
    picks: the current picks.json, which the page starts from; root: the project folder relative to the page, where
    `focalcut reel serve` answers api/picks (served, the page saves picks.json itself)."""
    w = 320 if fmt == "landscape" else 180
    cards, shots = [], []
    for i, s in enumerate(plan["shots"]):
        u = s["unit"]
        a = s["src_in"] - u.get("export_t0", 0.0)
        tags = " ".join(filter(None, [s["act"], s["kind"], "+".join(s["fx"]),
                                      s.get("transition") and f"⇢ {s['transition']}"]))
        img = f'<img src="{html.escape(quote(thumbs[i]))}" loading="lazy" alt="">' if thumbs[i] else ""
        cards.append(f'<figure data-i="{i}">{img}<figcaption><b>{i + 1}</b> {html.escape(u["id"])} '
                     f'<span class=t>{mmss(a)}–{mmss(a + s["length"])}</span><br><small>{html.escape(tags)}</small>'
                     f'<code title="copy into picks.json">{html.escape(pick_line(s))}</code>'
                     f'<span class=bar><button data-act=scrub>Scrub source ▸</button>'
                     f'<button data-act=ban>Ban</button><button data-act=more>More +</button></span>'
                     f'<span class=st></span></figcaption></figure>')
        shots.append(dict(id=u["id"], a=round(a, 2), b=round(a + s["length"], 2), kind=s["kind"]))
    data = json.dumps(dict(shots=shots, sources=sources or {}, picks=picks or {}, key=key, root=root)).replace("</", "<\\/")
    return (TEMPLATE.replace("%W%", str(w)).replace("%FMT%", html.escape(fmt)).replace("%N%", str(len(shots)))
            .replace("%CARDS%", "".join(cards)).replace("%DATA%", data).replace("%NAV%", nav)
            .replace("%MENU%", browse.MENU_CSS))


TEMPLATE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>%FMT% reel review</title>
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--card:#1c1c1c;--line:#333;--acc:#f5c542;--add:#3ddc84;--ban:#ff5c5c;--mom:#7aa7ff}
@media (prefers-color-scheme:light){:root{--bg:#f6f6f6;--fg:#111;--mut:#555;--card:#fff;--line:#ddd;--acc:#a06c00;
--add:#0a8a4a;--ban:#c62828;--mom:#2457c5}}
body{background:var(--bg);color:var(--fg);font:13px system-ui;margin:16px}
h1{font-size:18px;margin:0 0 6px} p{color:var(--mut);max-width:80ch}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(%W%px,1fr));gap:12px}
figure{margin:0;background:var(--card);border-radius:8px;overflow:hidden;border:2px solid transparent}
figure.banned{border-color:var(--ban);opacity:.6} figure.changed{border-color:var(--add)}
img{width:100%;display:block} figcaption{padding:6px 8px} .t{color:var(--acc)}
small,.st{color:var(--mut)} .st{display:block;font-size:11px}
code{display:block;margin-top:4px;font-size:11px;color:var(--mut);user-select:all;overflow-wrap:anywhere;cursor:copy}
button{font:inherit;background:var(--line);color:var(--fg);border:0;border-radius:5px;padding:4px 8px;cursor:pointer;
margin:4px 4px 0 0} button:hover{outline:1px solid var(--acc)} .bar{display:block}
#top{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5;border-bottom:1px solid var(--line);
margin-bottom:12px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
#count{color:var(--acc);margin-right:8px}%MENU%
dialog{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:12px;
width:min(960px,96vw);max-height:96vh}
dialog::backdrop{background:#000a}
#vid{width:100%;max-height:58vh;background:#000;display:block}
.tl{position:relative;background:var(--line);border-radius:6px;margin:6px 0;cursor:ew-resize;touch-action:none;
user-select:none;overflow:hidden}
#tl{height:44px} #zt{height:88px;background:var(--bg);border:1px solid var(--line)}
.tl i{position:absolute;top:0;bottom:0;border-radius:3px;pointer-events:none}
.tl .cur{background:var(--acc);opacity:.85} .tl .oth{background:var(--mut);opacity:.55}
.tl .mom{width:3px;background:var(--mom);top:auto;height:40%} .tl .add{width:4px;background:var(--add);z-index:2}
.tl .add::after{content:"";position:absolute;top:0;left:-5px;border:7px solid transparent;border-top-color:var(--add)}
.tl .ban{background:var(--ban);opacity:.35} .tl .pre{background:var(--add);opacity:.4}
.tl .io{background:var(--fg);opacity:.15;border-left:2px solid var(--fg);border-right:2px solid var(--fg)}
.tl .ph{width:2px;background:var(--fg);z-index:3} .tl .ph::before{content:"";position:absolute;top:0;left:-6px;
border:7px solid transparent;border-top-color:var(--fg)}
.tl .win{border:2px solid var(--acc);background:transparent;z-index:1}
.tl .tk{width:1px;background:var(--mut);top:auto;height:10px;opacity:.6} .tl .tk.big{height:20px;opacity:.9}
.tl .lb{top:auto;bottom:22px;font-size:10px;color:var(--mut);transform:translateX(-50%);background:none}
#hover{position:fixed;pointer-events:none;background:var(--fg);color:var(--bg);font-size:11px;padding:1px 5px;
border-radius:3px;display:none;z-index:10;font-variant-numeric:tabular-nums}
#help{font-size:12px;color:var(--mut);margin:6px 0 0;line-height:1.9}
.row{display:flex;flex-wrap:wrap;gap:4px;align-items:center} #pos{font-variant-numeric:tabular-nums;color:var(--acc)}
#alts{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;max-height:110px;overflow:auto}
#alts button{background:transparent;border:1px solid var(--mom)} #alts button.on{background:var(--mom);color:#fff}
#marks{font-size:12px;color:var(--mut);margin-top:6px} #marks b{color:var(--add)}
kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:11px}
pre{white-space:pre-wrap;font-size:11px;background:var(--card);padding:8px;border-radius:6px;max-height:40vh;overflow:auto}
</style>
<div id=top><h1>%FMT% reel — %N% shots</h1>%NAV%<span id=count></span>
<button id=copy>Copy picks.json</button><button id=dl>Download picks.json</button>
<button id=clear>Reset to file</button><span id=saved></span>
<details><summary>show picks</summary><pre id=json></pre></details></div>
<p>Times are in each export file's own timeline. <b>Scrub source</b> opens the export the shot comes from: the
<span style="color:var(--acc)">yellow</span> block is this shot, grey blocks are the reel's other shots from that
video, <span style="color:var(--mom)">blue</span> ticks are the moments the planner found. Mark extra shots anywhere
with <kbd>M</kbd> (as many as you like), swap this shot to the playhead or an <kbd>I</kbd>/<kbd>O</kbd> range
(<kbd>S</kbd>), ban it (<kbd>B</kbd>), or ask for more from the video. Drag either strip to scrub; the lower one is
zoomed around the playhead for frame-accurate picks (scroll or <kbd>=</kbd>/<kbd>−</kbd> to zoom); J/K/L shuttle. Then save the
picks as <code style="display:inline">focalcut/reel/picks.json</code> and re-run
<code style="display:inline">--steps plan,render,xml,review</code>.</p>
<main>%CARDS%</main>
<dialog id=dlg><div class=row style="justify-content:space-between"><b id=title></b>
<span><button id=helpb title="keyboard shortcuts (?)">Keys ?</button> <button id=close>Close ✕</button></span></div>
<video id=vid preload=auto playsinline muted></video>
<div id=tl class=tl title="whole video: drag to scrub; the outlined box is the zoomed strip below"></div>
<div id=zt class=tl title="zoomed around the playhead: drag to scrub precisely; scroll to zoom in / out"></div>
<div class=row><span id=pos>0:00.00</span> <span id=speed style="color:var(--mut)"></span>
<button id=back>−1 s</button><button id=fb>◀ frame</button><button id=play>Play ▸</button><button id=ff>frame ▶</button>
<button id=fwd>+1 s</button>
<select id=rate title="playback speed"><option value=0.25>¼×</option><option value=0.5>½×</option>
<option value=1 selected>1×</option><option value=2>2×</option></select>
<button id=zin title="zoom in (=)">Zoom +</button><button id=zout title="zoom out (−)">Zoom −</button>
<button id=goshot>Back to this shot</button></div>
<div class=row><button id=add style="background:var(--add);color:#000">+ Mark extra shot (M)</button>
<button id=swap>⇄ Use playhead / in–out instead of this shot (S)</button>
<button id=pref title="prefer the in–out range without banning this shot (P)">Prefer in–out (P)</button>
<button id=bandlg>Ban this shot (B)</button>
<button id=moredlg>More from this video +</button><button id=lessdlg>−</button></div>
<div id=alts></div><div id=marks></div>
<div id=help><kbd>Space</kbd>/<kbd>K</kbd> play-pause · <kbd>L</kbd> play, again for 2× 4× · <kbd>J</kbd> rewind,
again faster · <kbd>←</kbd>/<kbd>→</kbd> or <kbd>,</kbd>/<kbd>.</kbd> one frame · <kbd>Shift</kbd>+arrows 1 s ·
<kbd>0</kbd>–<kbd>9</kbd> jump to 0–90% · <kbd>Home</kbd>/<kbd>End</kbd><br>
<kbd>M</kbd> (or <kbd>A</kbd>) mark an extra shot — as many as you like · <kbd>[</kbd>/<kbd>]</kbd> previous / next
marker or moment · <kbd>Backspace</kbd> delete the marker at the playhead · <kbd>I</kbd>/<kbd>O</kbd> in / out ·
<kbd>X</kbd> clear in–out · <kbd>S</kbd> swap this shot · <kbd>P</kbd> prefer in–out · <kbd>B</kbd> ban ·
<kbd>=</kbd>/<kbd>−</kbd> or scroll: zoom the strip · <kbd>Esc</kbd> close</div></dialog>
<div id=hover></div>
<script>
const D=%DATA%;
const mmss=x=>{x=Math.max(0,x);const m=Math.floor(x/60);return m+":"+(x-60*m).toFixed(1).padStart(4,"0")};
const store={get(){try{return JSON.parse(localStorage.getItem("picks:"+D.key))}catch(e){return null}},
 set(v){try{localStorage.setItem("picks:"+D.key,JSON.stringify(v))}catch(e){}},
 del(){try{localStorage.removeItem("picks:"+D.key)}catch(e){}}};
let P=store.get()||JSON.parse(JSON.stringify(D.picks));
const ent=id=>(P[id]&&typeof P[id]==="object"&&!Array.isArray(P[id]))?P[id]:(P[id]={});
const esc=x=>String(x).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
function tidy(){for(const k in P){const e=P[k];if(k[0]=="_"||!e||typeof e!=="object"||Array.isArray(e))continue;
 for(const f of ["ban","prefer","add"])if(e[f]&&!e[f].length)delete e[f];if(e.more===0)delete e.more;
 if(!Object.keys(e).length)delete P[k]}}
// served by `focalcut reel serve`: picks.json is read and written directly; opened as a file: kept in this browser
let server=false,timer=null;const saved=t=>{const e=document.getElementById("saved");if(e)e.textContent=t};
function save(){tidy();if(server){clearTimeout(timer);timer=setTimeout(push,250);saved("saving…")}else store.set(P);render()}
function push(){fetch(D.root+"api/picks",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(P,null,1)})
 .then(r=>saved(r.ok?"saved to picks.json ✓":"could not save ("+r.status+")")).catch(e=>saved("could not save: "+e.message))}
fetch(D.root+"api/picks",{cache:"no-store"}).then(r=>r.ok?r.json():Promise.reject()).then(j=>{
 if(!j||typeof j!=="object"||Array.isArray(j))return;P=j;server=true;(document.getElementById("allp")||{}).hidden=false;saved("picks.json — saves as you go");render()}).catch(()=>{});
function range(a,b){return mmss(a)+"-"+mmss(b)}
const secs=v=>{if(typeof v==="number")return v;const p=String(v).split(":");return p.length>1?(+p[0])*60+(+p[1]):+p[0]};
const span=v=>{if(Array.isArray(v))return v.map(secs);const s=String(v);const i=s.indexOf("-");
 return i>0?[secs(s.slice(0,i)),secs(s.slice(i+1))]:[secs(s),secs(s)]};
function shotRange(i){const s=D.shots[i];return range(s.a,s.b)}
function isBanned(i){const s=D.shots[i],e=P[s.id];return !!(e&&Array.isArray(e.ban)&&(e.ban.includes("all")||
 e.ban.some(v=>{const[a,b]=span(v);return a<=s.a+0.05&&b>=s.b-0.05})))}
function covers(v,s){if(v=="all")return true;const[a,b]=span(v);return a<=s.a+0.05&&b>=s.b-0.05}
function toggleBan(i){const s=D.shots[i],e=ent(s.id),r=shotRange(i);e.ban=e.ban||[];
 const k=e.ban.indexOf(r);if(k>=0)e.ban.splice(k,1);
 else if(isBanned(i)){const wide=e.ban.filter(v=>covers(v,s));                 // a wider ban ("all", a range)
  if(!confirm("Lift the ban "+wide.join(", ")+" on "+s.id+"?"))return;e.ban=e.ban.filter(v=>!covers(v,s))}
 else e.ban.push(r);save()}
function more(id,d){const e=ent(id);e.more=Math.max(0,(e.more||0)+d);save()}
function render(){
 document.querySelectorAll("figure").forEach(f=>{const i=+f.dataset.i,s=D.shots[i],e=P[s.id]||{};
  f.classList.toggle("banned",isBanned(i));
  const bits=[];if(e.add&&e.add.length)bits.push("+"+e.add.length+" extra");if(e.more)bits.push("more ×"+e.more);
  if(e.prefer&&e.prefer.length)bits.push("prefer "+e.prefer.join(", "));
  f.classList.toggle("changed",!isBanned(i)&&bits.length>0);
  f.querySelector(".st").textContent=bits.join(" · ");
  f.querySelector("[data-act=ban]").textContent=isBanned(i)?"Unban":"Ban"});
 const n=Object.keys(P).filter(k=>k[0]!="_").length;document.getElementById("count").textContent=n+" clip"+(n==1?"":"s")+" with picks";
 document.getElementById("json").textContent=JSON.stringify(P,null,1);
 if(dlg.open)draw()}
document.querySelector("main").addEventListener("click",ev=>{const b=ev.target.closest("button");if(!b)return;
 const i=+b.closest("figure").dataset.i;const a=b.dataset.act;
 if(a=="ban")toggleBan(i);else if(a=="more")more(D.shots[i].id,1);else if(a=="scrub")openShot(i)});
document.getElementById("copy").onclick=()=>{const t=JSON.stringify(P,null,1);
 (navigator.clipboard?navigator.clipboard.writeText(t):Promise.reject()).catch(()=>{
  const r=document.createRange();r.selectNodeContents(document.getElementById("json"));
  getSelection().removeAllRanges();getSelection().addRange(r);document.execCommand("copy")});
 document.getElementById("copy").textContent="Copied ✓";setTimeout(()=>document.getElementById("copy").textContent="Copy picks.json",1500)};
document.getElementById("dl").onclick=()=>{const a=document.createElement("a");
 a.href=URL.createObjectURL(new Blob([JSON.stringify(P,null,1)],{type:"application/json"}));a.download="picks.json";a.click()};
document.getElementById("clear").onclick=()=>{if(confirm("Drop this page's changes and start again from picks.json?")){
 store.del();P=JSON.parse(JSON.stringify(D.picks));render()}};
// --- scrubber: an overview strip and a zoomed strip (as in Resolve's cut page), JKL shuttle and I/O/M marks as in
// Premiere and Final Cut, frame steps and 0-9 jumps as on YouTube ---
const dlg=document.getElementById("dlg"),vid=document.getElementById("vid"),tl=document.getElementById("tl"),
 zt=document.getElementById("zt"),hov=document.getElementById("hover");
const FR=1/50;let cur=null,zoomW=8,inPt=null,outPt=null,shuttle=0,rew=null,drag=null;
const now=()=>vid.currentTime||0;
function seek(t){if(rew)stopShuttle();vid.currentTime=Math.min(Math.max(0,t),dur()-0.01);draw()}
function openShot(i){cur=i;inPt=outPt=null;stopShuttle();const s=D.shots[i],src=D.sources[s.id]||{};
 document.getElementById("title").textContent=(i+1)+" · "+s.id+" · "+s.kind+" "+shotRange(i);
 if(src.video){if(vid.getAttribute("src")!==src.video)vid.src=src.video;vid.style.display=""}
 else{vid.removeAttribute("src");vid.load();vid.style.display="none"}
 const go=()=>{vid.currentTime=s.a;draw()};vid.readyState>=1?go():vid.addEventListener("loadedmetadata",go,{once:true});
 const al=document.getElementById("alts");al.innerHTML="";
 (src.moments||[]).forEach(([t,k,r])=>{const b=document.createElement("button");b.textContent=mmss(t)+" "+k;
  b.dataset.t=t;b.title="rank "+r+" — jump there";b.onclick=()=>seek(t-0.8);al.appendChild(b)});
 dlg.showModal();draw()}
function dur(){return vid.duration&&isFinite(vid.duration)?vid.duration:
 Math.max(...D.shots.filter(x=>x.id==D.shots[cur].id).map(x=>x.b),...((D.sources[D.shots[cur].id]||{}).moments||[[1]]).map(m=>m[0]))+5}
function zwin(){const L=dur(),w=Math.min(zoomW,L);let a=(drag&&drag.el===zt)?drag.a:now()-w/2;
 a=Math.min(Math.max(0,a),Math.max(0,L-w));return[a,a+w]}
function marks(){const s=D.shots[cur],e=P[s.id]||{},L=dur(),out=[];
 (e.ban||[]).forEach(v=>out.push(v=="all"?["ban",0,L,"banned"]:["ban",...span(v),"banned "+v]));
 (e.prefer||[]).forEach(v=>out.push(["pre",...span(v),"prefer "+v]));
 D.shots.forEach((x,k)=>{if(x.id==s.id&&k!=cur)out.push(["oth",x.a,x.b,"shot "+(k+1)])});
 out.push(["cur",s.a,s.b,"this shot"]);
 if(inPt!==null||outPt!==null){const a=inPt??now(),b=outPt??now();out.push(["io",Math.min(a,b),Math.max(a,b),"in–out"])}
 ((D.sources[s.id]||{}).moments||[]).forEach(([t,k])=>out.push(["mom",t,null,k+" "+mmss(t)]));
 (e.add||[]).forEach(v=>out.push(["add",span(v)[0],null,"extra shot "+v]));
 return out}
function paint(el,a,b,ticks){const W=b-a,pct=x=>(100*(x-a)/W).toFixed(3)+"%";let h="";
 if(ticks){const step=W<=4?0.2:W<=12?0.5:W<=40?1:5,lab=W<=4?1:W<=12?2:W<=40?5:30;
  for(let t=Math.ceil(a/step)*step;t<=b;t+=step){const big=Math.abs(t/lab-Math.round(t/lab))<1e-6;
   h+=`<i class="tk${big?" big":""}" style="left:${pct(t)}"></i>`;if(big)h+=`<i class=lb style="left:${pct(t)}">${mmss(t)}</i>`}}
 for(const[c,x0,x1,t]of marks()){if((x1??x0)<a||x0>b)continue;
  h+=x1===null?`<i class="${c}" title="${esc(t)}" style="left:${pct(x0)}"></i>`:
   `<i class="${c}" title="${esc(t)}" style="left:${pct(Math.max(x0,a))};width:calc(${pct(Math.min(x1,b))} - ${pct(Math.max(x0,a))} + 2px)"></i>`}
 if(!ticks){const[za,zb]=zwin();h+=`<i class=win style="left:${pct(za)};width:${(100*(zb-za)/W).toFixed(3)}%"></i>`}
 h+=`<i class=ph style="left:${pct(now())}"></i>`;el.innerHTML=h}
function draw(){if(cur===null)return;const s=D.shots[cur],e=P[s.id]||{};
 paint(tl,0,dur(),false);paint(zt,...zwin(),true);
 {const t=now(),m=Math.floor(t/60);document.getElementById("pos").textContent=
  m+":"+(t-60*m).toFixed(2).padStart(5,"0")+"  f"+Math.round(t*50)}
 document.getElementById("speed").textContent=rew?"◀◀ "+rew.n+"×":shuttle>1?"▶▶ "+shuttle+"×":(inPt!==null||outPt!==null)?
  "in "+(inPt===null?"–":mmss(inPt))+" · out "+(outPt===null?"–":mmss(outPt)):"";
 document.querySelectorAll("#alts button").forEach(b=>b.classList.toggle("on",Math.abs(+b.dataset.t-now())<0.6));
 const m=document.getElementById("marks");const adds=e.add||[];
 m.innerHTML=(adds.length?`Extra shots (${adds.length}): `+adds.map((v,k)=>`<b class=jm data-t="${span(v)[0]}" style="cursor:pointer">${esc(v)}</b> <button data-k=${k} class=rm>✕</button>`).join(" "):"No extra shots marked — press M at each moment you want.")+
  (e.more?` · more from this video ×${e.more}`:"")+(isBanned(cur)?" · <span style='color:var(--ban)'>this shot is banned</span>":"");
 m.querySelectorAll(".rm").forEach(b=>b.onclick=()=>{e.add.splice(+b.dataset.k,1);save()});
 m.querySelectorAll(".jm").forEach(b=>b.onclick=()=>seek(+b.dataset.t))}
// drag to scrub: the overview maps the whole video; the zoomed strip holds its window still while you drag
function at(el,ev){const r=el.getBoundingClientRect(),f=Math.min(Math.max((ev.clientX-r.left)/r.width,0),1);
 const[a,b]=el===tl?[0,dur()]:zwin();return a+f*(b-a)}
for(const el of [tl,zt]){
 el.addEventListener("pointerdown",ev=>{vid.pause();stopShuttle();el.setPointerCapture(ev.pointerId);
  const[a]=zwin();drag={el,a};seek(at(el,ev))});
 el.addEventListener("pointermove",ev=>{const t=at(el,ev);hov.style.display="block";hov.textContent=mmss(t);
  hov.style.left=(ev.clientX+10)+"px";hov.style.top=(el.getBoundingClientRect().top-20)+"px";if(drag&&drag.el===el)seek(t)});
 el.addEventListener("pointerup",()=>{drag=null;draw()});
 el.addEventListener("pointerleave",()=>{hov.style.display="none"});
 el.addEventListener("wheel",ev=>{ev.preventDefault();if(ev.shiftKey||Math.abs(ev.deltaX)>Math.abs(ev.deltaY)){
   seek(now()+(ev.deltaX||ev.deltaY)/100*zoomW/8)}else zoom(ev.deltaY>0?1.4:1/1.4)},{passive:false})}
function zoom(f){zoomW=Math.min(Math.max(zoomW*f,1),Math.max(dur(),2));draw()}
vid.addEventListener("timeupdate",draw);vid.addEventListener("seeked",draw);
let raf=null;vid.addEventListener("play",()=>{const loop=()=>{draw();if(!vid.paused)raf=requestAnimationFrame(loop)};loop()});
function stopShuttle(){if(rew){clearInterval(rew.id);rew=null}shuttle=0;vid.playbackRate=+document.getElementById("rate").value}
function playL(){if(rew){stopShuttle()}const base=+document.getElementById("rate").value;
 shuttle=vid.paused?1:Math.min(shuttle*2||2,8);vid.playbackRate=base*shuttle;vid.play();draw()}
function playJ(){vid.pause();shuttle=0;const n=rew?Math.min(rew.n*2,8):1;if(rew)clearInterval(rew.id);
 rew={n,id:setInterval(()=>{if(now()<=0)return stopShuttle();vid.currentTime=Math.max(0,now()-0.04*n);draw()},40)};draw()}
function pause(){vid.pause();stopShuttle();draw()}
const step=d=>{pause();seek(now()+d)};
function pts(){const s=D.shots[cur],e=P[s.id]||{};return[...D.shots.filter(x=>x.id==s.id).map(x=>x.a),
 ...((D.sources[s.id]||{}).moments||[]).map(m=>m[0]),...(e.add||[]).map(v=>span(v)[0])].sort((a,b)=>a-b)}
function jump(dir){const t=now(),p=pts();const x=dir>0?p.find(v=>v>t+0.05):p.reverse().find(v=>v<t-0.05);if(x!==undefined)seek(x)}
document.getElementById("back").onclick=()=>step(-1);document.getElementById("fwd").onclick=()=>step(1);
document.getElementById("fb").onclick=()=>step(-FR);document.getElementById("ff").onclick=()=>step(FR);
document.getElementById("play").onclick=()=>vid.paused?playL():pause();
document.getElementById("rate").onchange=ev=>{vid.playbackRate=+ev.target.value};
document.getElementById("zin").onclick=()=>zoom(1/1.5);document.getElementById("zout").onclick=()=>zoom(1.5);
document.getElementById("goshot").onclick=()=>seek(D.shots[cur].a);
document.getElementById("helpb").onclick=()=>document.getElementById("help").scrollIntoView({block:"nearest"});
document.getElementById("close").onclick=()=>dlg.close();
dlg.addEventListener("close",()=>{pause();hov.style.display="none"});
function addHere(){const e=ent(D.shots[cur].id);e.add=e.add||[];const n=now(),m=Math.floor(n/60);
 const t=m+":"+(n-60*m).toFixed(2).padStart(5,"0");                              // frame-accurate
 if(!e.add.some(v=>Math.abs(span(v)[0]-now())<0.15))e.add.push(t);e.add.sort((a,b)=>span(a)[0]-span(b)[0]);save()}
function delHere(){const e=P[D.shots[cur].id];if(!e||!e.add)return;let k=-1,best=0.6;
 e.add.forEach((v,j)=>{const d=Math.abs(span(v)[0]-now());if(d<best){best=d;k=j}});if(k>=0){e.add.splice(k,1);save()}}
function io(){if(inPt===null&&outPt===null)return null;const a=inPt??now(),b=outPt??now();
 return Math.abs(b-a)<0.2?null:[Math.min(a,b),Math.max(a,b)]}
function swapHere(){const s=D.shots[cur],e=ent(s.id),r=io()||[Math.max(0,now()-0.6),now()+0.6];
 e.ban=e.ban||[];e.prefer=e.prefer||[];if(!e.ban.includes(shotRange(cur)))e.ban.push(shotRange(cur));
 e.prefer.push(range(...r));inPt=outPt=null;save()}
function preferIO(){const r=io();if(!r)return;const e=ent(D.shots[cur].id);e.prefer=e.prefer||[];e.prefer.push(range(...r));
 inPt=outPt=null;save()}
document.getElementById("add").onclick=addHere;document.getElementById("swap").onclick=swapHere;
document.getElementById("pref").onclick=preferIO;
document.getElementById("bandlg").onclick=()=>toggleBan(cur);
document.getElementById("moredlg").onclick=()=>more(D.shots[cur].id,1);
document.getElementById("lessdlg").onclick=()=>more(D.shots[cur].id,-1);
window.addEventListener("keydown",ev=>{if(!dlg.open||ev.target.tagName=="SELECT"||ev.ctrlKey||ev.metaKey||ev.altKey)return;
 const k=ev.key,K=k.toLowerCase();let done=true;
 if(k==" "||K=="k")vid.paused&&!rew?playL():pause();
 else if(K=="l")playL();else if(K=="j")playJ();
 else if(k=="ArrowLeft"||k==",")step(ev.shiftKey?-1:-FR);
 else if(k=="ArrowRight"||k==".")step(ev.shiftKey?1:FR);
 else if(k=="Home")seek(0);else if(k=="End")seek(dur());
 else if(/^[0-9]$/.test(k))seek(dur()*(+k)/10);
 else if(K=="m"||K=="a")addHere();
 else if(k=="Backspace"||k=="Delete")delHere();
 else if(k=="[")jump(-1);else if(k=="]")jump(1);
 else if(K=="i"){inPt=now();draw()}else if(K=="o"){outPt=now();draw()}else if(K=="x"){inPt=outPt=null;draw()}
 else if(K=="s")swapHere();else if(K=="p")preferIO();else if(K=="b")toggleBan(cur);
 else if(k=="="||k=="+")zoom(1/1.5);else if(k=="-"||k=="_")zoom(1.5);
 else if(k=="?")document.getElementById("help").scrollIntoView({block:"nearest"});
 else done=false;
 if(done){ev.preventDefault();ev.stopPropagation()}},true);
</script>"""


def sources(project, plan, fmt, page_dir):
    """Per unit in the plan: its export file (relative to the page; browsers play the H.264 exports, not the cameras'
    10-bit 4:2:2 HEVC) and the moments the planner found in it, in the export's own timeline, best first."""
    moments = {r["unit"]["id"]: r for r in project.load("moments.json")} if os.path.exists(
        project.path("moments.json")) else {}
    key = "score_p" if fmt == "portrait" else "score_l"
    out = {}
    for s in plan["shots"]:
        u = s["unit"]
        if u["id"] in out:
            continue
        x0 = u.get("export_t0", 0.0)
        f = os.path.join(project.export_dir, u.get("category", ""), u["id"].split("#")[0] + ".mp4")
        r = moments.get(u["id"], {})
        ms = sorted(r.get("moments", []), key=lambda m: -m.get(key, m.get("score", 0)))
        # a joined B-roll piece ("#n") has no export of its own: its times are its own clip's, not the joined file's
        ok = os.path.exists(f) and "#" not in u["id"]
        out[u["id"]] = dict(video=quote(os.path.relpath(f, page_dir).replace(os.sep, "/")) if ok else None,
                            moments=[[round(m["t"] - x0, 2), m["kind"], round(m.get(key, m.get("score", 0)), 2)]
                                     for m in ms if m["t"] - x0 >= 0])
    return out


def run(project, fmt, cut=None) -> str:
    cut = cut or fmt
    plan = project.load(f"plan_{cut}.json")
    fmt = plan.get("format", fmt)
    video = project.path("out", f"{project.out_stem(cut, fmt)}.mp4")
    out = project.path("review", cut)
    os.makedirs(out, exist_ok=True)
    w = 320 if fmt == "landscape" else 180
    thumbs = []
    for i, s in enumerate(plan["shots"]):
        jpg = os.path.join(out, f"{i + 1:03d}.jpg")
        if os.path.exists(video):
            src, t = video, s["rec_in"] + s["length"] * 0.5
        else:
            src, t = project.decode_from(s["unit"]["source"], s["src_in"] + s["length"] * 0.5)
            src = src if os.path.isabs(src) else os.path.join(project.folder, src)
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1",
                            "-vf", f"scale={w}:-2", "-q:v", "4", jpg])
        thumbs.append(f"{cut}/{i + 1:03d}.jpg" if r.returncode == 0 and os.path.exists(jpg) else None)
    path = project.path("review", f"{cut}.html")
    with open(path, "w") as f:
        f.write(page(plan, fmt, thumbs, sources(project, plan, fmt, os.path.dirname(path)), project.picks(),
                     key=f"{project.name}/{cut}", root=browse.api_root(project),
                     nav=browse.menu(project, cut)))
    print(f"[reel] review: {path}")
    return path
