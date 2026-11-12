// The source browser. This file: what its three views share (the reel being worked on, filmstrips, timelines, the
// players) and the Clips view -- every clip, one at a time, over a zoomable timeline where the editor marks
// selections. cut.js is Cut & picks, frame.js the portrait framing, main.js the views, the keys and the start.
const TW = D.tile[0], TH = D.tile[1], FR = 1 / 50;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

// --- the reel being worked on: one picks.json serves the project's reel ("main") and its cuts ----------------------
let reel = ls.get("reel") || "all";
if (!RID.includes(reel)) reel = "all";
const reelName = () => (reel == "all" ? "every reel" : rlab(reel));
function useReel() {
  const r = reel == "all" ? "main" : reel;
  D.clips.forEach((c) => {
    const u = (c.R || {})[r];
    if (u) [c.L, c.P] = [u.L, u.P];
  });
  $("reelsw").innerHTML = RS.length > 1
    ? "<b>Reel</b>" + [["all", "All"], ...RS.map((x) => [x.id, x.label])]
        .map(([k, l]) => `<button data-reel="${esc(k)}" class="${k == reel ? "on" : ""}">${esc(l)}</button>`).join("")
    : "";
}
function setReel(r, quiet) {
  if (r == reel || (r != "all" && !RID.includes(r))) return;
  reel = r;
  ls.set("reel", reel);
  useReel();
  place.clear();
  fcur = null;
  if (!quiet) toast(reel == "all" ? "All reels: picks and bans go in every reel" : `${rlab(reel)}: picks, bans, the opening and the ending are for this reel only`);
  renderAll();
}
$("reelsw").onclick = (ev) => {
  const b = ev.target.closest("[data-reel]");
  if (b) setReel(b.dataset.reel);
};
useReel();

// --- filmstrips and timelines ----------------------------------------------------------------------------------------
// a clip's frame at t as a background, h tall; in a box W wide, a narrower (vertical) frame sits in the middle, its
// neighbours in the strip clipped off
const tileOf = (c) => c.tile || [TW, TH];
const tileAt = (c, t) => Math.min(c.n - 1, Math.max(0, Math.floor(t / c.step)));
const thumbAt = (c, t, h, W) => {
  const [tw, th] = tileOf(c), k = tileAt(c, t), s = h / th, pad = W ? Math.max(0, (W - tw * s) / 2) : 0;
  return `background-image:url('${c.strip}');background-size:${c.n * tw * s}px ${h}px;background-position:-${k * tw * s}px 0` +
    (pad ? `;box-sizing:border-box;padding-left:${pad}px;padding-right:${pad}px;background-origin:content-box;background-clip:content-box` : "");
};
const X = (t, a, b) => ((100 * (t - a)) / (b - a)).toFixed(3) + "%";
function wspan(a, b, A, B) {
  const x0 = Math.max(a, A), x1 = Math.min(b, B);
  return `left:${X(x0, A, B)};width:${((100 * (x1 - x0)) / (B - A)).toFixed(3)}%`;
}
function tAt(el, ev, A, B) {
  const r = el.getBoundingClientRect();
  return A + clamp((ev.clientX - r.left) / r.width, 0, 1) * (B - A);
}
// the filmstrip of clip c from A to B across a canvas h tall
function film(canvas, A, B, h, c, img) {
  const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  canvas.width = Math.max(1, r.width * dpr);
  canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.fillStyle = "#000";
  g.fillRect(0, 0, canvas.width, canvas.height);
  if (!img.complete || !img.naturalWidth) return;
  const [tw_, th_] = tileOf(c), tw = ((h * tw_) / th_) * dpr, W = canvas.width;
  for (let x = 0; x < W; x += tw) {
    const t = A + ((x + tw / 2) / W) * (B - A);
    if (t > c.dur) break;
    g.drawImage(img, tileAt(c, t) * tw_, 0, tw_, th_, x, 0, tw, h * dpr);
  }
}
const strips = new Map(); // a clip's filmstrip image, loaded once
function stripOf(c, then) {
  let im = strips.get(c.id);
  if (!im) {
    im = new Image();
    strips.set(c.id, im);
    im.src = c.strip;
  }
  if (then && !im.complete) im.addEventListener("load", then, { once: true });
  return im;
}
const hov = $("hover");
function showHover(ev, c, t, el) {
  hov.style.display = "block";
  hov.querySelector("div").setAttribute("style", thumbAt(c, t, 108, 192));
  hov.querySelector("span").textContent = fmt(t);
  hov.style.left = Math.min(innerWidth - 205, Math.max(4, ev.clientX - 100)) + "px";
  hov.style.top = el.getBoundingClientRect().top - 136 + "px";
}
const hideHover = () => (hov.style.display = "none");

// --- players: a video may not play at all (opened as a file with no export beside it, a codec the browser lacks),
// so each keeps the time it was last sent to, and the filmstrip stands in for the picture -----------------------------
// scrubbing: one seek in flight per video, the next goes to wherever the pointer is by then (a new currentTime on
// every mouse move restarts the seek, so the picture would never catch up until the mouse stops)
function goTo(v, t) {
  v._at = t;
  if (v.seeking) return void (v._want = t);
  v._want = null;
  try {
    v.currentTime = t;
  } catch (e) {}
}
function load(v, c, t) {
  if (v.dataset.c !== c.id) {
    v.dataset.c = c.id;
    v.src = c.video;
  }
  v._want = null;
  v._at = t;
  try {
    v.currentTime = t;
  } catch (e) {}
}
const tOf = (v) => v._want ?? ((v.readyState >= 1 ? v.currentTime : v._at) || 0);
const noVideo = (v) => !!(v.error || (v.readyState < 2 && !v.seeking) || !v.videoWidth);
for (const v of document.querySelectorAll("video")) {
  v.addEventListener("seeked", () => {
    if (v._want != null) {
      const t = v._want;
      v._want = null;
      v.currentTime = t;
    }
  });
  v.addEventListener("loadedmetadata", () => {
    if (v._at != null && Math.abs(v.currentTime - v._at) > 0.02) v.currentTime = v._at;
  });
}
// the frame of clip c at t from its filmstrip, where the video would be
function stillDraw(st, wrap, v, c, t) {
  const no = noVideo(v);
  st.classList.toggle("on", no);
  if (!no) return;
  const r = wrap.getBoundingClientRect(), [tw, th] = tileOf(c), h = Math.min(r.height, (r.width * th) / tw);
  st.setAttribute("style", thumbAt(c, t, h) + `;width:${(h * tw) / th}px;height:${h}px;left:${(r.width - (h * tw) / th) / 2}px;top:${(r.height - h) / 2}px`);
}
// where the picture sits inside its box (letterboxed), in the box's pixels
function vrect(wrap, v, c) {
  const R = wrap.getBoundingClientRect(), ar = v.videoWidth && v.videoHeight ? v.videoWidth / v.videoHeight : c.ar || TW / TH;
  const w = Math.min(R.width, R.height * ar), h = w / ar;
  return { l: (R.width - w) / 2, t: (R.height - h) / 2, w, h };
}
const starBar = (id, a, b, attr) => {
  const n = starsOf(id, a, b);
  return `<span class=stars title="how much you want it: the planner keeps and lengthens the most starred first (click again to clear)">` +
    [1, 2, 3, 4, 5].map((i) => `<b data-star=${i} ${attr} title="${i} star${i > 1 ? "s" : ""} — key ${i}">${i <= n ? "★" : "☆"}</b>`).join("") + `</span>`;
};
function setStars(id, a, b, n) {
  putStars(id, a, b, n);
  save();
  toast(n ? `${"★".repeat(n)}${"☆".repeat(5 - n)}  ${id} ${fmt(a)}–${fmt(b)}` : "Stars cleared");
}
const bansOf = (id) => vals(id, "ban", reel);
const banSpans = (id) => bansOf(id).map((v) => (v == "all" ? [-1, 1e9] : span(v)));
const isOut = (bans, a, b) => bans.some(([u, v]) => (a + b) / 2 >= u && (a + b) / 2 <= v);

// --- Clips: the list -------------------------------------------------------------------------------------------------
let cur = 0, cat = "all", onlySel = false, rub = null;
const seen = new Set(ls.get("seen") || []);
const C = () => D.clips[cur];
const mine = (id) => sels(id).filter(([a, b]) => inReel(id, a, b, reel));
$("filters").innerHTML = ["all", ...new Set(D.clips.map((c) => c.cat))]
  .map((c) => `<button data-c="${esc(c)}" class="${c == "all" ? "on" : ""}">${esc(c)}</button>`).join("") +
  `<button id=onlysel title="only clips with selections">● with selections</button>`;
$("filters").onclick = (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  if (b.id == "onlysel") b.classList.toggle("on", (onlySel = !onlySel));
  else {
    cat = b.dataset.c;
    $("filters").querySelectorAll("[data-c]").forEach((x) => x.classList.toggle("on", x === b));
  }
  list();
};
const visible = () => D.clips.map((c, i) => i).filter((i) => (cat == "all" || D.clips[i].cat == cat) && (!onlySel || mine(D.clips[i].id).length));
let listCur = -1;
function list() {
  $("list").innerHTML = visible().map((i) => {
    const c = D.clips[i], n = mine(c.id).length, tot = mine(c.id).reduce((a, [x, y]) => a + y - x, 0), bn = bansOf(c.id).includes("all");
    return `<div class="ci${i == cur ? " cur" : ""}${seen.has(c.id) ? " seen" : ""}${bn ? " out" : ""}" data-i=${i}>
 <div class=th style="${thumbAt(c, c.dur * 0.3, 54, 96)}">${n ? `<b>${n}</b>` : ""}</div><div><div class=nm>${esc(c.id)}</div>
 <div class=meta><span class=chip>${esc(c.cat)}</span> ${esc(c.clock)} · ${short(c.dur)}${n ? ` · <span style="color:var(--sel)">● ${short(tot)}</span>` : ""}</div>
 <div class=meta>${c.L.length ? `<span style="color:var(--land)">L${c.L.length}</span> ` : ""}${c.P.length ? `<span style="color:var(--port)">P${c.P.length}</span>` : ""}${bn ? " banned" : ""}</div></div></div>`;
  }).join("");
  if (listCur != cur) {            // (only when the clip changes: a save must not jump a list being read)
    listCur = cur;
    const el = $("list").querySelector(".cur");
    if (el) el.scrollIntoView({ block: "nearest" });
  }
}
$("list").onclick = (ev) => {
  const d = ev.target.closest(".ci");
  if (d) go(+d.dataset.i);
};

// --- Clips: the player and its timeline ----------------------------------------------------------------------------
const vid = $("vid"), tl = $("tl"), ov = $("ov"), L = $("lsel");
let va = 0, vb = 1, inPt = null, outPt = null, act = -1, loop = false, playOne = false, shuttle = 0, rew = null, drag = null;
const now = () => tOf(vid), dur = () => C().dur || vid.duration || 1;
function go(i, t) {
  if (i < 0 || i >= D.clips.length) return;
  cur = i;
  const c = C();
  seen.add(c.id);
  ls.set("seen", [...seen]);
  ls.set("clip", c.id);
  inPt = outPt = null;
  act = -1;
  loop = false;
  stopShuttle();
  vid.pause();
  load(vid, c, t || 0);
  va = 0;
  vb = c.dur;
  stripOf(c, paint);
  $("cname").textContent = c.id;
  $("cmeta").innerHTML = `<span class=chip>${esc(c.cat)}</span> ${esc(c.tag)} · ${esc(c.clock)} · ${short(c.dur)} · clip ${i + 1} of ${D.clips.length}`;
  renderAll();
  paint();
}
function seek(t) {
  if (rew) stopShuttle();
  goTo(vid, clamp(t, 0, dur() - 0.02));
  follow();
  draw();
}
function follow() {
  const t = now(), w = vb - va;
  if (t < va || t > vb) {
    va = clamp(t - w * 0.1, 0, Math.max(0, dur() - w));
    vb = va + w;
    paint();
  }
}
const Xv = (t) => X(t, va, vb), wv = (a, b) => wspan(a, b, va, vb);
function paint() {
  const c = C();
  if (!c || curView != 1) return;
  film($("strip").querySelector("canvas"), va, vb, 80, c, stripOf(c));
  film(ov.querySelector("canvas"), 0, dur(), 30, c, stripOf(c));
  draw();
}
function draw() {
  const c = C();
  if (!c || curView != 1) return;
  const W = vb - va, t = now(), ss = sels(c.id);
  // ruler
  const stp = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60].find((s) => W / s <= 60) || 120, lab = [0.5, 1, 2, 5, 10, 15, 30, 60, 120].find((s) => W / s <= 10) || 300;
  let h = "";
  for (let x = Math.ceil(va / stp) * stp; x <= vb; x += stp) {
    const big = Math.abs(x / lab - Math.round(x / lab)) < 1e-6;
    h += `<i class="tk${big ? " big" : ""}" style="left:${Xv(x)}"></i>`;
    if (big) h += `<i class=lb style="left:${Xv(x)}">${W < 15 ? fmt(x) : short(x)}</i>`;
  }
  $("ruler").innerHTML = h;
  // over the filmstrip: moments and bans
  const S = $("strip");
  S.querySelectorAll("i").forEach((e) => e.remove());
  let s = "";
  c.moments.forEach(([m, k]) => {
    if (m >= va && m <= vb) s += `<i class=mom style="left:${Xv(m)}" title="${esc(k)} ${short(m)}"></i>`;
  });
  bansOf(c.id).forEach((v) => {
    const [a, b] = v == "all" ? [0, c.dur] : span(v);
    if (b > va && a < vb) s += `<i class=ban style="${wv(a, b)}" title="banned ${esc(v)}"></i>`;
  });
  S.insertAdjacentHTML("beforeend", s);
  // the selections lane
  L.querySelectorAll("i").forEach((e) => e.remove());
  s = "";
  ss.forEach(([a, b], k) => {
    if (b <= va || a >= vb) return;
    const o = onlyOf(c.id, a, b), st = starsOf(c.id, a, b);
    s += `<i class="blk${k == act ? " act" : ""}${inReel(c.id, a, b, reel) ? "" : " away"}" data-k=${k} style="${wv(a, b)}"
 title="${fmt(a)} – ${fmt(b)} (${(b - a).toFixed(2)} s)${RS.length > 1 ? " · " + (o ? o.map(rlab).join(" + ") + " only" : "every reel") : ""}"><i class="h l"></i>${k + 1} · ${(b - a).toFixed(1)}s${st ? " " + "★".repeat(st) : ""}<i class="h r"></i></i>`;
  });
  points(c.id).forEach((p) => {
    if (p >= va && p <= vb) s += `<i class=pt style="left:${Xv(p)}" title="marked moment ${fmt(p)}"></i>`;
  });
  const io = pend();
  if (io) s += `<i class=pend style="${wv(...io)}"></i>`;
  if (!ss.length && !io) s += `<i class=empty>drag here to pick a range · or I / O then Enter · or click a reel shot below to take it</i>`;
  L.insertAdjacentHTML("beforeend", s);
  // the reel lanes: the shots the reel takes from this clip now
  const bans = banSpans(c.id);
  for (const [id, arr] of [["lland", c.L], ["lport", c.P]]) {
    const e = $(id);
    e.querySelectorAll("i").forEach((x) => x.remove());
    e.insertAdjacentHTML("beforeend", arr.filter(([a, b]) => b > va && a < vb).map(([a, b, n]) =>
      `<i class="use${isOut(bans, a, b) ? " out" : ""}" data-a=${a} data-b=${b} style="${wv(a, b)}" title="reel shot ${n}: ${short(a)}–${short(b)} — click to pick it, ✕ to take it out; drag along the lanes to take out every shot in a range">${n}<b class=ux title="take this shot out of the cut (bans its stretch)">✕</b></i>`).join(""));
  }
  if (rub && rub.moved) {
    const [a, b] = [Math.min(rub.t0, rub.t1), Math.max(rub.t0, rub.t1)];
    for (const id of ["lland", "lport"]) $(id).insertAdjacentHTML("beforeend", `<i class=rub style="${wv(a, b)}"></i>`);
  }
  fkeys(c.id).forEach(([k]) => {
    if (k >= va && k <= vb) $("lport").insertAdjacentHTML("beforeend", `<i class=fk style="left:${Xv(k)}" title="your pan of the portrait crop at ${fmt(k)}"></i>`);
  });
  folkeys(c.id).forEach(([k]) => {
    if (k >= va && k <= vb) $("lport").insertAdjacentHTML("beforeend", `<i class="fk fl" style="left:${Xv(k)}" title="following a person from ${fmt(k)}"></i>`);
  });
  // marks across the whole timeline
  tl.querySelectorAll(":scope>i").forEach((e) => e.remove());
  let m = "";
  if (inPt !== null && inPt >= va && inPt <= vb) m += `<i class=inm style="left:${Xv(inPt)}"></i>`;
  if (outPt !== null && outPt >= va && outPt <= vb) m += `<i class=outm style="left:${Xv(outPt)}"></i>`;
  if (t >= va && t <= vb) m += `<i class=ph style="left:${Xv(t)}"></i>`;
  tl.insertAdjacentHTML("beforeend", m);
  // the overview
  ov.querySelectorAll("i").forEach((e) => e.remove());
  const D0 = dur();
  let o = "";
  ss.forEach(([a, b]) => (o += `<i class=blk style="${wspan(a, b, 0, D0)};top:4px;bottom:4px;pointer-events:none"></i>`));
  o += `<i class=win style="${wspan(va, vb, 0, D0)}"></i><i class=ph style="left:${X(t, 0, D0)}"></i>`;
  ov.insertAdjacentHTML("beforeend", o);
  $("tc").textContent = fmt(t) + "  /  " + short(D0);
  stillDraw($("still"), $("vwrap"), vid, c, t);
  $("io").textContent = inPt === null && outPt === null ? "" :
    "in " + (inPt === null ? "—" : fmt(inPt)) + "  out " + (outPt === null ? "—" : fmt(outPt)) + (io ? "  (" + (io[1] - io[0]).toFixed(2) + " s)" : "");
  $("play").textContent = vid.paused && !rew ? "▶ Play" : "❚❚ Pause";
  $("banc").textContent = bansOf(c.id).includes("all") ? "Unban clip" : "Ban clip";
  $("badd").disabled = $("bban").disabled = !io;
  frameDraw($("fr"), $("vwrap"), vid, c, ...rangeFor(c, t), t);
}
function pend() {
  if (inPt === null && outPt === null) return null;
  const a = inPt ?? now(), b = outPt ?? now();
  return Math.abs(b - a) < 0.05 ? null : [Math.min(a, b), Math.max(a, b)];
}
function mysel() {
  const c = C(), ss = sels(c.id);
  $("mysel").innerHTML = ss.length
    ? `<table><tr><th>#</th><th>in</th><th>out</th><th>length</th><th>stars</th>${RS.length > 1 ? "<th>in the reels</th>" : ""}<th></th></tr>` +
      ss.map(([a, b], k) => `<tr class="${k == act ? "act" : ""}${inReel(c.id, a, b, reel) ? "" : " away"}"><td>${k + 1}</td><td>${fmt(a)}</td><td>${fmt(b)}</td><td>${(b - a).toFixed(2)} s</td>
 <td>${starBar(c.id, a, b, `data-k=${k}`)}</td>${RS.length > 1 ? "<td>" + RS.map((r) => `<label class=rk title="in the ${esc(r.label)} reel"><input type=checkbox data-r="${esc(r.id)}" data-k=${k}${inReel(c.id, a, b, r.id) ? " checked" : ""}> ${esc(r.label)}</label>`).join(" ") + "</td>" : ""}
 <td><button data-p=${k}>▶ Play</button> <button data-l=${k}>Loop</button> <button data-d=${k} class=danger title="remove this selection${reel == "all" ? "" : " from " + esc(rlab(reel))}">✕</button></td></tr>`).join("") + `</table>`
    : `<p class=mut style="margin:0">No selections in this clip yet.</p>`;
}
$("mysel").onchange = (ev) => {
  const i = ev.target, x = sels(C().id)[+i.dataset.k];
  if (!i.dataset.r || !x) return;
  if (!setIn(C().id, x[0], x[1], i.dataset.r, i.checked)) {
    toast("A pick is in one reel at least — ✕ removes it", "warn");
    return renderAll();
  }
  save();
  toast(`${fmt(x[0])}–${fmt(x[1])} ${i.checked ? "is now in" : "taken out of"} ${rlab(i.dataset.r)}`);
};
$("mysel").onclick = (ev) => {
  const c = C(), ss = sels(c.id), st = ev.target.closest("[data-star]");
  if (st) {
    const x = ss[+st.dataset.k], n = +st.dataset.star;
    if (x) setStars(c.id, x[0], x[1], starsOf(c.id, x[0], x[1]) == n ? 0 : n);
    return;
  }
  const b = ev.target.closest("button");
  if (!b) return;
  if (b.dataset.d !== undefined) return takeOut(ss[+b.dataset.d]);
  const k = +(b.dataset.p ?? b.dataset.l);
  act = k;
  loop = b.dataset.l !== undefined;
  playOne = !loop;
  seek(ss[k][0]);
  vid.play();
  renderAll();
};
// a selection out of the reel being worked on (it stays in the others)
function takeOut(x) {
  if (!x) return;
  const left = removeSel(C().id, x[0], x[1], reel);
  act = -1;
  save();
  toast(left.length ? `Taken out of ${rlab(reel)} — still in ${left.map(rlab).join(", ")}` : `Selection ${fmt(x[0])}–${fmt(x[1])} removed`);
}
function pickRange(a, b) {
  const c = C(), fresh = addSel(c.id, a, b, reel);
  inPt = outPt = null;
  act = sels(c.id).findIndex(([x, y]) => near2(x, Math.min(a, b)) && near2(y, Math.max(a, b)));
  if (fresh) {
    save();
    toast(`Selection ${fmt(Math.min(a, b))}–${fmt(Math.max(a, b))} added${reel == "all" ? "" : " for " + rlab(reel)}`);
  } else renderAll();
}
// what's in play: the planner's shots in this reel, and the editor's own calls
function counts() {
  const n = (f) => D.clips.reduce((a, c) => a + f(c), 0);
  const bc = n((c) => (bansOf(c.id).includes("all") ? 1 : 0)), br = n((c) => bansOf(c.id).filter((v) => v != "all").length);
  const framed = n((c) => c.P.filter(([a, b]) => framingOf(c.id, a, b) != "auto").length);
  const it = (v, label, title, cls = "") => `<span class="${cls}" title="${esc(title)}"><b>${v}</b> ${esc(label)}</span>`;
  $("counts").innerHTML = [
    it(n((c) => c.L.length), "landscape shots", "in the current landscape cut of " + (reel == "all" ? "the main reel" : rlab(reel))),
    it(n((c) => c.P.length), "portrait shots", "in the current portrait cut of " + (reel == "all" ? "the main reel" : rlab(reel))),
    it(n((c) => mine(c.id).length + points(c.id).length), "picks", "your selections and marked moments for " + reelName() + ": the next re-plan must use them", "y"),
    framed ? it(framed, "framed by you", "portrait shots you panned or made follow a person", "f") : "",
    bc + br ? it(bc + br, "bans", bc + " whole clips, " + br + " stretches", "x") : "",
    it(D.clips.length - D.clips.filter((c) => seen.has(c.id)).length, "clips not seen", "clips you have not opened yet"),
  ].filter(Boolean).join(" · ");
}
// the timeline's mouse: scrub on the ruler, the filmstrip and the overview; make, move and trim in the selections lane
for (const el of [$("ruler"), $("strip")]) {
  el.addEventListener("pointerdown", (ev) => {
    pause();
    el.setPointerCapture(ev.pointerId);
    drag = { k: "scrub", el };
    seek(tAt(el, ev, va, vb));
  });
  el.addEventListener("pointermove", (ev) => {
    const t = tAt(el, ev, va, vb);
    showHover(ev, C(), t, tl);
    if (drag && drag.el === el) seek(t);
  });
}
ov.addEventListener("pointerdown", (ev) => {
  ov.setPointerCapture(ev.pointerId);
  drag = { k: "ov", el: ov };
  movewin(tAt(ov, ev, 0, dur()));
});
ov.addEventListener("pointermove", (ev) => {
  const t = tAt(ov, ev, 0, dur());
  showHover(ev, C(), t, ov);
  if (drag && drag.k == "ov") movewin(t);
});
function movewin(t) {
  const w = vb - va;
  va = clamp(t - w / 2, 0, Math.max(0, dur() - w));
  vb = va + w;
  if (now() < va || now() > vb) seek(t);
  paint();
}
// drags in the lanes snap to the playhead (where it was when the drag began) within 10 px
const near = (el, t, p) => p != null && (Math.abs(t - p) * el.getBoundingClientRect().width) / (vb - va) < 10;
const snapT = (el, ev, p = now()) => {
  const t = tAt(el, ev, va, vb);
  return near(el, t, p) ? p : t;
};
// a selection's edge within 8 px of the pointer, inside or just outside the block (the nearer one wins)
function edgeAt(ev) {
  const r = L.getBoundingClientRect(), px = r.width / (vb - va), x = ev.clientX - r.left;
  let best = null, bd = 8;
  sels(C().id).forEach(([a, b], k) => {
    for (const [t, side] of [[a, "trimL"], [b, "trimR"]]) {
      const d = Math.abs((t - va) * px - x);
      if (d <= bd) [bd, best] = [d, { k, side }];
    }
  });
  return best;
}
L.addEventListener("pointerdown", (ev) => {
  if (ev.button) return;
  pause();
  L.setPointerCapture(ev.pointerId);
  const ph = now(), t = snapT(L, ev, ph), b = ev.target.closest(".blk"), e = edgeAt(ev);
  if (e || b) {
    const k = e ? e.k : +b.dataset.k, ss = sels(C().id);
    act = k;
    drag = { k: e ? e.side : "move", i: k, t0: tAt(L, ev, va, vb), orig: ss[k].slice(), cur: ss[k].slice(), moved: false, ph };
  } else drag = { k: "new", a: t, b: t, ph };
  draw();
});
L.addEventListener("pointermove", (ev) => {
  showHover(ev, C(), tAt(L, ev, va, vb), tl);
  L.style.cursor = drag ? (drag.k.startsWith("trim") ? "ew-resize" : "") : edgeAt(ev) ? "ew-resize" : "";
  if (!drag || drag.el) return;
  const t = snapT(L, ev, drag.ph);
  if (drag.k == "new") {
    drag.b = t;
    inPt = Math.min(drag.a, t);
    outPt = Math.max(drag.a, t);
    return seek(t);
  }
  const o = drag.orig, d = t - drag.t0;
  if (Math.abs(d) > 0.03) drag.moved = true;
  let [a, b] = o;
  if (drag.k == "move") {
    const w = b - a;
    a = clamp(o[0] + tAt(L, ev, va, vb) - drag.t0, 0, dur() - w);
    if (near(L, a, drag.ph)) a = drag.ph;
    else if (near(L, a + w, drag.ph)) a = drag.ph - w;
    b = a + w;
  } else if (drag.k == "trimL") seek((a = clamp(t, 0, b - 0.1)));
  else seek((b = clamp(t, a + 0.1, dur())));
  drag.cur = [a, b];
  const blk = L.querySelector(`.blk[data-k="${drag.i}"]`);
  if (blk) {
    blk.setAttribute("style", wv(a, b));
    blk.childNodes[1].textContent = `${drag.i + 1} · ${(b - a).toFixed(1)}s`;
  }
});
L.addEventListener("pointerup", (ev) => {
  if (!drag || drag.el) return;
  const c = C(), d = drag;
  drag = null;
  if (d.k == "new") {
    inPt = outPt = null;
    if (Math.abs(d.b - d.a) >= 0.2) pickRange(d.a, d.b);
    else seek(tAt(L, ev, va, vb));
    return;
  }
  if (d.moved && moveSel(c.id, ...d.orig, ...d.cur)) {
    act = sels(c.id).findIndex(([x]) => near2(x, d.cur[0]));
    save();
  } else {
    seek(d.orig[0]);
    renderAll();
  }
});
L.addEventListener("pointercancel", () => {
  drag = null;
  inPt = outPt = null;
  draw();
});
addEventListener("pointerup", () => {
  if (drag && drag.el) {
    drag = null;
    draw();
  }
});
// the reel lanes: ✕ on a shot takes it out of the cut; drag along either lane to take out every landscape and
// portrait shot in the range (bans their stretches; one undo step; the ends snap to the playhead); a plain click
// makes the shot a pick
function banShots(c, shots) {
  let n = 0;
  shots.forEach(([a, b]) => {
    if (addVal(c.id, "ban", rkey(a, b), reel)) n++;
  });
  if (n) save();
  return n;
}
for (const id of ["lland", "lport"]) {
  const el = $(id);
  el.addEventListener("pointerdown", (ev) => {
    if (ev.button || ev.target.closest(".ux")) return;
    el.setPointerCapture(ev.pointerId);
    rub = { t0: snapT(el, ev), t1: snapT(el, ev), x: ev.clientX, moved: false };
  });
  el.addEventListener("pointermove", (ev) => {
    showHover(ev, C(), tAt(el, ev, va, vb), tl);
    if (!rub || rub.done) return;
    rub.t1 = snapT(el, ev);
    if (Math.abs(ev.clientX - rub.x) > 4) rub.moved = true;
    if (rub.moved) draw();
  });
  el.addEventListener("pointerup", () => {
    const r = rub;
    if (!r || r.done) return;
    if (!r.moved) return void (rub = null);
    rub = { moved: false, done: true };
    const c = C(), [a, b] = [Math.min(r.t0, r.t1), Math.max(r.t0, r.t1)];
    const n = banShots(c, [...c.L, ...c.P].filter(([x, y]) => (x + y) / 2 >= a && (x + y) / 2 <= b));
    toast(n ? `Took ${n} shot${n == 1 ? "" : "s"} out of the cut${reel == "all" ? "" : " of " + rlab(reel)} — Ctrl+Z undoes` : "No reel shots in that range", n ? "ok" : "warn");
    draw();
  });
  el.addEventListener("click", (ev) => {
    if (rub && rub.done) return void (rub = null);
    const c = C(), x = ev.target.closest(".ux"), u = ev.target.closest(".use");
    if (x) {
      const n = banShots(c, [[+u.dataset.a, +u.dataset.b]]);
      return toast(n ? `Shot taken out of the cut${reel == "all" ? "" : " of " + rlab(reel)} — Ctrl+Z undoes` : "Already out");
    }
    if (!u) return;
    pickRange(+u.dataset.a, +u.dataset.b);
    seek(+u.dataset.a);
  });
}
for (const e of ["error", "loadeddata", "seeking", "seeked", "pause"]) vid.addEventListener(e, () => draw());
for (const el of [tl, ov]) el.addEventListener("pointerleave", hideHover);
tl.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const r = tl.getBoundingClientRect(), f = (ev.clientX - r.left) / r.width;
  if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) {
    const w = vb - va, d = ((ev.deltaX || ev.deltaY) / 600) * w;
    va = clamp(va + d, 0, Math.max(0, dur() - w));
    vb = va + w;
  } else zoom(ev.deltaY > 0 ? 1.25 : 0.8, va + f * (vb - va));
  paint();
}, { passive: false });
function zoom(k, at) {
  const D0 = dur(), w = clamp((vb - va) * k, 0.5, D0);
  at = at ?? now();
  const f = (at - va) / (vb - va || 1);
  va = clamp(at - f * w, 0, D0 - w);
  vb = va + w;
  paint();
}
const fit = () => {
  va = 0;
  vb = dur();
  paint();
};
// playback
function stopShuttle() {
  if (rew) {
    clearInterval(rew.id);
    rew = null;
  }
  shuttle = 0;
  vid.playbackRate = +$("rate").value;
}
function pause() {
  vid.pause();
  stopShuttle();
  draw();
}
function playL() {
  if (rew) stopShuttle();
  shuttle = vid.paused ? 1 : Math.min((shuttle || 1) * 2, 8);
  vid.playbackRate = +$("rate").value * shuttle;
  vid.play().catch(() => toast("This clip cannot play here — the filmstrip stands in; scrub the timeline", "warn"));
}
function playJ() {
  vid.pause();
  shuttle = 0;
  const n = rew ? Math.min(rew.n * 2, 8) : 1;
  if (rew) clearInterval(rew.id);
  rew = { n, id: setInterval(() => {
    if (now() <= 0) return stopShuttle();
    goTo(vid, Math.max(0, now() - 0.04 * n));
    follow();
    draw();
  }, 40) };
  draw();
}
const stepBy = (d) => {
  pause();
  seek(now() + d);
};
const togglePlay = () => (vid.paused && !rew ? ((playOne = false), playL()) : pause());
vid.addEventListener("play", () => {
  const f = () => {
    const ss = sels(C().id);
    if (loop && act >= 0 && ss[act] && now() >= ss[act][1]) vid.currentTime = ss[act][0];
    else if (playOne && act >= 0 && ss[act] && now() >= ss[act][1]) {
      vid.pause();
      playOne = false;
    }
    follow();
    draw();
    if (!vid.paused) requestAnimationFrame(f);
  };
  f();
});
$("play").onclick = togglePlay;
$("fb").onclick = () => stepBy(-FR);
$("ff").onclick = () => stepBy(FR);
$("rate").onchange = () => {
  vid.playbackRate = +$("rate").value;
  $("rate").blur();
};
$("sound").onclick = () => {
  const m = !vid.muted;
  document.querySelectorAll("video").forEach((v) => (v.muted = m));
  $("sound").textContent = m ? "Sound off" : "Sound on";
  $("sound").classList.toggle("on", !m);
};
$("zfit").onclick = fit;
$("zin").onclick = () => zoom(0.6);
$("zout").onclick = () => zoom(1.6);
$("prevc").onclick = () => nav(-1);
$("nextc").onclick = () => nav(1);
function nav(d) {
  const v = visible(), k = v.indexOf(cur), j = k < 0 ? v[0] : v[clamp(k + d, 0, v.length - 1)];
  if (j !== undefined && j !== cur) go(j);
}
function addIO() {
  const io = pend();
  if (!io) return toast("Mark an in (I) and an out (O) first", "warn");
  pickRange(...io);
}
function delHere() {
  const ss = sels(C().id), t = now(), hit = ([a, b]) => t >= a - 0.05 && t <= b + 0.05;
  const k = act >= 0 && ss[act] && hit(ss[act]) ? act : ss.findIndex((x) => hit(x) && inReel(C().id, x[0], x[1], reel));
  if (k < 0) return toast("No selection at the playhead to remove", "warn");
  takeOut(ss[k]);
}
function banIO() {
  const io = pend();
  if (!io) return toast("Mark an in (I) and an out (O) around what to ban first", "warn");
  addVal(C().id, "ban", rkey(...io), reel);
  inPt = outPt = null;
  save();
  toast(`Banned ${fmt(io[0])}–${fmt(io[1])}${reel == "all" ? "" : " for " + rlab(reel)} — Ctrl+Z undoes`);
}
$("banc").onclick = () => {
  const id = C().id, on = bansOf(id).includes("all");
  if (on) delVal(id, "ban", "all", reel);
  else addVal(id, "ban", "all", reel);
  save();
  toast(on ? `${id} can be used again` : `${id} banned${reel == "all" ? "" : " for " + rlab(reel)} — Ctrl+Z undoes`);
};
const markIn = () => {
  inPt = now();
  if (outPt !== null && outPt <= inPt) outPt = null;
  draw();
};
const markOut = () => {
  outPt = now();
  if (inPt !== null && inPt >= outPt) inPt = null;
  draw();
};
$("bin").onclick = markIn;
$("bout").onclick = markOut;
$("badd").onclick = addIO;
$("bban").onclick = banIO;
function marks() {
  const c = C();
  return [...sels(c.id).flat(), ...points(c.id), ...c.moments.map((m) => m[0])].sort((a, b) => a - b);
}
function jump(d) {
  const t = now(), p = marks(), x = d > 0 ? p.find((v) => v > t + 0.03) : p.reverse().find((v) => v < t - 0.03);
  if (x !== undefined) seek(x);
}
// star the selection at the playhead (else the one in hand)
function starHere(n) {
  const id = C().id, ss = sels(id), t = now(), hit = ([a, b]) => t >= a - 0.05 && t <= b + 0.05;
  let j = act >= 0 && ss[act] && hit(ss[act]) ? act : ss.findIndex(hit);
  if (j < 0) j = act;
  if (j < 0 || !ss[j]) return toast("Put the playhead on a selection to star it", "warn");
  setStars(id, ss[j][0], ss[j][1], starsOf(id, ss[j][0], ss[j][1]) == n ? 0 : n);
}
function loopHere() {
  const ss = sels(C().id);
  if (act < 0) act = ss.findIndex(([a, b]) => now() >= a && now() <= b);
  if (act < 0) return toast("Put the playhead on a selection to loop it", "warn");
  loop = !loop;
  seek(ss[act][0]);
  vid.play();
}
