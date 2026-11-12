const TW = D.tile[0],
  TH = D.tile[1],
  FR = 1 / 50;
const $ = (id) => document.getElementById(id);
const fmt = (x) => {
  x = Math.max(0, x);
  const m = Math.floor(x / 60);
  return m + ":" + (x - 60 * m).toFixed(2).padStart(5, "0");
};
const short = (x) => {
  x = Math.max(0, x);
  const m = Math.floor(x / 60);
  return m + ":" + (x - 60 * m).toFixed(1).padStart(4, "0");
};
const esc = (x) =>
  String(x).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const secs = (v) => {
  if (typeof v === "number") return v;
  const p = String(v).trim().split(":");
  return p.reduce((a, x) => a * 60 + +x, 0);
};
const span = (v) => {
  if (Array.isArray(v)) return v.map(secs);
  const s = String(v),
    i = s.indexOf("-", 1);
  return i > 0 ? [secs(s.slice(0, i)), secs(s.slice(i + 1))] : [secs(s), secs(s)];
};
const isRange = (v) => {
  const [a, b] = span(v);
  return b - a > 0.05;
};
// --- picks: saved to picks.json by `focalcut reel serve`, else kept in this browser ---
const LS = "sources:" + D.key;
const ls = {
  get(k) {
    try {
      return JSON.parse(localStorage.getItem(LS + k));
    } catch (e) {
      return null;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem(LS + k, JSON.stringify(v));
    } catch (e) {}
  },
};
let P = JSON.parse(JSON.stringify(D.picks)),
  server = false,
  timer = null;
const ent = (id) => (P[id] && typeof P[id] === "object" && !Array.isArray(P[id]) ? P[id] : (P[id] = {}));

function tidy() {
  for (const k in P) {
    const e = P[k];
    if (k[0] == "_" || !e || typeof e !== "object" || Array.isArray(e)) continue;
    pruneTags(e);
    if (e.stars) {
      const ss = (e.add || []).map(span);
      for (const q in e.stars) {
        const [u, v] = span(q);
        if (!ss.some(([a, b]) => near2(a, u) && near2(b, v))) delete e.stars[q];
      }
      if (!Object.keys(e.stars).length) delete e.stars;
    }
    for (const f of ["ban", "prefer", "add", "portrait", "follow"]) if (e[f] && !e[f].length) delete e[f];
    if (e.more === 0) delete e.more;
    if (!Object.keys(e).length) delete P[k];
  }
}
function status(t, c) {
  const s = $("saved");
  s.textContent = t;
  s.className = c || "";
}
const hist = [];
let lastSnap = JSON.stringify(P);
function save() {
  tidy();
  {
    const now_ = JSON.stringify(P);
    if (now_ !== lastSnap) {
      hist.push(lastSnap);
      if (hist.length > 200) hist.shift();
      lastSnap = now_;
    }
  }
  if (server) {
    status("saving…");
    clearTimeout(timer);
    timer = setTimeout(push, 250);
  } else {
    ls.set("picks", P);
    status("kept in this browser — run `focalcut reel serve` to save to picks.json", "err");
  }
  renderAll();
}
async function push() {
  try {
    const r = await fetch(D.root + "api/picks", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(P, null, 1),
    });
    if (!r.ok) throw new Error(r.status);
    status("saved to picks.json ✓", "ok");
    if (staleCrop) {
      const id = staleCrop;
      staleCrop = null;
      for (const k of [...crops.keys()]) if (k.startsWith(id + "|")) crops.delete(k);
      lastCrop.delete(id);
      draw();
      pdraw();
    }
  } catch (e) {
    status("could not save: " + e.message, "err");
  }
}
async function boot() {
  try {
    const r = await fetch(D.root + "api/picks", { cache: "no-store" });
    if (!r.ok) throw 0;
    P = await r.json();
    server = true;
    lastSnap = JSON.stringify(P);
    ($("allp") || {}).hidden = false;
    status("picks.json — saves as you go", "ok");
  } catch (e) {
    const l = ls.get("picks");
    if (l) P = l;
    status("opened as a file: changes stay in this browser", "err");
  }
  const last = ls.get("clip");
  go(
    Math.max(
      0,
      D.clips.findIndex((c) => c.id == last),
    ),
    true,
  );
  framing();
}
// sources.html#frame=<reel>: straight to framing that reel's portrait cut -- Cut & picks, the portrait cut's shots in
// order, the crop over the picture: drag to pan it, G then a click to follow a dancer, ←/→ from shot to shot
function framing() {
  const m = /frame=([\w-]+)/.exec(location.hash);
  if (!m) return;
  if (RID.includes(m[1]) && reel !== m[1]) {
    reel = m[1];
    ls.set("reel", reel);
    useReel();
    place.clear();
  }
  pfmt = "P";
  ls.set("pfmt", "P");
  if (!frameOn) toggleFrame();
  view(2);
  pick(0, false);
  status(
    `framing the ${reel == "all" ? "" : rlab(reel) + " "}portrait cut — drag the picture to pan · G then click a dancer to follow · ←/→ shot to shot · Clear focus to undo`,
    "ok",
  );
}
addEventListener("hashchange", framing);
// selections of a clip: [[a,b],...] sorted; written back as "m:ss.ss-m:ss.ss" (single times stay as they were)
function sels(id) {
  return ((P[id] || {}).add || [])
    .filter(isRange)
    .map(span)
    .sort((x, y) => x[0] - y[0]);
}
function points(id) {
  return ((P[id] || {}).add || []).filter((v) => !isRange(v)).map((v) => span(v)[0]);
}
function setSels(id, list, quiet) {
  const e = ent(id);
  const pts = (e.add || []).filter((v) => !isRange(v));
  list = list.map(([a, b]) => [Math.max(0, Math.min(a, b)), Math.max(a, b)]).filter(([a, b]) => b - a >= 0.1);
  // each keeps its reels (the old selection it overlaps most: moved or trimmed); a new one is for the reel being picked
  // for; one taken away while picking for a reel leaves that reel only, and stays in the others
  const old = sels(id).map(([a, b]) => [a, b, onlyOf(id, a, b), starsOf(id, a, b)]);
  const items = list.map(([a, b]) => {
    let best = null,
      ov = 0;
    old.forEach(([u, v, o]) => {
      const x = Math.min(b, v) - Math.max(a, u);
      if (x > ov) {
        ov = x;
        best = o;
      }
    });
    let st = 0,
      ov2 = 0;
    old.forEach(([u, v, , n]) => {
      const x = Math.min(b, v) - Math.max(a, u);
      if (x > ov2) {
        ov2 = x;
        st = n;
      }
    });
    return { a, b, t: ov > 0 ? best : reel == "all" ? null : [reel], st };
  });
  if (reel != "all")
    old.forEach(([u, v, o]) => {
      if (list.some(([a, b]) => Math.min(b, v) - Math.max(a, u) > 0)) return;
      const t = (o || RID).filter((r) => r != reel);
      if (t.length && RID.length > 1) items.push({ a: u, b: v, t, st: starsOf(id, u, v) });
    });
  items.sort((x, y) => x.a - y.a);
  if (e.only) for (const k in e.only) if (tagParts(k)[0] == "add" && isRange(k)) delete e.only[k];
  e.add = [...pts, ...items.map((x) => fmt(x.a) + "-" + fmt(x.b))];
  items.forEach((x) => setTags(id, "add", fmt(x.a) + "-" + fmt(x.b), x.t));
  const so = {};
  items.forEach((x) => {
    if (x.st) so[fmt(x.a) + "-" + fmt(x.b)] = x.st;
  });
  if (Object.keys(so).length) e.stars = so;
  else delete e.stars;
  if (!quiet) save();
}
function banned(id) {
  return vals(id, "ban", reel);
}
// reels: one picks.json serves the project's reel ("main") and its cuts (reel.json "cuts"); a clip's "only" maps a
// selection to the reels it is for, and one not listed is for every reel
const rlab = (r) => (RS.find((x) => x.id == r) || { label: r }).label;
let reel = ls.get("reel") || "all";
if (!RID.includes(reel)) reel = "all";
const rkey = (a, b) => fmt(a) + "-" + fmt(b);
const onlyOf = (id, a, b) => tagsOf(id, "add", rkey(a, b));
const inReel = (id, a, b, r = reel) => isIn(id, "add", rkey(a, b), r);
function setIn(id, a, b, r, on) {
  let v = onlyOf(id, a, b) || RID.slice();
  v = on ? [...new Set([...v, r])] : v.filter((x) => x != r);
  if (!v.length) {
    status("a pick is in one reel at least — ✕ removes it", "err");
    return renderAll();
  }
  // (its opening / ending role follows it out of a reel)
  if (!on)
    for (const f of ["open", "end"]) {
      const w = roleIn(id, f, r);
      if (w && same(w, a, b)) clearRole(id, f, r);
    }
  setTags(id, "add", rkey(a, b), v);
  save();
}
// stars: how much the editor wants a pick (1-5; a clip's "stars": {range: n}, every reel): the planner keeps and
// lengthens the most starred first and drops the least starred first (unrated counts as 3)
function starsOf(id, a, b) {
  const o = (P[id] || {}).stars;
  if (!o) return 0;
  for (const k in o) {
    const [u, v] = span(k);
    if (near2(u, a) && near2(v, b)) return o[k];
  }
  return 0;
}
function setStars(id, a, b, n) {
  const e = ent(id),
    o = {};
  for (const k in e.stars || {}) {
    const [u, v] = span(k);
    if (!(near2(u, a) && near2(v, b))) o[k] = e.stars[k];
  }
  if (n) o[rkey(a, b)] = n;
  if (Object.keys(o).length) e.stars = o;
  else delete e.stars;
  save();
  status(n ? `${"★".repeat(n)} ${fmt(a)}–${fmt(b)}` : "stars cleared", "ok");
}
const starBar = (id, a, b, attr) => {
  const n = starsOf(id, a, b);
  return (
    `<span class=stars title="how much you want it (1–5; click again to clear)">` +
    [1, 2, 3, 4, 5]
      .map(
        (i) =>
          `<b data-star=${i} ${attr} title="${i} star${i > 1 ? "s" : ""} — key ${i}">${i <= n ? "★" : "☆"}<sub>${i}</sub></b>`,
      )
      .join("") +
    `</span>`
  );
};
function useReel() {
  const r = reel == "all" ? "main" : reel;
  D.clips.forEach((c) => {
    const u = (c.R || {})[r];
    if (u) {
      c.L = u.L;
      c.P = u.P;
    }
  });
  $("reelsw").innerHTML =
    RS.length > 1
      ? "<b>Reel</b>" +
        [["all", "All"], ...RS.map((x) => [x.id, x.label])]
          .map(([k, l]) => `<button data-reel="${esc(k)}" class="${k == reel ? "on" : ""}">${esc(l)}</button>`)
          .join("")
      : "";
}
$("reelsw").onclick = (ev) => {
  const b = ev.target.closest("[data-reel]");
  if (!b) return;
  reel = b.dataset.reel;
  ls.set("reel", reel);
  useReel();
  place.clear();
  status(
    reel == "all"
      ? "all reels: picks and bans go in every reel"
      : `${rlab(reel)}: picks, bans, opening and ending are for this reel only`,
    "ok",
  );
  renderAll();
};
useReel();
// one pick opens the reel and one ends it: the clip's "open" / "end" is that pick's in-out (it stays a selection too)
const same = (v, a, b) => {
  if (!v) return false;
  const [u, w] = span(v);
  return Math.abs(u - a) < 0.01 && Math.abs(w - b) < 0.01;
};
function roleOf(id, a, b) {
  return same(roleIn(id, "open", reel), a, b) ? "open" : same(roleIn(id, "end", reel), a, b) ? "end" : null;
}
// (per reel: picking for one reel sets or clears that reel's; a clip holds one opening and one ending)
function setRole(x, f) {
  const on = same(roleIn(x.c.id, f, reel), x.a, x.b);
  for (const k in P) {
    const e = P[k];
    if (k[0] != "_" && e && typeof e == "object" && !Array.isArray(e) && roleIn(k, f, reel)) clearRole(k, f, reel);
  }
  if (!on) {
    const e = ent(x.c.id);
    if (typeof e[f] == "string") {
      setTags(x.c.id, f, e[f], null);
      delete e[f];
    }
    e[f] = fmt(x.a) + "-" + fmt(x.b);
    setTags(x.c.id, f, e[f], reel == "all" ? null : [reel]);
    if (x.k < 0) {
      const ss = sels(x.c.id);
      ss.push([x.a, x.b]);
      return setSels(x.c.id, ss);
    }
  }
  save();
  status(
    on
      ? `no ${f == "open" ? "opening" : "ending"} pick: the planner chooses`
      : `${x.c.id} ${fmt(x.a)} ${f == "open" ? "opens" : "ends"} the reel`,
    "ok",
  );
}
// --- clip list ---
let cur = 0,
  cat = "all",
  onlySel = false,
  rub = null;
const seen = new Set(ls.get("seen") || []);
const C = () => D.clips[cur];
const cats = ["all", ...new Set(D.clips.map((c) => c.cat))];
$("filters").innerHTML =
  cats.map((c) => `<button data-c="${esc(c)}" class="${c == "all" ? "on" : ""}">${esc(c)}</button>`).join("") +
  `<button id=onlysel title="only clips with selections">● selected</button>`;
$("filters").onclick = (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  if (b.id == "onlysel") {
    onlySel = !onlySel;
    b.classList.toggle("on", onlySel);
  } else {
    cat = b.dataset.c;
    $("filters")
      .querySelectorAll("[data-c]")
      .forEach((x) => x.classList.toggle("on", x === b));
  }
  list();
};
const visible = () =>
  D.clips
    .map((c, i) => i)
    .filter((i) => (cat == "all" || D.clips[i].cat == cat) && (!onlySel || sels(D.clips[i].id).length));
// a clip's frame at t as a background, h tall; in a box W wide, a narrower (vertical) frame sits in the middle, its
// neighbours in the strip clipped off
const tileOf = (c) => c.tile || [TW, TH];
const thumbAt = (c, t, h, W) => {
  const [tw, th] = tileOf(c),
    k = Math.min(c.n - 1, Math.max(0, Math.floor(t / c.step))),
    s = h / th,
    pad = W ? Math.max(0, (W - tw * s) / 2) : 0;
  return (
    `background-image:url('${c.strip}');background-size:${c.n * tw * s}px ${h}px;background-position:-${k * tw * s}px 0` +
    (pad
      ? `;box-sizing:border-box;padding-left:${pad}px;padding-right:${pad}px;background-origin:content-box;background-clip:content-box`
      : "")
  );
};
const mine = (id) => sels(id).filter(([a, b]) => inReel(id, a, b));
function list() {
  $("list").innerHTML = visible()
    .map((i) => {
      const c = D.clips[i],
        n = mine(c.id).length,
        tot = sels(c.id).reduce((a, [x, y]) => a + y - x, 0),
        bn = banned(c.id).includes("all");
      return `<div class="ci${i == cur ? " cur" : ""}${seen.has(c.id) ? " seen" : ""}" data-i=${i} style="${bn ? "opacity:.4" : ""}">
 <div class=th style="${thumbAt(c, c.dur * 0.3, 54, 96)}">${n ? `<b>${n}</b>` : ""}</div><div><div class=nm>${esc(c.id)}</div>
 <div class=meta><span class=chip>${esc(c.cat)}</span> ${esc(c.clock)} · ${short(c.dur)}${n ? ` · <span style="color:var(--sel)">● ${short(tot)}</span>` : ""}</div>
 <div class=meta>${c.L.length ? `<span style="color:var(--land)">L${c.L.length}</span> ` : ""}${c.P.length ? `<span style="color:var(--port)">P${c.P.length}</span>` : ""}${bn ? " banned" : ""}</div></div></div>`;
    })
    .join("");
  const el = $("list").querySelector(".cur");
  if (el) el.scrollIntoView({ block: "nearest" });
}
$("list").onclick = (ev) => {
  const d = ev.target.closest(".ci");
  if (d) go(+d.dataset.i);
};
// --- player + timeline ---
const vid = $("vid"),
  tl = $("tl"),
  ov = $("ov"),
  hov = $("hover");
let va = 0,
  vb = 1,
  inPt = null,
  outPt = null,
  act = -1,
  loop = false,
  shuttle = 0,
  rew = null,
  drag = null,
  img = new Image();
// scrubbing: one seek in flight per video, the next goes to wherever the pointer is by then (a new currentTime on every
// mouse move restarts the seek, so the picture would never catch up until the mouse stops)
function goTo(v, t) {
  if (v.seeking) {
    v._want = t;
    return;
  }
  v._want = null;
  v.currentTime = t;
}
for (const v of [$("vid"), $("pv")])
  v.addEventListener("seeked", () => {
    if (v._want != null) {
      const t = v._want;
      v._want = null;
      v.currentTime = t;
    }
  });
const now = () => vid._want ?? (vid.currentTime || 0),
  dur = () => C().dur || vid.duration || 1;
function go(i, first) {
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
  vid.src = c.video;
  va = 0;
  vb = c.dur;
  img = new Image();
  img.onload = paint;
  img.src = c.strip;
  $("cname").textContent = c.id;
  $("cmeta").innerHTML = `<span class=chip>${esc(c.cat)}</span> ${esc(c.tag)} · ${esc(c.clock)}
  · ${short(c.dur)} · clip ${i + 1} of ${D.clips.length}`;
  vid.addEventListener(
    "loadedmetadata",
    () => {
      vid.currentTime = 0;
      renderAll();
    },
    { once: true },
  );
  renderAll();
}
function seek(t) {
  if (rew) stopShuttle();
  goTo(vid, Math.min(Math.max(0, t), dur() - 0.02));
  follow();
  draw();
}
function follow() {
  const t = now(),
    w = vb - va;
  if (t < va || t > vb) {
    va = Math.max(0, Math.min(t - w * 0.1, dur() - w));
    vb = va + w;
    paint();
  }
}
const X = (t, a = va, b = vb) => ((100 * (t - a)) / (b - a)).toFixed(3) + "%";
function wspan(a, b, A = va, B = vb) {
  const x0 = Math.max(a, A),
    x1 = Math.min(b, B);
  return `left:${X(x0, A, B)};width:${((100 * (x1 - x0)) / (B - A)).toFixed(3)}%`;
}
function film(canvas, A, B, h, c = C(), im = img) {
  const r = canvas.getBoundingClientRect(),
    dpr = devicePixelRatio || 1,
    img = im;
  canvas.width = Math.max(1, r.width * dpr);
  canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.fillStyle = "#000";
  g.fillRect(0, 0, canvas.width, canvas.height);
  if (!img.complete || !img.naturalWidth) return;
  const [TW_, TH_] = tileOf(c),
    tw = ((h * TW_) / TH_) * dpr,
    W = canvas.width;
  for (let x = 0; x < W; x += tw) {
    const t = A + ((x + tw / 2) / W) * (B - A);
    if (t > c.dur) break;
    const k = Math.min(c.n - 1, Math.max(0, Math.floor(t / c.step)));
    g.drawImage(img, k * TW_, 0, TW_, TH_, x, 0, tw, h * dpr);
  }
}
function paint() {
  film($("strip").querySelector("canvas"), va, vb, 80);
  film(ov.querySelector("canvas"), 0, dur(), 30);
  draw();
}
function draw() {
  const c = C();
  if (!c) return;
  const W = vb - va,
    t = now();
  // ruler
  const step = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60].find((s) => W / s <= 60) || 120,
    lab = [0.5, 1, 2, 5, 10, 15, 30, 60, 120].find((s) => W / s <= 10) || 300;
  let h = "";
  for (let x = Math.ceil(va / step) * step; x <= vb; x += step) {
    const big = Math.abs(x / lab - Math.round(x / lab)) < 1e-6;
    h += `<i class="tk${big ? " big" : ""}" style="left:${X(x)}"></i>`;
    if (big) h += `<i class=lb style="left:${X(x)}">${W < 15 ? fmt(x) : short(x)}</i>`;
  }
  $("ruler").innerHTML = h;
  // strip overlays: moments, bans, in/out, playhead
  const S = $("strip");
  S.querySelectorAll("i").forEach((e) => e.remove());
  let s = "";
  c.moments.forEach(([m, k]) => {
    if (m >= va && m <= vb) s += `<i class=mom style="left:${X(m)}" title="${esc(k)} ${short(m)}"></i>`;
  });
  banned(c.id).forEach((v) => {
    const [a, b] = v == "all" ? [0, c.dur] : span(v);
    if (b > va && a < vb) s += `<i class=ban style="${wspan(a, b)}" title="banned ${esc(v)}"></i>`;
  });
  S.insertAdjacentHTML("beforeend", s);
  // selections lane
  const L = $("lsel");
  L.querySelectorAll("i").forEach((e) => e.remove());
  s = "";
  sels(c.id).forEach(([a, b], k) => {
    const o = onlyOf(c.id, a, b);
    if (b > va && a < vb)
      s += `<i class="blk${k == act ? " act" : ""}${inReel(c.id, a, b) ? "" : " away"}" data-k=${k} style="${wspan(a, b)}"
  title="${fmt(a)} – ${fmt(b)} (${(b - a).toFixed(2)} s)${RS.length > 1 ? " · " + (o ? o.map(rlab).join(" + ") + " only" : "every reel") : ""}"><i class="h l"></i>${k + 1} · ${(b - a).toFixed(1)}s${starsOf(c.id, a, b) ? " " + "★".repeat(starsOf(c.id, a, b)) : ""}<i class="h r"></i></i>`;
  });
  points(c.id).forEach((p) => {
    if (p >= va && p <= vb) s += `<i class=pt style="left:${X(p)}" title="marked moment ${fmt(p)}"></i>`;
  });
  const io = pend();
  if (io) s += `<i class=pend style="${wspan(...io)}"></i>`;
  if (!sels(c.id).length && !io)
    s += `<i class=empty>drag here to pick a range · or I / O then Enter · or click a reel shot below to take it</i>`;
  L.insertAdjacentHTML("beforeend", s);
  for (const [id, arr] of [
    ["lland", c.L],
    ["lport", c.P],
  ]) {
    const e = $(id);
    e.querySelectorAll("i").forEach((x) => x.remove());
    const bans = banned(c.id).map((v) => (v == "all" ? [-1, 1e9] : span(v))),
      out = (a, b) => bans.some(([u, v]) => (a + b) / 2 >= u && (a + b) / 2 <= v);
    e.insertAdjacentHTML(
      "beforeend",
      arr
        .filter(([a, b]) => b > va && a < vb)
        .map(
          ([a, b, n]) =>
            `<i class="use${out(a, b) ? " out" : ""}" data-a=${a} data-b=${b} style="${wspan(a, b)}" title="reel shot ${n}: ${short(a)}–${short(b)} — click to pick it, ✕ to take it out; drag along the lanes to take out every shot in a range">${n}<b class=ux title="take this shot out of the cut (bans its stretch)">✕</b></i>`,
        )
        .join(""),
    );
  }
  if (rub && rub.moved) {
    const [a, b] = [Math.min(rub.t0, rub.t1), Math.max(rub.t0, rub.t1)];
    for (const id of ["lland", "lport"])
      $(id).insertAdjacentHTML("beforeend", `<i class=rub style="${wspan(a, b)}"></i>`);
  }
  fkeys(c.id).forEach(([k]) => {
    if (k >= va && k <= vb)
      $("lport").insertAdjacentHTML("beforeend", `<i class=fk style="left:${X(k)}" title="focal point ${fmt(k)}"></i>`);
  });
  folkeys(c.id).forEach(([k]) => {
    if (k >= va && k <= vb)
      $("lport").insertAdjacentHTML(
        "beforeend",
        `<i class="fk fl" style="left:${X(k)}" title="following a person from ${fmt(k)}"></i>`,
      );
  });
  // markers spanning the timeline
  tl.querySelectorAll(":scope>i").forEach((e) => e.remove());
  let m = "";
  if (inPt !== null && inPt >= va && inPt <= vb) m += `<i class=inm style="left:${X(inPt)}"></i>`;
  if (outPt !== null && outPt >= va && outPt <= vb) m += `<i class=outm style="left:${X(outPt)}"></i>`;
  if (t >= va && t <= vb) m += `<i class=ph style="left:${X(t)}"></i>`;
  tl.insertAdjacentHTML("beforeend", m);
  // overview
  ov.querySelectorAll("i").forEach((e) => e.remove());
  let o = "";
  const D0 = dur();
  sels(c.id).forEach(
    ([a, b]) => (o += `<i class=blk style="${wspan(a, b, 0, D0)};top:4px;bottom:4px;pointer-events:none"></i>`),
  );
  o += `<i class=win style="${wspan(va, vb, 0, D0)}"></i><i class=ph style="left:${X(t, 0, D0)}"></i>`;
  ov.insertAdjacentHTML("beforeend", o);
  $("tc").textContent = fmt(t) + "  /  " + short(D0);
  const st = $("still"),
    noVid = vid.error || (vid.readyState < 2 && !vid.seeking) || !vid.videoWidth;
  st.classList.toggle("on", !!noVid);
  if (noVid) {
    const r = $("vwrap").getBoundingClientRect(),
      [tw, th] = tileOf(c),
      h = Math.min(r.height, (r.width * th) / tw);
    st.setAttribute(
      "style",
      thumbAt(c, t, h) +
        `;width:${(h * tw) / th}px;height:${h}px;left:${(r.width - (h * tw) / th) / 2}px;top:${(r.height - h) / 2}px`,
    );
  }
  $("io").textContent =
    inPt === null && outPt === null
      ? ""
      : "in " +
        (inPt === null ? "—" : fmt(inPt)) +
        "  out " +
        (outPt === null ? "—" : fmt(outPt)) +
        (io ? "  (" + (io[1] - io[0]).toFixed(2) + " s)" : "");
  $("play").textContent = vid.paused && !rew ? "▶ Play" : "❚❚ Pause";
  $("banc").textContent = banned(c.id).includes("all") ? "Unban clip" : "Ban clip";
  frameDraw($("fr"), $("vwrap"), vid, c, ...rangeFor(c, t), t);
}
function pend() {
  if (inPt === null && outPt === null) return null;
  const a = inPt ?? now(),
    b = outPt ?? now();
  return Math.abs(b - a) < 0.05 ? null : [Math.min(a, b), Math.max(a, b)];
}
function mysel() {
  const c = C(),
    ss = sels(c.id);
  $("mysel").innerHTML = ss.length
    ? `<table><tr><th>#</th><th>in</th><th>out</th><th>length</th><th></th></tr>` +
      ss
        .map(
          ([a, b], k) =>
            `<tr class="${k == act ? "act" : ""}"><td>${k + 1}</td><td>${fmt(a)}</td><td>${fmt(b)}</td><td>${(b - a).toFixed(2)} s</td>
  <td>${starBar(c.id, a, b, `data-k=${k}`)}${RS.length > 1 ? RS.map((r) => `<label class=rk><input type=checkbox data-r="${esc(r.id)}" data-k=${k}${inReel(c.id, a, b, r.id) ? " checked" : ""}> ${esc(r.label)}</label>`).join("") : ""}<button data-p=${k}>▶ play</button> <button data-l=${k}>↻ loop</button> <button data-d=${k}>✕</button></td></tr>`,
        )
        .join("") +
      `</table>`
    : `<p style="color:var(--mut);margin:0">No selections in this clip yet: <kbd>I</kbd> at the start, <kbd>O</kbd>
  at the end, <kbd>Enter</kbd> — or drag in the selections lane.</p>`;
}
$("mysel").onchange = (ev) => {
  const i = ev.target;
  if (!i.dataset.r) return;
  const ss = sels(C().id),
    x = ss[+i.dataset.k];
  if (x) setIn(C().id, x[0], x[1], i.dataset.r, i.checked);
};
$("mysel").onclick = (ev) => {
  const st = ev.target.closest("[data-star]");
  if (st) {
    const ss = sels(C().id),
      x = ss[+st.dataset.k];
    if (x) {
      const n = +st.dataset.star;
      setStars(C().id, x[0], x[1], starsOf(C().id, x[0], x[1]) == n ? 0 : n);
    }
    return;
  }
  const b = ev.target.closest("button");
  if (!b) return;
  const ss = sels(C().id);
  if (b.dataset.d !== undefined) {
    ss.splice(+b.dataset.d, 1);
    act = -1;
    setSels(C().id, ss);
  } else {
    const k = +(b.dataset.p ?? b.dataset.l);
    act = k;
    loop = b.dataset.l !== undefined;
    seek(ss[k][0]);
    vid.play();
    renderAll();
  }
};
function renderAll() {
  list();
  draw();
  mysel();
  selview();
  counts();
}
// what's in play: the planner's shots per reel, and the editor's own calls
function counts() {
  const n = (f) => D.clips.reduce((a, c) => a + f(c), 0),
    L = n((c) => c.L.length),
    Pn = n((c) => c.P.length),
    my = n((c) => mine(c.id).length + points(c.id).length),
    fk = n((c) => fkeys(c.id).length),
    bc = n((c) => (banned(c.id).includes("all") ? 1 : 0)),
    br = n((c) => banned(c.id).filter((v) => v != "all").length),
    more = n((c) => (moreIn(c.id, reel) ? 1 : 0)),
    unseen = D.clips.length - D.clips.filter((c) => seen.has(c.id)).length;
  const it = (v, t, cls = "") => `<span class="${cls}" title="${t}"><b>${v}</b> ${t.split(":")[0]}</span>`;
  $("counts").innerHTML = [
    it(L, "landscape shots: in the current landscape cut"),
    it(Pn, "portrait shots: in the current portrait cut"),
    it(
      my,
      "your picks" +
        (reel == "all" ? "" : " for " + rlab(reel)) +
        ": selections and marked moments the next re-plan must use",
      "y",
    ),
    fk ? it(fk, "focus points: where the portrait crop goes", "f") : "",
    bc + br ? it(bc + br, "bans: " + bc + " whole clips, " + br + " stretches", "x") : "",
    more ? it(more, "clips asked for more") : "",
    it(unseen, "clips not looked at yet"),
  ]
    .filter(Boolean)
    .join(" · ");
}
// timeline mouse: scrub on ruler/strip/overview; make, move and trim selections in the selections lane
function tAt(el, ev, A = va, B = vb) {
  const r = el.getBoundingClientRect();
  return A + Math.min(Math.max((ev.clientX - r.left) / r.width, 0), 1) * (B - A);
}
function showHover(ev, t, el) {
  const c = C();
  hov.style.display = "block";
  hov.querySelector("div").setAttribute("style", thumbAt(c, t, 108, 192));
  hov.querySelector("span").textContent = fmt(t);
  const r = el.getBoundingClientRect();
  hov.style.left = Math.min(innerWidth - 205, Math.max(4, ev.clientX - 100)) + "px";
  hov.style.top = r.top - 136 + "px";
}
for (const el of [$("ruler"), $("strip")]) {
  el.addEventListener("pointerdown", (ev) => {
    pause();
    el.setPointerCapture(ev.pointerId);
    drag = { k: "scrub", el };
    seek(tAt(el, ev));
  });
  el.addEventListener("pointermove", (ev) => {
    const t = tAt(el, ev);
    showHover(ev, t, tl);
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
  showHover(ev, t, ov);
  if (drag && drag.k == "ov") movewin(t);
});
function movewin(t) {
  const w = vb - va;
  va = Math.max(0, Math.min(t - w / 2, dur() - w));
  vb = va + w;
  if (now() < va || now() > vb) seek(t);
  paint();
}
// drags in the lanes snap to the playhead (where it was when the drag began) within 10 px
const near = (el, t, p) => p != null && (Math.abs(t - p) * el.getBoundingClientRect().width) / (vb - va) < 10;
const snapT = (el, ev, p = now()) => {
  const t = tAt(el, ev);
  return near(el, t, p) ? p : t;
};
const L = $("lsel");
// a selection's edge within 8 px of the pointer, inside or just outside the block (the nearer one wins)
function edgeAt(ev) {
  const r = L.getBoundingClientRect(),
    px = (r.right - r.left) / (vb - va),
    x = ev.clientX - r.left;
  let best = null,
    bd = 8;
  sels(C().id).forEach(([a, b], k) => {
    for (const [t, side] of [
      [a, "trimL"],
      [b, "trimR"],
    ]) {
      const d = Math.abs((t - va) * px - x);
      if (d <= bd) {
        bd = d;
        best = { k, side };
      }
    }
  });
  return best;
}
L.addEventListener("pointerdown", (ev) => {
  pause();
  L.setPointerCapture(ev.pointerId);
  const ph = now(),
    t = snapT(L, ev, ph),
    b = ev.target.closest(".blk"),
    e = edgeAt(ev);
  if (e || b) {
    const k = e ? e.k : +b.dataset.k,
      ss = sels(C().id);
    act = k;
    drag = { k: e ? e.side : "move", i: k, t0: tAt(L, ev), orig: ss[k].slice(), ss, moved: false, ph };
  } else drag = { k: "new", a: t, b: t, ph };
  draw();
});
L.addEventListener("pointermove", (ev) => {
  showHover(ev, tAt(L, ev), tl);
  L.style.cursor = drag ? (drag.k.startsWith("trim") ? "ew-resize" : "") : edgeAt(ev) ? "ew-resize" : "";
  if (!drag || drag.el) return;
  const t = snapT(L, ev, drag.ph);
  if (drag.k == "new") {
    drag.b = t;
    inPt = Math.min(drag.a, t);
    outPt = Math.max(drag.a, t);
    seek(t);
  } else {
    const o = drag.orig,
      d = t - drag.t0;
    if (Math.abs(d) > 0.03) drag.moved = true;
    let [a, b] = o;
    if (drag.k == "move") {
      const w = b - a;
      a = Math.max(0, Math.min(o[0] + tAt(L, ev) - drag.t0, dur() - w));
      if (near(L, a, drag.ph)) a = drag.ph;
      else if (near(L, a + w, drag.ph)) a = drag.ph - w;
      b = a + w;
    } else if (drag.k == "trimL") {
      a = Math.max(0, Math.min(t, b - 0.1));
      seek(a);
    } else {
      b = Math.min(dur(), Math.max(t, a + 0.1));
      seek(b);
    }
    drag.ss[drag.i] = [a, b];
    const blk = L.querySelector(`.blk[data-k="${drag.i}"]`);
    if (blk) {
      blk.setAttribute("style", wspan(a, b));
      blk.childNodes[1].textContent = `${drag.i + 1} · ${(b - a).toFixed(1)}s`;
    }
  }
});
L.addEventListener("pointerup", (ev) => {
  if (!drag || drag.el) return;
  const c = C();
  if (drag.k == "new") {
    if (Math.abs(drag.b - drag.a) >= 0.2) {
      const ss = sels(c.id);
      ss.push([Math.min(drag.a, drag.b), Math.max(drag.a, drag.b)]);
      inPt = outPt = null;
      drag = null;
      setSels(c.id, ss);
      act = sels(c.id).findIndex(([a]) => Math.abs(a - Math.min(...ss[ss.length - 1])) < 0.01);
      renderAll();
    } else {
      inPt = outPt = null;
      drag = null;
      seek(tAt(L, ev));
    }
    return;
  }
  const d = drag;
  drag = null;
  if (d.moved) setSels(c.id, d.ss);
  else seek(d.orig[0]);
  mysel();
});
addEventListener("pointerup", () => {
  if (drag && drag.el) {
    drag = null;
    draw();
  }
});
// the reel lanes: ✕ on a shot takes it out of the cut; drag along either lane to take out every landscape and portrait
// shot in the range (bans their stretches; one undo step; the ends snap to the playhead); a plain click makes it a pick
function banShots(c, list) {
  let n = 0;
  list.forEach(([a, b]) => {
    if (addVal(c.id, "ban", fmt(a) + "-" + fmt(b), reel)) n++;
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
    showHover(ev, tAt(el, ev), tl);
    if (!rub) return;
    rub.t1 = snapT(el, ev);
    if (Math.abs(ev.clientX - rub.x) > 4) rub.moved = true;
    if (rub.moved) draw();
  });
  el.addEventListener("pointerup", () => {
    const r = rub;
    if (!r) return;
    if (!r.moved) {
      rub = null;
      return;
    }
    rub = { moved: false, done: true };
    const c = C(),
      [a, b] = [Math.min(r.t0, r.t1), Math.max(r.t0, r.t1)];
    const hit = [...c.L, ...c.P].filter(([x, y]) => (x + y) / 2 >= a && (x + y) / 2 <= b);
    const n = banShots(c, hit);
    status(
      n ? `took ${n} shot${n == 1 ? "" : "s"} out of the cut — Ctrl+Z undoes` : "no reel shots in that range",
      n ? "ok" : "",
    );
    draw();
  });
}
for (const id of ["lland", "lport"])
  $(id).addEventListener("click", (ev) => {
    if (rub && rub.done) {
      rub = null;
      return;
    }
    const x = ev.target.closest(".ux");
    if (x) {
      const u = x.closest(".use"),
        n = banShots(C(), [[+u.dataset.a, +u.dataset.b]]);
      status(n ? "took the shot out of the cut — Ctrl+Z undoes" : "already out", "ok");
      return;
    }
    const u = ev.target.closest(".use");
    if (!u) return;
    const a = +u.dataset.a,
      b = +u.dataset.b,
      c = C(),
      ss = sels(c.id);
    if (!ss.some(([x, y]) => Math.abs(x - a) < 0.05 && Math.abs(y - b) < 0.05)) ss.push([a, b]);
    setSels(c.id, ss);
    act = sels(c.id).findIndex(([x]) => Math.abs(x - a) < 0.01);
    seek(a);
    renderAll();
  });
vid.addEventListener("error", draw);
vid.addEventListener("loadeddata", draw);
vid.addEventListener("seeking", draw);
for (const el of [tl, ov]) el.addEventListener("pointerleave", () => (hov.style.display = "none"));
tl.addEventListener(
  "wheel",
  (ev) => {
    ev.preventDefault();
    const r = tl.getBoundingClientRect(),
      f = (ev.clientX - r.left) / r.width;
    if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) {
      const w = vb - va,
        d = ((ev.deltaX || ev.deltaY) / 600) * w;
      va = Math.max(0, Math.min(va + d, dur() - w));
      vb = va + w;
    } else zoom(ev.deltaY > 0 ? 1.25 : 0.8, va + f * (vb - va));
    paint();
  },
  { passive: false },
);
function zoom(k, at) {
  const D0 = dur(),
    w = Math.min(Math.max((vb - va) * k, 0.5), D0);
  at = at ?? now();
  const f = (at - va) / (vb - va || 1);
  va = Math.max(0, Math.min(at - f * w, D0 - w));
  vb = va + w;
  paint();
}
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
  vid.play();
}
function playJ() {
  vid.pause();
  shuttle = 0;
  const n = rew ? Math.min(rew.n * 2, 8) : 1;
  if (rew) clearInterval(rew.id);
  rew = {
    n,
    id: setInterval(() => {
      if (now() <= 0) return stopShuttle();
      goTo(vid, Math.max(0, now() - 0.04 * n));
      follow();
      draw();
    }, 40),
  };
  draw();
}
const step = (d) => {
  pause();
  seek(now() + d);
};
vid.addEventListener("play", () => {
  const f = () => {
    const ss = sels(C().id);
    if (loop && act >= 0 && ss[act] && now() >= ss[act][1]) vid.currentTime = ss[act][0];
    else if (!loop && act >= 0 && ss[act] && playOne && now() >= ss[act][1]) {
      vid.pause();
      playOne = false;
    }
    follow();
    draw();
    if (!vid.paused) requestAnimationFrame(f);
  };
  f();
});
let playOne = false;
vid.addEventListener("pause", draw);
vid.addEventListener("seeked", draw);
$("play").onclick = () => (vid.paused && !rew ? playL() : pause());
$("fb").onclick = () => step(-FR);
$("ff").onclick = () => step(FR);
$("rate").onchange = () => (vid.playbackRate = +$("rate").value);
$("sound").onclick = () => {
  vid.muted = !vid.muted;
  $("pv").muted = vid.muted;
  $("sound").textContent = vid.muted ? "🔇" : "🔊";
};
$("zfit").onclick = () => {
  va = 0;
  vb = dur();
  paint();
};
$("zin").onclick = () => zoom(0.6);
$("zout").onclick = () => zoom(1.6);
$("prevc").onclick = () => nav(-1);
$("nextc").onclick = () => nav(1);
function nav(d) {
  const v = visible(),
    k = v.indexOf(cur);
  const j = k < 0 ? v[0] : v[Math.min(Math.max(k + d, 0), v.length - 1)];
  if (j !== undefined && j !== cur) go(j);
}
function addSel() {
  const io = pend();
  if (!io) return status("mark an in (I) and an out (O) first", "err");
  const c = C(),
    ss = sels(c.id);
  ss.push(io);
  inPt = outPt = null;
  setSels(c.id, ss);
  act = sels(c.id).findIndex(([a]) => Math.abs(a - io[0]) < 0.01);
  renderAll();
}
function delHere() {
  const c = C(),
    ss = sels(c.id),
    t = now();
  let k =
    act >= 0 && ss[act] && t >= ss[act][0] - 0.05 && t <= ss[act][1] + 0.05
      ? act
      : ss.findIndex(([a, b]) => t >= a - 0.05 && t <= b + 0.05);
  if (k < 0) return;
  ss.splice(k, 1);
  act = -1;
  setSels(c.id, ss);
}
function banIO() {
  const io = pend();
  if (!io) return status("mark an in–out to ban first", "err");
  addVal(C().id, "ban", short(io[0]) + "-" + short(io[1]), reel);
  inPt = outPt = null;
  save();
}
$("banc").onclick = () => {
  const id = C().id;
  if (banned(id).includes("all")) delVal(id, "ban", "all", reel);
  else addVal(id, "ban", "all", reel);
  save();
};
$("bin").onclick = () => {
  inPt = now();
  draw();
};
$("bout").onclick = () => {
  outPt = now();
  draw();
};
$("badd").onclick = addSel;
function marks() {
  const c = C();
  return [...sels(c.id).flat(), ...points(c.id), ...c.moments.map((m) => m[0])].sort((a, b) => a - b);
}
function jump(d) {
  const t = now(),
    p = marks();
  const x = d > 0 ? p.find((v) => v > t + 0.03) : p.reverse().find((v) => v < t - 0.03);
  if (x !== undefined) seek(x);
}
// --- picks, in the order they play in the cut ---
// Each pick is matched to the reel shot that uses it (landscape first); picks the planner has not placed yet come
// after, in the order they were filmed. Tweaks keep a pick's place until the next re-plan.
const pv = $("pv");
let order = [],
  pk = null,
  mode = null,
  pa = 0,
  pb = 1,
  pimg = new Image(),
  pdrag = null;
const place = new Map();
const pkey = (id, a) => id + "|" + a.toFixed(2);
let pfmt = ls.get("pfmt") || "L";
function shotOf(c, a, b) {
  for (const [fmt, arr] of pfmt == "P"
    ? [
        ["P", c.P],
        ["L", c.L],
      ]
    : [
        ["L", c.L],
        ["P", c.P],
      ]) {
    let best = null,
      ov = 0;
    arr.forEach(([x, y, n]) => {
      const o = Math.min(b, y) - Math.max(a, x);
      if (o > ov + 1e-6) {
        ov = o;
        best = n;
      }
    });
    if (best !== null && ov >= Math.min(0.3, (b - a) / 2)) return { fmt, n: best };
  }
  return null;
}
// the chosen cut's shots come in too (k -1: not a pick yet): editing one makes it a pick, removing one bans its stretch
function allSels() {
  const out = [];
  D.clips.forEach((c, i) =>
    sels(c.id).forEach(([a, b], k) => {
      if (!inReel(c.id, a, b)) return;
      const key = pfmt + "|" + pkey(c.id, a);
      if (!place.has(key)) place.set(key, shotOf(c, a, b));
      out.push({ c, i, k, a, b, shot: place.get(key) });
    }),
  );
  if (pfmt != "picks") {
    const have = new Set(out.filter((x) => x.shot && x.shot.fmt == pfmt).map((x) => x.shot.n));
    D.clips.forEach((c, i) => {
      const bans = banned(c.id).map((v) => (v == "all" ? [-1, 1e9] : span(v)));
      c[pfmt].forEach(([a, b, n]) => {
        if (!have.has(n) && !bans.some(([u, v]) => (a + b) / 2 >= u && (a + b) / 2 <= v))
          out.push({ c, i, k: -1, a, b, shot: { fmt: pfmt, n }, cut: true });
      });
    });
  }
  const rank = (x) => (x.shot ? (x.shot.fmt == (pfmt == "P" ? "P" : "L") ? 0 : 1e4) + x.shot.n : 1e9 + x.i * 1e3 + x.a);
  return out.sort((x, y) => rank(x) - rank(y));
}
$("pfmts").onclick = (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  pfmt = b.dataset.f;
  ls.set("pfmt", pfmt);
  pk = null;
  pv.pause();
  selview();
};
// several shots at once: tick a card's box (shift-click for a range), then drop them all in one go (one undo step)
const chosen = new Set();
let lastCk = -1;
const ckey = (x) => pfmt + "|" + x.c.id + "|" + x.a.toFixed(2) + "|" + x.b.toFixed(2);
function tick(j, range) {
  const x = order[j];
  if (!x) return;
  const on = !chosen.has(ckey(x));
  const [lo, hi] = range && lastCk >= 0 ? [Math.min(lastCk, j), Math.max(lastCk, j)] : [j, j];
  for (let k = lo; k <= hi; k++) {
    const y = order[k];
    if (y) on ? chosen.add(ckey(y)) : chosen.delete(ckey(y));
  }
  lastCk = j;
  selview();
}
function dropMany(xs) {
  const gone = new Map();
  xs.forEach((x) => {
    if (x.k >= 0) {
      if (!gone.has(x.c.id)) gone.set(x.c.id, []);
      gone.get(x.c.id).push([x.a, x.b]);
    } else addVal(x.c.id, "ban", fmt(x.a) + "-" + fmt(x.b), reel);
  });
  gone.forEach((ab, id) => {
    for (const [a, b] of ab)
      for (const f of ["open", "end"]) if (same(roleIn(id, f, reel), a, b)) clearRole(id, f, reel);
    setSels(
      id,
      sels(id).filter(([a, b]) => !ab.some(([u, v]) => Math.abs(a - u) < 0.002 && Math.abs(b - v) < 0.002)),
      true,
    );
  });
  save();
}
$("bulkall").onclick = () => {
  order.forEach((x) => {
    if (x.cut) chosen.add(ckey(x));
  });
  selview();
};
$("bulkclr").onclick = () => {
  chosen.clear();
  lastCk = -1;
  selview();
};
$("bulkdrop").onclick = () => {
  const xs = order.filter((x) => chosen.has(ckey(x)));
  if (!xs.length) return;
  if (pk && xs.some((x) => x.c.id == pk.id && Math.abs(x.a - pk.a) < 0.002)) {
    pk = null;
    pv.pause();
  }
  chosen.clear();
  lastCk = -1;
  dropMany(xs);
  status(`dropped ${xs.length} shot${xs.length == 1 ? "" : "s"} — Ctrl+Z undoes`, "ok");
};
const pj = () => (pk ? order.findIndex((x) => x.c.id == pk.id && Math.abs(x.a - pk.a) < 0.002) : -1);
const PX = () => order[pj()];
function selview() {
  order = allSels();
  const live = new Set(order.map(ckey));
  [...chosen].forEach((k) => {
    if (!live.has(k)) chosen.delete(k);
  });
  $("bulk").classList.toggle("on", chosen.size > 0);
  $("cards").classList.toggle("picking", chosen.size > 0);
  $("bulkn").textContent = chosen.size + " selected";
  const tot = order.reduce((s, x) => s + x.b - x.a, 0),
    j = pj(),
    placed = order.filter((x) => x.shot).length;
  const np = order.filter((x) => !x.cut).length,
    nc = order.length - np;
  $("pfmts")
    .querySelectorAll("button")
    .forEach((b) => b.classList.toggle("on", b.dataset.f == pfmt));
  $("seltitle").textContent =
    (pfmt == "picks"
      ? ""
      : `${nc + order.filter((x) => !x.cut && x.shot && x.shot.fmt == pfmt).length} shots in the ${pfmt == "P" ? "portrait" : "landscape"} cut · `) +
    np +
    " pick" +
    (np == 1 ? "" : "s");
  $("seltot").textContent =
    short(tot) +
    " of footage · in cut order" +
    (order.length > placed ? ` · ${order.length - placed} not in the cut yet (I place them on the next re-plan)` : "") +
    (nc ? " · trim, slip or drag a shot to make it a pick; ✕ bans it" : "");
  $("seq").innerHTML = order
    .map(
      (x, k) =>
        `<i data-j=${k} class="${k == j ? "act" : ""}${chosen.has(ckey(x)) ? " chk" : ""}${x.shot ? "" : " new"}${x.cut ? " cut" + (pfmt == "P" ? " p" : "") : ""}" style="flex:${(x.b - x.a).toFixed(2)}" title="${esc(x.c.id)} ${fmt(x.a)}–${fmt(x.b)}"></i>`,
    )
    .join("");
  let h = "",
    sep = false;
  order.forEach((x, k) => {
    if (!x.shot && !sep) {
      sep = true;
      h += `<div class=catgrp>Not in the cut yet — placed on the next re-plan</div>`;
    }
    const ck = chosen.has(ckey(x));
    h += `<div class="card${k == j ? " act" : ""}${x.cut ? " cut" : ""}${ck ? " chk" : ""}" data-j=${k}><div class=th data-j=${k}></div>
  <div class=bd><div><b>${x.shot ? (x.shot.fmt == "L" ? "Shot " : "Portrait shot ") + x.shot.n : "New"}</b> ${x.cut ? "" : "<span class=tag>pick</span> "}${!x.cut && starsOf(x.c.id, x.a, x.b) ? `<span style="color:var(--io)">${"★".repeat(starsOf(x.c.id, x.a, x.b))}</span> ` : ""}${!x.cut && reel == "all" && onlyOf(x.c.id, x.a, x.b) ? `<span class=tag>${esc(onlyOf(x.c.id, x.a, x.b).map(rlab).join(" + "))} only</span> ` : ""}${{ open: "<span class=tag>★ opening</span> ", end: "<span class=tag>⚑ ending</span> " }[roleOf(x.c.id, x.a, x.b)] || ""}· ${esc(x.c.id)}<br>
  <small>${fmt(x.a)} – ${fmt(x.b)} · <b>${(x.b - x.a).toFixed(2)} s</b> · ${esc(x.c.clock)}</small></div>
  <div><button data-del=${k} title="${x.cut ? "take this shot out of the cut (bans its stretch)" : "remove this pick"}">✕</button></div></div></div>`;
  });
  $("cards").innerHTML = h || `<p style="color:var(--mut)">No picks yet — choose footage in Sources.</p>`;
  const ths = $("cards").querySelectorAll(".th"),
    tw0 = (ths[0] && ths[0].clientWidth) || 200;
  ths.forEach((el) => {
    const x = order[+el.dataset.j],
      hh = (tw0 * TH) / TW;
    el.setAttribute("style", thumbAt(x.c, x.a, hh, tw0));
    el.innerHTML = `<span>${k1(el)}</span><input type=checkbox class=ck data-ck=${el.dataset.j} title="select (shift-click: a range)"${chosen.has(ckey(x)) ? " checked" : ""}>`;
  });
  function k1(el) {
    const x = order[+el.dataset.j];
    return +el.dataset.j + 1 + " · " + (x.b - x.a).toFixed(1) + "s";
  }
  $("ped").style.display = PX() ? "block" : "none";
  const px = PX();
  $("pstars").innerHTML = px && px.k >= 0 ? starBar(px.c.id, px.a, px.b, "") : "";
  $("preels").innerHTML =
    px && px.k >= 0 && RS.length > 1
      ? "In " +
        RS.map(
          (r) =>
            `<button data-reel="${esc(r.id)}" class="${inReel(px.c.id, px.a, px.b, r.id) ? "on" : ""}" title="put this pick in / take it out of the ${esc(r.label)} reel">${esc(r.label)}</button>`,
        ).join("")
      : "";
  if (px) pdraw();
}
$("cards").onclick = (ev) => {
  const c = ev.target.closest("[data-ck]");
  if (c) {
    ev.stopPropagation();
    ev.preventDefault();
    tick(+c.dataset.ck, ev.shiftKey);
    return;
  }
  const cd = ev.target.closest(".card");
  if (cd && (ev.shiftKey || ev.ctrlKey || ev.metaKey) && !ev.target.closest("[data-del]")) {
    tick(+cd.dataset.j, ev.shiftKey);
    return;
  }
  const b = ev.target.closest("[data-del],.card");
  if (!b) return;
  const x = order[+b.dataset.j ?? +b.dataset.del];
  if (b.dataset.del !== undefined) {
    ev.stopPropagation();
    const y = order[+b.dataset.del];
    if (pk && pk.id == y.c.id && Math.abs(pk.a - y.a) < 0.002) {
      pk = null;
      pv.pause();
    }
    drop(y);
  } else pick(+b.dataset.j, false);
};
$("seq").onclick = (ev) => {
  const i = ev.target.closest("i");
  if (i) pick(+i.dataset.j, true);
};
function pick(j, play) {
  const x = order[j];
  if (!x) return stopAll();
  pk = { id: x.c.id, a: x.a };
  const ctx = Math.max(2, (x.b - x.a) * 0.75);
  pa = Math.max(0, x.a - ctx);
  pb = Math.min(x.c.dur, x.b + ctx);
  pimg = new Image();
  pimg.onload = pdraw;
  pimg.src = x.c.strip;
  const start = () => {
    pv.currentTime = x.a;
    if (play) pv.play();
    pdraw();
  };
  if (pv.dataset.c !== x.c.id) {
    pv.dataset.c = x.c.id;
    pv.src = x.c.video;
    pv.addEventListener("loadedmetadata", start, { once: true });
  } else start();
  selview();
  const card = $("cards").querySelector(`.card[data-j="${j}"]`);
  if (card) card.scrollIntoView({ block: "nearest" });
}
// change the current pick's in / out (lengthen, trim, slip); it keeps its place in the cut
function edit(a, b) {
  const x = PX();
  if (!x) return;
  const L = x.c.dur;
  if (b - a < 0.1) return;
  if (a < 0) {
    b -= a;
    a = 0;
  }
  if (b > L) {
    a -= b - L;
    b = L;
  }
  a = Math.max(0, a);
  const e0 = P[x.c.id] || {};
  for (const f of ["open", "end"])
    if (same(e0[f], x.a, x.b)) {
      const t = tagsOf(x.c.id, f, e0[f]); // (its role goes with it)
      setTags(x.c.id, f, e0[f], null);
      e0[f] = fmt(a) + "-" + fmt(b);
      setTags(x.c.id, f, e0[f], t);
    }
  const ss = sels(x.c.id);
  if (x.k >= 0) ss[x.k] = [a, b];
  else ss.push([a, b]);
  place.set(pfmt + "|" + pkey(x.c.id, a), x.shot);
  pk = { id: x.c.id, a };
  if (a < pa || b > pb) {
    const ctx = Math.max(2, (b - a) * 0.75);
    pa = Math.max(0, a - ctx);
    pb = Math.min(L, b + ctx);
  }
  setSels(x.c.id, ss);
}
const pnow = () => pv._want ?? (pv.currentTime || 0);
function pdraw() {
  const x = PX();
  if (!x) return;
  const T = $("pt"),
    W = pb - pa,
    t = pnow();
  film($("pstrip").querySelector("canvas"), pa, pb, 64, x.c, pimg);
  T.querySelectorAll(":scope>i,#pstrip>i").forEach((e) => e.remove());
  let a = x.a,
    b = x.b;
  if (pdrag && pdrag.cur) [a, b] = pdrag.cur;
  let h = `<i class=dim style="${wspan(pa, a, pa, pb)}"></i><i class=dim style="${wspan(b, pb, pa, pb)}"></i>`;
  sels(x.c.id).forEach(([u, v], k) => {
    if (k != x.k && v > pa && u < pb) h += `<i class=oth style="${wspan(u, v, pa, pb)}" title="another pick"></i>`;
  });
  h += `<i class="blk act" data-k=0 style="${wspan(a, b, pa, pb)}"><i class="h l"></i>${(b - a).toFixed(2)} s<i class="h r"></i></i>`;
  fkeys(x.c.id).forEach(([k]) => {
    if (k >= pa && k <= pb) h += `<i class=fk style="left:${X(k, pa, pb)}" title="focal point ${fmt(k)}"></i>`;
  });
  if (t >= pa && t <= pb) h += `<i class=ph style="left:${X(t, pa, pb)}"></i>`;
  $("pstrip").insertAdjacentHTML("beforeend", h);
  const step = [0.1, 0.2, 0.5, 1, 2, 5, 10].find((s) => W / s <= 40) || 30;
  let r = "";
  for (let v = Math.ceil(pa / step) * step; v <= pb; v += step) r += `<i class="tk" style="left:${X(v, pa, pb)}"></i>`;
  $("pruler").innerHTML =
    r +
    `<i class=lb style="left:${X(a, pa, pb)}">in ${fmt(a)}</i><i class=lb style="left:${X(b, pa, pb)}">out ${fmt(b)}</i>`;
  $("plen").textContent = (b - a).toFixed(2) + " s";
  const ro = roleOf(x.c.id, x.a, x.b);
  $("pfirst").classList.toggle("on", ro == "open");
  $("plast").classList.toggle("on", ro == "end");
  $("pvtc").textContent = pj() + 1 + "/" + order.length + " · " + x.c.id + " · " + fmt(t);
  $("pplay").textContent = pv.paused ? "▶ Play pick" : "❚❚ Pause";
  $("ploop").classList.toggle("on", mode == "loop");
  const st = $("pstill"),
    no = pv.error || (pv.readyState < 2 && !pv.seeking) || !pv.videoWidth;
  st.classList.toggle("on", !!no);
  if (no) {
    const R = $("pvwrap").getBoundingClientRect(),
      [tw, th] = tileOf(x.c),
      hh = Math.min(R.height, (R.width * th) / tw);
    st.setAttribute(
      "style",
      thumbAt(x.c, t, hh) +
        `;width:${(hh * tw) / th}px;height:${hh}px;left:${(R.width - (hh * tw) / th) / 2}px;top:${(R.height - hh) / 2}px`,
    );
  }
  frameDraw($("pfr"), $("pvwrap"), pv, x.c, x.a, x.b, t);
}
const PT = $("pstrip");
PT.addEventListener("pointerdown", (ev) => {
  const x = PX();
  if (!x) return;
  pv.pause();
  PT.setPointerCapture(ev.pointerId);
  const t = tAt(PT, ev, pa, pb),
    blk = ev.target.closest(".blk");
  pdrag = blk
    ? {
        k: ev.target.classList.contains("l") ? "l" : ev.target.classList.contains("r") ? "r" : "slip",
        t0: t,
        o: [x.a, x.b],
        cur: [x.a, x.b],
      }
    : { k: "scrub" };
  if (!blk) goTo(pv, t);
  pdraw();
});
PT.addEventListener("pointermove", (ev) => {
  const x = PX();
  if (!x) return;
  const t = tAt(PT, ev, pa, pb);
  hov.style.display = "block";
  hov.querySelector("div").setAttribute("style", thumbAt(x.c, t, 108, 192));
  hov.querySelector("span").textContent = fmt(t);
  const R = PT.getBoundingClientRect();
  hov.style.left = Math.min(innerWidth - 205, Math.max(4, ev.clientX - 100)) + "px";
  hov.style.top = R.top - 136 + "px";
  if (!pdrag) return;
  if (pdrag.k == "scrub") {
    goTo(pv, t);
    pdraw();
    return;
  }
  let [a, b] = pdrag.o;
  const d = t - pdrag.t0;
  if (pdrag.k == "slip") {
    const w = b - a;
    a = Math.max(0, Math.min(a + d, x.c.dur - w));
    b = a + w;
    goTo(pv, a);
  } else if (pdrag.k == "l") {
    a = Math.max(0, Math.min(t, b - 0.1));
    goTo(pv, a);
  } else {
    b = Math.min(x.c.dur, Math.max(t, a + 0.1));
    goTo(pv, b);
  }
  pdrag.cur = [a, b];
  pdraw();
});
PT.addEventListener("pointerup", () => {
  const d = pdrag;
  pdrag = null;
  if (d && d.cur && (d.cur[0] !== d.o[0] || d.cur[1] !== d.o[1])) edit(...d.cur);
  else pdraw();
});
PT.addEventListener("pointerleave", () => (hov.style.display = "none"));
PT.addEventListener(
  "wheel",
  (ev) => {
    ev.preventDefault();
    const x = PX();
    if (!x) return;
    const R = PT.getBoundingClientRect(),
      f = (ev.clientX - R.left) / R.width,
      at = pa + f * (pb - pa),
      w = Math.min(Math.max((pb - pa) * (ev.deltaY > 0 ? 1.25 : 0.8), 1), x.c.dur);
    pa = Math.max(0, Math.min(at - f * w, x.c.dur - w));
    pb = pa + w;
    pdraw();
  },
  { passive: false },
);
pv.addEventListener("timeupdate", () => {
  const x = PX();
  if (!x) {
    return;
  }
  if (!pv.paused && pnow() >= x.b - 0.02) {
    if (mode == "loop") pv.currentTime = x.a;
    else if (mode == "all") {
      const j = pj() + 1;
      order[j] ? pick(j, true) : stopAll();
    } else pv.pause();
  }
  pdraw();
});
for (const e of ["seeked", "pause", "play", "loadeddata", "error"]) pv.addEventListener(e, pdraw);
pv.addEventListener("play", () => {
  const f = () => {
    pdraw();
    const x = PX();
    if (x && mode == "loop" && pnow() >= x.b - 0.02) pv.currentTime = x.a;
    if (!pv.paused) requestAnimationFrame(f);
  };
  f();
});
function playPick() {
  const x = PX();
  if (!x) return order.length && pick(0, true);
  if (!pv.paused) return pv.pause();
  if (mode != "all" && mode != "loop") mode = "one";
  if (pnow() < x.a - 0.05 || pnow() >= x.b - 0.05) pv.currentTime = x.a;
  pv.play();
}
function stopAll() {
  pv.pause();
  mode = null;
  pdraw();
}
function nudge(which, d) {
  const x = PX();
  if (!x) return;
  let [a, b] = [x.a, x.b];
  if (which == "in") a = Math.min(a + d, b - 0.1);
  else if (which == "out") b = Math.max(b + d, a + 0.1);
  else {
    a += d;
    b += d;
  }
  edit(a, b);
  pv.currentTime = which == "out" ? b : a;
}
$("ped").onclick = (ev) => {
  const st = ev.target.closest("[data-star]");
  if (st) {
    const x = PX();
    if (x && x.k >= 0) {
      const n = +st.dataset.star;
      setStars(x.c.id, x.a, x.b, starsOf(x.c.id, x.a, x.b) == n ? 0 : n);
    }
    return;
  }
  const b = ev.target.closest("button");
  if (!b) return;
  const x = PX();
  if (b.dataset.n) {
    const [w, d] = b.dataset.n.split(":");
    nudge(w, +d * (ev.shiftKey ? 10 : 1));
  } else if (b.id == "pplay") playPick();
  else if (b.id == "ploop") {
    mode = mode == "loop" ? "one" : "loop";
    playPick();
  } else if (b.id == "pprev") pick(Math.max(0, pj() - 1), !pv.paused);
  else if (b.id == "pnext") pick(Math.min(order.length - 1, pj() + 1), !pv.paused);
  else if (b.id == "pin") edit(Math.min(pnow(), x.b - 0.1), x.b);
  else if (b.id == "pout") edit(x.a, Math.max(pnow(), x.a + 0.1));
  else if (b.id == "popen") {
    view(1);
    go(x.i);
    act = x.k;
    vid.addEventListener("loadedmetadata", () => seek(x.a), { once: true });
  } else if (b.dataset.reel) {
    if (x.k >= 0) {
      const r = b.dataset.reel,
        on = !inReel(x.c.id, x.a, x.b, r);
      if (!on && r == reel && (onlyOf(x.c.id, x.a, x.b) || RID).length > 1) {
        pk = null;
        pv.pause();
      }
      setIn(x.c.id, x.a, x.b, r, on);
    }
  } else if (b.id == "pfirst") setRole(x, "open");
  else if (b.id == "plast") setRole(x, "end");
  else if (b.id == "pdel") {
    const j = pj();
    pk = null;
    pv.pause();
    drop(x);
    if (order[j]) pick(j, false);
  }
};
// a pick goes; a shot of the cut is banned, so the next re-plan puts something else there
function drop(x) {
  if (x.k >= 0) {
    for (const f of ["open", "end"]) if (same(roleIn(x.c.id, f, reel), x.a, x.b)) clearRole(x.c.id, f, reel);
    const ss = sels(x.c.id);
    ss.splice(x.k, 1);
    setSels(x.c.id, ss);
  } else {
    addVal(x.c.id, "ban", fmt(x.a) + "-" + fmt(x.b), reel);
    save();
  }
}
$("playall").onclick = () => {
  mode = "all";
  const j = Math.max(0, pj());
  pick(j, true);
};
$("stopall").onclick = stopAll;
// --- portrait framing: the crop the planner derives (served by /api/crop) and the editor's focal points, picks
// "portrait": [["m:ss.ss", x, y], ...] (fractions of the frame); the reel's crop centre follows them, its size as derived
let frameOn = !!ls.get("frame");
const crops = new Map();
const fkeys = (id) => ((P[id] || {}).portrait || []).map(([t, x, y]) => [secs(t), +x, +y]).sort((p, q) => p[0] - q[0]);
// the person to follow: picks "follow": [["m:ss.ss", x, y], ...] where the editor clicked them; the server's crop
// (/api/crop) then tracks their face through the shot, so the page asks for it again once the click is saved
const folkeys = (id) => ((P[id] || {}).follow || []).map(([t, x, y]) => [secs(t), +x, +y]).sort((p, q) => p[0] - q[0]);
let followOn = false,
  staleCrop = null;
function setFollow(id, t, x, y, a, b) {
  const e = ent(id),
    ks = folkeys(id).filter((k) => k[0] < a - 0.5 || k[0] > b + 0.5);
  if (x != null) ks.push([t, x, y]);
  if (ks.length) e.follow = ks.sort((p, q) => p[0] - q[0]).map(([t, x, y]) => [fmt(t), +x.toFixed(4), +y.toFixed(4)]);
  else delete e.follow;
  staleCrop = id;
  save();
}
function followMode(on) {
  followOn = on;
  if (on && !frameOn) toggleFrame();
  document.querySelectorAll(".fol").forEach((b) => b.classList.toggle("on", on));
  document.querySelectorAll(".fr").forEach((f) => f.classList.toggle("follow", on));
  if (on) status("click the dancer to follow", "ok");
}
const putKeys = (id, ks) => {
  ent(id).portrait = ks.sort((p, q) => p[0] - q[0]).map(([t, x, y]) => [fmt(t), +x.toFixed(4), +y.toFixed(4)]);
};
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const lerp = (ts, vs, t) => {
  if (t <= ts[0]) return vs[0];
  for (let i = 1; i < ts.length; i++)
    if (t <= ts[i]) {
      const f = (t - ts[i - 1]) / (ts[i] - ts[i - 1] || 1);
      return vs[i - 1] + f * (vs[i] - vs[i - 1]);
    }
  return vs[vs.length - 1];
};
// one request at a time (each is worked out by the server): a drag asks for a new stretch on every move, so only the
// latest one waiting is fetched next, and the clip's last crop stands in meanwhile. Returns [crop, its a] or a state.
let cropBusy = false,
  cropNext = null;
const lastCrop = new Map();
function fetchCrop(c, a, b, k) {
  cropBusy = true;
  fetch(`${D.root}api/crop?id=${encodeURIComponent(c.id)}&a=${a.toFixed(3)}&b=${b.toFixed(3)}&fmt=portrait`)
    .then((r) => (r.ok ? r.json() : Promise.reject()))
    .then((j) => {
      crops.set(k, [j, a]);
      lastCrop.set(c.id, [j, a]);
    })
    .catch(() => crops.set(k, false))
    .then(() => {
      cropBusy = false;
      const n = cropNext;
      cropNext = null;
      if (n && !crops.has(n[3])) fetchCrop(...n);
      draw();
      pdraw();
    });
}
function derived(c, a, b) {
  const k = c.id + "|" + a.toFixed(2) + "|" + b.toFixed(2);
  if (crops.has(k)) return crops.get(k) ?? lastCrop.get(c.id);
  if (!server) {
    crops.set(k, false);
    return false;
  }
  if (cropBusy) cropNext = [c, a, b, k];
  else {
    crops.set(k, undefined);
    fetchCrop(c, a, b, k);
  }
  return lastCrop.get(c.id);
}
// where the frame is worked out from: the selection or portrait reel shot under the playhead, else the 3 s around it
function rangeFor(c, t) {
  const s = sels(c.id).find(([a, b]) => t >= a - 0.02 && t <= b + 0.02) || c.P.find(([a, b]) => t >= a && t <= b);
  if (s) return [s[0], s[1]];
  const a = Math.max(0, Math.min(Math.floor(t / 3) * 3, c.dur - 3));
  return [a, Math.min(a + 3, c.dur)];
}
function frameAt(c, a, b, t) {
  const g = derived(c, a, b);
  if (g === undefined) return "wait";
  if (g === false) return "off";
  const [cr, a0] = g;
  if (cr === null) return "fills";
  const ts = cr.path.map((q) => a0 + q[0]),
    d = {
      x: lerp(
        ts,
        cr.path.map((q) => q[1]),
        t,
      ),
      y: lerp(
        ts,
        cr.path.map((q) => q[2]),
        t,
      ),
    };
  const ks = fkeys(c.id).filter((k) => k[0] >= a - 0.5 && k[0] <= b + 0.5);
  let o = null;
  if (ks.length) {
    const kt = ks.map((k) => k[0]);
    o = {
      x: clamp(
        lerp(
          kt,
          ks.map((k) => k[1]),
          t,
        ),
        cr.w / 2,
        1 - cr.w / 2,
      ),
      y: clamp(
        lerp(
          kt,
          ks.map((k) => k[2]),
          t,
        ),
        cr.h / 2,
        1 - cr.h / 2,
      ),
    };
  }
  return { d, o, w: cr.w, h: cr.h, z: cr.z, fol: cr.follow };
}
function vrect(wrap, v) {
  const R = wrap.getBoundingClientRect(),
    ar = v.videoWidth && v.videoHeight ? v.videoWidth / v.videoHeight : C()?.ar || TW / TH;
  const w = Math.min(R.width, R.height * ar),
    h = w / ar;
  return { l: (R.width - w) / 2, t: (R.height - h) / 2, w, h };
}
function frameDraw(el, wrap, v, c, a, b, t) {
  el.classList.toggle("on", frameOn);
  $("fhelp").hidden = !frameOn;
  document.querySelectorAll(".frt").forEach((x) => x.classList.toggle("on", frameOn));
  if (!frameOn) {
    el.innerHTML = "";
    return;
  }
  const F = frameAt(c, a, b, t);
  if (typeof F == "string") {
    el.innerHTML = `<span class=frl>${
      {
        fills: "this clip fills the portrait frame as it is — no crop to place",
        off: "the portrait frame needs the page served by <code>focalcut reel serve</code>",
        wait: "working out the portrait crop…",
      }[F]
    }</span>`;
    return;
  }
  const r = vrect(wrap, v),
    box = (p, cls) =>
      `<i class="fbox ${cls}" style="left:${r.l + (p.x - F.w / 2) * r.w}px;top:${r.t + (p.y - F.h / 2) * r.h}px;width:${F.w * r.w}px;height:${F.h * r.h}px"></i>`;
  const A = F.o || F.d,
    dx = F.o ? F.o.x - F.d.x : 0,
    pin = folkeys(c.id).find((k) => Math.abs(k[0] - t) <= 0.6);
  el.innerHTML =
    box(A, "act") +
    (F.o ? box(F.d, "der") : "") +
    `<i class=fdot style="left:${r.l + A.x * r.w}px;top:${r.t + A.y * r.h}px"></i>` +
    (pin
      ? `<i class=fpin style="left:${r.l + pin[1] * r.w}px;top:${r.t + pin[2] * r.h}px" title="the person you chose to follow"></i>`
      : "") +
    `<span class=frl>${F.fol == "face" ? "<b>following the person you clicked</b> · " : F.fol == "point" ? "<b>no face found there: held on your click</b> · " : ""}planner x ${F.d.x.toFixed(2)} y ${F.d.y.toFixed(2)}${F.z > 1.001 ? " · zoom " + F.z.toFixed(2) : ""}` +
    (F.o
      ? ` → <b>your focus x ${F.o.x.toFixed(2)} y ${F.o.y.toFixed(2)} (${dx >= 0 ? "+" : ""}${dx.toFixed(2)})</b>`
      : " · drag to set a focal point") +
    `</span>`;
}
function toggleFrame() {
  frameOn = !frameOn;
  ls.set("frame", frameOn);
  draw();
  pdraw();
}
// drag on the picture: a focal point at the playhead (one within ¼ s of it moves instead)
function frameDrag(el, wrap, v, ctx, redraw, stop) {
  let x = null;
  const put = (ev) => {
    const r = vrect(wrap, v),
      R = wrap.getBoundingClientRect();
    const ks = fkeys(x.c.id).filter((k) => Math.abs(k[0] - x.t) > 0.25);
    ks.push([x.t, clamp((ev.clientX - R.left - r.l) / r.w, 0, 1), clamp((ev.clientY - R.top - r.t) / r.h, 0, 1)]);
    putKeys(x.c.id, ks);
    redraw();
  };
  el.addEventListener("pointerdown", (ev) => {
    if (!frameOn) return;
    x = ctx();
    if (!x) return;
    stop();
    if (followOn) {
      const r = vrect(wrap, v),
        R = wrap.getBoundingClientRect(),
        [a, b] = x.r || rangeFor(x.c, x.t);
      setFollow(
        x.c.id,
        x.t,
        clamp((ev.clientX - R.left - r.l) / r.w, 0, 1),
        clamp((ev.clientY - R.top - r.t) / r.h, 0, 1),
        a,
        b,
      );
      x = null;
      followMode(false);
      status("following that person — the crop updates in a moment", "ok");
      return;
    }
    el.setPointerCapture(ev.pointerId);
    put(ev);
  });
  el.addEventListener("pointermove", (ev) => {
    if (x) put(ev);
  });
  el.addEventListener("pointerup", () => {
    if (x) {
      x = null;
      save();
    }
  });
}
frameDrag($("fr"), $("vwrap"), vid, () => ({ c: C(), t: now() }), draw, pause);
frameDrag(
  $("pfr"),
  $("pvwrap"),
  pv,
  () => PX() && { c: PX().c, t: pnow(), r: [PX().a, PX().b] },
  pdraw,
  () => pv.pause(),
);
function clearFocus(c, a, b) {
  const ks = fkeys(c.id),
    kept = ks.filter((k) => k[0] < a - 0.5 || k[0] > b + 0.5),
    fl = folkeys(c.id).some((k) => k[0] >= a - 0.5 && k[0] <= b + 0.5);
  if (kept.length == ks.length && !fl) return status("no focal points here");
  putKeys(c.id, kept);
  if (fl) setFollow(c.id, 0, null, null, a, b);
  else save();
}
$("bfollow").onclick = $("pfollow").onclick = () => followMode(!followOn);
$("bframe").onclick = $("pframe").onclick = toggleFrame;
$("fclr").onclick = () => clearFocus(C(), ...rangeFor(C(), now()));
$("pfclr").onclick = () => {
  const x = PX();
  if (x) clearFocus(x.c, x.a, x.b);
};
// views
function view(n) {
  $("clipsview").style.display = n == 1 ? "grid" : "none";
  $("selview").style.display = n == 2 ? "grid" : "none";
  $("v1").classList.toggle("on", n == 1);
  $("v2").classList.toggle("on", n == 2);
  if (n == 2) {
    pause();
    selview();
  } else {
    pv.pause();
    paint();
  }
}
$("v1").onclick = () => view(1);
$("v2").onclick = () => view(2);
$("copy").onclick = () => {
  navigator.clipboard && navigator.clipboard.writeText(JSON.stringify(P, null, 1));
  $("copy").textContent = "Copied ✓";
  setTimeout(() => ($("copy").textContent = "Copy picks"), 1500);
};
$("dl").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(P, null, 1)], { type: "application/json" }));
  a.download = "picks.json";
  a.click();
};
addEventListener("resize", () => paint());
function undo() {
  if (!hist.length) return status("nothing to undo");
  P = JSON.parse(hist.pop());
  lastSnap = JSON.stringify(P);
  tidy();
  if (server) {
    clearTimeout(timer);
    timer = setTimeout(push, 250);
  } else ls.set("picks", P);
  status("undone");
  renderAll();
}
addEventListener(
  "keydown",
  (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() == "z") {
      ev.preventDefault();
      undo();
      return;
    }
  },
  true,
);
addEventListener(
  "keydown",
  (ev) => {
    if (ev.target.tagName == "SELECT" || ev.target.tagName == "INPUT" || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const k = ev.key,
      K = k.toLowerCase();
    let done = true;
    if (k == "Tab") view($("selview").style.display == "grid" ? 1 : 2);
    else if (K == "f") toggleFrame();
    else if (K == "g") followMode(!followOn);
    else if (k == "Escape" && followOn) followMode(false);
    else if ($("selview").style.display == "grid") {
      const x = PX();
      if (chosen.size && (k == "Backspace" || k == "Delete")) $("bulkdrop").click();
      else if (k == "Escape" && chosen.size) $("bulkclr").click();
      else if (K == "a" && !ev.shiftKey) $("bulkall").click();
      else if (K == "x" && x) tick(pj(), false);
      else if (k == " ") playPick();
      else if (k == "Escape") stopAll();
      else if (k == "ArrowRight" || k == "ArrowDown") pick(Math.min(order.length - 1, pj() + 1), !pv.paused);
      else if (k == "ArrowLeft" || k == "ArrowUp") pick(Math.max(0, pj() - 1), !pv.paused);
      else if (!x) done = false;
      else if (K == "i") edit(Math.min(pnow(), x.b - 0.1), x.b);
      else if (K == "o") edit(x.a, Math.max(pnow(), x.a + 0.1));
      else if (k == "," || k == "<") nudge("slip", ev.shiftKey ? -0.5 : -0.02);
      else if (k == "." || k == ">") nudge("slip", ev.shiftKey ? 0.5 : 0.02);
      else if (k == "\\") {
        mode = mode == "loop" ? "one" : "loop";
        playPick();
      } else if (k == "Backspace" || k == "Delete") $("pdel").click();
      else if (/^[1-5]$/.test(k) && x.k >= 0) {
        const n = +k;
        setStars(x.c.id, x.a, x.b, starsOf(x.c.id, x.a, x.b) == n ? 0 : n);
      } else done = false;
    } else if (/^[1-5]$/.test(k)) {
      const ss = sels(C().id),
        t = now();
      let j =
        act >= 0 && ss[act] && t >= ss[act][0] - 0.05 && t <= ss[act][1] + 0.05
          ? act
          : ss.findIndex(([a, b]) => t >= a - 0.05 && t <= b + 0.05);
      if (j < 0) j = act;
      if (j >= 0 && ss[j]) {
        const n = +k;
        setStars(C().id, ss[j][0], ss[j][1], starsOf(C().id, ss[j][0], ss[j][1]) == n ? 0 : n);
      } else status("put the playhead on a selection to star it", "err");
    } else if (k == " " || K == "k") vid.paused && !rew ? ((playOne = false), playL()) : pause();
    else if (K == "l") playL();
    else if (K == "j") playJ();
    else if (k == "ArrowLeft" || k == ",") step(ev.shiftKey ? -1 : -FR);
    else if (k == "ArrowRight" || k == ".") step(ev.shiftKey ? 1 : FR);
    else if (k == "ArrowUp" || k == "PageUp") nav(-1);
    else if (k == "ArrowDown" || k == "PageDown") nav(1);
    else if (K == "i") {
      inPt = now();
      if (outPt !== null && outPt <= inPt) outPt = null;
      draw();
    } else if (K == "o") {
      outPt = now();
      if (inPt !== null && inPt >= outPt) inPt = null;
      draw();
    } else if (k == "Enter" || K == "a") addSel();
    else if (K == "x") {
      inPt = outPt = null;
      draw();
    } else if (k == "Backspace" || k == "Delete") delHere();
    else if (k == "[") jump(-1);
    else if (k == "]") jump(1);
    else if (k == "\\") {
      const ss = sels(C().id);
      if (act < 0) act = ss.findIndex(([a, b]) => now() >= a && now() <= b);
      if (act >= 0) {
        loop = !loop;
        seek(ss[act][0]);
        vid.play();
      }
    } else if (K == "b") banIO();
    else if (k == "=" || k == "+") zoom(0.6);
    else if (k == "-" || k == "_") zoom(1.6);
    else if (k == "0") {
      va = 0;
      vb = dur();
      paint();
    } else if (k == "Home") seek(0);
    else if (k == "End") seek(dur());
    else done = false;
    if (done) {
      ev.preventDefault();
      ev.stopPropagation();
    }
  },
  true,
);
boot();
