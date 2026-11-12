const mmss = (x) => {
  x = Math.max(0, x);
  const m = Math.floor(x / 60);
  return m + ":" + (x - 60 * m).toFixed(1).padStart(4, "0");
};
const store = {
  get() {
    try {
      return JSON.parse(localStorage.getItem("picks:" + D.key));
    } catch (e) {
      return null;
    }
  },
  set(v) {
    try {
      localStorage.setItem("picks:" + D.key, JSON.stringify(v));
    } catch (e) {}
  },
  del() {
    try {
      localStorage.removeItem("picks:" + D.key);
    } catch (e) {}
  },
};
let P = store.get() || JSON.parse(JSON.stringify(D.picks));
const ent = (id) => (P[id] && typeof P[id] === "object" && !Array.isArray(P[id]) ? P[id] : (P[id] = {}));

const R = D.reel || "main"; // picks made on this page are for this reel; the others' are left alone
const esc = (x) =>
  String(x).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
function tidy() {
  for (const k in P) {
    const e = P[k];
    if (k[0] == "_" || !e || typeof e !== "object" || Array.isArray(e)) continue;
    pruneTags(e);
    for (const f of ["ban", "prefer", "add"]) if (e[f] && !e[f].length) delete e[f];
    if (e.more === 0) delete e.more;
    if (!Object.keys(e).length) delete P[k];
  }
}
// served by `focalcut reel serve`: picks.json is read and written directly; opened as a file: kept in this browser
let server = false,
  timer = null;
const saved = (t) => {
  const e = document.getElementById("saved");
  if (e) e.textContent = t;
};
function save() {
  tidy();
  if (server) {
    clearTimeout(timer);
    timer = setTimeout(push, 250);
    saved("saving…");
  } else store.set(P);
  render();
}
function push() {
  fetch(D.root + "api/picks", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(P, null, 1),
  })
    .then((r) => saved(r.ok ? "saved to picks.json ✓" : "could not save (" + r.status + ")"))
    .catch((e) => saved("could not save: " + e.message));
}
fetch(D.root + "api/picks", { cache: "no-store" })
  .then((r) => (r.ok ? r.json() : Promise.reject()))
  .then((j) => {
    if (!j || typeof j !== "object" || Array.isArray(j)) return;
    P = j;
    server = true;
    (document.getElementById("allp") || {}).hidden = false;
    saved("picks.json — saves as you go");
    render();
  })
  .catch(() => {});
function range(a, b) {
  return mmss(a) + "-" + mmss(b);
}
const secs = (v) => {
  if (typeof v === "number") return v;
  const p = String(v).split(":");
  return p.length > 1 ? +p[0] * 60 + +p[1] : +p[0];
};
const span = (v) => {
  if (Array.isArray(v)) return v.map(secs);
  const s = String(v);
  const i = s.indexOf("-");
  return i > 0 ? [secs(s.slice(0, i)), secs(s.slice(i + 1))] : [secs(s), secs(s)];
};
function shotRange(i) {
  const s = D.shots[i];
  return range(s.a, s.b);
}
function isBanned(i) {
  const s = D.shots[i];
  return vals(s.id, "ban", R).some((v) => covers(v, s));
}
function covers(v, s) {
  if (v == "all") return true;
  const [a, b] = span(v);
  return a <= s.a + 0.05 && b >= s.b - 0.05;
}
function toggleBan(i) {
  const s = D.shots[i],
    r = shotRange(i),
    mine = vals(s.id, "ban", R);
  if (mine.some((v) => valEq("ban", v, r))) delVal(s.id, "ban", r, R);
  else if (isBanned(i)) {
    const wide = mine.filter((v) => covers(v, s)); // a wider ban ("all", a range)
    if (!confirm("Lift the ban " + wide.join(", ") + " on " + s.id + "?")) return;
    wide.forEach((v) => delVal(s.id, "ban", v, R));
  } else addVal(s.id, "ban", r, R);
  save();
}
function more(id, d) {
  addMore(id, d, R);
  save();
}
function render() {
  document.querySelectorAll("figure").forEach((f) => {
    const i = +f.dataset.i,
      s = D.shots[i],
      e = P[s.id] || {};
    f.classList.toggle("banned", isBanned(i));
    const bits = [],
      ad = vals(s.id, "add", R),
      pr = vals(s.id, "prefer", R),
      mo = moreIn(s.id, R);
    if (ad.length) bits.push("+" + ad.length + " extra");
    if (mo) bits.push("more ×" + mo);
    if (pr.length) bits.push("prefer " + pr.join(", "));
    f.classList.toggle("changed", !isBanned(i) && bits.length > 0);
    f.querySelector(".st").textContent = bits.join(" · ");
    f.querySelector("[data-act=ban]").textContent = isBanned(i) ? "Unban" : "Ban";
  });
  const n = Object.keys(P).filter((k) => k[0] != "_").length;
  document.getElementById("count").textContent = n + " clip" + (n == 1 ? "" : "s") + " with picks";
  document.getElementById("json").textContent = JSON.stringify(P, null, 1);
  if (dlg.open) draw();
}
document.querySelector("main").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  const i = +b.closest("figure").dataset.i;
  const a = b.dataset.act;
  if (a == "ban") toggleBan(i);
  else if (a == "more") more(D.shots[i].id, 1);
  else if (a == "scrub") openShot(i);
});
document.getElementById("copy").onclick = () => {
  const t = JSON.stringify(P, null, 1);
  (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).catch(() => {
    const r = document.createRange();
    r.selectNodeContents(document.getElementById("json"));
    getSelection().removeAllRanges();
    getSelection().addRange(r);
    document.execCommand("copy");
  });
  document.getElementById("copy").textContent = "Copied ✓";
  setTimeout(() => (document.getElementById("copy").textContent = "Copy picks.json"), 1500);
};
document.getElementById("dl").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(P, null, 1)], { type: "application/json" }));
  a.download = "picks.json";
  a.click();
};
document.getElementById("clear").onclick = () => {
  if (confirm("Drop this page's changes and start again from picks.json?")) {
    store.del();
    P = JSON.parse(JSON.stringify(D.picks));
    render();
  }
};
// --- scrubber: an overview strip and a zoomed strip (as in Resolve's cut page), JKL shuttle and I/O/M marks as in
// Premiere and Final Cut, frame steps and 0-9 jumps as on YouTube ---
const dlg = document.getElementById("dlg"),
  vid = document.getElementById("vid"),
  tl = document.getElementById("tl"),
  zt = document.getElementById("zt"),
  hov = document.getElementById("hover");
const FR = 1 / 50;
let cur = null,
  zoomW = 8,
  inPt = null,
  outPt = null,
  shuttle = 0,
  rew = null,
  drag = null;
const now = () => vid.currentTime || 0;
function seek(t) {
  if (rew) stopShuttle();
  vid.currentTime = Math.min(Math.max(0, t), dur() - 0.01);
  draw();
}
function openShot(i) {
  cur = i;
  inPt = outPt = null;
  stopShuttle();
  const s = D.shots[i],
    src = D.sources[s.id] || {};
  document.getElementById("title").textContent = i + 1 + " · " + s.id + " · " + s.kind + " " + shotRange(i);
  if (src.video) {
    if (vid.getAttribute("src") !== src.video) vid.src = src.video;
    vid.style.display = "";
  } else {
    vid.removeAttribute("src");
    vid.load();
    vid.style.display = "none";
  }
  const go = () => {
    vid.currentTime = s.a;
    draw();
  };
  vid.readyState >= 1 ? go() : vid.addEventListener("loadedmetadata", go, { once: true });
  const al = document.getElementById("alts");
  al.innerHTML = "";
  (src.moments || []).forEach(([t, k, r]) => {
    const b = document.createElement("button");
    b.textContent = mmss(t) + " " + k;
    b.dataset.t = t;
    b.title = "rank " + r + " — jump there";
    b.onclick = () => seek(t - 0.8);
    al.appendChild(b);
  });
  dlg.showModal();
  draw();
}
function dur() {
  return vid.duration && isFinite(vid.duration)
    ? vid.duration
    : Math.max(
        ...D.shots.filter((x) => x.id == D.shots[cur].id).map((x) => x.b),
        ...((D.sources[D.shots[cur].id] || {}).moments || [[1]]).map((m) => m[0]),
      ) + 5;
}
function zwin() {
  const L = dur(),
    w = Math.min(zoomW, L);
  let a = drag && drag.el === zt ? drag.a : now() - w / 2;
  a = Math.min(Math.max(0, a), Math.max(0, L - w));
  return [a, a + w];
}
function marks() {
  const s = D.shots[cur],
    e = P[s.id] || {},
    L = dur(),
    out = [];
  vals(s.id, "ban", R).forEach((v) =>
    out.push(v == "all" ? ["ban", 0, L, "banned"] : ["ban", ...span(v), "banned " + v]),
  );
  vals(s.id, "prefer", R).forEach((v) => out.push(["pre", ...span(v), "prefer " + v]));
  D.shots.forEach((x, k) => {
    if (x.id == s.id && k != cur) out.push(["oth", x.a, x.b, "shot " + (k + 1)]);
  });
  out.push(["cur", s.a, s.b, "this shot"]);
  if (inPt !== null || outPt !== null) {
    const a = inPt ?? now(),
      b = outPt ?? now();
    out.push(["io", Math.min(a, b), Math.max(a, b), "in–out"]);
  }
  ((D.sources[s.id] || {}).moments || []).forEach(([t, k]) => out.push(["mom", t, null, k + " " + mmss(t)]));
  vals(s.id, "add", R).forEach((v) => out.push(["add", span(v)[0], null, "extra shot " + v]));
  return out;
}
function paint(el, a, b, ticks) {
  const W = b - a,
    pct = (x) => ((100 * (x - a)) / W).toFixed(3) + "%";
  let h = "";
  if (ticks) {
    const step = W <= 4 ? 0.2 : W <= 12 ? 0.5 : W <= 40 ? 1 : 5,
      lab = W <= 4 ? 1 : W <= 12 ? 2 : W <= 40 ? 5 : 30;
    for (let t = Math.ceil(a / step) * step; t <= b; t += step) {
      const big = Math.abs(t / lab - Math.round(t / lab)) < 1e-6;
      h += `<i class="tk${big ? " big" : ""}" style="left:${pct(t)}"></i>`;
      if (big) h += `<i class=lb style="left:${pct(t)}">${mmss(t)}</i>`;
    }
  }
  for (const [c, x0, x1, t] of marks()) {
    if ((x1 ?? x0) < a || x0 > b) continue;
    h +=
      x1 === null
        ? `<i class="${c}" title="${esc(t)}" style="left:${pct(x0)}"></i>`
        : `<i class="${c}" title="${esc(t)}" style="left:${pct(Math.max(x0, a))};width:calc(${pct(Math.min(x1, b))} - ${pct(Math.max(x0, a))} + 2px)"></i>`;
  }
  if (!ticks) {
    const [za, zb] = zwin();
    h += `<i class=win style="left:${pct(za)};width:${((100 * (zb - za)) / W).toFixed(3)}%"></i>`;
  }
  h += `<i class=ph style="left:${pct(now())}"></i>`;
  el.innerHTML = h;
}
function draw() {
  if (cur === null) return;
  const s = D.shots[cur],
    e = P[s.id] || {};
  paint(tl, 0, dur(), false);
  paint(zt, ...zwin(), true);
  {
    const t = now(),
      m = Math.floor(t / 60);
    document.getElementById("pos").textContent =
      m + ":" + (t - 60 * m).toFixed(2).padStart(5, "0") + "  f" + Math.round(t * 50);
  }
  document.getElementById("speed").textContent = rew
    ? "◀◀ " + rew.n + "×"
    : shuttle > 1
      ? "▶▶ " + shuttle + "×"
      : inPt !== null || outPt !== null
        ? "in " + (inPt === null ? "–" : mmss(inPt)) + " · out " + (outPt === null ? "–" : mmss(outPt))
        : "";
  document
    .querySelectorAll("#alts button")
    .forEach((b) => b.classList.toggle("on", Math.abs(+b.dataset.t - now()) < 0.6));
  const m = document.getElementById("marks");
  const adds = vals(s.id, "add", R);
  m.innerHTML =
    (adds.length
      ? `Extra shots (${adds.length}): ` +
        adds
          .map(
            (v, k) =>
              `<b class=jm data-t="${span(v)[0]}" style="cursor:pointer">${esc(v)}</b> <button data-k=${k} class=rm>✕</button>`,
          )
          .join(" ")
      : "No extra shots marked — press M at each moment you want.") +
    (moreIn(s.id, R) ? ` · more from this video ×${moreIn(s.id, R)}` : "") +
    (isBanned(cur) ? " · <span style='color:var(--ban)'>this shot is banned</span>" : "");
  m.querySelectorAll(".rm").forEach(
    (b) =>
      (b.onclick = () => {
        delVal(s.id, "add", adds[+b.dataset.k], R);
        save();
      }),
  );
  m.querySelectorAll(".jm").forEach((b) => (b.onclick = () => seek(+b.dataset.t)));
}
// drag to scrub: the overview maps the whole video; the zoomed strip holds its window still while you drag
function at(el, ev) {
  const r = el.getBoundingClientRect(),
    f = Math.min(Math.max((ev.clientX - r.left) / r.width, 0), 1);
  const [a, b] = el === tl ? [0, dur()] : zwin();
  return a + f * (b - a);
}
for (const el of [tl, zt]) {
  el.addEventListener("pointerdown", (ev) => {
    vid.pause();
    stopShuttle();
    el.setPointerCapture(ev.pointerId);
    const [a] = zwin();
    drag = { el, a };
    seek(at(el, ev));
  });
  el.addEventListener("pointermove", (ev) => {
    const t = at(el, ev);
    hov.style.display = "block";
    hov.textContent = mmss(t);
    hov.style.left = ev.clientX + 10 + "px";
    hov.style.top = el.getBoundingClientRect().top - 20 + "px";
    if (drag && drag.el === el) seek(t);
  });
  el.addEventListener("pointerup", () => {
    drag = null;
    draw();
  });
  el.addEventListener("pointerleave", () => {
    hov.style.display = "none";
  });
  el.addEventListener(
    "wheel",
    (ev) => {
      ev.preventDefault();
      if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) {
        seek(now() + (((ev.deltaX || ev.deltaY) / 100) * zoomW) / 8);
      } else zoom(ev.deltaY > 0 ? 1.4 : 1 / 1.4);
    },
    { passive: false },
  );
}
function zoom(f) {
  zoomW = Math.min(Math.max(zoomW * f, 1), Math.max(dur(), 2));
  draw();
}
vid.addEventListener("timeupdate", draw);
vid.addEventListener("seeked", draw);
let raf = null;
vid.addEventListener("play", () => {
  const loop = () => {
    draw();
    if (!vid.paused) raf = requestAnimationFrame(loop);
  };
  loop();
});
function stopShuttle() {
  if (rew) {
    clearInterval(rew.id);
    rew = null;
  }
  shuttle = 0;
  vid.playbackRate = +document.getElementById("rate").value;
}
function playL() {
  if (rew) {
    stopShuttle();
  }
  const base = +document.getElementById("rate").value;
  shuttle = vid.paused ? 1 : Math.min(shuttle * 2 || 2, 8);
  vid.playbackRate = base * shuttle;
  vid.play();
  draw();
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
      vid.currentTime = Math.max(0, now() - 0.04 * n);
      draw();
    }, 40),
  };
  draw();
}
function pause() {
  vid.pause();
  stopShuttle();
  draw();
}
const step = (d) => {
  pause();
  seek(now() + d);
};
function pts() {
  const s = D.shots[cur],
    e = P[s.id] || {};
  return [
    ...D.shots.filter((x) => x.id == s.id).map((x) => x.a),
    ...((D.sources[s.id] || {}).moments || []).map((m) => m[0]),
    ...vals(s.id, "add", R).map((v) => span(v)[0]),
  ].sort((a, b) => a - b);
}
function jump(dir) {
  const t = now(),
    p = pts();
  const x = dir > 0 ? p.find((v) => v > t + 0.05) : p.reverse().find((v) => v < t - 0.05);
  if (x !== undefined) seek(x);
}
document.getElementById("back").onclick = () => step(-1);
document.getElementById("fwd").onclick = () => step(1);
document.getElementById("fb").onclick = () => step(-FR);
document.getElementById("ff").onclick = () => step(FR);
document.getElementById("play").onclick = () => (vid.paused ? playL() : pause());
document.getElementById("rate").onchange = (ev) => {
  vid.playbackRate = +ev.target.value;
};
document.getElementById("zin").onclick = () => zoom(1 / 1.5);
document.getElementById("zout").onclick = () => zoom(1.5);
document.getElementById("goshot").onclick = () => seek(D.shots[cur].a);
document.getElementById("helpb").onclick = () => document.getElementById("help").scrollIntoView({ block: "nearest" });
document.getElementById("close").onclick = () => dlg.close();
dlg.addEventListener("close", () => {
  pause();
  hov.style.display = "none";
});
function addHere() {
  const id = D.shots[cur].id,
    e = ent(id);
  e.add = e.add || [];
  const n = now(),
    m = Math.floor(n / 60);
  const t = m + ":" + (n - 60 * m).toFixed(2).padStart(5, "0"); // frame-accurate
  const near = e.add.find((v) => Math.abs(span(v)[0] - now()) < 0.15);
  addVal(id, "add", near === undefined ? t : near, R);
  e.add.sort((a, b) => span(a)[0] - span(b)[0]);
  save();
}
function delHere() {
  const id = D.shots[cur].id;
  let k = null,
    best = 0.6;
  vals(id, "add", R).forEach((v) => {
    const d = Math.abs(span(v)[0] - now());
    if (d < best) {
      best = d;
      k = v;
    }
  });
  if (k !== null) {
    delVal(id, "add", k, R);
    save();
  }
}
function io() {
  if (inPt === null && outPt === null) return null;
  const a = inPt ?? now(),
    b = outPt ?? now();
  return Math.abs(b - a) < 0.2 ? null : [Math.min(a, b), Math.max(a, b)];
}
function swapHere() {
  const s = D.shots[cur],
    r = io() || [Math.max(0, now() - 0.6), now() + 0.6];
  addVal(s.id, "ban", shotRange(cur), R);
  addVal(s.id, "prefer", range(...r), R);
  inPt = outPt = null;
  save();
}
function preferIO() {
  const r = io();
  if (!r) return;
  addVal(D.shots[cur].id, "prefer", range(...r), R);
  inPt = outPt = null;
  save();
}
document.getElementById("add").onclick = addHere;
document.getElementById("swap").onclick = swapHere;
document.getElementById("pref").onclick = preferIO;
document.getElementById("bandlg").onclick = () => toggleBan(cur);
document.getElementById("moredlg").onclick = () => more(D.shots[cur].id, 1);
document.getElementById("lessdlg").onclick = () => more(D.shots[cur].id, -1);
window.addEventListener(
  "keydown",
  (ev) => {
    if (!dlg.open || ev.target.tagName == "SELECT" || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const k = ev.key,
      K = k.toLowerCase();
    let done = true;
    if (k == " " || K == "k") vid.paused && !rew ? playL() : pause();
    else if (K == "l") playL();
    else if (K == "j") playJ();
    else if (k == "ArrowLeft" || k == ",") step(ev.shiftKey ? -1 : -FR);
    else if (k == "ArrowRight" || k == ".") step(ev.shiftKey ? 1 : FR);
    else if (k == "Home") seek(0);
    else if (k == "End") seek(dur());
    else if (/^[0-9]$/.test(k)) seek((dur() * +k) / 10);
    else if (K == "m" || K == "a") addHere();
    else if (k == "Backspace" || k == "Delete") delHere();
    else if (k == "[") jump(-1);
    else if (k == "]") jump(1);
    else if (K == "i") {
      inPt = now();
      draw();
    } else if (K == "o") {
      outPt = now();
      draw();
    } else if (K == "x") {
      inPt = outPt = null;
      draw();
    } else if (K == "s") swapHere();
    else if (K == "p") preferIO();
    else if (K == "b") toggleBan(cur);
    else if (k == "=" || k == "+") zoom(1 / 1.5);
    else if (k == "-" || k == "_") zoom(1.5);
    else if (k == "?") document.getElementById("help").scrollIntoView({ block: "nearest" });
    else done = false;
    if (done) {
      ev.preventDefault();
      ev.stopPropagation();
    }
  },
  true,
);
