// A reel's review page: every shot of the cut as a card (ban it, ask for more from its clip), and a scrubber on the
// clip a shot comes from -- an overview strip and a zoomed strip (as on Resolve's cut page), J/K/L shuttle and I/O/M
// marks (as in Premiere and Final Cut) -- to mark extra shots, swap the shot for another moment or prefer a range.
// Picks made here are for this page's reel (D.reel); the other reels' are left alone.
App.ls = "picks:" + D.key + ":";
const R = D.reel || "main";
const shotRange = (i) => rkey(D.shots[i].a, D.shots[i].b);
const covers = (v, s) => {
  if (v == "all") return true;
  const [a, b] = span(v);
  return a <= s.a + 0.06 && b >= s.b - 0.06;
};
const isBanned = (i) => vals(D.shots[i].id, "ban", R).some((v) => covers(v, D.shots[i]));
function toggleBan(i) {
  const s = D.shots[i], wide = vals(s.id, "ban", R).filter((v) => covers(v, s));
  if (wide.length) {
    const big = wide.filter((v) => v == "all" || span(v)[1] - span(v)[0] > s.b - s.a + 0.5);   // a ban of more than this shot
    if (big.length && !confirm(`Lift the ban ${big.join(", ")} on ${s.id}? It covers more than this shot.`)) return;
    wide.forEach((v) => delVal(s.id, "ban", v, R));
    save();
    toast(`Shot ${i + 1} is back in`);
  } else {
    addVal(s.id, "ban", shotRange(i), R);
    save();
    toast(`Shot ${i + 1} banned for ${rlab(R)}: the next re-plan puts something else there — Ctrl+Z undoes`);
  }
}
function more(id, d) {
  addMore(id, d, R);
  save();
  toast(moreIn(id, R) ? `${id}: ${moreIn(id, R)} more shot${moreIn(id, R) == 1 ? "" : "s"} asked for` : `${id}: no extra shots asked for`);
}
function render() {
  document.querySelectorAll("figure").forEach((f) => {
    const i = +f.dataset.i, s = D.shots[i], ban = isBanned(i);
    const bits = [], pr = vals(s.id, "prefer", R), mo = moreIn(s.id, R);
    if (ban) bits.push("banned");
    if (mo) bits.push("more ×" + mo);
    if (pr.length) bits.push("prefer " + pr.join(", "));
    f.classList.toggle("banned", ban);
    f.classList.toggle("changed", !ban && (mo > 0 || pr.length > 0));
    f.querySelector(".st").textContent = bits.join(" · ");
    f.querySelector("[data-act=ban]").textContent = ban ? "Unban" : "Ban";
    f.querySelector("[data-act=less]").hidden = !mo;
  });
  const n = D.shots.filter((s, i) => isBanned(i)).length;
  $("count").textContent = n ? `${n} banned` : "";
  if (dlg.open) draw();
}
App.render = render;
App.onBoot = render;
document.querySelector("main").addEventListener("click", (ev) => {
  const f = ev.target.closest("figure");
  if (!f) return;
  const i = +f.dataset.i, b = ev.target.closest("button"), a = b ? b.dataset.act : ev.target.tagName == "IMG" ? "scrub" : null;
  if (a == "ban") toggleBan(i);
  else if (a == "more") more(D.shots[i].id, 1);
  else if (a == "less") more(D.shots[i].id, -1);
  else if (a == "scrub") openShot(i);
});

// --- the scrubber ----------------------------------------------------------------------------------------------------
const dlg = $("dlg"), vid = $("vid"), tl = $("tl"), zt = $("zt"), hov = $("hover");
const FR = 1 / 50;
let cur = null, zoomW = 8, inPt = null, outPt = null, shuttle = 0, rew = null, drag = null;
// (a clip that cannot play here keeps the time it was sent to, so marks still land where the strip says)
const now = () => (vid.readyState >= 1 ? vid.currentTime : vid._at) || 0;
function seek(t) {
  if (rew) stopShuttle();
  vid._at = Math.min(Math.max(0, t), dur() - 0.01);
  try {
    vid.currentTime = vid._at;
  } catch (e) {}
  draw();
}
function openShot(i) {
  cur = i;
  inPt = outPt = null;
  stopShuttle();
  const s = D.shots[i], src = D.sources[s.id] || {};
  $("title").textContent = `${i + 1} · ${s.id} · ${s.kind} · ${fmt(s.a)}–${fmt(s.b)}`;
  if (src.video) {
    if (vid.getAttribute("src") !== src.video) vid.src = src.video;
    vid.style.display = "";
  } else {
    vid.removeAttribute("src");
    vid.load();
    vid.style.display = "none";
  }
  vid._at = s.a;
  const go = () => {
    if (cur === i) vid.currentTime = vid._at;
    draw();
  };
  vid.readyState >= 1 ? go() : vid.addEventListener("loadedmetadata", go, { once: true });
  $("alts").innerHTML = (src.moments || []).map(([t, k, r]) => `<button data-t=${t} title="rank ${r} — jump there">${short(t)} ${esc(k)}</button>`).join("");
  dlg.showModal();
  draw();
}
$("alts").onclick = (ev) => {
  const b = ev.target.closest("button");
  if (b) seek(+b.dataset.t - 0.8);
};
function dur() {
  if (vid.duration && isFinite(vid.duration)) return vid.duration;
  const id = D.shots[cur].id;
  return Math.max(...D.shots.filter((x) => x.id == id).map((x) => x.b), ...((D.sources[id] || {}).moments || [[1]]).map((m) => m[0])) + 5;
}
function zwin() {
  const Ld = dur(), w = Math.min(zoomW, Ld);
  let a = drag && drag.el === zt ? drag.a : now() - w / 2;
  a = Math.min(Math.max(0, a), Math.max(0, Ld - w));
  return [a, a + w];
}
function marks() {
  const s = D.shots[cur], Ld = dur(), out = [];
  vals(s.id, "ban", R).forEach((v) => out.push(v == "all" ? ["ban", 0, Ld, "banned"] : ["ban", ...span(v), "banned " + v]));
  vals(s.id, "prefer", R).forEach((v) => out.push(["pre", ...span(v), "prefer " + v]));
  D.shots.forEach((x, k) => {
    if (x.id == s.id && k != cur) out.push(["oth", x.a, x.b, "shot " + (k + 1)]);
  });
  out.push(["cur", s.a, s.b, "this shot"]);
  if (inPt !== null || outPt !== null) {
    const a = inPt ?? now(), b = outPt ?? now();
    out.push(["io", Math.min(a, b), Math.max(a, b), "in–out"]);
  }
  ((D.sources[s.id] || {}).moments || []).forEach(([t, k]) => out.push(["mom", t, null, k + " " + short(t)]));
  vals(s.id, "add", R).forEach((v) => (isRange(v) ? out.push(["pre", ...span(v), "your pick " + v]) : out.push(["add", span(v)[0], null, "extra shot " + v])));
  return out;
}
function paint(el, a, b, ticks) {
  const W = b - a, pct = (x) => ((100 * (x - a)) / W).toFixed(3) + "%";
  let h = "";
  if (ticks) {
    const stp = W <= 4 ? 0.2 : W <= 12 ? 0.5 : W <= 40 ? 1 : 5, lab = W <= 4 ? 1 : W <= 12 ? 2 : W <= 40 ? 5 : 30;
    for (let t = Math.ceil(a / stp) * stp; t <= b; t += stp) {
      const big = Math.abs(t / lab - Math.round(t / lab)) < 1e-6;
      h += `<i class="tk${big ? " big" : ""}" style="left:${pct(t)}"></i>`;
      if (big) h += `<i class=lb style="left:${pct(t)}">${short(t)}</i>`;
    }
  }
  for (const [c, x0, x1, t] of marks()) {
    if ((x1 ?? x0) < a || x0 > b) continue;
    h += x1 === null ? `<i class="${c}" title="${esc(t)}" style="left:${pct(x0)}"></i>`
      : `<i class="${c}" title="${esc(t)}" style="left:${pct(Math.max(x0, a))};width:calc(${pct(Math.min(x1, b))} - ${pct(Math.max(x0, a))} + 2px)"></i>`;
  }
  if (!ticks) {
    const [za, zb] = zwin();
    h += `<i class=win style="left:${pct(za)};width:${((100 * (zb - za)) / W).toFixed(3)}%"></i>`;
  }
  el.innerHTML = h + `<i class=ph style="left:${pct(now())}"></i>`;
}
function draw() {
  if (cur === null) return;
  const s = D.shots[cur], t = now(), adds = vals(s.id, "add", R).filter((v) => !isRange(v)), mo = moreIn(s.id, R);
  paint(tl, 0, dur(), false);
  paint(zt, ...zwin(), true);
  $("pos").textContent = fmt(t) + "  f" + Math.round(t * 50);
  $("speed").textContent = rew ? "◀◀ " + rew.n + "×" : shuttle > 1 ? "▶▶ " + shuttle + "×"
    : inPt !== null || outPt !== null ? "in " + (inPt === null ? "–" : fmt(inPt)) + " · out " + (outPt === null ? "–" : fmt(outPt)) : "";
  $("play").textContent = vid.paused && !rew ? "▶ Play" : "❚❚ Pause";
  $("bandlg").firstChild.textContent = isBanned(cur) ? "Unban this shot " : "Ban this shot ";
  $("lessdlg").disabled = !mo;
  document.querySelectorAll("#alts button").forEach((b) => b.classList.toggle("on", Math.abs(+b.dataset.t - t) < 0.6));
  $("marks").innerHTML = (adds.length
    ? `Extra shots (${adds.length}): ` + adds.map((v, k) => `<b data-t="${span(v)[0]}">${esc(v)}</b> <button data-k=${k} title="remove">✕</button>`).join(" ")
    : "No extra shots marked — press M at each moment you want.") +
    (mo ? ` · more from this video ×${mo}` : "") + (isBanned(cur) ? ` · <span style="color:var(--ban)">this shot is banned</span>` : "");
}
$("marks").onclick = (ev) => {
  const s = D.shots[cur], b = ev.target.closest("button"), j = ev.target.closest("b");
  if (b) {
    delVal(s.id, "add", vals(s.id, "add", R).filter((v) => !isRange(v))[+b.dataset.k], R);
    save();
  } else if (j) seek(+j.dataset.t);
};
// drag to scrub: the overview maps the whole video; the zoomed strip holds its window still while you drag
function at(el, ev) {
  const r = el.getBoundingClientRect(), f = Math.min(Math.max((ev.clientX - r.left) / r.width, 0), 1), [a, b] = el === tl ? [0, dur()] : zwin();
  return a + f * (b - a);
}
for (const el of [tl, zt]) {
  el.addEventListener("pointerdown", (ev) => {
    vid.pause();
    stopShuttle();
    el.setPointerCapture(ev.pointerId);
    drag = { el, a: zwin()[0] };
    seek(at(el, ev));
  });
  el.addEventListener("pointermove", (ev) => {
    const t = at(el, ev);
    hov.style.display = "block";
    hov.textContent = fmt(t);
    hov.style.left = ev.clientX + 10 + "px";
    hov.style.top = el.getBoundingClientRect().top - 20 + "px";
    if (drag && drag.el === el) seek(t);
  });
  el.addEventListener("pointerup", () => {
    drag = null;
    draw();
  });
  el.addEventListener("pointerleave", () => (hov.style.display = "none"));
  el.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) seek(now() + (((ev.deltaX || ev.deltaY) / 100) * zoomW) / 8);
    else zoom(ev.deltaY > 0 ? 1.4 : 1 / 1.4);
  }, { passive: false });
}
function zoom(f) {
  zoomW = Math.min(Math.max(zoomW * f, 1), Math.max(dur(), 2));
  draw();
}
for (const e of ["timeupdate", "seeked", "pause"]) vid.addEventListener(e, draw);
vid.addEventListener("play", () => {
  const f = () => {
    draw();
    if (!vid.paused) requestAnimationFrame(f);
  };
  f();
});
function stopShuttle() {
  if (rew) {
    clearInterval(rew.id);
    rew = null;
  }
  shuttle = 0;
  vid.playbackRate = +$("rate").value;
}
function playL() {
  if (rew) stopShuttle();
  shuttle = vid.paused ? 1 : Math.min(shuttle * 2 || 2, 8);
  vid.playbackRate = +$("rate").value * shuttle;
  vid.play().catch(() => toast("This clip cannot play here", "warn"));
  draw();
}
function playJ() {
  vid.pause();
  shuttle = 0;
  const n = rew ? Math.min(rew.n * 2, 8) : 1;
  if (rew) clearInterval(rew.id);
  rew = { n, id: setInterval(() => {
    if (now() <= 0) return stopShuttle();
    vid.currentTime = Math.max(0, now() - 0.04 * n);
    draw();
  }, 40) };
  draw();
}
function pause() {
  vid.pause();
  stopShuttle();
  draw();
}
const stepBy = (d) => {
  pause();
  seek(now() + d);
};
function pts() {
  const s = D.shots[cur];
  return [...D.shots.filter((x) => x.id == s.id).map((x) => x.a), ...((D.sources[s.id] || {}).moments || []).map((m) => m[0]),
    ...vals(s.id, "add", R).map((v) => span(v)[0])].sort((a, b) => a - b);
}
function jump(dir) {
  const t = now(), p = pts(), x = dir > 0 ? p.find((v) => v > t + 0.05) : p.reverse().find((v) => v < t - 0.05);
  if (x !== undefined) seek(x);
}
function addHere() {
  const id = D.shots[cur].id, near = vals(id, "add", "all").find((v) => !isRange(v) && Math.abs(span(v)[0] - now()) < 0.15);
  const fresh = addVal(id, "add", near === undefined ? fmt(now()) : near, R);
  ent(id).add.sort((a, b) => span(a)[0] - span(b)[0]);
  save();
  toast(fresh ? `Extra shot marked at ${fmt(now())} for ${rlab(R)}` : "Already marked here", fresh ? "ok" : "warn");
}
function delHere() {
  const id = D.shots[cur].id;
  let k = null, best = 0.6;
  vals(id, "add", R).filter((v) => !isRange(v)).forEach((v) => {
    const d = Math.abs(span(v)[0] - now());
    if (d < best) [best, k] = [d, v];
  });
  if (k === null) return toast("No marker at the playhead", "warn");
  delVal(id, "add", k, R);
  save();
  toast("Marker removed");
}
function io() {
  if (inPt === null && outPt === null) return null;
  const a = inPt ?? now(), b = outPt ?? now();
  return Math.abs(b - a) < 0.2 ? null : [Math.min(a, b), Math.max(a, b)];
}
function swapHere() {
  const s = D.shots[cur], r = io() || [Math.max(0, now() - 0.6), now() + 0.6];
  addVal(s.id, "ban", shotRange(cur), R);
  addVal(s.id, "prefer", rkey(...r), R);
  inPt = outPt = null;
  save();
  toast(`Shot ${cur + 1} banned; ${fmt(r[0])}–${fmt(r[1])} preferred instead — Ctrl+Z undoes`);
}
function preferIO() {
  const r = io();
  if (!r) return toast("Mark an in (I) and an out (O) first", "warn");
  addVal(D.shots[cur].id, "prefer", rkey(...r), R);
  inPt = outPt = null;
  save();
  toast(`${fmt(r[0])}–${fmt(r[1])} preferred`);
}
$("back").onclick = () => stepBy(-1);
$("fwd").onclick = () => stepBy(1);
$("fb").onclick = () => stepBy(-FR);
$("ff").onclick = () => stepBy(FR);
$("play").onclick = () => (vid.paused && !rew ? playL() : pause());
$("rate").onchange = (ev) => {
  vid.playbackRate = +ev.target.value;
  ev.target.blur();
};
$("zin").onclick = () => zoom(1 / 1.5);
$("zout").onclick = () => zoom(1.5);
$("goshot").onclick = () => seek(D.shots[cur].a);
$("close").onclick = () => dlg.close();
dlg.addEventListener("close", () => {
  pause();
  hov.style.display = "none";
});
$("add").onclick = addHere;
$("swap").onclick = swapHere;
$("pref").onclick = preferIO;
$("bandlg").onclick = () => toggleBan(cur);
$("moredlg").onclick = () => more(D.shots[cur].id, 1);
$("lessdlg").onclick = () => more(D.shots[cur].id, -1);
addEventListener("keydown", (ev) => {
  if (!dlg.open || $("help").open || typing(ev) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = ev.key, K = k.toLowerCase();
  if (k == " " || K == "k") vid.paused && !rew ? playL() : pause();
  else if (K == "l") playL();
  else if (K == "j") playJ();
  else if (k == "ArrowLeft" || k == ",") stepBy(ev.shiftKey ? -1 : -FR);
  else if (k == "ArrowRight" || k == ".") stepBy(ev.shiftKey ? 1 : FR);
  else if (k == "Home") seek(0);
  else if (k == "End") seek(dur());
  else if (/^[0-9]$/.test(k)) seek((dur() * +k) / 10);
  else if (K == "m" || K == "a") addHere();
  else if (k == "Backspace" || k == "Delete") delHere();
  else if (k == "[") jump(-1);
  else if (k == "]") jump(1);
  else if (K == "i") (inPt = now()), draw();
  else if (K == "o") (outPt = now()), draw();
  else if (K == "x") (inPt = outPt = null), draw();
  else if (K == "s") swapHere();
  else if (K == "p") preferIO();
  else if (K == "b") toggleBan(cur);
  else if (k == "=" || k == "+") zoom(1 / 1.5);
  else if (k == "-" || k == "_") zoom(1.5);
  else return;
  ev.preventDefault();
  ev.stopPropagation();
}, true);
boot();
