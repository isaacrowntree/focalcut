// Cut & picks: the reel's shots in the order they play, the editor's picks among them. The list is on the left; the
// shot in hand plays on the right over its own strip, where it is trimmed, slipped, starred, given to reels, made the
// opening or the ending, or dropped.
// Each pick is matched to the reel shot that uses it; picks the planner has not placed yet come after, in the order
// they were filmed. A change keeps a pick's place until the next re-plan.
const pv = $("pv"), PT = $("pstrip");
let order = [], pk = null, mode = null, pa = 0, pb = 1, pdrag = null;
const place = new Map();
const pkey = (id, a) => id + "|" + a.toFixed(2);
let pfmt = ls.get("pfmt") || "L";
function shotOf(c, a, b) {
  for (const [f, arr] of pfmt == "P" ? [["P", c.P], ["L", c.L]] : [["L", c.L], ["P", c.P]]) {
    let best = null, ovl = 0;
    arr.forEach(([x, y, n]) => {
      const o = Math.min(b, y) - Math.max(a, x);
      if (o > ovl + 1e-6) [ovl, best] = [o, n];
    });
    if (best !== null && ovl >= Math.min(0.3, (b - a) / 2)) return { fmt: f, n: best };
  }
  return null;
}
// the chosen cut's shots come in too (k -1: not a pick yet): changing one makes it a pick, removing one bans its stretch
function allSels() {
  const out = [];
  D.clips.forEach((c, i) => sels(c.id).forEach(([a, b], k) => {
    if (!inReel(c.id, a, b, reel)) return;
    const key = pfmt + "|" + pkey(c.id, a);
    if (!place.has(key)) place.set(key, shotOf(c, a, b));
    out.push({ c, i, k, a, b, shot: place.get(key) });
  }));
  if (pfmt != "picks") {
    const have = new Set(out.filter((x) => x.shot && x.shot.fmt == pfmt).map((x) => x.shot.n));
    D.clips.forEach((c, i) => {
      const bans = banSpans(c.id);
      c[pfmt].forEach(([a, b, n]) => {
        if (!have.has(n) && !isOut(bans, a, b)) out.push({ c, i, k: -1, a, b, shot: { fmt: pfmt, n }, cut: true });
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
  const on = !chosen.has(ckey(x)), [lo, hi] = range && lastCk >= 0 ? [Math.min(lastCk, j), Math.max(lastCk, j)] : [j, j];
  for (let k = lo; k <= hi; k++) if (order[k]) on ? chosen.add(ckey(order[k])) : chosen.delete(ckey(order[k]));
  lastCk = j;
  selview();
}
// a pick goes (from the reel being worked on); a shot of the cut is banned, so the next re-plan puts something else there
function dropOne(x) {
  if (x.k >= 0) return removeSel(x.c.id, x.a, x.b, reel);
  addVal(x.c.id, "ban", rkey(x.a, x.b), reel);
  return [];
}
function drop(xs) {
  if (pk && xs.some((x) => x.c.id == pk.id && Math.abs(x.a - pk.a) < 0.002)) {
    pk = null;
    pv.pause();
  }
  const picks = xs.filter((x) => x.k >= 0).length, cuts = xs.length - picks;
  xs.forEach(dropOne);
  save();
  toast([picks && `${picks} pick${picks == 1 ? "" : "s"} removed${reel == "all" ? "" : " from " + rlab(reel)}`,
    cuts && `${cuts} planner's shot${cuts == 1 ? "" : "s"} banned`].filter(Boolean).join(" · ") + " — Ctrl+Z undoes");
}
function tickAll() {
  order.forEach((x) => x.cut && chosen.add(ckey(x)));
  selview();
}
function untick() {
  chosen.clear();
  lastCk = -1;
  selview();
}
function dropTicked() {
  const xs = order.filter((x) => chosen.has(ckey(x)));
  if (!xs.length) return;
  chosen.clear();
  lastCk = -1;
  drop(xs);
}
$("bulkall").onclick = tickAll;
$("bulkclr").onclick = untick;
$("bulkdrop").onclick = dropTicked;
const pj = () => (pk ? order.findIndex((x) => x.c.id == pk.id && Math.abs(x.a - pk.a) < 0.002) : -1);
const PX = () => order[pj()];
function selview() {
  if (curView != 2) return;
  order = allSels();
  const live = new Set(order.map(ckey));
  [...chosen].forEach((k) => live.has(k) || chosen.delete(k));
  $("bulk").classList.toggle("on", chosen.size > 0);
  $("cards").classList.toggle("picking", chosen.size > 0);
  $("bulkn").textContent = chosen.size + " ticked";
  const tot = order.reduce((s, x) => s + x.b - x.a, 0), j = pj(), placed = order.filter((x) => x.shot).length;
  const np = order.filter((x) => !x.cut).length, nc = order.length - np, which = reel == "all" ? "main" : reel;
  $("pfmts").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.f == pfmt));
  $("bulkall").hidden = pfmt == "picks" || !nc;
  $("seltitle").textContent = pfmt == "picks" ? `Your ${np} pick${np == 1 ? "" : "s"}${reel == "all" ? "" : " for " + rlab(reel)}`
    : `${rlab(which)} ${pfmt == "P" ? "portrait" : "landscape"} cut · ${nc + order.filter((x) => !x.cut && x.shot && x.shot.fmt == pfmt).length} shots, ${np} of them your picks`;
  $("seltot").textContent = short(tot) + " of footage, in cut order" +
    (order.length > placed ? ` · ${order.length - placed} not in the cut yet (placed by the next re-plan)` : "");
  $("seq").innerHTML = order.map((x, k) =>
    `<i data-j=${k} class="${k == j ? "act" : ""}${chosen.has(ckey(x)) ? " chk" : ""}${x.shot ? "" : " new"}${x.cut ? " cut" + (pfmt == "P" ? " p" : "") : ""}" style="flex:${(x.b - x.a).toFixed(2)}" title="${k + 1} · ${esc(x.c.id)} ${fmt(x.a)}–${fmt(x.b)}"></i>`).join("");
  // (a card's picture is cut from the clip's filmstrip at the card's own width)
  const cw = $("cards").clientWidth - 16, cols = Math.max(1, Math.floor((cw + 8) / 158)), CARD_W = (cw - (cols - 1) * 8) / cols - 2, CARD_H = (CARD_W * TH) / TW;
  let h = "", sep = false;
  order.forEach((x, k) => {
    if (!x.shot && !sep) {
      sep = true;
      h += `<div class=catgrp>Not in the cut yet — placed by the next re-plan</div>`;
    }
    const st = !x.cut && starsOf(x.c.id, x.a, x.b), only = !x.cut && reel == "all" && onlyOf(x.c.id, x.a, x.b), role = roleOf(x.c.id, x.a, x.b, reel);
    const fr = pfmt == "P" && x.shot ? framingOf(x.c.id, x.a, x.b) : "auto";
    h += `<div class="card${k == j ? " act" : ""}${x.cut ? " cut" : ""}${chosen.has(ckey(x)) ? " chk" : ""}" data-j=${k}>
 <div class=th style="${thumbAt(x.c, x.a, CARD_H, CARD_W)}"><span>${k + 1} · ${(x.b - x.a).toFixed(1)}s</span><input type=checkbox class=ck data-ck=${k} title="tick (shift-click: a range)"${chosen.has(ckey(x)) ? " checked" : ""}></div>
 <div class=bd><b>${x.shot ? (x.shot.fmt == "L" ? "Shot " : "Portrait ") + x.shot.n : "New"}</b> ${x.cut ? "" : "<span class=tag>pick</span> "}${st ? `<span class=st>${"★".repeat(st)}</span> ` : ""}${
      role ? `<span class="tag role">${role == "open" ? "opens" : "ends"}</span> ` : ""}${only ? `<span class=tag>${esc(only.map(rlab).join(" + "))} only</span> ` : ""}${
      fr != "auto" ? `<span class="badge ${fr}">${fr == "pan" ? "panned" : "following"}</span>` : ""}
 <small class=nm>${esc(x.c.id)}</small><small>${fmt(x.a)} – ${fmt(x.b)} · ${esc(x.c.clock)}</small></div>
 <button data-del=${k} class=danger title="${x.cut ? "take this shot out of the cut (bans its stretch)" : "remove this pick"}">✕</button></div>`;
  });
  $("cards").innerHTML = h || `<p class=emptystate>${pfmt == "picks" ? "No picks yet — mark ranges in Clips." : "No cut planned for this reel yet."}</p>`;
  const px = PX();
  $("ped").classList.toggle("on", !!px);
  $("pedempty").hidden = !!px;
  $("pstars").innerHTML = px && px.k >= 0 ? starBar(px.c.id, px.a, px.b, "") : px ? `<span title="star it and it becomes your pick">${starBar(px.c.id, px.a, px.b, "")}</span>` : "";
  $("preels").innerHTML = px && px.k >= 0 && RS.length > 1
    ? "In " + RS.map((r) => `<button data-reel="${esc(r.id)}" class="${inReel(px.c.id, px.a, px.b, r.id) ? "on" : ""}" title="put this pick in / take it out of the ${esc(r.label)} reel">${esc(r.label)}</button>`).join("")
    : "";
  $("pdel").textContent = px && px.cut ? "✕ Ban shot" : "✕ Remove";
  if (px) pdraw();
}
$("cards").onclick = (ev) => {
  const c = ev.target.closest("[data-ck]");
  if (c) {
    ev.preventDefault();
    return tick(+c.dataset.ck, ev.shiftKey);
  }
  const del = ev.target.closest("[data-del]"), cd = ev.target.closest(".card");
  if (del) return drop([order[+del.dataset.del]]);
  if (!cd) return;
  if (ev.shiftKey || ev.ctrlKey || ev.metaKey) return tick(+cd.dataset.j, ev.shiftKey);
  pick(+cd.dataset.j, false);
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
  stripOf(x.c, pdraw);
  load(pv, x.c, x.a);
  if (play) pv.play().catch(() => {});
  selview();
  const card = $("cards").querySelector(`.card[data-j="${j}"]`);
  if (card) card.scrollIntoView({ block: "nearest" });
}
const pstep = (d) => pick(clamp(pj() + d, 0, order.length - 1), !pv.paused);
// change the shot's in / out (lengthen, trim, slip): a planner's shot becomes a pick; it keeps its place in the cut
function edit(a, b) {
  const x = PX();
  if (!x || b - a < 0.1) return;
  const Ld = x.c.dur;
  if (a < 0) [a, b] = [0, b - a];
  if (b > Ld) [a, b] = [a - (b - Ld), Ld];
  a = Math.max(0, a);
  if (x.k >= 0 ? !moveSel(x.c.id, x.a, x.b, a, b) : !addSel(x.c.id, a, b, reel)) return;
  place.set(pfmt + "|" + pkey(x.c.id, a), x.shot);
  pk = { id: x.c.id, a };
  if (a < pa || b > pb) {
    const ctx = Math.max(2, (b - a) * 0.75);
    pa = Math.max(0, a - ctx);
    pb = Math.min(Ld, b + ctx);
  }
  save();
}
const pnow = () => tOf(pv);
function pdraw() {
  const x = PX();
  if (!x || curView != 2) return;
  const W = pb - pa, t = pnow(), Xp = (v) => X(v, pa, pb);
  film(PT.querySelector("canvas"), pa, pb, 64, x.c, stripOf(x.c));
  $("pt").querySelectorAll(":scope>i,#pstrip>i").forEach((e) => e.remove());
  let [a, b] = pdrag && pdrag.cur ? pdrag.cur : [x.a, x.b];
  let h = `<i class=dim style="${wspan(pa, a, pa, pb)}"></i><i class=dim style="${wspan(b, pb, pa, pb)}"></i>`;
  sels(x.c.id).forEach(([u, v], k) => {
    if (k != x.k && v > pa && u < pb) h += `<i class=oth style="${wspan(u, v, pa, pb)}" title="another pick"></i>`;
  });
  h += `<i class="blk act" data-k=0 style="${wspan(a, b, pa, pb)}"><i class="h l"></i>${(b - a).toFixed(2)} s<i class="h r"></i></i>`;
  fkeys(x.c.id).forEach(([k]) => {
    if (k >= pa && k <= pb) h += `<i class=fk style="left:${Xp(k)}" title="your pan of the portrait crop at ${fmt(k)}"></i>`;
  });
  folkeys(x.c.id).forEach(([k]) => {
    if (k >= pa && k <= pb) h += `<i class="fk fl" style="left:${Xp(k)}" title="following a person from ${fmt(k)}"></i>`;
  });
  if (t >= pa && t <= pb) h += `<i class=ph style="left:${Xp(t)}"></i>`;
  PT.insertAdjacentHTML("beforeend", h);
  const stp = [0.1, 0.2, 0.5, 1, 2, 5, 10].find((s) => W / s <= 40) || 30;
  let r = "";
  for (let v = Math.ceil(pa / stp) * stp; v <= pb; v += stp) r += `<i class=tk style="left:${Xp(v)}"></i>`;
  $("pruler").innerHTML = r + `<i class=lb style="left:${Xp(a)}">in ${fmt(a)}</i><i class=lb style="left:${Xp(b)}">out ${fmt(b)}</i>`;
  $("plen").textContent = (b - a).toFixed(2) + " s";
  const ro = roleOf(x.c.id, x.a, x.b, reel);
  $("pfirst").classList.toggle("on", ro == "open");
  $("plast").classList.toggle("on", ro == "end");
  $("pvtc").textContent = `${pj() + 1} / ${order.length} · ${x.c.id} · ${fmt(t)}`;
  $("pplay").textContent = pv.paused ? "▶ Play" : "❚❚ Pause";
  $("ploop").classList.toggle("on", mode == "loop");
  stillDraw($("pstill"), $("pvwrap"), pv, x.c, t);
  frameDraw($("pfr"), $("pvwrap"), pv, x.c, x.a, x.b, t);
}
PT.addEventListener("pointerdown", (ev) => {
  const x = PX();
  if (!x || ev.button) return;
  pv.pause();
  PT.setPointerCapture(ev.pointerId);
  const t = tAt(PT, ev, pa, pb), blk = ev.target.closest(".blk"), cl = ev.target.classList;
  pdrag = blk ? { k: cl.contains("l") ? "l" : cl.contains("r") ? "r" : "slip", t0: t, o: [x.a, x.b], cur: [x.a, x.b] } : { k: "scrub" };
  if (!blk) goTo(pv, t);
  pdraw();
});
PT.addEventListener("pointermove", (ev) => {
  const x = PX();
  if (!x) return;
  const t = tAt(PT, ev, pa, pb);
  showHover(ev, x.c, t, PT);
  if (!pdrag) return;
  if (pdrag.k == "scrub") {
    goTo(pv, t);
    return pdraw();
  }
  let [a, b] = pdrag.o;
  if (pdrag.k == "slip") {
    const w = b - a;
    a = clamp(a + t - pdrag.t0, 0, x.c.dur - w);
    b = a + w;
    goTo(pv, a);
  } else if (pdrag.k == "l") goTo(pv, (a = clamp(t, 0, b - 0.1)));
  else goTo(pv, (b = clamp(t, a + 0.1, x.c.dur)));
  pdrag.cur = [a, b];
  pdraw();
});
PT.addEventListener("pointerup", () => {
  const d = pdrag;
  pdrag = null;
  if (d && d.cur && (d.cur[0] !== d.o[0] || d.cur[1] !== d.o[1])) edit(...d.cur);
  else pdraw();
});
PT.addEventListener("pointercancel", () => {
  pdrag = null;
  pdraw();
});
PT.addEventListener("pointerleave", hideHover);
PT.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const x = PX();
  if (!x) return;
  const R = PT.getBoundingClientRect(), f = (ev.clientX - R.left) / R.width, at = pa + f * (pb - pa);
  const w = clamp((pb - pa) * (ev.deltaY > 0 ? 1.25 : 0.8), 1, x.c.dur);
  pa = clamp(at - f * w, 0, x.c.dur - w);
  pb = pa + w;
  pdraw();
}, { passive: false });
pv.addEventListener("timeupdate", () => {
  const x = PX();
  if (!x || curView != 2) return;
  if (!pv.paused && pnow() >= x.b - 0.02) {
    if (mode == "loop") pv.currentTime = x.a;
    else if (mode == "all") order[pj() + 1] ? pick(pj() + 1, true) : stopAll();
    else pv.pause();
  }
  pdraw();
});
for (const e of ["seeked", "pause", "play", "loadeddata", "error"]) pv.addEventListener(e, () => pdraw());
pv.addEventListener("play", () => {
  const f = () => {
    if (curView != 2) return;
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
  if (pnow() < x.a - 0.05 || pnow() >= x.b - 0.05) goTo(pv, x.a);
  pv.play().catch(() => toast("This clip cannot play here — the filmstrip stands in; scrub the strip", "warn"));
}
function stopAll() {
  pv.pause();
  mode = null;
  pdraw();
}
const loopPick = () => {
  mode = mode == "loop" ? "one" : "loop";
  playPick();
};
function nudge(which, d) {
  const x = PX();
  if (!x) return;
  let [a, b] = [x.a, x.b];
  if (which == "in") a = Math.min(a + d, b - 0.1);
  else if (which == "out") b = Math.max(b + d, a + 0.1);
  else [a, b] = [a + d, b + d];
  edit(a, b);
  const y = PX();
  if (y) goTo(pv, which == "out" ? y.b : y.a);
}
const pIn = () => PX() && edit(Math.min(pnow(), PX().b - 0.1), PX().b);
const pOut = () => PX() && edit(PX().a, Math.max(pnow(), PX().a + 0.1));
function pRemove() {
  const x = PX(), j = pj();
  if (!x) return;
  drop([x]);
  if (order[j]) pick(j, false);
}
function pStar(n) {
  const x = PX();
  if (!x) return;
  if (x.k < 0) addSel(x.c.id, x.a, x.b, reel);       // (starring a planner's shot makes it a pick)
  setStars(x.c.id, x.a, x.b, starsOf(x.c.id, x.a, x.b) == n ? 0 : n);
}
function pRole(f) {
  const x = PX();
  if (!x) return;
  // (a clip holds one opening and one ending: another reel's, on another pick of this clip, gives way)
  const what = f == "open" ? "open" : "end", was = (P[x.c.id] || {})[f];
  const lost = reel != "all" && typeof was == "string" && !same(was, x.a, x.b) ? (tagsOf(x.c.id, f, was) || RID).filter((r) => r != reel) : [];
  const on = setRole(x.c.id, x.a, x.b, f, reel);
  save();
  if (on && lost.length) toast(`${x.c.id} ${fmt(x.a)} ${what}s ${rlab(reel)} — and ${lost.map(rlab).join(", ")} no longer ${what}s on ${was}: a clip holds one ${what}ing. Ctrl+Z undoes`, "warn");
  else toast(on ? `${x.c.id} ${fmt(x.a)} ${what}s ${reel == "all" ? "every reel" : rlab(reel)}` : `No ${what}ing pick${reel == "all" ? "" : " for " + rlab(reel)}: the planner chooses`);
}
$("ped").onclick = (ev) => {
  const st = ev.target.closest("[data-star]");
  if (st) return pStar(+st.dataset.star);
  const b = ev.target.closest("button"), x = PX();
  if (!b || !x) return;
  if (b.dataset.n) {
    const [w, d] = b.dataset.n.split(":");
    nudge(w, +d * (ev.shiftKey ? 10 : 1));
  } else if (b.id == "pplay") playPick();
  else if (b.id == "ploop") loopPick();
  else if (b.id == "pprev") pstep(-1);
  else if (b.id == "pnext") pstep(1);
  else if (b.id == "pin") pIn();
  else if (b.id == "pout") pOut();
  else if (b.id == "popen") openInClips(x.i, x.a, x.k);
  else if (b.dataset.reel) {
    const r = b.dataset.reel, on = !inReel(x.c.id, x.a, x.b, r);
    if (!setIn(x.c.id, x.a, x.b, r, on)) return toast("A pick is in one reel at least — ✕ removes it", "warn");
    if (!on && r == reel) {
      pk = null;
      pv.pause();
    }
    save();
    toast(`${x.c.id} ${fmt(x.a)} ${on ? "is now in" : "taken out of"} ${rlab(r)}`);
  } else if (b.id == "pfirst") pRole("open");
  else if (b.id == "plast") pRole("end");
  else if (b.id == "pdel") pRemove();
};
$("playall").onclick = () => {
  mode = "all";
  pick(Math.max(0, pj()), true);
};
$("stopall").onclick = stopAll;
