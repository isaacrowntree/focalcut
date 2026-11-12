// Portrait framing: where the 9:16 crop sits on a 16:9 picture. The planner works a crop out for every shot
// (`focalcut reel serve` answers api/crop; the cut's own crops come with the page, so a page opened as a file still
// shows them); the editor moves it two ways, both kept per clip in picks.json and applied by the next re-plan:
//   "portrait": [["m:ss.ss", x, y], ...]  pans -- the crop's centre at a time (fractions of the frame); one pan in a
//                                          shot holds it there, several and the crop glides between them
//   "follow":   [["m:ss.ss", x, y], ...]  a person, where the editor clicked them: the planner tracks their face
// The frame is drawn over the picture in every view (F); the Frame portrait view goes through a reel's portrait cut
// shot by shot with the result beside it.
let frameOn = !!ls.get("frame"), followOn = false, staleCrop = null, finding = null;
const crops = new Map(), lastCrop = new Map();
let cropBusy = false, cropNext = null;
const lerp = (ts, vs, t) => {
  if (t <= ts[0]) return vs[0];
  for (let i = 1; i < ts.length; i++)
    if (t <= ts[i]) return vs[i - 1] + ((t - ts[i - 1]) / (ts[i] - ts[i - 1] || 1)) * (vs[i] - vs[i - 1]);
  return vs[vs.length - 1];
};
// one request at a time (each is worked out by the server): only the latest one waiting is fetched next
function fetchCrop(c, a, b, k) {
  cropBusy = true;
  fetch(`${D.root}api/crop?id=${encodeURIComponent(c.id)}&a=${a.toFixed(3)}&b=${b.toFixed(3)}&fmt=portrait`)
    .then((r) => r.json().then((j) => (r.ok ? j : Promise.reject(new Error((j && j.error) || r.status)))))
    .then((j) => {
      crops.set(k, [j, a]);
      lastCrop.set(c.id, [j, a]);
    })
    .catch((e) => crops.set(k, { err: String(e.message || e) }))
    .then(() => {
      cropBusy = false;
      if (finding && finding.k == k) finding = null;
      const n = cropNext;
      cropNext = null;
      if (n && !crops.has(n[3])) fetchCrop(...n);
      redrawFrames();
    });
}
const cropKey = (c, a, b) => c.id + "|" + a.toFixed(2) + "|" + b.toFixed(2);
// the crop as the cut has it, when this stretch is one of its portrait shots
function planCrop(c, a, b) {
  const s = c.P.find(([x, y]) => Math.abs(x - a) < 0.03 && Math.abs(y - b) < 0.03);
  return s && s.length > 3 ? [s[3], s[0]] : null;
}
// [crop, the time its path starts at], or {err}, or undefined while the server works it out
function derived(c, a, b) {
  const k = cropKey(c, a, b), hit = crops.get(k);
  if (hit && !hit.err) return hit;
  const stand = planCrop(c, a, b) || lastCrop.get(c.id);
  if (hit) return stand || hit;                 // (the server could not work it out: the cut's own crop, if any)
  if (!server) return stand || { err: "off" };
  if (!crops.has(k)) {
    if (cropBusy) cropNext = [c, a, b, k];
    else {
      crops.set(k, undefined);
      fetchCrop(c, a, b, k);
    }
  }
  return stand;
}
// where the frame is worked out from: the selection or portrait reel shot under the playhead, else the 3 s around it
function rangeFor(c, t) {
  const s = sels(c.id).find(([a, b]) => t >= a - 0.02 && t <= b + 0.02) || c.P.find(([a, b]) => t >= a && t <= b);
  if (s) return [s[0], s[1]];
  const a = Math.max(0, Math.min(Math.floor(t / 3) * 3, c.dur - 3));
  return [a, Math.min(a + 3, c.dur)];
}
// the frame at t of a shot [a, b]: {d: the planner's centre, o: the editor's pan (or null), w, h, z, fol, kind}, or
// "wait" / "fills" (the clip is portrait already) / "off" (not served) / {err}
function frameAt(c, a, b, t) {
  const g = derived(c, a, b);
  if (g === undefined) return "wait";
  if (g.err) return g.err == "off" ? "off" : g;
  const [cr, a0] = g;
  if (cr === null) return "fills";
  const ts = cr.path.map((q) => a0 + q[0]), d = { x: lerp(ts, cr.path.map((q) => q[1]), t), y: lerp(ts, cr.path.map((q) => q[2]), t) };
  const ks = inShot(fkeys(c.id), a, b);
  let o = null;
  if (ks.length) {
    const kt = ks.map((k) => k[0]);
    o = { x: clamp(lerp(kt, ks.map((k) => k[1]), t), cr.w / 2, 1 - cr.w / 2), y: clamp(lerp(kt, ks.map((k) => k[2]), t), cr.h / 2, 1 - cr.h / 2) };
  }
  return { d, o, w: cr.w, h: cr.h, z: cr.z, fol: cr.follow, kind: framingOf(c.id, a, b), n: ks.length };
}
// what frames the shot, in words; warn: it did not go as asked
function framingText(c, a, b, F) {
  const kind = framingOf(c.id, a, b);
  if (finding && finding.id == c.id && Math.abs(finding.a - a) < 0.03) return { kind, text: "Finding the person you clicked…", warn: true };
  if (kind == "pan") return { kind, text: F && F.n > 1 ? `Your pan — gliding through ${F.n} points` : "Your pan" };
  if (kind == "follow") {
    if (!server) return { kind, text: "Following the person you clicked (shown after the next re-plan)" };
    if (F && F.fol == "point") return { kind, text: "No face found where you clicked — the crop holds on your click", warn: true };
    return { kind, text: "Following the person you clicked" };
  }
  return { kind, text: "The planner's framing" };
}
function frameDraw(el, wrap, v, c, a, b, t) {
  const on = frameOn || el.id == "ffr";
  if (el.id != "ffr") {
    el.classList.toggle("on", on);
    $("fhelp").hidden = !frameOn || curView != 1;
    document.querySelectorAll(".frt").forEach((x) => x.classList.toggle("on", frameOn));
  }
  if (!on) return void (el.innerHTML = "");
  const F = frameAt(c, a, b, t);
  if (typeof F == "string" || F.err) {
    el.innerHTML = `<span class=frl>${{
      fills: "This clip fills the portrait frame as it is — there is no crop to place",
      off: "The portrait crop of this stretch needs the page served by <code>focalcut reel serve</code>",
      wait: "Working out the portrait crop…" }[F] || "Could not work out the portrait crop: " + esc(F.err)}</span>`;
    return F;
  }
  const r = vrect(wrap, v, c), A = F.o || F.d, say = framingText(c, a, b, F);
  const box = (p, cls) => `<i class="fbox ${cls}" style="left:${r.l + (p.x - F.w / 2) * r.w}px;top:${r.t + (p.y - F.h / 2) * r.h}px;width:${F.w * r.w}px;height:${F.h * r.h}px"></i>`;
  const pin = inShot(folkeys(c.id), a, b).find((k) => Math.abs(k[0] - t) <= 0.6);
  el.innerHTML = box(A, "act " + say.kind) + (F.o ? box(F.d, "der") : "") +
    (pin ? `<i class=fpin style="left:${r.l + pin[1] * r.w}px;top:${r.t + pin[2] * r.h}px" title="the person you chose to follow"></i>` : "") +
    (el.id == "ffr" ? "" : `<span class=frl><b>${esc(say.text)}</b>${F.z > 1.001 ? " · zoom " + F.z.toFixed(2) : ""}${say.kind == "auto" ? " · drag to pan, G then click a dancer to follow" : ""}</span>`);
  return F;
}
function toggleFrame() {
  frameOn = !frameOn;
  ls.set("frame", frameOn);
  if (!frameOn) followMode(false);
  redrawFrames();
}
function followMode(on) {
  followOn = on;
  if (on && !frameOn) toggleFrame();
  document.querySelectorAll(".fol").forEach((b) => b.classList.toggle("on", on));
  document.querySelectorAll(".fr").forEach((f) => f.classList.toggle("follow", on));
  if (on) toast("Click the dancer to follow (Esc: never mind)");
}
function redrawFrames() {
  if (curView == 1) draw();
  else if (curView == 2) pdraw();
  else fdraw();
}
// the crop of shot x = {c, a, b, t} panned to (cx, cy). whole: one pan for the whole shot (its other pans go); else a
// pan at the playhead (one within ¼ s of it moves instead). A pan takes over from a person followed in the shot.
function panTo(x, cx, cy, whole) {
  const id = x.c.id, ks = whole ? fkeys(id).filter((k) => !inShot([k], x.a, x.b).length) : fkeys(id).filter((k) => Math.abs(k[0] - x.t) > 0.25);
  ks.push([whole ? (x.a + x.b) / 2 : x.t, cx, cy]);
  putKeys(id, "portrait", ks);
  unfollow(x);
}
function unfollow(x) {
  const fl = folkeys(x.c.id), kept = fl.filter((k) => !inShot([k], x.a, x.b).length);
  if (kept.length == fl.length) return false;
  putKeys(x.c.id, "follow", kept);
  staleCrop = x.c.id;
  return true;
}
// the person at (mx, my) followed through shot x: the planner tracks their face (the server's crop shows it once the
// click is saved). The shot's pans go, so the crop is seen to follow.
function followAt(x, mx, my) {
  const id = x.c.id;
  putKeys(id, "follow", [...folkeys(id).filter((k) => !inShot([k], x.a, x.b).length), [x.t, mx, my]]);
  putKeys(id, "portrait", fkeys(id).filter((k) => !inShot([k], x.a, x.b).length));
  staleCrop = id;
  if (server) finding = { id, a: x.a, k: cropKey(x.c, x.a, x.b) };
  followMode(false);
  save();
  toast(server ? "Following the person you clicked — Ctrl+Z undoes" : "Following the person you clicked: kept in this browser, shown after the next re-plan");
  flashState();
}
function clearFocus(c, a, b) {
  const x = { c, a, b }, ks = fkeys(c.id), kept = ks.filter((k) => !inShot([k], a, b).length), fl = unfollow(x);
  if (kept.length == ks.length && !fl) return toast("This stretch already has the planner's framing", "warn");
  putKeys(c.id, "portrait", kept);
  save();
  toast("Back to the planner's framing — Ctrl+Z undoes");
  flashState();
}
// once a change is saved the server's crops of that clip are worked out again (a followed person changes them)
App.onSaved = () => {
  if (!staleCrop) return;
  const id = staleCrop;
  staleCrop = null;
  for (const k of [...crops.keys()]) if (k.startsWith(id + "|")) crops.delete(k);
  redrawFrames();
};
// the frame's mouse: drag the bright frame to pan it (a drag begun outside it brings it under the pointer); a click
// follows the person there when follow is armed (G) or the view says so (opts.clickFollows), else centres the crop
function frameMouse(el, wrap, v, ctx, redraw, stop, opts = {}) {
  let g = null;
  const pos = (ev) => {
    const r = vrect(wrap, v, g.x.c), R = wrap.getBoundingClientRect();
    return [clamp((ev.clientX - R.left - r.l) / r.w, 0, 1), clamp((ev.clientY - R.top - r.t) / r.h, 0, 1)];
  };
  const pan = (ev) => {
    const F = g.F, [mx, my] = pos(ev);
    const cx = g.inside ? g.c0[0] + mx - g.m[0] : mx, cy = g.inside ? g.c0[1] + my - g.m[1] : my;
    panTo(g.x, clamp(cx, F.w / 2, 1 - F.w / 2), clamp(cy, F.h / 2, 1 - F.h / 2), opts.whole && opts.whole());
    g.dirty = true;
    redraw();
  };
  el.addEventListener("pointerdown", (ev) => {
    if (!el.classList.contains("on") || ev.button) return;
    const x = ctx();
    if (!x) return;
    stop();
    g = { x, F: frameAt(x.c, x.a, x.b, x.t), moved: false, px: ev.clientX, py: ev.clientY };
    g.m = pos(ev);
    if (typeof g.F == "object" && !g.F.err) {
      const A = g.F.o || g.F.d;
      g.c0 = [A.x, A.y];
      g.inside = Math.abs(g.m[0] - A.x) <= g.F.w / 2 && Math.abs(g.m[1] - A.y) <= g.F.h / 2;
    } else g.F = null;
    el.setPointerCapture(ev.pointerId);
  });
  el.addEventListener("pointermove", (ev) => {
    if (!g) return;
    if (!g.moved && Math.hypot(ev.clientX - g.px, ev.clientY - g.py) < 4) return;
    g.moved = true;
    if (followOn || !g.F) return;
    el.classList.add("drag");
    pan(ev);
  });
  const end = (ev) => {
    if (!g) return;
    const d = g, follows = followOn || opts.clickFollows;
    el.classList.remove("drag");
    if (ev.type == "pointercancel") {
      g = null;
      return;
    }
    if (d.moved && !followOn) {
      g = null;
      if (!d.dirty) return toast("There is no portrait crop to pan here", "warn");
      save();
      toast("Crop panned — Ctrl+Z undoes");
      flashState();
    } else if (follows) {
      g = null;
      followAt(d.x, ...d.m);
    } else if (d.F) {
      d.inside = false;
      pan(ev);
      g = null;
      save();
      toast("Crop centred there — Ctrl+Z undoes");
    } else g = null;
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
}
frameMouse($("fr"), $("vwrap"), vid, () => ({ c: C(), t: now(), a: rangeFor(C(), now())[0], b: rangeFor(C(), now())[1] }), draw, pause);
frameMouse($("pfr"), $("pvwrap"), pv, () => PX() && { c: PX().c, t: pnow(), a: PX().a, b: PX().b }, pdraw, () => pv.pause());
$("bfollow").onclick = $("pfollow").onclick = () => followMode(!followOn);
$("bframe").onclick = $("pframe").onclick = toggleFrame;
$("fclr").onclick = () => clearFocus(C(), ...rangeFor(C(), now()));
$("pfclr").onclick = () => PX() && clearFocus(PX().c, PX().a, PX().b);

// --- Frame portrait: the reel's portrait cut, shot by shot -----------------------------------------------------------
const fv = $("fv"), FT = $("fstrip");
let fcur = null, forder = [], ffilter = "all", fmode = ls.get("fmode") || "shot", fscrub = false;
function fshots() {
  const out = [];
  D.clips.forEach((c, i) => c.P.forEach(([a, b, n, crop]) => out.push({ c, i, a, b, n, crop })));
  return out.sort((x, y) => x.n - y.n);
}
const isCur = (x) => fcur && x.c.id == fcur.id && Math.abs(x.a - fcur.a) < 0.002;
const FJ = () => forder.findIndex(isCur), FX = () => forder[FJ()];
const fnow = () => tOf(fv);
// a shot's crop at its middle as a small 9:16 picture, cut from the clip's filmstrip
function cropThumb(x, W, H) {
  const [tw, th] = tileOf(x.c), t = (x.a + x.b) / 2, k = tileAt(x.c, t), cr = x.crop;
  if (!cr) return thumbAt(x.c, t, (W * th) / tw) + ";background-position-y:center";
  const ks = inShot(fkeys(x.c.id), x.a, x.b), ts = cr.path.map((q) => x.a + q[0]);
  let cx = lerp(ts, cr.path.map((q) => q[1]), t), cy = lerp(ts, cr.path.map((q) => q[2]), t);
  if (ks.length) [cx, cy] = [lerp(ks.map((q) => q[0]), ks.map((q) => q[1]), t), lerp(ks.map((q) => q[0]), ks.map((q) => q[2]), t)];
  cx = clamp(cx, cr.w / 2, 1 - cr.w / 2);
  cy = clamp(cy, cr.h / 2, 1 - cr.h / 2);
  const s = W / (cr.w * tw);
  return `background-image:url('${x.c.strip}');background-size:${x.c.n * tw * s}px ${th * s}px;background-position:-${(k * tw + (cx - cr.w / 2) * tw) * s}px -${(cy - cr.h / 2) * th * s}px`;
}
const BADGE = { pan: "panned", follow: "following", auto: "planner" };
function frameview() {
  if (curView != 3) return;
  const all = fshots(), which = reel == "all" ? "main" : reel, kind = (x) => framingOf(x.c.id, x.a, x.b);
  const framed = all.filter((x) => kind(x) != "auto").length;
  forder = all.filter((x) => ffilter == "all" || isCur(x) || (ffilter == "mine") == (kind(x) != "auto"));
  $("ftitle").textContent = `${rlab(which)} portrait cut`;
  $("fcount").textContent = `${all.length} shots · ${framed} framed by you`;
  $("ffilter").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.ff == ffilter));
  $("fshots").innerHTML = forder.map((x, j) => {
    const k = kind(x);
    return `<div class="fs${isCur(x) ? " cur" : ""}" data-j=${j}><div class=th style="${cropThumb(x, 45, 80)}"></div>
 <div><div class=nm>${x.n} · ${esc(x.c.id)}</div><small>${fmt(x.a)} – ${fmt(x.b)} · ${(x.b - x.a).toFixed(1)} s</small>
 <span class="badge ${k}">${x.crop === null ? "fills the frame" : BADGE[k]}</span></div></div>`;
  }).join("") || `<p class=emptystate>${all.length ? "No shots like that." : ""}</p>`;
  $("fempty").hidden = !!all.length;
  $("fbody").hidden = !all.length;
  if (!all.length) {
    $("fempty").innerHTML = `There is no portrait cut of <b>${esc(rlab(which))}</b> yet.<br>Plan it (<code>focalcut reel run … --steps plan,review</code>) and this view goes through it shot by shot.`;
    return;
  }
  if (FJ() < 0) return fpick(0);
  fdraw();
}
function fpick(j, play) {
  const x = forder[j];
  if (!x) return;
  fcur = { id: x.c.id, a: x.a };
  finding = null;
  stripOf(x.c, fdraw);
  load(fv, x.c, x.a);
  if (play) fv.play().catch(() => {});
  frameview();
  const el = $("fshots").querySelector(".fs.cur");
  if (el) el.scrollIntoView({ block: "nearest" });
}
const fstep = (d) => fpick(clamp(FJ() + d, 0, forder.length - 1), !fv.paused);
const fwhole = (x) => fmode == "shot" && inShot(fkeys(x.c.id), x.a, x.b).length < 2;
let lastSay = "";
function flashState() {
  const s = $("fstate");
  s.classList.remove("flash");
  void s.offsetWidth;
  s.classList.add("flash");
}
function fdraw() {
  const x = FX();
  if (!x || curView != 3) return;
  const t = fnow(), pad = 0.5, A = Math.max(0, x.a - pad), B = Math.min(x.c.dur, x.b + pad);
  stillDraw($("fstill"), $("fwrap"), fv, x.c, t);
  const F = frameDraw($("ffr"), $("fwrap"), fv, x.c, x.a, x.b, t), say = framingText(x.c, x.a, x.b, typeof F == "object" ? F : null);
  $("fname").textContent = `Shot ${x.n} · ${x.c.id}`;
  $("fmeta").textContent = `${fmt(x.a)} – ${fmt(x.b)} · ${(x.b - x.a).toFixed(2)} s · ${FJ() + 1} of ${forder.length}`;
  const st = $("fstate"), text = F == "fills" ? "Fills the frame as it is" : say.text;
  st.textContent = text;
  st.className = "fstate " + (F == "fills" ? "" : say.kind) + (say.warn ? " warn" : "") + (st.classList.contains("flash") ? " flash" : "");
  if (lastSay && lastSay != x.c.id + x.a + text && lastSay.startsWith(x.c.id + x.a)) flashState();
  lastSay = x.c.id + x.a + text;
  $("fprev").className = F == "fills" ? "" : say.kind;
  $("ftc").textContent = fmt(t);
  $("fplay").textContent = fv.paused ? "▶ Play" : "❚❚ Pause";
  $("fmode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", (b.dataset.fm == "shot") == fwhole(x)));
  $("freset").disabled = say.kind == "auto";
  preview(x, F, t);
  film(FT.querySelector("canvas"), A, B, 56, x.c, stripOf(x.c));
  FT.querySelectorAll("i").forEach((e) => e.remove());
  let h = `<i class=dim style="${wspan(A, x.a, A, B)}"></i><i class=dim style="${wspan(x.b, B, A, B)}"></i>`;
  inShot(fkeys(x.c.id), x.a, x.b).forEach(([k]) => (h += `<i class=fk style="left:${X(clamp(k, A, B), A, B)}" title="your pan at ${fmt(k)}"></i>`));
  inShot(folkeys(x.c.id), x.a, x.b).forEach(([k]) => (h += `<i class="fk fl" style="left:${X(clamp(k, A, B), A, B)}" title="following a person from ${fmt(k)}"></i>`));
  h += `<i class=ph style="left:${X(clamp(t, A, B), A, B)}"></i>`;
  FT.insertAdjacentHTML("beforeend", h);
}
// what the reel shows: the crop cut out of the picture (the video's frame, else the filmstrip's)
function preview(x, F, t) {
  const cv = $("fcanvas"), g = cv.getContext("2d"), W = cv.width, H = cv.height;
  g.fillStyle = "#000";
  g.fillRect(0, 0, W, H);
  const vidOk = !noVideo(fv), im = stripOf(x.c), [tw, th] = tileOf(x.c);
  const src = vidOk ? [fv, 0, 0, fv.videoWidth, fv.videoHeight] : im.complete && im.naturalWidth ? [im, tileAt(x.c, t) * tw, 0, tw, th] : null;
  if (!src) return;
  const [el, sx, sy, sw, sh] = src;
  try {
    if (typeof F != "object" || F.err) {
      const s = Math.min(W / sw, H / sh);        // (no crop: the whole picture, fitted)
      g.drawImage(el, sx, sy, sw, sh, (W - sw * s) / 2, (H - sh * s) / 2, sw * s, sh * s);
    } else {
      const A = F.o || F.d;
      g.drawImage(el, sx + (A.x - F.w / 2) * sw, sy + (A.y - F.h / 2) * sh, F.w * sw, F.h * sh, 0, 0, W, H);
    }
  } catch (e) {}
}
$("fshots").onclick = (ev) => {
  const d = ev.target.closest(".fs");
  if (d) fpick(+d.dataset.j, false);
};
$("ffilter").onclick = (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  ffilter = b.dataset.ff;
  frameview();
};
$("fmode").onclick = (ev) => {
  const b = ev.target.closest("button"), x = FX();
  if (!b || !x) return;
  fmode = b.dataset.fm;
  ls.set("fmode", fmode);
  if (fmode == "shot" && inShot(fkeys(x.c.id), x.a, x.b).length > 1) toast("This shot glides through several pans: the next drag replaces them with one for the whole shot", "warn");
  fdraw();
};
function fplay() {
  const x = FX();
  if (!x) return;
  if (!fv.paused) return fv.pause();
  if (fnow() < x.a - 0.05 || fnow() >= x.b - 0.05) goTo(fv, x.a);
  fv.play().catch(() => toast("This clip cannot play here — the filmstrip stands in; scrub the strip under the picture", "warn"));
}
function fnudge(d) {
  const x = FX();
  if (!x) return;
  fv.pause();
  goTo(fv, clamp(fnow() + d, x.a, x.b - 0.02));
  fdraw();
}
const freset = () => FX() && clearFocus(FX().c, FX().a, FX().b);
$("fprevb").onclick = () => fstep(-1);
$("fnextb").onclick = () => fstep(1);
$("fplay").onclick = fplay;
$("freset").onclick = freset;
$("fopen").onclick = () => FX() && openInClips(FX().i, FX().a, -1);
const floopBack = () => {
  const x = FX();
  if (x && !fv.paused && fnow() >= x.b - 0.02) fv.currentTime = x.a;
};
fv.addEventListener("play", () => {
  const f = () => {
    if (curView != 3) return;
    floopBack();
    fdraw();
    if (!fv.paused) requestAnimationFrame(f);
  };
  f();
});
fv.addEventListener("timeupdate", floopBack);
for (const e of ["seeked", "pause", "loadeddata", "error"]) fv.addEventListener(e, () => fdraw());
const fscrubTo = (ev) => {
  const x = FX(), A = Math.max(0, x.a - 0.5), B = Math.min(x.c.dur, x.b + 0.5);
  goTo(fv, clamp(tAt(FT, ev, A, B), x.a, x.b - 0.02));
  fdraw();
};
FT.addEventListener("pointerdown", (ev) => {
  if (!FX() || ev.button) return;
  fv.pause();
  FT.setPointerCapture(ev.pointerId);
  fscrub = true;
  fscrubTo(ev);
});
FT.addEventListener("pointermove", (ev) => fscrub && fscrubTo(ev));
for (const e of ["pointerup", "pointercancel"]) FT.addEventListener(e, () => (fscrub = false));
frameMouse($("ffr"), $("fwrap"), fv, () => FX() && { c: FX().c, t: fnow(), a: FX().a, b: FX().b }, fdraw, () => fv.pause(),
  { clickFollows: true, whole: () => fwhole(FX()) });
