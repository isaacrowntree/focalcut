// per-reel picks: one picks.json serves the project's reel ("main") and its cuts. A clip's "only" gives a value to
// some reels: its key is the "add" range as written, or "<field>:<value>" (ban, prefer, open, end; "more:"); a value
// not listed is in every reel. R is the reel a page edits for ("all": every reel).
const RS = D.reels || [],
  RID = RS.map((r) => r.id),
  near2 = (u, v) => Math.abs(u - v) < 0.006;
const TF = /^(ban|prefer|open|end|more):(.*)$/;
function valEq(f, x, v) {
  if (f == "more") return true;
  if (x == "all" || v == "all") return x == v;
  const [a, b] = span(x),
    [c, d] = span(v);
  return near2(a, c) && near2(b, d);
}
function tagParts(k) {
  const m = TF.exec(k);
  return m ? [m[1], m[2]] : ["add", k];
}
function tagsOf(id, f, v) {
  const o = (P[id] || {}).only;
  if (!o) return null;
  for (const k in o) {
    const [kf, kv] = tagParts(k);
    if (kf == f && valEq(f, kv, v)) return Array.isArray(o[k]) ? o[k] : null;
  }
  return null;
}
function setTags(id, f, v, list) {
  const e = ent(id),
    o = {};
  for (const k in e.only || {}) {
    const [kf, kv] = tagParts(k);
    if (!(kf == f && valEq(f, kv, v))) o[k] = e.only[k];
  }
  if (list && RID.length > 1 && !RID.every((r) => list.includes(r)))
    o[f == "add" ? v : f + ":" + (f == "more" ? "" : v)] = RID.filter((r) => list.includes(r));
  if (Object.keys(o).length) e.only = o;
  else delete e.only;
}
function isIn(id, f, v, R) {
  if (R == "all") return true;
  const t = tagsOf(id, f, v);
  return !t || t.includes(R);
}
function vals(id, f, R) {
  return ((P[id] || {})[f] || []).filter((x) => isIn(id, f, x, R));
}
// add a value for reel R (already there for other reels: R joins them); take it out of R (gone when no reel is left)
function addVal(id, f, v, R) {
  const e = ent(id),
    L = (e[f] = e[f] || []),
    x = L.find((y) => valEq(f, y, v));
  if (x === undefined) {
    L.push(v);
    setTags(id, f, v, R == "all" ? null : [R]);
    return true;
  }
  const t = tagsOf(id, f, x);
  if (R == "all") {
    setTags(id, f, x, null);
    return !!t;
  }
  if (t && !t.includes(R)) {
    setTags(id, f, x, [...t, R]);
    return true;
  }
  return false;
}
function delVal(id, f, v, R) {
  const e = P[id];
  if (!e || !e[f]) return;
  const k = e[f].findIndex((y) => valEq(f, y, v));
  if (k < 0) return;
  const x = e[f][k],
    t = (tagsOf(id, f, x) || RID).filter((r) => r != R);
  if (R == "all" || !t.length) {
    e[f].splice(k, 1);
    setTags(id, f, x, null);
  } else setTags(id, f, x, t);
}
// "more" (more shots from the clip) and "open" / "end" (the pick a reel opens / ends on: one value per clip)
function moreIn(id, R) {
  const e = P[id] || {};
  return e.more && isIn(id, "more", "", R) ? e.more : 0;
}
function addMore(id, d, R) {
  const e = ent(id),
    had = e.more || 0,
    t = tagsOf(id, "more", "");
  e.more = Math.max(0, had + d);
  if (!had) setTags(id, "more", "", R == "all" ? null : [R]);
  else if (d > 0 && t && R != "all" && !t.includes(R)) setTags(id, "more", "", [...t, R]);
}
function roleIn(id, f, R) {
  const e = P[id] || {};
  return typeof e[f] == "string" && isIn(id, f, e[f], R) ? e[f] : null;
}
function clearRole(id, f, R) {
  const e = P[id];
  if (!e || typeof e[f] != "string") return;
  const t = (tagsOf(id, f, e[f]) || RID).filter((r) => r != R);
  if (R == "all" || !t.length) {
    setTags(id, f, e[f], null);
    delete e[f];
  } else setTags(id, f, e[f], t);
}
// a clip's tags whose value is gone go too
function pruneTags(e) {
  if (!e.only) return;
  for (const k in e.only) {
    const [f, v] = tagParts(k);
    const live =
      f == "more"
        ? !!e.more
        : f == "open" || f == "end"
          ? typeof e[f] == "string" && valEq(f, e[f], v)
          : (e[f] || []).some((x) => valEq(f, x, v));
    if (!live) delete e.only[k];
  }
  if (!Object.keys(e.only).length) delete e.only;
}
