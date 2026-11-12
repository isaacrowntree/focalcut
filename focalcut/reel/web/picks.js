// The picks model, shared by every review page (and run under Node by tests/test_web.py): times, the per-reel tags
// of a clip's "only" map, selections with their stars and opening / ending roles, and the merge of two edits of
// one picks.json. It reads the page's data `D` (D.reels) and edits the picks in `P`; nothing here touches the page.
//
// picks.json, per export file:  {"add": ["1:12.40-1:15.10", "0:41.00"], "ban": ["0:00-0:20", "all"],
//   "prefer": [...], "more": 2, "open": "<range>", "end": "<range>", "stars": {"<range>": 1-5},
//   "portrait": [["m:ss.ss", x, y], ...], "follow": [["m:ss.ss", x, y], ...], "only": {"<key>": ["main", "short"]}}

// --- times -------------------------------------------------------------------------------------------------------
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
const secs = (v) => (typeof v === "number" ? v : String(v).trim().split(":").reduce((a, x) => a * 60 + +x, 0));
// "1:12.40-1:15.10" -> [72.4, 75.1]; a single time -> [t, t]
const span = (v) => {
  if (Array.isArray(v)) return v.map(secs);
  const s = String(v), i = s.indexOf("-", 1);
  return i > 0 ? [secs(s.slice(0, i)), secs(s.slice(i + 1))] : [secs(s), secs(s)];
};
const isRange = (v) => {
  const [a, b] = span(v);
  return b - a > 0.05;
};
const rkey = (a, b) => fmt(a) + "-" + fmt(b);
const near2 = (u, v) => Math.abs(u - v) < 0.006;
const same = (v, a, b) => {
  if (!v) return false;
  const [u, w] = span(v);
  return Math.abs(u - a) < 0.01 && Math.abs(w - b) < 0.01;
};
const esc = (x) => String(x).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// --- the picks and the reels they serve ----------------------------------------------------------------------------
let P = {};
const RS = (typeof D !== "undefined" && D.reels) || [], RID = RS.map((r) => r.id);
const rlab = (r) => (RS.find((x) => x.id == r) || { label: r }).label;
const isClip = (k, e) => k[0] != "_" && e && typeof e === "object" && !Array.isArray(e);
const ent = (id) => (isClip(id, P[id]) ? P[id] : (P[id] = {}));

// per-reel picks: one picks.json serves the project's reel ("main") and its cuts. A clip's "only" gives a value to
// some reels: its key is the "add" range as written, or "<field>:<value>" (ban, prefer, open, end; "more:"); a value
// not listed is in every reel. R is the reel a page edits for ("all": every reel).
const TF = /^(ban|prefer|open|end|more):(.*)$/;
function valEq(f, x, v) {
  if (f == "more") return true;
  if (x == "all" || v == "all") return x == v;
  const [a, b] = span(x), [c, d] = span(v);
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
  const e = ent(id), o = {};
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
  const e = ent(id), L = (e[f] = e[f] || []), x = L.find((y) => valEq(f, y, v));
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
  const x = e[f][k], t = (tagsOf(id, f, x) || RID).filter((r) => r != R);
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
  const e = ent(id), had = e.more || 0, t = tagsOf(id, "more", "");
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
    const live = f == "more" ? !!e.more
      : f == "open" || f == "end" ? typeof e[f] == "string" && valEq(f, e[f], v)
      : (e[f] || []).some((x) => valEq(f, x, v));
    if (!live) delete e.only[k];
  }
  if (!Object.keys(e.only).length) delete e.only;
}

// --- selections: a clip's "add" ranges ("m:ss.ss-m:ss.ss"); single times ("marked moments") stay as written --------
function sels(id) {
  return ((P[id] || {}).add || []).filter(isRange).map(span).sort((x, y) => x[0] - y[0]);
}
function points(id) {
  return ((P[id] || {}).add || []).filter((v) => !isRange(v)).map((v) => span(v)[0]);
}
const onlyOf = (id, a, b) => tagsOf(id, "add", rkey(a, b));
const inReel = (id, a, b, R) => isIn(id, "add", rkey(a, b), R);
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
function putStars(id, a, b, n) {
  const e = ent(id), o = {};
  for (const k in e.stars || {}) {
    const [u, v] = span(k);
    if (!(near2(u, a) && near2(v, b))) o[k] = e.stars[k];
  }
  if (n) o[rkey(a, b)] = n;
  if (Object.keys(o).length) e.stars = o;
  else delete e.stars;
}
// the "add" value as written for a selection, or undefined
const written = (id, a, b) => ((P[id] || {}).add || []).find((v) => isRange(v) && valEq("add", v, rkey(a, b)));
function sortAdds(e) {
  const pts = (e.add || []).filter((v) => !isRange(v)), rs = (e.add || []).filter(isRange);
  e.add = [...pts, ...rs.sort((x, y) => span(x)[0] - span(y)[0])];
}
// a new selection for reel R (one already there for other reels: R joins it). True when something changed.
function addSel(id, a, b, R) {
  [a, b] = [Math.max(0, Math.min(a, b)), Math.max(a, b)];
  if (b - a < 0.1) return false;
  const had = written(id, a, b), ok = addVal(id, "add", had ?? rkey(a, b), R);
  sortAdds(ent(id));
  return ok;
}
// a selection moved or trimmed: it stays the same pick -- its reels, its stars and its opening / ending role go
// with it. (Another selection already at the new range takes it in.)
function moveSel(id, a, b, a2, b2) {
  const e = P[id], old = written(id, a, b);
  if (!e || old === undefined || b2 - a2 < 0.1) return false;
  const key = rkey(a2, b2);
  if (valEq("add", old, key)) return false;
  const tags = tagsOf(id, "add", old), st = starsOf(id, a, b), dup = written(id, a2, b2);
  for (const f of ["open", "end"])
    if (same(e[f], a, b)) {
      const t = tagsOf(id, f, e[f]);
      setTags(id, f, e[f], null);
      e[f] = key;
      setTags(id, f, key, t);
    }
  setTags(id, "add", old, null);
  putStars(id, a, b, 0);
  if (dup !== undefined) e.add.splice(e.add.indexOf(old), 1);
  else {
    e.add[e.add.indexOf(old)] = key;
    setTags(id, "add", key, tags);
  }
  if (st) putStars(id, a2, b2, st);
  sortAdds(e);
  return true;
}
// a selection taken out of reel R: gone when R was its last reel (or R is "all"), else it stays for the others.
// Its role in R goes with it. Returns the reels it is still in ([]: gone).
function removeSel(id, a, b, R) {
  const e = P[id], old = written(id, a, b);
  if (!e || old === undefined) return [];
  const left = R == "all" ? [] : (tagsOf(id, "add", old) || RID).filter((r) => r != R);
  for (const f of ["open", "end"]) if (same(roleIn(id, f, R), a, b)) clearRole(id, f, R);
  if (!left.length) {
    for (const f of ["open", "end"]) if (same(e[f], a, b)) clearRole(id, f, "all");
    e.add.splice(e.add.indexOf(old), 1);
    setTags(id, "add", old, null);
    putStars(id, a, b, 0);
  } else setTags(id, "add", old, left);
  return left;
}
// put a selection in / take it out of one reel; false when that would leave it in none (remove it instead)
function setIn(id, a, b, r, on) {
  let v = onlyOf(id, a, b) || RID.slice();
  v = on ? [...new Set([...v, r])] : v.filter((x) => x != r);
  if (!v.length) return false;
  if (!on) for (const f of ["open", "end"]) if (same(roleIn(id, f, r), a, b)) clearRole(id, f, r);
  setTags(id, "add", written(id, a, b) ?? rkey(a, b), v);
  return true;
}
// one pick opens a reel and one ends it: the clip's "open" / "end" is that pick's in-out (it stays a selection too);
// per reel, and a clip holds one opening and one ending. Returns true when the role was set, false when cleared.
function roleOf(id, a, b, R) {
  return same(roleIn(id, "open", R), a, b) ? "open" : same(roleIn(id, "end", R), a, b) ? "end" : null;
}
function setRole(id, a, b, f, R) {
  const on = same(roleIn(id, f, R), a, b);
  for (const k in P) if (isClip(k, P[k]) && roleIn(k, f, R)) clearRole(k, f, R);
  if (on) return false;
  const e = ent(id);
  if (typeof e[f] == "string") {
    setTags(id, f, e[f], null);
    delete e[f];
  }
  e[f] = rkey(a, b);
  setTags(id, f, e[f], R == "all" ? null : [R]);
  addSel(id, a, b, R);
  return true;
}

// --- portrait framing: focal points ("portrait") and the person to follow ("follow"), [["m:ss.ss", x, y], ...] ------
const REACH = 0.5; // a point steers a shot that plays within this many seconds of it (plan.FOCUS_REACH)
const keysOf = (id, f) => ((P[id] || {})[f] || []).map(([t, x, y]) => [secs(t), +x, +y]).sort((p, q) => p[0] - q[0]);
const fkeys = (id) => keysOf(id, "portrait"), folkeys = (id) => keysOf(id, "follow");
const inShot = (ks, a, b) => ks.filter((k) => k[0] >= a - REACH && k[0] <= b + REACH);
function putKeys(id, f, ks) {
  const e = ent(id);
  if (ks.length) e[f] = ks.sort((p, q) => p[0] - q[0]).map(([t, x, y]) => [fmt(t), +x.toFixed(4), +y.toFixed(4)]);
  else delete e[f];
}
// what frames a shot: "follow" (a person the editor clicked), "pan" (their focal points) or "auto" (the planner)
function framingOf(id, a, b) {
  return inShot(fkeys(id), a, b).length ? "pan" : inShot(folkeys(id), a, b).length ? "follow" : "auto";
}

// --- the whole file --------------------------------------------------------------------------------------------------
function tidy() {
  for (const k in P) {
    const e = P[k];
    if (!isClip(k, e)) continue;
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
// two edits of one picks.json (base: what both started from) put together, value by value: what only one side
// changed is that side's; maps (a clip, its "only", its "stars") merge key by key and lists (its "add", "ban", pans)
// item by item, each side's additions and removals both kept; a single value both changed differently is mine.
// {picks, theirs: clips the other edit changed, both: [where both changed one value, e.g. "C1.more"]}
function merge3(base, mine, theirs) {
  const J = JSON.stringify, both = [], obj = (v) => v && typeof v == "object" && !Array.isArray(v);
  const m3 = (b, m, t, path) => {
    const jb = J(b), jm = J(m), jt = J(t);
    if (jm === jb || jm === jt) return t;
    if (jt === jb) return m;
    if (obj(m) && obj(t) && (b === undefined || obj(b))) {
      const out = {};
      for (const k of new Set([...Object.keys(t), ...Object.keys(m)])) {
        const v = m3((b || {})[k], m[k], t[k], path ? path + "." + k : k);
        if (v !== undefined) out[k] = v;
      }
      return out;
    }
    if (Array.isArray(m) && Array.isArray(t) && (b === undefined || Array.isArray(b))) {
      const has = (list) => new Set((list || []).map((x) => J(x))), B = has(b), M = has(m), T = has(t);
      return [...t.filter((x) => !(B.has(J(x)) && !M.has(J(x)))), ...m.filter((x) => !B.has(J(x)) && !T.has(J(x)))];
    }
    both.push(path);
    return m;
  };
  const picks = JSON.parse(J(m3(base, mine, theirs, "") || {}));
  const taken = Object.keys({ ...base, ...theirs }).filter((k) => J(base[k]) !== J(theirs[k])).length;
  return { picks, theirs: taken, both };
}
