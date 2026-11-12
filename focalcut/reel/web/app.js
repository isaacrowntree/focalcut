// What every review page shares: where the picks are kept (picks.json through `focalcut reel serve`, else this
// browser), saving with undo / redo, what the page tells the editor (the save state in the header, a toast for
// what an action did), the theme and the keys sheet. A page sets App.render (redraw after any change) and calls boot().
const $ = (id) => document.getElementById(id);
const App = { render() {}, ls: "sources:" + D.key, onBoot() {} };
const ls = {
  get(k) {
    try {
      return JSON.parse(localStorage.getItem(App.ls + k));
    } catch (e) {
      return null;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem(App.ls + k, JSON.stringify(v));
    } catch (e) {}
  },
  del(k) {
    try {
      localStorage.removeItem(App.ls + k);
    } catch (e) {}
  },
};

// --- what the page says ----------------------------------------------------------------------------------------------
// the save state, always in the header: ok (saved), busy, err (not saved), local (kept in this browser)
function saveState(text, kind) {
  const s = $("saved");
  if (!s) return;
  s.textContent = text;
  s.className = "save " + (kind || "");
  s.title = { local: "Opened as a file: run `focalcut reel serve <folder>` and open the page from it to save to picks.json" }[kind] || "";
}
// what an action just did: shown for a few seconds over the page (kind: ok, warn, err)
let toastTimer = null;
function toast(text, kind) {
  let t = $("toast");
  if (!t) {
    t = document.body.appendChild(document.createElement("div"));
    t.id = "toast";
    t.setAttribute("role", "status");
  }
  t.textContent = text;
  t.className = "on " + (kind || "ok");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = kind || "ok"), kind == "err" ? 6000 : 3500);
}
const status = toast; // (the name the pages grew up with)

// --- picks: saved to picks.json by `focalcut reel serve`, else kept in this browser ---------------------------------
P = JSON.parse(JSON.stringify(D.picks || {}));
let server = false;
const Sync = { etag: null, base: JSON.stringify(P), timer: null, busy: false, again: false };
const hist = [], fut = [];
let lastSnap = JSON.stringify(P);
const api = () => D.root + "api/picks";
function save() {
  tidy();
  const now_ = JSON.stringify(P);
  if (now_ !== lastSnap) {
    hist.push([lastSnap, now_]);
    if (hist.length > 300) hist.shift();
    fut.length = 0;
    lastSnap = now_;
  }
  commit();
  App.render();
}
function commit() {
  if (server) {
    saveState("Saving…", "busy");
    clearTimeout(Sync.timer);
    Sync.timer = setTimeout(push, 250);
  } else {
    ls.set("picks", P);
    saveState("Kept in this browser only", "local");
  }
  buttons();
}
// the file is replaced only if it is still the one this page last saw (If-Match); one changed meanwhile by another
// page, or by hand, is merged clip by clip (picks.js merge3) and saved again
async function push(keepalive) {
  clearTimeout(Sync.timer);
  Sync.timer = null;
  if (Sync.busy) return void (Sync.again = true);
  Sync.busy = true;
  const sent = JSON.stringify(P);
  try {
    const r = await fetch(api(), {
      method: "PUT", keepalive: keepalive === true,
      headers: { "Content-Type": "application/json", ...(Sync.etag ? { "If-Match": Sync.etag } : {}) },
      body: JSON.stringify(P, null, 1),
    });
    if (r.status == 409) {
      const theirs = await r.json(), m = merge3(JSON.parse(Sync.base), P, theirs);
      Sync.etag = r.headers.get("ETag");
      Sync.base = JSON.stringify(theirs);
      P = m.picks;
      tidy();
      lastSnap = JSON.stringify(P);
      Sync.again = true;
      toast(`picks.json was changed somewhere else — merged with your changes` +
        (m.both.length ? ` (kept yours for ${m.both.join(", ")})` : ""), "warn");
      App.render();
    } else if (!r.ok) throw new Error("the server answered " + r.status);
    else {
      Sync.etag = r.headers.get("ETag");
      Sync.base = sent;
      if (!Sync.again && !Sync.timer) saveState("Saved to picks.json", "ok");
      App.onSaved && App.onSaved();
    }
  } catch (e) {
    saveState("Not saved — " + e.message, "err");
    toast("Could not save to picks.json: " + e.message + ". Your changes are still on this page; the next change tries again.", "err");
  }
  Sync.busy = false;
  if (Sync.again) {
    Sync.again = false;
    push();
  }
}
// back on this page with nothing waiting to be saved: take the file as it is now (another page may have changed it)
async function refresh() {
  if (!server || Sync.busy || Sync.timer || document.hidden) return;
  try {
    const r = await fetch(api(), { cache: "no-store", headers: Sync.etag ? { "If-None-Match": Sync.etag } : {} });
    if (!r.ok) return;
    const j = await r.json(), s = JSON.stringify(j);
    if (Sync.timer || Sync.busy) return;           // (a change was made meanwhile: its save merges instead)
    Sync.etag = r.headers.get("ETag");
    if (s === Sync.base) return;
    P = j;
    Sync.base = lastSnap = s;
    toast("picks.json changed on disk — showing it as it is now");
    buttons();
    App.render();
  } catch (e) {}
}
async function boot() {
  try {
    const r = await fetch(api(), { cache: "no-store" });
    if (!r.ok) throw 0;
    const j = await r.json();
    if (!j || typeof j !== "object" || Array.isArray(j)) throw 0;
    P = j;
    server = true;
    Sync.etag = r.headers.get("ETag");
    Sync.base = lastSnap = JSON.stringify(P);
    document.querySelectorAll("[data-served]").forEach((e) => (e.hidden = false));
    saveState("Saved to picks.json", "ok");
  } catch (e) {
    const l = ls.get("picks");
    if (l && typeof l === "object") P = l;
    lastSnap = JSON.stringify(P);
    saveState(l ? "Kept in this browser only" : "Opened as a file", "local");
    document.querySelectorAll("[data-local]").forEach((e) => (e.hidden = false));
  }
  document.documentElement.classList.toggle("served", server);
  buttons();
  App.onBoot();
}
addEventListener("focus", refresh);
document.addEventListener("visibilitychange", refresh);
addEventListener("pagehide", () => {
  if (server && Sync.timer) push(true);
});
// each change is kept as (before, after), and undone by taking just that change back out of the picks as they are
// now -- so an undo after another page's edit was merged in leaves that edit alone
function step(from, to, what) {
  if (!from.length) return toast("Nothing to " + what, "warn");
  const e = from.pop(), [was, then] = what == "undo" ? [e[1], e[0]] : e;
  to.push(e);
  P = merge3(JSON.parse(was), JSON.parse(then), P).picks;
  tidy();
  lastSnap = JSON.stringify(P);
  commit();
  toast(what == "undo" ? "Undone" : "Redone");
  App.render();
}
const undo = () => step(hist, fut, "undo"), redo = () => step(fut, hist, "redo");
function buttons() {
  if ($("undo")) $("undo").disabled = !hist.length;
  if ($("redo")) $("redo").disabled = !fut.length;
  const r = $("reset");
  if (r) r.hidden = server || !ls.get("picks");
}
// the picks kept in this browser dropped: back to the ones the page was made with (opened as a file)
function resetLocal() {
  if (!confirm("Drop the changes kept in this browser and start again from the picks this page was made with?")) return;
  ls.del("picks");
  P = JSON.parse(JSON.stringify(D.picks || {}));
  lastSnap = JSON.stringify(P);
  hist.length = fut.length = 0;
  saveState("Opened as a file", "local");
  buttons();
  App.render();
}
function picksText() {
  return JSON.stringify(P, null, 1);
}
function copyPicks(b) {
  const done = () => {
    const was = b.textContent;
    b.textContent = "Copied";
    setTimeout(() => (b.textContent = was), 1500);
  };
  (navigator.clipboard ? navigator.clipboard.writeText(picksText()) : Promise.reject()).then(done, () => {
    const t = document.body.appendChild(document.createElement("textarea"));
    t.value = picksText();
    t.select();
    document.execCommand("copy");
    t.remove();
    done();
  });
}
function downloadPicks() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([picksText()], { type: "application/json" }));
  a.download = "picks.json";
  a.click();
}

// --- theme: the system's unless the editor chose one ---------------------------------------------------------------
function theme(t) {
  if (t === undefined) {
    try {
      t = localStorage.getItem("focalcut:theme") || "auto";
    } catch (e) {
      t = "auto";
    }
  } else
    try {
      localStorage.setItem("focalcut:theme", t);
    } catch (e) {}
  if (t == "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  document.querySelectorAll("[data-theme-set]").forEach((b) => b.classList.toggle("on", b.dataset.themeSet == t));
}
theme();

// --- the header's shared buttons and keys ----------------------------------------------------------------------------
document.addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b) {
    if (!ev.target.closest("details.menu")) document.querySelectorAll("details.menu[open]").forEach((d) => (d.open = false));
    return;
  }
  if (b.id == "undo") undo();
  else if (b.id == "redo") redo();
  else if (b.id == "copy") copyPicks(b);
  else if (b.id == "dl") downloadPicks();
  else if (b.id == "reset") resetLocal();
  else if (b.dataset.themeSet) theme(b.dataset.themeSet);
  else if (b.id == "helpb") help(true);
  else if (b.id == "helpx") help(false);
});
function help(on) {
  const h = $("help");
  if (!h) return;
  if (on === undefined) on = !h.open;
  if (on && !h.open) h.showModal();
  else if (!on && h.open) h.close();
}
// keys typed into a field are the field's; a tick box or a button never swallows the page's keys
const typing = (ev) => {
  const t = ev.target, n = t.tagName;
  return n == "TEXTAREA" || n == "SELECT" || t.isContentEditable || (n == "INPUT" && !/^(checkbox|radio|button|range)$/.test(t.type));
};
addEventListener("keydown", (ev) => {
  if (typing(ev)) return;
  const z = ev.key.toLowerCase() == "z", y = ev.key.toLowerCase() == "y";
  if ((ev.ctrlKey || ev.metaKey) && (z || y)) {
    ev.preventDefault();
    ev.stopImmediatePropagation();
    z && !ev.shiftKey ? undo() : redo();
  } else if (ev.key == "?" && !ev.ctrlKey && !ev.metaKey) {
    ev.preventDefault();
    ev.stopImmediatePropagation();
    help();
  }
}, true);
