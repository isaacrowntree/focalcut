"""The review pages' front end (focalcut/reel/web/): the pages are put together whole, the server keeps two pages'
edits of one picks.json apart, the picks model does what the planner reads (under Node, when there is one), and the
source browser itself works in a headless Chromium (when there is one: FOCALCUT_CHROME, or Playwright's)."""
import glob
import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from focalcut.reel import browse, review, serve, webui
from focalcut.reel.project import Project, only_reel

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = shutil.which("node")
CROP = dict(w=0.3164, h=1.0, z=1.0, path=[[0.0, 0.5, 0.5], [1.0, 0.55, 0.5]])
REELS = [{"id": "main", "label": "Main 2:46"}, {"id": "short", "label": "Short 1:30"}]
PICKS = {"C1": {"add": ["0:10.00-0:12.00", "0:50.00-0:52.00"], "only": {"0:50.00-0:52.00": ["main"]},
                "stars": {"0:10.00-0:12.00": 4}, "open": "0:10.00-0:12.00", "ban": ["0:30.00-0:31.00"]},
         "_note": "kept as it is"}


def clip(cid, dur, main, short, cat="night"):
    return dict(id=cid, cat=cat, tag="", clock="21:00", sort=cid, dur=dur, video=f"../../../export/{cid}.mp4",
                ar=16 / 9, strip=f"sources/{cid}.jpg", tile=[160, 90], step=1.0, n=int(dur), moments=[[2.0, "spin"]],
                L=main[0], P=main[1], R={"main": dict(L=main[0], P=main[1]), "short": dict(L=short[0], P=short[1])})


def project(tmp_path, video=False):
    """A project as the server sees it, with a sources page and a portrait review page of its "short" cut made from
    hand-written clips (no footage: the pages fall back to their filmstrips, which are missing too). video: C1 gets
    a test pattern a headless Chromium can play (VP8, whatever the file is called), when ffmpeg can make one."""
    folder = tmp_path / "Night Out"
    os.makedirs(folder / "focalcut" / "reel" / "review")
    if video and shutil.which("ffmpeg"):
        os.makedirs(folder / "export")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25:duration=60",
                        "-c:v", "libvpx", "-b:v", "120k", "-g", "5", "-f", "webm", str(folder / "export" / "C1.mp4")],
                       check=False)
    (folder / "focalcut" / "reel" / "reel.json").write_text(json.dumps({"cuts": {"short": {}}}))
    (folder / "focalcut" / "reel" / "picks.json").write_text(json.dumps(PICKS, indent=1))
    pr = Project(str(folder))
    clips = [clip("C1", 60.0, ([[10.0, 12.0, 1], [30.0, 31.0, 3]], [[10.0, 12.0, 1, CROP], [40.0, 41.5, 2, CROP]]),
                  ([[20.0, 21.0, 1]], [[20.0, 22.0, 1, CROP]])),
             clip("C2", 30.0, ([[5.0, 6.0, 2]], []), ([], [[3.0, 4.0, 2, CROP]]), cat="day")]
    nav = browse.menu(pr, "sources")
    (folder / "focalcut" / "reel" / "review" / "sources.html").write_text(
        browse.page(clips, PICKS, key="t", title="Night Out", root=browse.api_root(pr), nav=nav, reels=REELS))
    shots = [dict(unit=dict(id="C1", export_t0=0.0), src_in=20.0, length=2.0, act="night", kind="spin", fx=[]),
             dict(unit=dict(id="C2", export_t0=0.0), src_in=3.0, length=1.0, act="night", kind="dip", fx=[])]
    (folder / "focalcut" / "reel" / "review" / "short_portrait.html").write_text(
        review.page(dict(shots=shots), "portrait", [None, None], picks=PICKS, key="t/short_portrait",
                    root=browse.api_root(pr), nav=nav, project="Night Out", reel="short", reels=REELS))
    return pr


def served(pr):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.handler(pr))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


def http(url, data=None, method=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


# --- the pages ---------------------------------------------------------------------------------------------------------
def test_pages_are_whole_and_carry_the_front_end_version(tmp_path):
    pr = project(tmp_path)
    for name, needs in (("sources.html", ("id=frameview", "function merge3", "function frameMouse", "api/crop")),
                        ("short_portrait.html", ("Short 1:30", "function merge3", "function openShot",
                                                 'href="sources.html#frame=short&shot=1"'))):
        path = pr.path("review", name)
        page = open(path).read()
        assert all(n in page for n in needs), name
        assert page.count("<script>") == 2 and "%" + "NAV%" not in page and "%" + "TITLE%" not in page
        assert '"/api/' not in page and "`/api/" not in page             # relative: works under any server prefix
        assert "http://" not in page and "https://" not in page          # nothing fetched: works opened as a file
        assert webui.stamp_of(path) == webui.version()
    assert webui.stamp_of(str(tmp_path / "none.html")) is None


def test_every_script_parses():
    if not NODE:
        pytest.skip("no node")
    for f in sorted(glob.glob(os.path.join(webui.WEB, "*.js"))):
        r = subprocess.run([NODE, "--check", f], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_a_portrait_shot_carries_its_crop_to_the_page():
    plan = dict(shots=[dict(unit=dict(id="C1"), src_in=12.0, length=1.0,
                            crop=dict(w=0.3, h=1.0, z=1.2, follow="face", fill=0.9, path=[[0.0, 0.51234, 0.5]])),
                       dict(unit=dict(id="C1"), src_in=20.0, length=2.0, crop=None)])
    uses = browse._uses({"portrait": plan, "landscape": plan}, "C1", 2.0)
    assert uses["landscape"] == [[10.0, 11.0, 1], [18.0, 20.0, 2]]
    assert uses["portrait"] == [[10.0, 11.0, 1, dict(w=0.3, h=1.0, z=1.2, follow="face", path=[[0.0, 0.512, 0.5]])],
                                [18.0, 20.0, 2, None]]


# --- the server --------------------------------------------------------------------------------------------------------
def test_two_pages_cannot_overwrite_each_other(tmp_path):
    pr = project(tmp_path)
    srv, base = served(pr)

    def put(o, tag=None):
        return http(base + "api/picks", json.dumps(o).encode(), "PUT",
                    {"Content-Type": "application/json", **({"If-Match": tag} if tag else {})})
    try:
        st, body, h = http(base + "api/picks")
        tag = h["ETag"]
        assert st == 200 and json.loads(body) == PICKS
        assert http(base + "api/picks", headers={"If-None-Match": tag})[0] == 304
        st, _, h2 = put({**PICKS, "C2": {"ban": ["all"]}}, tag)                 # page A saves
        assert st == 200 and h2["ETag"] != tag
        st, body, h3 = put({**PICKS, "C3": {"more": 1}}, tag)                   # page B, from the same start: refused
        assert st == 409 and json.loads(body)["C2"] == {"ban": ["all"]} and h3["ETag"] == h2["ETag"]
        assert put({**json.loads(body), "C3": {"more": 1}}, h3["ETag"])[0] == 200   # ... merged, it goes through
        assert put({"C9": {}})[0] == 200                                        # no If-Match: as before
        assert put([1, 2])[0] == 400
    finally:
        srv.shutdown()
    assert json.load(open(pr.path("picks.json"))) == {"C9": {}}
    assert "C3" in json.load(open(pr.path("picks.json.bak")))
    hist = os.listdir(pr.path("picks_history"))                                 # the session's start is kept
    assert len(hist) == 1 and json.load(open(pr.path("picks_history", hist[0]))) == PICKS


def test_files_are_revalidated_and_a_failed_crop_says_why(tmp_path):
    pr = project(tmp_path)
    os.makedirs(pr.path("review", "sources"))
    with open(pr.path("review", "sources", "C1.jpg"), "wb") as f:
        f.write(b"jpeg")
    with open(pr.path("units.json"), "w") as f:
        json.dump([dict(id="C1", score="C1", export_t0=0.0)], f)
    srv, base = served(pr)
    try:
        url = base + "focalcut/reel/review/sources/C1.jpg"
        st, body, h = http(url)
        assert st == 200 and body == b"jpeg" and h["Cache-Control"] == "no-cache" and h["Last-Modified"]
        assert http(url, headers={"If-Modified-Since": h["Last-Modified"]})[0] == 304
        assert http(url, headers={"Range": "bytes=0-1"})[1] == b"jp"
        st, body, _ = http(base + "api/crop?id=C1&a=1&b=2")                     # no scores: an answer, not a hang-up
        assert st in (200, 500) and (st == 200 or "error" in json.loads(body))
        assert http(base + "api/crop?id=nope&a=1&b=2")[0] == 400
    finally:
        srv.shutdown()


def test_serve_makes_a_stale_sources_page_again(tmp_path, monkeypatch):
    pr = project(tmp_path)
    page = pr.path("review", "sources.html")
    made = []
    monkeypatch.setattr(browse, "run", lambda p: made.append(p.folder))
    monkeypatch.setattr(serve.ThreadingHTTPServer, "serve_forever", lambda self: None)
    serve.run(pr, port=0)
    assert made == []                                                           # this version's page: left alone
    with open(page, "w") as f:
        f.write("<!doctype html><title>an old page</title>")
    serve.run(pr, port=0)
    assert made == [pr.folder]


# --- the picks model, as the planner reads what it writes ---------------------------------------------------------------
MODEL = r"""
const vm = require("vm"), fs = require("fs");
const ctx = vm.createContext({ D: { reels: [{ id: "main", label: "Main" }, { id: "short", label: "Short" }] }, console });
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
const run = (code) => vm.runInContext(code, ctx);
const out = {};
run(`P = ${process.argv[3]}`);
// a pick made for Short; moved: its reel, stars and role go with it
run(`addSel("C1", 20, 25, "short"); putStars("C1", 20, 25, 5); setRole("C1", 20, 25, "end", "short");
     moveSel("C1", 20, 25, 33, 38.5); tidy()`);
out.moved = run(`JSON.parse(JSON.stringify(P.C1))`);
// the every-reel pick taken out of Short: it stays for Main, and so does the opening it holds
out.left = run(`removeSel("C1", 10, 12, "short")`);
// a pick in one reel cannot be unticked out of it; removing it there removes it, its stars and its role
out.last = run(`setIn("C1", 33, 38.5, "short", false)`);
out.gone = run(`removeSel("C1", 33, 38.5, "short")`);
run(`tidy()`);
out.after = run(`JSON.parse(JSON.stringify(P))`);
// framing: a pan and a followed person in one shot; times and ranges as written
run(`putKeys("C1", "portrait", [[20.5, 0.4, 0.5]]); putKeys("C1", "follow", [[41, 0.6, 0.3]])`);
out.framing = [run(`framingOf("C1", 20, 22)`), run(`framingOf("C1", 40, 41.5)`), run(`framingOf("C1", 30, 31)`)];
out.span = [run(`span("1:12.40-1:15.10")`), run(`span("1:02:03.5")`), run(`rkey(72.4, 75.1)`)];
// two edits of one file
out.merge = run(`merge3({a: 1, b: 1, c: 1}, {a: 2, b: 1}, {a: 1, b: 3, c: 1, d: 4})`);
out.clash = run(`merge3({a: 1}, {a: 2}, {a: 3})`);
console.log(JSON.stringify(out));
"""


def test_picks_model_under_node(tmp_path):
    if not NODE:
        pytest.skip("no node")
    script = tmp_path / "model.js"
    script.write_text(MODEL)
    r = subprocess.run([NODE, str(script), os.path.join(webui.WEB, "picks.js"), json.dumps(PICKS)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    moved = out["moved"]
    assert "0:33.00-0:38.50" in moved["add"] and "0:20.00-0:25.00" not in moved["add"]
    assert moved["only"]["0:33.00-0:38.50"] == ["short"] and moved["stars"]["0:33.00-0:38.50"] == 5
    assert moved["end"] == "0:33.00-0:38.50" and moved["only"]["end:0:33.00-0:38.50"] == ["short"]
    assert only_reel(moved, "short")["end"] == "0:33.00-0:38.50" and "end" not in only_reel(moved, "main")
    assert out["left"] == ["main"] and out["last"] is False and out["gone"] == []
    c1 = out["after"]["C1"]
    assert c1["add"] == ["0:10.00-0:12.00", "0:50.00-0:52.00"] and "end" not in c1 and c1["stars"] == {"0:10.00-0:12.00": 4}
    assert c1["only"] == {"0:50.00-0:52.00": ["main"], "0:10.00-0:12.00": ["main"], "open:0:10.00-0:12.00": ["main"]}
    assert "add" not in only_reel(c1, "short") and "open" not in only_reel(c1, "short")
    assert only_reel(c1, "main")["open"] == "0:10.00-0:12.00" and out["after"]["_note"] == "kept as it is"
    assert out["framing"] == ["pan", "follow", "auto"]
    assert out["span"] == [[72.4, 75.1], [3723.5, 3723.5], "1:12.40-1:15.10"]
    assert out["merge"] == {"picks": {"a": 2, "b": 3, "d": 4}, "theirs": 2, "both": []}
    assert out["clash"] == {"picks": {"a": 2}, "theirs": 1, "both": ["a"]}


# --- the pages in a browser ----------------------------------------------------------------------------------------------
def chrome():
    found = [os.environ.get("FOCALCUT_CHROME"), *sorted(glob.glob(os.path.expanduser(
        "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")), reverse=True),
        shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("google-chrome")]
    return next((f for f in found if f and os.path.exists(f)), None)


def test_the_pages_in_a_browser(tmp_path):
    if not NODE or not chrome() or int(subprocess.run([NODE, "-p", "process.versions.node.split('.')[0]"],
                                                      capture_output=True, text=True).stdout or 0) < 22:
        pytest.skip("needs node 22+ and a Chromium (FOCALCUT_CHROME)")
    pr = project(tmp_path, video=True)
    srv, base = served(pr)
    try:
        r = subprocess.run([NODE, os.path.join(HERE, "ui", "pages.mjs"), base], capture_output=True, text=True,
                           timeout=240, env={**os.environ, "CHROME": chrome(), "SCRATCH": str(tmp_path)})
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert json.load(open(pr.path("picks.json"))) == PICKS                      # every step was undone
    short = Project(pr.folder, "short").picks()["C1"]                           # ... and the planner reads it per reel
    assert short["add"] == ["0:10.00-0:12.00"] and short["open"] == "0:10.00-0:12.00"
