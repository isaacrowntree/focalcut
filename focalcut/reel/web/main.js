// The source browser's views, its keys and its start.
let curView = 0;
const VIEWS = { 1: "clipsview", 2: "selview", 3: "frameview" }, HASH = { 1: "clips", 2: "cut", 3: "frame" };
function view(n, quiet) {
  if (n == curView) return;
  vid.pause();
  stopShuttle();
  pv.pause();
  fv.pause();
  followMode(false);
  curView = n;
  for (const k in VIEWS) {
    $(VIEWS[k]).classList.toggle("on", k == n);
    $("v" + k).classList.toggle("on", k == n);
    $("v" + k).setAttribute("aria-selected", k == n);
  }
  ls.set("view", n);
  if (!quiet) history.replaceState(null, "", "#" + HASH[n] + (n == 3 && reel != "all" ? "=" + reel : ""));
  renderAll();
  if (n == 1) paint();
}
$("views").onclick = (ev) => {
  const b = ev.target.closest("button");
  if (b) view(+b.id.slice(1));
};
function renderAll() {
  if (curView == 1) {
    list();
    draw();
    mysel();
  }
  selview();
  frameview();
  counts();
}
App.render = renderAll;
function openInClips(i, t, k) {
  view(1);
  go(i, t);
  act = k;
  seek(t);
  renderAll();
}
// the address says where the page is: #clips, #cut, #frame or #frame=<reel> (the portrait framing of that reel: the
// menu's "Frame … portrait" links)
function route() {
  const m = /^#(clips|cut|frame)(?:=([\w-]+))?(?:&shot=(\d+))?/.exec(location.hash);
  if (!m) return false;
  if (m[2]) setReel(m[2], true);
  view({ clips: 1, cut: 2, frame: 3 }[m[1]], true);
  if (m[3] && curView == 3) {                      // (a review page's "Frame" link: straight to that shot)
    ffilter = "all";
    frameview();
    fpick(Math.max(0, forder.findIndex((x) => x.n == +m[3])), false);
  }
  return true;
}
addEventListener("hashchange", route);
App.onBoot = () => {
  useReel();
  const last = ls.get("clip");
  curView = 0;
  go(Math.max(0, D.clips.findIndex((c) => c.id == last)));
  if (!route()) view(ls.get("view") || 1, true);
  if (curView == 3) toast(`Framing the ${rlab(reel == "all" ? "main" : reel)} portrait cut: drag to pan the crop, click a dancer to follow them, ← → from shot to shot`);
};
addEventListener("resize", () => {
  if (curView == 1) paint();
  else redrawFrames();
});

// --- keys ------------------------------------------------------------------------------------------------------------
const star = (k) => /^[1-5]$/.test(k);
const KEYS = {
  1(k, K, ev) {
    if (star(k)) starHere(+k);
    else if (k == " " || K == "k") togglePlay();
    else if (K == "l") playL();
    else if (K == "j") playJ();
    else if (k == "ArrowLeft" || k == ",") stepBy(ev.shiftKey ? -1 : -FR);
    else if (k == "ArrowRight" || k == ".") stepBy(ev.shiftKey ? 1 : FR);
    else if (k == "ArrowUp" || k == "PageUp") nav(-1);
    else if (k == "ArrowDown" || k == "PageDown") nav(1);
    else if (K == "i") markIn();
    else if (K == "o") markOut();
    else if (k == "Enter") addIO();
    else if (K == "x") {
      inPt = outPt = null;
      draw();
    } else if (k == "Backspace" || k == "Delete") delHere();
    else if (k == "[") jump(-1);
    else if (k == "]") jump(1);
    else if (k == "\\") loopHere();
    else if (K == "b") banIO();
    else if (k == "=" || k == "+") zoom(0.6);
    else if (k == "-" || k == "_") zoom(1.6);
    else if (k == "0") fit();
    else if (k == "Home") seek(0);
    else if (k == "End") seek(dur());
    else if (k == "Escape") pause();
    else return false;
  },
  2(k, K, ev) {
    const x = PX();
    if (chosen.size && (k == "Backspace" || k == "Delete")) dropTicked();
    else if (k == "Escape" && chosen.size) untick();
    else if (K == "a" && !ev.shiftKey) tickAll();
    else if (K == "x" && x) tick(pj(), false);
    else if (k == " ") playPick();
    else if (k == "Escape") stopAll();
    else if (k == "ArrowRight" || k == "ArrowDown") pstep(1);
    else if (k == "ArrowLeft" || k == "ArrowUp") pstep(-1);
    else if (!x) return false;
    else if (K == "i") pIn();
    else if (K == "o") pOut();
    else if (k == "," || k == "<") nudge("slip", ev.shiftKey ? -0.5 : -0.02);
    else if (k == "." || k == ">") nudge("slip", ev.shiftKey ? 0.5 : 0.02);
    else if (k == "\\") loopPick();
    else if (k == "Backspace" || k == "Delete") pRemove();
    else if (star(k)) pStar(+k);
    else return false;
  },
  3(k, K, ev) {
    if (k == " " || K == "k") fplay();
    else if (k == "ArrowRight" || k == "ArrowDown") fstep(1);
    else if (k == "ArrowLeft" || k == "ArrowUp") fstep(-1);
    else if (k == "," || k == "<") fnudge(ev.shiftKey ? -0.5 : -FR);
    else if (k == "." || k == ">") fnudge(ev.shiftKey ? 0.5 : FR);
    else if (k == "Backspace" || k == "Delete") freset();
    else if (k == "Home") fpick(0, false);
    else if (k == "End") fpick(forder.length - 1, false);
    else if (k == "Escape") fv.pause();
    else return false;
  },
};
addEventListener("keydown", (ev) => {
  if (typing(ev) || ev.ctrlKey || ev.metaKey || ev.altKey || $("help").open) return;
  const k = ev.key, K = k.toLowerCase(), el = document.activeElement;
  let done = true;
  if (k == "Tab") view(((curView + (ev.shiftKey ? 1 : 0)) % 3) + 1);
  else if (K == "f" && curView != 3) toggleFrame();
  else if (K == "g") curView == 3 ? toast("Click a dancer to follow them through this shot") : followMode(!followOn);
  else if (k == "Escape" && followOn) followMode(false);
  else done = KEYS[curView](k, K, ev) !== false;
  if (!done) return;
  ev.preventDefault();
  ev.stopPropagation();
  if (el && el != document.body) el.blur();        // (a key is the page's, not the button's last clicked)
}, true);
boot();
