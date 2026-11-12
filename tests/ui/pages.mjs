// The source browser and a review page driven in a headless Chromium, on the small project tests/test_web.py makes
// (two clips, two reels, no footage: the pages stand on their filmstrip fallbacks). Every step checks what landed in
// picks.json through the server, and the run ends with the file as it began.
//     node tests/ui/pages.mjs http://127.0.0.1:<port>/        (CHROME: the browser; run by pytest)
import { launch, picks } from "./cdp.mjs";
const B = process.argv[2], U = B + "focalcut/reel/review/";
const fails = [];
const ok = (cond, what, got) => {
  if (!cond) fails.push(what + (got === undefined ? "" : " — got " + JSON.stringify(got)));
  console.log((cond ? "ok   " : "FAIL ") + what);
};
const orig = await picks(B);
const p = await launch({ width: 1440, height: 900 });
const saved = async () => { await p.sleep(450); return picks(B); };
const text = (id) => p.eval(`document.getElementById("${id}").textContent`);
const lane = async (sel, t, dur = 60) => p.at(sel, t / dur, 0.5);
const undoAll = async () => { for (let i = 0; i < 40 && !(await p.eval(`document.getElementById("undo").disabled`)); i++) await p.key("z", { mods: 2 }); return saved(); };

await p.goto(U + "sources.html", 800); await p.eval(`localStorage.clear()`);
await p.goto(U + "sources.html#clips", 1500);
ok((await text("saved")) == "Saved to picks.json", "served: the header says the picks are saved", await text("saved"));
ok(JSON.stringify(await saved()) == JSON.stringify(orig), "opening the page leaves picks.json alone");
ok(await p.eval(`document.querySelectorAll("#list .ci").length`) == 2, "both clips are listed");
const playable = await p.eval(`vid.videoWidth > 0`);
console.log(playable ? "     (C1 plays: the players are checked too)" : "     (no playable clip: the players stand on their filmstrips)");
if (playable) {
  await p.eval(`seek(11)`); await p.key(" "); await p.sleep(900);
  ok(await p.eval(`!vid.paused && vid.currentTime > 11.3`) && /Pause/.test(await text("play")), "Space plays from the playhead");
  await p.key(" ");
  ok(await p.eval(`vid.paused`), "Space pauses");
  await p.eval(`seek(0)`);
}

// --- Clips: the Short reel; make, move, star and remove a selection ---
await p.clickOn(`#reelsw [data-reel=short]`);
ok(await p.eval(`reel`) == "short" && /picks/.test(await text("counts")), "the reel switch chooses Short");
ok(await p.eval(`document.querySelectorAll("#lsel .blk.away").length`) == 1, "the Main-only pick is faded in Short");
let [x0, y] = await lane("#lsel", 20), [x1] = await lane("#lsel", 25);
await p.drag(x0, y, x1, y);
let s = await saved(), add = s.C1.add.find((v) => v.startsWith("0:20."));
ok(add == "0:20.00-0:25.00" && JSON.stringify(s.C1.only[add]) == '["short"]', "a drag in the lane makes a selection for Short", s.C1);
ok(/added for Short/.test(await text("toast")), "… and says so", await text("toast"));
[x0] = await lane("#lsel", 22); [x1] = await lane("#lsel", 35);
await p.drag(x0, y, x1, y);
s = await saved();
ok(s.C1.add.includes("0:33.00-0:38.00") && !s.C1.add.includes(add) && JSON.stringify(s.C1.only["0:33.00-0:38.00"]) == '["short"]' && !(add in s.C1.only),
  "dragging it past its own length moves it: same pick, same reel, nothing left behind", s.C1);
await p.eval(`seek(34)`); await p.key("5");
s = await saved();
ok(s.C1.stars["0:33.00-0:38.00"] == 5, "key 5 stars the selection at the playhead", s.C1.stars);
[x0] = await lane("#lsel", 38); [x1] = await lane("#lsel", 36.5);
await p.drag(x0 - 1, y, x1, y);
s = await saved();
const trimmed = s.C1.add.find((v) => v.startsWith("0:33.00-"));
ok(trimmed && trimmed != "0:33.00-0:38.00" && s.C1.stars[trimmed] == 5, "trimming it keeps its stars", s.C1);
await p.eval(`document.querySelector("#mysel tr.act").scrollIntoView({block:"nearest"})`);
await p.clickOn(`#mysel tr.act input[data-r=main]`);
await p.eval(`document.getElementById("main").scrollTop=0`);
s = await saved();
ok(!(trimmed in (s.C1.only || {})), "ticking Main puts it in every reel", s.C1.only);
await p.key(" ");
ok(await p.eval(`document.activeElement === document.body || document.activeElement.tagName != "INPUT"`), "a tick box does not keep the keys");
// the opening role follows a pick moved in Clips
await p.key("Tab");
ok(await p.eval(`curView`) == 2, "Tab goes to Cut & picks");
const j = await p.eval(`order.findIndex(x=>x.k>=0&&Math.abs(x.a-33)<0.01)`);
await p.eval(`pick(${j},false)`); await p.sleep(300);
ok(/0:33\.00/.test(await text("pvtc")), "the shot in hand shows at its in-point, video or not", await text("pvtc"));
await p.clickOn("#pfirst");
s = await saved();
ok(s.C1.open == trimmed && JSON.stringify(s.C1.only["open:" + trimmed]) == '["short"]', "Opens the reel: this pick opens Short", s.C1);
ok(/Main 2:46 no longer opens/.test(await text("toast")), "… and warns that Main's opening in the same clip gave way (a clip holds one)", await text("toast"));
await p.clickOn(`#ped [data-n="slip:0.5"]`);
s = await saved();
ok(s.C1.open.startsWith("0:33.50-") && s.C1.add.includes(s.C1.open) && s.C1.stars[s.C1.open] == 5, "slipping the pick takes its role and stars along", [s.C1.open, s.C1.add]);
await p.clickOn("#popen"); await p.sleep(400);
ok(await p.eval(`curView`) == 1 && Math.abs(await p.eval(`now()`) - 33.5) < 0.01, "Open in Clips lands on the shot", await p.eval(`now()`));
[x0] = await lane("#lsel", 34.5); [x1] = await lane("#lsel", 44.5);
await p.drag(x0, y, x1, y);
s = await saved();
ok(s.C1.open && s.C1.open.startsWith("0:43.50-") && s.C1.add.includes(s.C1.open) && s.C1.stars[s.C1.open] == 5,
  "moving the opening pick in Clips moves the opening with it (no orphan)", s.C1);
// removing an every-reel pick while on Short
await p.eval(`seek(11)`); await p.key("Backspace");
s = await saved();
ok(s.C1.add.includes("0:10.00-0:12.00") && JSON.stringify(s.C1.only["0:10.00-0:12.00"]) == '["main"]' && /still in Main/.test(await text("toast")),
  "Backspace on Short takes an every-reel pick out of Short only, and says it stays in Main", [s.C1.only, await text("toast")]);
// bans: I / O / B, ✕ on a reel shot
await p.eval(`seek(2)`); await p.key("i"); await p.eval(`seek(4)`); await p.key("o"); await p.key("b");
s = await saved();
ok(s.C1.ban.includes("0:02.00-0:04.00") && JSON.stringify(s.C1.only["ban:0:02.00-0:04.00"]) == '["short"]', "I, O, B bans the in–out for Short", s.C1.ban);
const use = await p.eval(`(()=>{const r=document.querySelector("#lland .use:not(.out)").getBoundingClientRect();return[r.left,r.top,r.width,r.height]})()`);
await p.mouse("mouseMoved", use[0] + use[2] / 2, use[1] + 8, { buttons: 0 });
await p.click(use[0] + use[2] - 6, use[1] + 8);
s = await saved();
ok(s.C1.ban.includes("0:20.00-0:21.00"), "✕ on a reel shot bans its stretch", s.C1.ban);
// undo all the way, redo one
s = await undoAll();
ok(JSON.stringify(s) == JSON.stringify(orig), "Ctrl+Z all the way back gives the file it started from");
await p.key("z", { mods: 10 });
s = await saved();
ok(s.C1.add.includes("0:20.00-0:25.00"), "Ctrl+Shift+Z redoes", s.C1.add);
s = await undoAll();

// --- Frame portrait: pan by dragging, follow by clicking, reset ---
await p.goto(U + "sources.html#frame=short", 300); await p.eval(`location.reload()`); await p.sleep(1500);
ok(await p.eval(`curView`) == 3 && await p.eval(`reel`) == "short" && await p.eval(`forder.length`) == 2, "#frame=short opens the Short portrait cut, shot by shot");
ok(/planner/.test(await text("fstate")), "a shot starts with the planner's framing", await text("fstate"));
if (playable) {
  ok(Math.abs(await p.eval(`fv.currentTime`) - 20) < 0.05, "the shot opens on its first frame", await p.eval(`fv.currentTime`));
  await p.key(" "); await p.sleep(2700);
  ok(await p.eval(`!fv.paused && fv.currentTime >= 19.95 && fv.currentTime <= 22.1`), "Space plays the shot and loops inside it", await p.eval(`fv.currentTime`));
  await p.key(" ");
  ok(await p.eval(`(()=>{const c=document.getElementById("fcanvas"),d=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<d.length;i+=400)if(d[i]+d[i+1]+d[i+2]>60)n++;return n})()`) > 100,
    "the result beside the picture is drawn from the video");
}
let box = await p.eval(`(()=>{const r=document.querySelector("#ffr .fbox.act").getBoundingClientRect();return[r.left,r.top,r.width,r.height]})()`);
await p.drag(box[0] + box[2] / 2, box[1] + box[3] / 2, box[0] + box[2] / 2 + 120, box[1] + box[3] / 2);
s = await saved();
const key = (s.C1.portrait || [])[0];
ok(key && key[0] == "0:21.00" && key[1] > 0.55, "dragging the frame pans the whole shot's crop", s.C1.portrait);
ok((await text("fstate")) == "Your pan" && /panned/.test(await text("toast")) && /panned/.test(await p.eval(`document.querySelector("#fshots .fs.cur .badge").textContent`)),
  "… and the page shows it took: the state, a toast, the shot's badge", [await text("fstate"), await text("toast")]);
const moved = await p.eval(`(()=>{const r=document.querySelector("#ffr .fbox.act").getBoundingClientRect();return r.left})()`);
ok(moved > box[0] + 60, "… and the frame is where it was dragged", [box[0], moved]);
const [cx, cy] = await p.at("#fwrap", 0.3, 0.4);
await p.click(cx, cy);
s = await saved(); await p.sleep(600);
ok(s.C1.follow && s.C1.follow.length == 1 && !s.C1.portrait, "a click follows the person there (the pan gives way)", s.C1);
ok(/Following|No face/.test(await text("fstate")) && /following/.test(await p.eval(`document.querySelector("#fshots .fs.cur .badge").textContent`)), "… and the page says so", await text("fstate"));
await p.key("ArrowRight");
ok(await p.eval(`FX().c.id`) == "C2" && await p.eval(`document.getElementById("fwrap").getBoundingClientRect().top`) > 0, "→ goes to the next shot, the picture stays on screen");
await p.key("ArrowLeft"); await p.key("Backspace");
s = await saved();
ok(!s.C1.follow && !s.C1.portrait && /planner/.test(await text("fstate")), "Backspace gives the shot back to the planner", s.C1);
s = await undoAll();
ok(JSON.stringify(s) == JSON.stringify(orig), "framing undone: the file as it started");

// --- two pages on one project: the review page bans, this page stars; neither is lost ---
const q = await launch({ width: 1200, height: 800 });
await q.goto(U + "short_portrait.html", 1200);
ok(/Short 1:30/.test(await q.eval(`document.querySelector("h1").textContent`)), "the review page names its reel");
await q.eval(`document.querySelector('figure[data-i="1"] [data-act=ban]').click()`);
s = await saved();
ok(s.C2 && s.C2.ban[0] == "0:03.00-0:04.00" && JSON.stringify(s.C2.only) == '{"ban:0:03.00-0:04.00":["short"]}', "the review page bans a shot for its reel, to the frame", s.C2);
await p.eval(`view(1);go(0);seek(11)`); await p.key("2");
s = await saved(); await p.sleep(500); s = await picks(B);
ok(s.C2 && s.C2.ban && s.C1.stars["0:10.00-0:12.00"] == 2, "a change on the other (stale) page merges: both are in the file", s);
await q.eval(`openShot(0)`); await q.sleep(300);
await q.key("m"); await q.sleep(500);
s = await picks(B);
ok(s.C1.add.includes("0:20.00") && s.C1.stars["0:10.00-0:12.00"] == 2, "the review page's marker merges with the star", s.C1);
await q.key("z", { mods: 2 }); await q.key("Escape"); await q.sleep(300);
await q.eval(`document.querySelector('figure[data-i="1"] [data-act=ban]').click()`); await q.sleep(600);
await p.eval(`window.dispatchEvent(new Event("focus"))`); await p.sleep(600);
await p.eval(`seek(11)`); await p.key("4");
s = await saved();
ok(JSON.stringify(s) == JSON.stringify(orig), "everything put back by hand: the file as it started", s);
ok(!p.errors.length && !q.errors.length, "no script errors", [p.errors, q.errors]);
await p.close(); await q.close();
if (fails.length) { console.log("\n" + fails.length + " failed:\n" + fails.join("\n")); process.exit(1); }
console.log("\nall passed");
