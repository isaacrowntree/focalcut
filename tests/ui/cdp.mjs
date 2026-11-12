// A small Chrome DevTools Protocol driver: no dependencies (Node 22's fetch + WebSocket).
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = process.env.CHROME || `${process.env.HOME}/.cache/ms-playwright/chromium-1134/chrome-linux/chrome`;
const sleep = ms => new Promise(r => setTimeout(r, ms));

export async function launch({ width = 1440, height = 900, port = 9300 + Math.floor(Math.random() * 500), dark = true } = {}) {
  const dir = mkdtempSync(join(process.env.SCRATCH || tmpdir(), "chrome-"));
  const proc = spawn(CHROME, ["--headless=new", "--no-sandbox", "--disable-gpu", `--remote-debugging-port=${port}`,
    `--user-data-dir=${dir}`, `--window-size=${width},${height}`, "--autoplay-policy=no-user-gesture-required",
    "--hide-scrollbars", "about:blank"], { stdio: "ignore" });
  let ws = null;
  for (let i = 0; i < 80 && !ws; i++) {
    try {
      const tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      const t = tabs.find(x => x.type == "page");
      if (t) ws = new WebSocket(t.webSocketDebuggerUrl);
    } catch (e) { await sleep(100); }
  }
  if (!ws) { proc.kill(); throw new Error("chrome did not start"); }
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const waiting = new Map(), logs = [], errors = [];
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.id && waiting.has(m.id)) { const [res, rej] = waiting.get(m.id); waiting.delete(m.id); m.error ? rej(new Error(m.error.message)) : res(m.result); }
    else if (m.method == "Runtime.consoleAPICalled") logs.push(m.params.type + ": " + m.params.args.map(a => a.value ?? a.description).join(" "));
    else if (m.method == "Runtime.exceptionThrown") errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  };
  const send = (method, params = {}) => new Promise((res, rej) => { waiting.set(++id, [res, rej]); ws.send(JSON.stringify({ id, method, params })); });
  await send("Page.enable"); await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: dark ? "dark" : "light" }] });
  const page = {
    send, logs, errors, sleep,
    async goto(url, wait = 1200) { await send("Page.navigate", { url }); await sleep(wait); },
    async eval(expr) {
      const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) throw new Error("eval: " + (r.exceptionDetails.exception?.description || r.exceptionDetails.text) + "\n  in: " + expr.slice(0, 200));
      return r.result.value;
    },
    async shot(file, full = false) {
      const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: full });
      writeFileSync(file, Buffer.from(r.data, "base64"));
    },
    async size(w, h, dark_ = dark) {
      await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile: w < 600 });
      await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: dark_ ? "dark" : "light" }] });
      await sleep(300);
    },
    async mouse(type, x, y, extra = {}) { await send("Input.dispatchMouseEvent", { type, x, y, button: "left", buttons: type == "mouseReleased" ? 0 : 1, clickCount: 1, ...extra }); },
    async click(x, y, mods = 0) { await page.mouse("mousePressed", x, y, { modifiers: mods }); await page.mouse("mouseReleased", x, y, { modifiers: mods }); await sleep(60); },
    async drag(x0, y0, x1, y1, steps = 6) {
      await page.mouse("mouseMoved", x0, y0, { buttons: 0 }); await page.mouse("mousePressed", x0, y0);
      for (let i = 1; i <= steps; i++) await page.mouse("mouseMoved", x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps);
      await page.mouse("mouseReleased", x1, y1); await sleep(60);
    },
    // the centre of an element (CSS selector), or a point a fraction (fx, fy) across it
    async at(sel, fx = 0.5, fy = 0.5) {
      const r = await page.eval(`(()=>{const e=document.querySelector(${JSON.stringify(sel)});if(!e)return null;const r=e.getBoundingClientRect();return[r.left,r.top,r.width,r.height]})()`);
      if (!r) throw new Error("no element " + sel);
      return [r[0] + r[2] * fx, r[1] + r[3] * fy];
    },
    async clickOn(sel, fx = 0.5, fy = 0.5, mods = 0) { const [x, y] = await page.at(sel, fx, fy); await page.click(x, y, mods); },
    async key(key, { code, mods = 0, text } = {}) {
      const vk = { Enter: 13, Tab: 9, Backspace: 8, Delete: 46, Escape: 27, ArrowLeft: 37, ArrowRight: 39, ArrowUp: 38, ArrowDown: 40, " ": 32 }[key] ?? key.toUpperCase().charCodeAt(0);
      const p = { key, code: code || (key.length == 1 ? "Key" + key.toUpperCase() : key), windowsVirtualKeyCode: vk, modifiers: mods };
      await send("Input.dispatchKeyEvent", { type: "rawKeyDown", ...p });
      if ((text ?? (key.length == 1 ? key : null)) && !(mods & 6)) await send("Input.dispatchKeyEvent", { type: "char", ...p, text: text ?? key });
      await send("Input.dispatchKeyEvent", { type: "keyUp", ...p }); await sleep(40);
    },
    async close() { send("Browser.close").catch(() => {}); await sleep(150); proc.kill(); ws.close(); },
  };
  return page;
}
export const picks = async base => (await fetch(base + "api/picks", { cache: "no-store" })).json();
