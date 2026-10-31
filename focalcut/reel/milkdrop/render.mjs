// Render MilkDrop presets (Butterchurn, WebGL in headless Chromium) frame by frame from a song, to a video file.
// Each frame gets the song's own 1024 samples at that instant and a fixed time step, so the layer is deterministic
// and frame-exact against the cut. No npm packages: Chrome is driven over its DevTools protocol (cdp.mjs).
//
//   node render.mjs job.json      job: {song, start, frames, fps, width, height, out, schedule: [[frame, preset,
//                                 blend seconds], ...], chrome, lib}  (lib: folder holding the Butterchurn builds)
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { launch } from "./cdp.mjs";

const job = JSON.parse(readFileSync(process.argv[2], "utf8"));
const { song, start, frames, fps, width, height, out, schedule, chrome, lib } = job;
const SR = 44100, N = 1024, BATCH = 25;

// the song as interleaved stereo 16-bit PCM
const pcm = await new Promise((res, rej) => {
  const p = spawn("ffmpeg", ["-v", "error", "-i", song, "-ac", "2", "-ar", String(SR), "-f", "s16le", "-"]);
  const chunks = []; p.stdout.on("data", (c) => chunks.push(c)); p.on("close", (c) => c ? rej(new Error("ffmpeg")) : res(Buffer.concat(chunks)));
});
const S = new Int16Array(pcm.buffer, pcm.byteOffset, pcm.length / 2);
const total = S.length / 2;

function levels(k) {                      // 1024 samples ending at this frame, as WebAudio-style bytes (128 = silence)
  const end = Math.round((start + k / fps) * SR);
  const L = new Uint8Array(N), R = new Uint8Array(N), M = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const j = end - N + i;
    const l = j >= 0 && j < total ? S[2 * j] : 0, r = j >= 0 && j < total ? S[2 * j + 1] : 0;
    L[i] = Math.max(0, Math.min(255, 128 + (l >> 8)));
    R[i] = Math.max(0, Math.min(255, 128 + (r >> 8)));
    M[i] = (L[i] + R[i]) >> 1;
  }
  return Buffer.concat([M, L, R]).toString("base64");
}

const env = { ...process.env, GALLIUM_DRIVER: "d3d12", MESA_D3D12_DEFAULT_ADAPTER_NAME: "NVIDIA",
  LD_LIBRARY_PATH: ["/usr/lib/wsl/lib", process.env.LD_LIBRARY_PATH].filter(Boolean).join(":") };
Object.assign(process.env, env);          // (the browser inherits it: WSL's GPU through Mesa's D3D12 driver)
const b = await launch(chrome, ["--use-gl=angle", "--use-angle=gl", "--enable-gpu", "--ignore-gpu-blocklist",
  "--enable-unsafe-swiftshader"]);
for (const f of ["butterchurn.min.js", "butterchurnPresets.min.js", "butterchurnPresetsExtra.min.js",
  "butterchurnPresetsExtra2.min.js"]) {
  await b.evaluate(readFileSync(`${lib}/${f}`, "utf8") + "\n;0");
}
const gpu = await b.evaluate(`(() => {
  const c = document.createElement('canvas'); c.width = ${width}; c.height = ${height}; document.body.appendChild(c);
  const bc = window.butterchurn.default || window.butterchurn;
  const all = {};
  for (const p of [window.butterchurnPresets, window.butterchurnPresetsExtra, window.butterchurnPresetsExtra2])
    if (p) Object.assign(all, (p.default || p).getPresets());
  window.presets = all;
  window.viz = bc.createVisualizer(new AudioContext(), c, {width: ${width}, height: ${height}, pixelRatio: 1,
                                                           textureRatio: 1});
  window.canvas = c;
  const dec = (s) => Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));
  window.renderBatch = (items) => items.map(([a, preset, blend]) => {
    if (preset) viz.loadPreset(presets[preset], blend);
    const u = dec(a), n = u.length / 3;
    viz.render({elapsedTime: ${1 / fps}, audioLevels: {timeByteArray: u.subarray(0, n), timeByteArrayL: u.subarray(n, 2 * n),
                                                       timeByteArrayR: u.subarray(2 * n)}});
    return c.toDataURL('image/jpeg', 0.93).slice(23);
  });
  const g = c.getContext('webgl2') || c.getContext('webgl');
  const e = g && g.getExtension('WEBGL_debug_renderer_info');
  return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown';
})()`);
if (job.list) {
  process.stdout.write(await b.evaluate("JSON.stringify(Object.keys(presets))"));
  b.close(); process.exit(0);
}
const missing = schedule.filter(([, p]) => p).map(([, p]) => p);
const have = await b.evaluate(`JSON.stringify(${JSON.stringify(missing)}.filter((p) => !(p in presets)))`);
if (JSON.parse(have).length) throw new Error("unknown presets: " + have);
process.stderr.write(`[milkdrop] ${gpu}\n`);

const enc = spawn("ffmpeg", ["-v", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", String(fps), "-i", "-",
  "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p", out], { stdio: ["pipe", "inherit", "inherit"] });
const at = new Map(schedule.map(([f, p, bl]) => [f, [p, bl]]));
const t0 = Date.now();
for (let k = 0; k < frames; k += BATCH) {
  const items = [];
  for (let i = k; i < Math.min(k + BATCH, frames); i++) {
    const [p, bl] = at.get(i) || [null, 0];
    items.push([levels(i), p, bl]);
  }
  const jpgs = await b.evaluate(`JSON.stringify(renderBatch(${JSON.stringify(items)}))`);
  for (const j of JSON.parse(jpgs)) {
    if (!enc.stdin.write(Buffer.from(j, "base64"))) await new Promise((r) => enc.stdin.once("drain", r));
  }
  if ((k / BATCH) % 10 === 0) process.stderr.write(`\r[milkdrop] ${k}/${frames} (${((k + 1) / ((Date.now() - t0) / 1000)).toFixed(1)} fps)`);
}
enc.stdin.end();
await new Promise((r) => enc.on("close", r));
b.close();
process.stderr.write(`\n[milkdrop] -> ${out} in ${((Date.now() - t0) / 1000).toFixed(0)}s\n`);
process.exit(0);
