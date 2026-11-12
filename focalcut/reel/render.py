"""Render a reel from plan_<fmt>.json: beat-locked shots, effect modules, transitions, the music, the stinger.

Shots are cut from the original camera files (or their configured proxies) at 50 fps and the project's size
(reel.json "resolution": 1080p, Instagram's ceiling, unless it says "4k" or another), each through the plan's crop:
a fixed zoom whose centre glides with the subject. Shots carrying effects run through focalcut.fx (speed curves
first, as they change how much source is read); transitions rewrite the tail of one shot and the head of the next
so the cuts stay on the beat. Output: out/<name>_recap_<fmt>.mp4 at -14 LUFS, plus out/<fmt>_shots.json (the final
file used for each shot, read by the Premiere export).

Where the time goes, and what runs beside what (measured: docs/render.md):

  shots       plain shots several at once (DECODERS: Quick Sync and CPU decodes, each with an NVENC session). A
              stabilised shot first has its camera shake measured (vidstab's pass 1: CPU alone, no encoder,
              STAB_JOBS at once, cached per source stretch in ~/.cache/focalcut/stab), then renders as soon as
              a slot is free. The effect shots (SAM 3, BiRefNet) go one at a time beside all of those, from the
              start. Every shot is cached in work/ by a key of all that shapes it
  overlay     the MilkDrop layer renders beside the shots (it needs the music and the acts, not the pictures); the
              composite runs on the GPU (composite.py), decode and encode on threads of their own
  the rest    transitions (two shots each, streamed: only the frames inside the blend are held), the mix and the
              mux are seconds; the shots are read as one by the overlay and joined straight into the delivered
              file, with no joined copies written in between

Each stage's wall-clock time is printed at the end and saved to out/<cut>_timings.json.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np

from . import exposure, hw

SPEED_FX = ("hero", "bullet", "montage", "flashin", "flashout", "jumpcut", "fit")

SIZE = {"landscape": (1920, 1080), "portrait": (1080, 1920)}   # 1080p: the default (reel.json "resolution")
FPS = 50
# shots rendered at once, by decoder: the Intel iGPU (hw.ffmpeg) leaves the CPU free for the filters, so it takes
# the most. Every shot holds an NVENC session while it renders and consumer drivers allow 12 at once: these, the
# effect shot beside them and the MilkDrop layer make 11
DECODERS = {"qsv": 6, "cpu": 3}
# vidstab's motion analysis is the heaviest thing a render does on the CPU (about half a core-second per 4K frame).
# It spreads over every core with OpenMP, at two thirds the efficiency of one thread; so each analysis gets
# STAB_THREADS and STAB_JOBS of them run at once (FOCALCUT_STAB_JOBS). Measured on 24 cores, 2,400 frames:
# 24 x 1 thread 52 s, 12 x 2 72 s, 8 x 3 67 s
STAB_THREADS = 1         # (render_shots gives a few analyses more threads each: a re-render of one changed shot)
# ... and each analysis holds about 0.6 GB (a stabilised 4K render 1.6 GB, a 4K effect shot 10 GB and more), so no
# more of them than a 2.5 GB share of the machine's memory each: 16 on a 40 GB machine. Past its memory a render
# swaps and takes twice as long
STAB_JOBS = max(2, min(os.cpu_count() or 4, int((hw.mem_gb("MemTotal") or 64) / 2.5)))
MEM_FLOOR = 4.0          # GB free below which a new analysis or stabilised render waits for others to finish
STAB_ANALYSIS = "full"   # reel.json "stab_analysis": "half" measures the shake on a half-size picture (7x less CPU;
#                          the stabilised picture then sits within a pixel or two of the full-size analysis's)


def __getattr__(name):                                      # render.ENC: the encoder arguments (found on first use)
    if name == "ENC":
        return hw.enc()
    raise AttributeError(name)
I, TP = -14.0, -1.5


def size(project, fmt) -> tuple:
    """(width, height) the project renders a format at: reel.json "resolution", else 1080p."""
    f = getattr(project, "size", None)
    return tuple(f(fmt)) if callable(f) else SIZE[fmt]


class Timer:
    """Wall-clock seconds per stage of a render; stages that run beside others are timed on their own."""

    def __init__(self):
        self.t, self.lock, self.t0 = {}, threading.Lock(), time.time()

    def add(self, stage, seconds):
        with self.lock:
            self.t[stage] = self.t.get(stage, 0.0) + seconds

    def __call__(self, stage):
        timer = self

        class Span:
            def __enter__(self):
                self.t = time.time()

            def __exit__(self, *exc):
                timer.add(stage, time.time() - self.t)
        return Span()

    def report(self) -> dict:
        return {"total": round(time.time() - self.t0, 1), **{k: round(v, 1) for k, v in self.t.items()}}


def crop_expr(path, k):
    """Piecewise-linear crop centre (fraction of the source) over shot time t, as an ffmpeg expression; k picks x (1)
    or y (2) from the path's [t, x, y] points."""
    pts = [(p[0], p[k]) for p in path]
    pieces = [f"({x0}+({x1}-{x0})*(t-{t0})/{max(t1 - t0, 1e-3)})"
              for (t0, x0), (t1, x1) in zip(pts[:-1], pts[1:])] + [f"{pts[-1][1]}"]
    ends = [p[0] for p in pts[1:]]                 # piece i holds while t < ends[i]; the last one after that

    def tree(lo, hi):                              # a balanced if() tree: ffmpeg refuses nesting past ~100 deep,
        if lo == hi:                               # which a chain hits on a long shot (one point per 0.2 s)
            return pieces[lo]
        mid = (lo + hi) // 2
        return f"if(lt(t\\,{ends[mid]})\\,{tree(lo, mid)}\\,{tree(mid + 1, hi)})"
    return tree(0, len(pieces) - 1)


STAB = os.path.expanduser("~/.cache/focalcut/stab")         # vidstab motion files (a path without spaces for ffmpeg)
_stab_locks, _stab_guard = {}, threading.Lock()


def stab(path, t, dur, rot):
    """Pass 1 of vidstab over exactly the stretch the shot decodes: the camera's shake, frame by frame."""
    os.makedirs(STAB, exist_ok=True)
    half = STAB_ANALYSIS == "half"
    name = hashlib.md5(f"{path}|{t:.3f}|{dur:.3f}|{rot}{'|half' if half else ''}".encode()).hexdigest()[:12] + ".trf"
    trf = os.path.join(STAB, name)
    with _stab_guard:                              # (the same stretch asked for twice at once is analysed once)
        lock = _stab_locks.setdefault(name, threading.Lock())
    with lock:
        if not os.path.exists(trf):                # run in STAB with a bare file name: no filter-string escaping
            pre = (rot + ",") if rot else ""
            pre += "scale=iw/2:ih/2:flags=bilinear," if half else ""
            part = f"{name}.{os.getpid()}.part"    # (another render analysing it too writes its own)
            # (below the effect shots and the encodes in the CPU's queue: they are what the render waits for)
            nice = ["nice", "-n", "10"] if shutil.which("nice") else []
            subprocess.run([*nice, "ffmpeg", "-v", "error", "-y", "-threads", str(STAB_THREADS), "-ss", f"{t:.3f}",
                            "-t", f"{dur:.3f}", "-i", path, "-an",
                            "-vf", f"{pre}vidstabdetect=shakiness=6:accuracy=12:result={part}", "-f", "null", "-"],
                           check=True, cwd=STAB, env={**os.environ, "OMP_NUM_THREADS": str(STAB_THREADS)})
            if half:
                scale_motion(os.path.join(STAB, part), 2)
            os.replace(os.path.join(STAB, part), trf)   # (a render stopped mid-pass leaves no half a motion file)
    return trf


def scale_motion(trf, k):
    """A vidstab motion file measured on a picture 1/k the size, rewritten for the full-size one: every local
    motion's vector, field position and field size times k. Only the text format ("VID.STAB 1") is understood."""
    import re
    with open(trf, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"VID.STAB 1"):
        raise RuntimeError("this ffmpeg's vidstab writes binary motion files: reel.json \"stab_analysis\": \"half\" "
                           "needs the text format (libvidstab 1.1.0)")
    lm = re.compile(r"\(LM (-?\d+) (-?\d+) (-?\d+) (-?\d+) (\d+) ")
    out = lm.sub(lambda m: "(LM " + " ".join(str(int(v) * k) for v in m.groups()) + " ", raw.decode())
    with open(trf, "w") as f:
        f.write(out)


def room(wait=180.0):
    """Hold a memory-heavy step back while the machine has under MEM_FLOOR GB free (others are finishing): for a
    while, never for good."""
    t0 = time.time()
    while (hw.mem_gb() or 1e9) < MEM_FLOOR and time.time() - t0 < wait:
        time.sleep(1.0)


def read_length(s) -> float:
    """Seconds of source a plain shot decodes: its length (more of it when paced) and two frames to spare."""
    return s["length"] * (s.get("speed") or 1.0) + 0.1


def shot_filter(s, fmt, project=None, dur=None):
    """The shot's filter chain: rotation, stabilisation (a shaky stretch: the plan's "stab"), the crop that follows
    the subject, the project's size at FPS."""
    W, H = size(project, fmt)
    u = s["unit"]
    f = [u["rotate"]] if u.get("rotate") else []
    if s.get("stab") and project is not None:               # smooth the camera, zoomed just enough to hide the
        path, t = project.decode_from(u["source"], s["src_in"])   # moving edges (4K has the pixels to spare)
        trf = stab(path, t, dur or s["length"] + 0.1, u.get("rotate"))
        trf = trf.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")   # filtergraph escaping
        f.append(f"vidstabtransform=input={trf}:smoothing=15:optzoom=1:zoomspeed=0.25:interpol=bicubic")
    c = s.get("crop")
    if c:
        f.append(f"crop=w=iw*{c['w']}:h=ih*{c['h']}:x=({crop_expr(c['path'], 1)})*iw-ow/2"
                 f":y=({crop_expr(c['path'], 2)})*ih-oh/2")
    if s.get("pull") or s.get("lift"):                      # a blown-out subject: its highlights pulled back;
        f.append(exposure.filt(s.get("pull"), s.get("lift")))   # a dark one at night: lifted
    p = s.get("speed") or 1.0                               # the shot's pace (plan "pace"): source read p times
    if p != 1.0 and not s.get("fx"):                        # as fast, the nearest frame each (a speed effect
        f.append(f"setpts=PTS/{p:.4f}")                     # applies it inside its curve instead)
    f += [f"scale={W}:{H}:flags=lanczos", f"fps={FPS}", "setsar=1", "format=yuv420p"]
    return ",".join(f)


def key(s, beat=None, size=None):
    """Cache name for a shot: changes whenever anything that shapes its pixels does (size: the frame it is rendered
    at, with the encode's settings; without one, the shot's identity alone -- the effects' random seed)."""
    u = s["unit"]
    k = json.dumps([u["id"], u["source"], u.get("rotate"), u["role"], s["src_in"], s["frames"], s.get("fx"),
                    s.get("crop"), s["peak"], s.get("logo"), s.get("stab"), s["kind"] == "closeup", beat, 3, hw.enc_tag()]
                   + (["half"] if s.get("stab") and STAB_ANALYSIS == "half" else [])
                   + ([s["span"]] if s.get("span") else []) + ([s["logo_image"]] if s.get("logo_image") else [])
                   + ([["pull", s["pull"]]] if s.get("pull") else [])
                   + ([["lift", s["lift"]]] if s.get("lift") else [])
                   + (["speed2"] if any(n in SPEED_FX for n in s.get("fx") or ()) else [])   # (1x tails)
                   + ([["pace", s["speed"]]] if (s.get("speed") or 1.0) != 1.0 else [])
                   + ([["size", *size, hw.ENC_VERSION]] if size else []),
                   sort_keys=True)
    return hashlib.md5(k.encode()).hexdigest()[:10]


def cut_shot(project, s, fmt, out, qsv=False):
    """A plain shot straight from the camera file: decoded on the Intel iGPU when qsv (and hw.ffmpeg can; a
    stabilised shot reads its motion file on the Linux side, so it stays on the CPU), encoded by hw.enc."""
    p = s.get("speed") or 1.0
    read = read_length(s)                                   # the source it plays (more of it when paced)
    if p != 1.0 and s.get("crop"):                          # follow the subject over all of it
        from .plan import reframe
        s = dict(s, crop=reframe(project, s["unit"], s["src_in"], s["length"] * p, fmt) or s["crop"])
    src, t = project.decode_from(s["unit"]["source"], s["src_in"])
    exe, dec, P, pre = hw.ffmpeg(src, qsv and not s.get("stab"))
    chain = pre + shot_filter(s, fmt, project, read)
    vf = ["-vf", chain]
    logo = project.logo(s.get("logo_image"))
    if s.get("logo") and logo:                              # the event's logo over the opening, or an act's title
        W, H = size(project, fmt)
        fade = ",fade=t=in:st=0.15:d=0.6:alpha=1" if s["logo"] == "in" else ""
        vf = ["-i", P(logo), "-filter_complex",
              f"[0:v]{chain},vignette=angle=PI/5,eq=brightness=-0.05[v];"
              f"[1:v]scale={int(W * (0.5 if W > H else 0.8))}:-1:flags=lanczos,format=rgba,loop=-1:1,fps={FPS}{fade}[l];"
              f"[v][l]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p"]
    hw.run_enc([exe, "-v", "error", "-y", *dec, "-ss", f"{t:.3f}", "-t", f"{read:.3f}",
                "-i", P(src), "-an", *vf, *hw.enc(), "-frames:v", str(s["frames"]), P(out)])


def fx_shot(project, s, fmt, beat, out):
    """Render a shot through its effect chain (speed curves first change how much source is read)."""
    from .. import fx
    from ..fx.speed import Speed
    from .plan import reframe
    W, H = size(project, fmt)
    u, L = s["unit"], s["length"]
    ctx = fx.Context(fps=FPS, beat=beat, peak=s["peak"], seed=int(key(s)[:4], 16), params={"group": s.get("group"), "closeup": s["kind"] == "closeup",
                                     "face": s["kind"] == "closeup" or (s.get("crop") or {}).get("face", 0) >= 0.5,
                                     "cap": 3.5 if u["role"] == "transition" else 1.5,   # dancing never past 1.5x
                                     "pace": s.get("speed") or 1.0})
    if s.get("span"):                              # an editor's selection: all of it, from their in-point
        ctx.params["src_len"] = s["span"][1] - s["span"][0]
    src_len, src_peak = L, s["peak"]
    for n in s["fx"]:
        e = fx.EFFECTS[n]()
        if isinstance(e, Speed):
            sp = e
            src_len, src_peak = e.source_length(L, ctx), e.source_peak(s["peak"], ctx, L)
    src_in = min(max(s["t"] - src_peak, u["t0"]), max(u["t1"] - src_len, u["t0"]))
    if "fit" in s["fx"]:
        src_in = min(max(s["span"][0], u["t0"]), max(u["t1"] - src_len, u["t0"]))
    elif src_len != L and abs(src_in - (s["t"] - src_peak)) > 1e-3:   # clamped at the unit's edge: slow down where
        ctx.peak = sp.out_time(s["t"] - src_in, L, ctx)               # the move now plays, not where it would have
    shot = dict(s, src_in=src_in, length=src_len)
    if s.get("crop"):                              # the source read changed: follow the subject over it instead
        shot["crop"] = reframe(project, u, src_in, src_len, fmt) or s["crop"]
    src, t = project.decode_from(u["source"], src_in)
    exe, dec, P, pre = hw.ffmpeg(src, not shot.get("stab"))
    raw = subprocess.run([exe, "-v", "error", *dec, "-ss", f"{t:.3f}", "-t", f"{src_len:.3f}", "-i", P(src), "-an",
                          "-vf", pre + shot_filter(shot, fmt, project, src_len), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = list(np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3))
    n_out = s["frames"]
    ctx.out_frames = n_out
    frames = fx.chain(frames, s["fx"], ctx)[:n_out]
    frames += [frames[-1]] * (n_out - len(frames))
    write_frames(frames, out)


def frame_count(path) -> int:
    return int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=nb_frames", "-of", "csv=p=0", path]).decode().strip().strip(","))


def frame_size(path) -> tuple:
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    return W, H


def read_frames(path, start=0, count=None):
    """A shot file's frames as RGB arrays: all of them, or count of them from frame start. Every decoded frame comes
    out once (ffmpeg's constant-rate output would repeat the first frame of a trimmed stretch to fill the time
    before it)."""
    W, H = frame_size(path)
    trim = [] if not start and count is None else \
        ["-vf", f"trim=start_frame={start}" + (f":end_frame={start + count}" if count is not None else "")
         + ",setpts=PTS-STARTPTS"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, *trim, "-fps_mode", "passthrough", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = list(np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3))
    if count is not None and len(frames) > count:
        raise RuntimeError(f"{path}: asked for {count} frames from {start}, decoded {len(frames)}")
    return frames


def write_frames(frames, out, pix_fmt="rgb24", size=None):
    """Frames (a list, or anything that yields arrays or raw bytes) encoded by hw.enc: RGB arrays, or with
    pix_fmt="yuv420p" and size=(W, H) raw 4:2:0 frames."""
    frames = iter(frames)
    first = next(frames)
    W, H = size or first.shape[1::-1]
    tags = hw.RAW_YUV if pix_fmt.startswith("yuv") else []
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{W}x{H}",
                          "-r", str(FPS), *tags, "-i", "-", *hw.enc(), out], stdin=subprocess.PIPE)
    try:
        for f in itertools.chain([first], frames):
            p.stdin.write(f if isinstance(f, (bytes, bytearray, memoryview)) else np.ascontiguousarray(f).data)
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait():
        if os.path.exists(out):
            os.remove(out)
        raise subprocess.CalledProcessError(p.returncode, "ffmpeg (write_frames)")


def rgb_to_yuv420(f) -> bytes:
    """An RGB frame as raw yuv420p, BT.709 limited range (what the shots are): luma per pixel, chroma averaged over
    each 2 x 2."""
    import cv2
    x = f.astype(np.float32)
    H, W = x.shape[:2]
    y = cv2.transform(x, np.float32([[0.2126, 0.7152, 0.0722]]))
    cb = ((x[..., 2] - y) * (224 / 255 / 1.8556)).reshape(H // 2, 2, W // 2, 2).mean((1, 3)) + 128
    cr = ((x[..., 0] - y) * (224 / 255 / 1.5748)).reshape(H // 2, 2, W // 2, 2).mean((1, 3)) + 128
    planes = (y * (219 / 255) + 16, cb, cr)
    return b"".join(np.clip(p + 0.5, 16, 240 if k else 235).astype(np.uint8).tobytes() for k, p in enumerate(planes))


def splice(src, out, at, new):
    """src re-encoded to out with its frames from `at` on replaced by `new` (RGB arrays). Streamed in the shot's own
    4:2:0, so a long shot at 4K is never held in memory whole and the frames outside the blend are re-encoded from
    exactly what was decoded: no trip through RGB."""
    W, H = frame_size(src)
    size_ = W * H * 3 // 2
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-fps_mode", "passthrough", "-f", "rawvideo",
                           "-pix_fmt", "yuv420p", "-"], stdout=subprocess.PIPE)

    def frames():
        k = 0
        while True:
            b = rd.stdout.read(size_)
            if len(b) < size_:
                break
            yield rgb_to_yuv420(new[k - at]) if at <= k < at + len(new) else b
            k += 1
    try:
        write_frames(frames(), out, "yuv420p", (W, H))
    finally:
        rd.stdout.close()
        rd.wait()


def render_shots(project, plan, fmt, work, draft=False, timer=None):
    """Every shot's file (cached by key): plain shots several at once -- DECODERS["qsv"] decoding on the Intel iGPU
    and DECODERS["cpu"] on the CPU, each encoding with NVENC where there is one -- while the effect shots (which run
    their own GPU models) go one at a time beside them. A stabilised shot's motion analysis comes first, on the CPU
    alone, STAB_JOBS of them at once."""
    import queue
    from concurrent.futures import ThreadPoolExecutor
    timer = timer or Timer()
    dim = size(project, fmt)
    n, done, lock = len(plan["shots"]), [0], threading.Lock()
    slots = queue.Queue()
    for kind, k in hw.decoders(DECODERS).items():
        for _ in range(k if kind == "cpu" or hw.win_ffmpeg() else 0):
            slots.put(kind)

    def tick():
        with lock:
            done[0] += 1
            print(f"\r[reel] {fmt}: shot {done[0]}/{n}", end="", flush=True)

    def plain(s, out):
        if not os.path.exists(out):
            kind = slots.get()
            try:
                if s.get("stab"):
                    room()
                cut_shot(project, s, fmt, out + ".part.mp4", qsv=kind == "qsv")
                os.replace(out + ".part.mp4", out)
            finally:
                slots.put(kind)
        tick()

    files, jobs, fxs = [], [], []
    for i, s in enumerate(plan["shots"]):
        fx = bool(s.get("fx")) and not draft
        out = os.path.join(work, f"shot_{i:03d}_{key(s, plan['beat'], dim)}{'_fx' if fx else ''}.mp4")
        files.append(out)
        (fxs if fx else jobs).append((s, out))
    failed = []

    def plains():                                           # the steady shots straight to a render slot; a shaky
        try:                                                # one once its pass 1 is done (CPU alone, no encoder
            global STAB_THREADS                             # session: those never wait for a slot)
            t0 = time.time()
            todo = sum(1 for s, out in jobs if s.get("stab") and not os.path.exists(out))
            njobs = int(os.environ.get("FOCALCUT_STAB_JOBS") or STAB_JOBS)
            STAB_THREADS = max(1, min(6, (os.cpu_count() or 4) // max(min(todo, njobs), 1)))   # (few to do: more each)
            with ThreadPoolExecutor(max(slots.qsize(), 1)) as ex, ThreadPoolExecutor(njobs) as ex2:
                def shaky(s, out):
                    if not os.path.exists(out):
                        room()
                        t1 = time.time()
                        path, t = project.decode_from(s["unit"]["source"], s["src_in"])
                        stab(path, t, read_length(s), s["unit"].get("rotate"))
                        timer.add("measuring camera shake (summed over shots)", time.time() - t1)
                    return ex.submit(plain, s, out)
                futs = [(ex2.submit(shaky, s, out) if s.get("stab") else ex.submit(plain, s, out)) for s, out in jobs]
                for (s, _), f in zip(jobs, futs, strict=True):
                    r = f.result()
                    if s.get("stab"):
                        r.result()
            timer.add("shots", time.time() - t0)
        except BaseException as e:                          # (raised by the main thread, below)
            failed.append(e)

    side = threading.Thread(target=plains, daemon=True)
    side.start()
    t0 = time.time()
    for s, out in fxs:                                      # the effect shots meanwhile, one at a time
        if failed:
            break
        if not os.path.exists(out):
            fx_shot(project, s, fmt, s.get("beat", plan["beat"]), out + ".part.mp4")
            os.replace(out + ".part.mp4", out)
            print(f"\n  {'+'.join(s['fx'])}: {s['unit']['id']} @ {s['t']}s", flush=True)
        tick()
    if fxs:
        timer.add("effect shots (beside the shots)", time.time() - t0)
    side.join()
    if failed:
        raise failed[0]
    print()
    return files


def apply_transitions(plan, files):
    """Rewrite the tail of the outgoing shot and the head of the incoming one at every planned transition. Only the
    frames inside the blend are held; the rest of both shots streams through."""
    from .. import fx
    for i, s in enumerate(plan["shots"]):
        name = s.get("transition")
        if not name or i == 0:
            continue
        tag = hashlib.md5((files[i - 1] + files[i] + name).encode()).hexdigest()[:6]
        a_out, b_out = files[i - 1].replace(".mp4", f"_o{tag}.mp4"), files[i].replace(".mp4", f"_i{tag}.mp4")
        if not (os.path.exists(a_out) and os.path.exists(b_out)):
            na, nb = frame_count(files[i - 1]), frame_count(files[i])
            ctx = fx.Context(fps=FPS, beat=s.get("beat", plan["beat"]), seed=i)
            tr = fx.TRANSITIONS[name]()
            n = min(tr.frames(ctx), na, nb) // 2 * 2
            tail, head = read_frames(files[i - 1], na - n // 2, n // 2), read_frames(files[i], 0, n // 2)
            if len(tail) != n // 2 or len(head) != n // 2:
                raise RuntimeError(f"transition {name} at shot {i}: read {len(tail)} + {len(head)} frames, not {n}")
            mid = tr.apply(tail, head, ctx)
            splice(files[i - 1], a_out + ".part.mp4", na - n // 2, mid[:n // 2])
            splice(files[i], b_out + ".part.mp4", 0, mid[n // 2:])
            os.replace(a_out + ".part.mp4", a_out); os.replace(b_out + ".part.mp4", b_out)
        files[i - 1], files[i] = a_out, b_out
    return files


def mix_audio(song, plan, stinger, out):
    """The song from its start point, fading under the stinger, plus the stinger's own sound."""
    ml, sa = plan["music_length"], plan["stinger_at"]
    graph = (f"[0:a]atrim=start={plan['music_start']}:duration={ml},asetpts=PTS-STARTPTS,"
             f"afade=t=out:st={sa:.3f}:d={max(ml - sa, 0.05):.3f},aresample=48000[m]")
    inputs = ["-i", song]
    if stinger:
        inputs += ["-i", stinger]
        graph += (f";[1:a]aresample=48000,adelay={int(sa * 1000)}|{int(sa * 1000)}[s];"
                  f"[m][s]amix=inputs=2:normalize=0:duration=longest[a]")
    else:
        graph += ";[m]anull[a]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[a]",
                    "-c:a", "pcm_s24le", out], check=True)


def run(project, fmt, draft=False, cut=None) -> str:
    """Render plan_<cut>.json (cut: a reel's format, or any other cut's name -- plan.place) to out/."""
    cut = cut or fmt
    plan = project.load(f"plan_{cut}.json")
    fmt = plan.get("format", fmt)
    music = project.load(project.music_file) if os.path.exists(project.path(project.music_file)) else {}
    W, H = size(project, fmt)
    timer = Timer()
    global STAB_ANALYSIS
    STAB_ANALYSIS = "half" if str(project.config.get("stab_analysis") or "").lower() == "half" else "full"
    outdir, work = project.path("out"), project.path("work", cut)
    os.makedirs(outdir, exist_ok=True); os.makedirs(work, exist_ok=True)
    marked = exposure.mark(project, plan)                    # reel.json "exposure": blown-out / dark subjects
    for what, k in (("highlights pulled", 0), ("night subjects lifted", 1)):
        done = {i: v[k] for i, v in marked.items() if v[k]}
        if done:
            print(f"[reel] {cut}: {what} on {len(done)} shots: "
                  + ", ".join(f"{i} {plan['shots'][i]['unit']['id']} {v:.2f}" for i, v in done.items()))
    from .velocity import speed_tails
    for line in speed_tails(plan):
        print(f"[reel] {cut}: WARNING speed: {line}")
    from ..fx import looks
    from . import overlay
    song = plan.get("song") or music.get("song")
    ov = overlay.options(looks.overlay(project.config, fmt))   # reel.json "overlay" over the look's own
    ahead, ahead_err = None, []
    if ov and not draft:                           # the MilkDrop layer needs the music and the acts, not the
        def layer():                               # pictures: it renders while the shots do
            t0 = time.time()
            try:
                overlay.prerender(work, plan, music, song, ov, W, H, FPS)
            except Exception as e:                 # (raised where the overlay would have rendered it)
                ahead_err.append(e)
            timer.add("milkdrop layer (beside the shots)", time.time() - t0)
        ahead = threading.Thread(target=layer, daemon=True)
        ahead.start()
    files = render_shots(project, plan, fmt, work, draft, timer)
    if not draft:
        with timer("transitions"):
            files = apply_transitions(plan, files)
    with open(os.path.join(outdir, f"{cut}_shots.json"), "w") as f:
        json.dump([os.path.relpath(p, project.reel_dir) for p in files], f, indent=1)
    if ov and not draft:                           # the visualiser layer over the cut (not the stinger)
        lst = os.path.join(work, "list_shots.txt")
        with open(lst, "w") as f:
            f.writelines("file '" + p.replace("'", "'\\''") + "'\n" for p in files)
        layered = os.path.join(work, "shots_overlay.mp4")
        with timer("milkdrop layer (waited for)"):
            ahead.join()
        if ahead_err:
            raise ahead_err[0]
        with timer("overlay"):
            overlay.apply(lst, layered, plan, music, song, ov, hw.enc(final=True), FPS)   # (the shots, read as
                                                                                           # one: no joined copy)
        files = [layered]
    stinger = project.stinger(fmt) if plan.get("stinger") else None
    with timer("stinger, join, mix and mux"):
        if stinger:                                    # the stinger as the last "shot", conformed to size and rate
            sting = os.path.join(work, f"stinger_{hw.enc_tag()}_{W}x{H}_{hw.ENC_VERSION}.mp4")
            if not os.path.exists(sting):
                hw.run_enc(["ffmpeg", "-v", "error", "-y", "-i", stinger, "-an",
                            "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS},setsar=1,format=yuv420p",
                            *hw.enc(final=True), sting + ".part.mp4"])
                os.replace(sting + ".part.mp4", sting)
            files = [*files, sting]
        lst = os.path.join(work, "list.txt")
        with open(lst, "w") as f:
            f.writelines("file '" + p.replace("'", "'\\''") + "'\n" for p in files)   # (concat's own quoting)
        mix = os.path.join(work, "mix.wav")
        mix_audio(song, plan, stinger, mix)
        err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", mix, "-af",
                              f"loudnorm=I={I}:TP={TP}:LRA=11:print_format=json", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        m = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
        final = os.path.join(outdir, f"{project.out_stem(cut, fmt)}{'_draft' if draft else ''}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", mix,
                        "-map", "0:v", "-map", "1:a",      # (the picture joined straight into the delivered file)
                        "-af", f"loudnorm=I={I}:TP={TP}:LRA=11:linear=true:measured_I={m['input_i']}:"
                               f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
                               f"measured_thresh={m['input_thresh']}:offset={m['target_offset']},aresample=48000",
                        "-c:v", "copy", "-bsf:v", hw.VUI, "-c:a", "aac", "-b:a", "320k", "-shortest",
                        "-movflags", "+faststart",
                        final + ".part.mp4"], check=True)
        os.replace(final + ".part.mp4", final)
    if not draft:                                            # the cut as rendered: what "keep": true re-cuts from
        project.dump(f"plan_{cut}_rendered.json", plan)
    rep = timer.report()
    with open(os.path.join(outdir, f"{cut}_timings.json"), "w") as f:
        json.dump(dict(rep, size=[W, H], frames=sum(s["frames"] for s in plan["shots"])), f, indent=1)
    print(f"[reel] {cut}: {W}x{H} in {rep['total']:.0f} s -- "
          + ", ".join(f"{k} {v:.0f} s" for k, v in rep.items() if k != "total" and v >= 0.5))
    print(f"[reel] -> {final}")
    return final
