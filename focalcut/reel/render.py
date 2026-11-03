"""Render a reel from plan_<fmt>.json: beat-locked shots, effect modules, transitions, the music, the stinger.

Shots are cut from the original camera files (or their configured proxies) at 1080p 50 fps (Instagram's ceiling),
each through the plan's crop: a fixed zoom whose centre glides with the subject. Shots carrying effects run through focalcut.fx (speed curves first, as
they change how much source is read); transitions rewrite the tail of one shot and the head of the next so the
cuts stay on the beat. Output: out/<name>_recap_<fmt>.mp4 at -14 LUFS, plus out/<fmt>_shots.json (the final file
used for each shot, read by the Premiere export).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np

from . import exposure, hw

SPEED_FX = ("hero", "bullet", "montage", "flashin", "flashout", "jumpcut", "fit")

SIZE = {"landscape": (1920, 1080), "portrait": (1080, 1920)}
FPS = 50
# shots rendered at once, by decoder: the Intel iGPU (hw.ffmpeg) leaves the CPU free for the filters, so it takes
# the most (measured on a 24-thread CPU: 3 + 3 kept the CPU at 85% and the iGPU's decoder at 40%)
DECODERS = {"qsv": 5, "cpu": 2}


def __getattr__(name):                                      # render.ENC: the encoder arguments (found on first use)
    if name == "ENC":
        return hw.enc()
    raise AttributeError(name)
I, TP = -14.0, -1.5


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


def stab(path, t, dur, rot):
    """Pass 1 of vidstab over exactly the stretch the shot decodes: the camera's shake, frame by frame."""
    os.makedirs(STAB, exist_ok=True)
    name = hashlib.md5(f"{path}|{t:.3f}|{dur:.3f}|{rot}".encode()).hexdigest()[:12] + ".trf"
    trf = os.path.join(STAB, name)
    if not os.path.exists(trf):                    # run in STAB with a bare file name: no filter-string escaping
        pre = (rot + ",") if rot else ""
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-t", f"{dur:.3f}", "-i", path, "-an", "-vf",
                        f"{pre}vidstabdetect=shakiness=6:accuracy=12:result={name}.part", "-f", "null", "-"],
                       check=True, cwd=STAB)
        os.replace(trf + ".part", trf)             # (a render stopped mid-pass leaves no half a motion file behind)
    return trf


def shot_filter(s, fmt, project=None, dur=None):
    """The shot's filter chain: rotation, stabilisation (a shaky stretch: the plan's "stab"), the crop that follows
    the subject, 1080p at FPS."""
    W, H = SIZE[fmt]
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
    if s.get("pull"):                                       # a blown-out subject: its highlights pulled back
        f.append(exposure.filt(s["pull"]))
    f += [f"scale={W}:{H}:flags=lanczos", f"fps={FPS}", "setsar=1", "format=yuv420p"]
    return ",".join(f)


def key(s, beat=None):
    """Cache name for a shot: changes whenever anything that shapes its pixels does."""
    u = s["unit"]
    k = json.dumps([u["id"], u["source"], u.get("rotate"), u["role"], s["src_in"], s["frames"], s.get("fx"),
                    s.get("crop"), s["peak"], s.get("logo"), s.get("stab"), s["kind"] == "closeup", beat, 3, hw.enc_tag()]
                   + ([s["span"]] if s.get("span") else []) + ([s["logo_image"]] if s.get("logo_image") else [])
                   + ([["pull", s["pull"]]] if s.get("pull") else [])
                   + (["speed2"] if any(n in SPEED_FX for n in s.get("fx") or ()) else []),   # (1x tails)
                   sort_keys=True)
    return hashlib.md5(k.encode()).hexdigest()[:10]


def cut_shot(project, s, fmt, out, qsv=False):
    """A plain shot straight from the camera file: decoded on the Intel iGPU when qsv (and hw.ffmpeg can; a
    stabilised shot reads its motion file on the Linux side, so it stays on the CPU), encoded by hw.enc."""
    src, t = project.decode_from(s["unit"]["source"], s["src_in"])
    exe, dec, P, pre = hw.ffmpeg(src, qsv and not s.get("stab"))
    chain = pre + shot_filter(s, fmt, project, s["length"] + 0.1)
    vf = ["-vf", chain]
    logo = project.logo(s.get("logo_image"))
    if s.get("logo") and logo:                              # the event's logo over the opening, or an act's title
        W, H = SIZE[fmt]
        fade = ",fade=t=in:st=0.15:d=0.6:alpha=1" if s["logo"] == "in" else ""
        vf = ["-i", P(logo), "-filter_complex",
              f"[0:v]{chain},vignette=angle=PI/5,eq=brightness=-0.05[v];"
              f"[1:v]scale={int(W * (0.5 if W > H else 0.8))}:-1,format=rgba,loop=-1:1,fps={FPS}{fade}[l];"
              f"[v][l]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p"]
    subprocess.run([exe, "-v", "error", "-y", *dec, "-ss", f"{t:.3f}", "-t", f"{s['length'] + 0.1:.3f}",
                    "-i", P(src), "-an", *vf, *hw.enc(), "-frames:v", str(s["frames"]), P(out)],
                   check=True)


def fx_shot(project, s, fmt, beat, out):
    """Render a shot through its effect chain (speed curves first change how much source is read)."""
    from .. import fx
    from ..fx.speed import Speed
    from .plan import reframe
    W, H = SIZE[fmt]
    u, L = s["unit"], s["length"]
    ctx = fx.Context(fps=FPS, beat=beat, peak=s["peak"], seed=int(key(s)[:4], 16), params={"group": s.get("group"), "closeup": s["kind"] == "closeup",
                                     "face": s["kind"] == "closeup" or (s.get("crop") or {}).get("face", 0) >= 0.5,
                                     "cap": 3.5 if u["role"] == "transition" else 1.5})   # dancing never past 1.5x
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


def read_frames(path):
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3))


def write_frames(frames, out):
    H, W = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", *hw.enc(), out], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        if os.path.exists(out):
            os.remove(out)
        raise subprocess.CalledProcessError(p.returncode, "ffmpeg (write_frames)")


def render_shots(project, plan, fmt, work, draft=False):
    """Every shot's file (cached by key): plain shots several at once -- DECODERS["qsv"] decoding on the Intel iGPU
    and DECODERS["cpu"] on the CPU, each encoding with NVENC where there is one -- while the effect shots (which run
    their own GPU models) go one at a time beside them."""
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
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
                cut_shot(project, s, fmt, out + ".part.mp4", qsv=kind == "qsv")
                os.replace(out + ".part.mp4", out)
            finally:
                slots.put(kind)
        tick()

    files, jobs, fxs = [], [], []
    for i, s in enumerate(plan["shots"]):
        fx = bool(s.get("fx")) and not draft
        out = os.path.join(work, f"shot_{i:03d}_{key(s, plan['beat'])}{'_fx' if fx else ''}.mp4")
        files.append(out)
        (fxs if fx else jobs).append((s, out))
    with ThreadPoolExecutor(max(slots.qsize(), 1)) as ex:
        futs = [ex.submit(plain, s, out) for s, out in jobs]
        for s, out in fxs:
            if not os.path.exists(out):
                fx_shot(project, s, fmt, s.get("beat", plan["beat"]), out + ".part.mp4")
                os.replace(out + ".part.mp4", out)
                print(f"\n  {'+'.join(s['fx'])}: {s['unit']['id']} @ {s['t']}s", flush=True)
            tick()
        for f in futs:
            f.result()
    print()
    return files


def apply_transitions(plan, files):
    """Rewrite the tail of the outgoing shot and the head of the incoming one at every planned transition."""
    from .. import fx
    for i, s in enumerate(plan["shots"]):
        name = s.get("transition")
        if not name or i == 0:
            continue
        tag = hashlib.md5((files[i - 1] + files[i] + name).encode()).hexdigest()[:6]
        a_out, b_out = files[i - 1].replace(".mp4", f"_o{tag}.mp4"), files[i].replace(".mp4", f"_i{tag}.mp4")
        if not (os.path.exists(a_out) and os.path.exists(b_out)):
            A, B = read_frames(files[i - 1]), read_frames(files[i])
            ctx = fx.Context(fps=FPS, beat=s.get("beat", plan["beat"]), seed=i)
            tr = fx.TRANSITIONS[name]()
            n = min(tr.frames(ctx), len(A), len(B)) // 2 * 2
            mid = tr.apply(A[len(A) - n // 2:], B[:n // 2], ctx)
            write_frames(A[:len(A) - n // 2] + mid[:n // 2], a_out)
            write_frames(mid[n // 2:] + B[n // 2:], b_out)
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
    W, H = SIZE[fmt]
    outdir, work = project.path("out"), project.path("work", cut)
    os.makedirs(outdir, exist_ok=True); os.makedirs(work, exist_ok=True)
    pulled = exposure.mark(project, plan)                    # reel.json "exposure": blown-out subjects
    if pulled:
        print(f"[reel] {cut}: highlights pulled on {len(pulled)} shots: "
              + ", ".join(f"{i} {plan['shots'][i]['unit']['id']} {p:.2f}" for i, p in pulled.items()))
    from .velocity import speed_tails
    for line in speed_tails(plan):
        print(f"[reel] {cut}: WARNING speed: {line}")
    files = render_shots(project, plan, fmt, work, draft)
    if not draft:
        files = apply_transitions(plan, files)
    with open(os.path.join(outdir, f"{cut}_shots.json"), "w") as f:
        json.dump([os.path.relpath(p, project.reel_dir) for p in files], f, indent=1)
    from ..fx import looks
    from . import overlay
    ov = overlay.options(looks.overlay(project.config, fmt))   # reel.json "overlay" over the look's own
    if ov and not draft:                           # the visualiser layer over the cut (not the stinger)
        lst = os.path.join(work, "list_shots.txt")
        with open(lst, "w") as f:
            f.writelines("file '" + p.replace("'", "'\\''") + "'\n" for p in files)
        joined, layered = os.path.join(work, "shots.mp4"), os.path.join(work, "shots_overlay.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined],
                       check=True)
        overlay.apply(joined, layered, plan, music, plan.get("song") or music["song"], ov, hw.enc(), FPS)
        files = [layered]
    stinger = project.stinger(fmt) if plan.get("stinger") else None
    if stinger:                                    # the stinger as the last "shot", conformed to size and rate
        sting = os.path.join(work, f"stinger_{hw.enc_tag()}.mp4")
        if not os.path.exists(sting):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", stinger, "-an",
                            "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS},setsar=1,format=yuv420p", *hw.enc(), sting],
                           check=True)
        files = [*files, sting]
    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as f:
        f.writelines("file '" + p.replace("'", "'\\''") + "'\n" for p in files)   # (concat's own quoting)
    video = os.path.join(work, "video.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video],
                   check=True)
    mix = os.path.join(work, "mix.wav")
    mix_audio(plan.get("song") or music["song"], plan, stinger, mix)
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", mix, "-af",
                          f"loudnorm=I={I}:TP={TP}:LRA=11:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
    final = os.path.join(outdir, f"{project.out_stem(cut, fmt)}{'_draft' if draft else ''}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", mix, "-map", "0:v", "-map", "1:a",
                    "-af", f"loudnorm=I={I}:TP={TP}:LRA=11:linear=true:measured_I={m['input_i']}:"
                           f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
                           f"measured_thresh={m['input_thresh']}:offset={m['target_offset']},aresample=48000",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart",
                    final + ".part.mp4"], check=True)
    os.replace(final + ".part.mp4", final)
    if not draft:                                            # the cut as rendered: what "keep": true re-cuts from
        project.dump(f"plan_{cut}_rendered.json", plan)
    print(f"[reel] -> {final}")
    return final
