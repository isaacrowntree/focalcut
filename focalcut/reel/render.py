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

SIZE = {"landscape": (1920, 1080), "portrait": (1080, 1920)}
FPS = 50
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
I, TP = -14.0, -1.5


def crop_expr(path, k):
    """Piecewise-linear crop centre (fraction of the source) over shot time t, as an ffmpeg expression; k picks x (1)
    or y (2) from the path's [t, x, y] points."""
    pts = [(p[0], p[k]) for p in path]
    expr = f"{pts[-1][1]}"
    for (t0, x0), (t1, x1) in reversed(list(zip(pts[:-1], pts[1:]))):
        seg = f"({x0}+({x1}-{x0})*(t-{t0})/{max(t1 - t0, 1e-3)})"
        expr = f"if(lt(t\\,{t1})\\,{seg}\\,{expr})"
    return expr


STAB = os.path.expanduser("~/.cache/focalcut/stab")         # vidstab motion files (a path without spaces for ffmpeg)


def stab(path, t, dur, rot):
    """Pass 1 of vidstab over exactly the stretch the shot decodes: the camera's shake, frame by frame."""
    os.makedirs(STAB, exist_ok=True)
    trf = os.path.join(STAB, hashlib.md5(f"{path}|{t:.3f}|{dur:.3f}|{rot}".encode()).hexdigest()[:12] + ".trf")
    if not os.path.exists(trf):
        pre = (rot + ",") if rot else ""
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-t", f"{dur:.3f}", "-i", path, "-an", "-vf",
                        f"{pre}vidstabdetect=shakiness=6:accuracy=12:result={trf}", "-f", "null", "-"], check=True)
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
        f.append(f"vidstabtransform=input={trf}:smoothing=15:optzoom=1:zoomspeed=0.25:interpol=bicubic")
    c = s.get("crop")
    if c:
        f.append(f"crop=w=iw*{c['w']}:h=ih*{c['h']}:x=({crop_expr(c['path'], 1)})*iw-ow/2"
                 f":y=({crop_expr(c['path'], 2)})*ih-oh/2")
    f += [f"scale={W}:{H}:flags=lanczos", f"fps={FPS}", "setsar=1", "format=yuv420p"]
    return ",".join(f)


def key(s):
    """Cache name for a shot: changes whenever anything that shapes its pixels does."""
    k = json.dumps([s["unit"]["id"], s["src_in"], s["frames"], s.get("fx"), s.get("crop"), s["peak"], s.get("logo"),
                    s.get("stab"), s["kind"] == "closeup", 2],
                   sort_keys=True)
    return hashlib.md5(k.encode()).hexdigest()[:10]


def cut_shot(project, s, fmt, out):
    src, t = project.decode_from(s["unit"]["source"], s["src_in"])
    chain = shot_filter(s, fmt, project, s["length"] + 0.1)
    vf = ["-vf", chain]
    logo = project.logo()
    if s.get("logo") and logo:                              # the event's logo, front and centre over the opening
        W, H = SIZE[fmt]
        fade = ",fade=t=in:st=0.15:d=0.6:alpha=1" if s["logo"] == "in" else ""
        vf = ["-i", logo, "-filter_complex",
              f"[0:v]{chain},vignette=angle=PI/5,eq=brightness=-0.05[v];"
              f"[1:v]scale={int(W * (0.5 if W > H else 0.8))}:-1,format=rgba,loop=-1:1,fps={FPS}{fade}[l];"
              f"[v][l]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-t", f"{s['length'] + 0.1:.3f}",
                    "-i", src, "-an", *vf, *ENC, "-frames:v", str(s["frames"]), out],
                   check=True)


def fx_shot(project, s, fmt, beat, out):
    """Render a shot through its effect chain (speed curves first change how much source is read)."""
    from .. import fx
    from ..fx.speed import Speed
    from .plan import reframe
    W, H = SIZE[fmt]
    u, L = s["unit"], s["length"]
    ctx = fx.Context(fps=FPS, beat=beat, peak=s["peak"], seed=int(key(s)[:4], 16), params={"group": s.get("group"), "closeup": s["kind"] == "closeup",
                                     "cap": 3.5 if u["role"] == "transition" else 1.5})   # dancing never past 1.5x
    src_len, src_peak = L, s["peak"]
    for n in s["fx"]:
        e = fx.EFFECTS[n]()
        if isinstance(e, Speed):
            src_len, src_peak = e.source_length(L, ctx), e.source_peak(s["peak"], ctx)
    src_in = min(max(s["t"] - src_peak, u["t0"]), max(u["t1"] - src_len, u["t0"]))
    shot = dict(s, src_in=src_in, length=src_len)
    if s.get("crop"):                              # the source read changed: follow the subject over it instead
        shot["crop"] = reframe(project, u, src_in, src_len, fmt) or s["crop"]
    src, t = project.decode_from(u["source"], src_in)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-t", f"{src_len:.3f}", "-i", src, "-an",
                          "-vf", shot_filter(shot, fmt, project, src_len), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
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
                          "-r", str(FPS), "-i", "-", *ENC, out], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()


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
            ctx = fx.Context(fps=FPS, beat=plan["beat"], seed=i)
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
    music = project.load("music.json") if os.path.exists(project.path("music.json")) else {}
    W, H = SIZE[fmt]
    outdir, work = project.path("out"), project.path("work", cut)
    os.makedirs(outdir, exist_ok=True); os.makedirs(work, exist_ok=True)
    files = []
    for i, s in enumerate(plan["shots"]):
        if s.get("fx") and not draft:
            out = os.path.join(work, f"shot_{i:03d}_{key(s)}_fx.mp4")
            if not os.path.exists(out):
                fx_shot(project, s, fmt, plan["beat"], out)
                print(f"\n  {'+'.join(s['fx'])}: {s['unit']['id']} @ {s['t']}s", flush=True)
        else:
            out = os.path.join(work, f"shot_{i:03d}_{key(s)}.mp4")
            if not os.path.exists(out):
                cut_shot(project, s, fmt, out)
        files.append(out)
        print(f"\r[reel] {fmt}: shot {i + 1}/{len(plan['shots'])}", end="", flush=True)
    print()
    if not draft:
        files = apply_transitions(plan, files)
    with open(os.path.join(outdir, f"{cut}_shots.json"), "w") as f:
        json.dump([os.path.relpath(p, project.reel_dir) for p in files], f, indent=1)
    stinger = project.stinger(fmt) if plan.get("stinger") else None
    if stinger:                                    # the stinger as the last "shot", conformed to size and rate
        sting = os.path.join(work, "stinger.mp4")
        if not os.path.exists(sting):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", stinger, "-an",
                            "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS},setsar=1,format=yuv420p", *ENC, sting],
                           check=True)
        files = [*files, sting]
    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in files)
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
    print(f"[reel] -> {final}")
    return final
