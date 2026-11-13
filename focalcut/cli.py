"""focalcut CLI — split video into focal-subject segments and export clips / a timeline.

Pass one file or many (a whole folder of clips). The SAM 3 model is loaded once and
reused across every clip, so batch runs amortise the load cost.

Only `cut`, `reel` (scoring, faces, effects) and `fx` need torch / opencv. The recommended contact-sheet pipeline
(`library`, `sheet`, `export`) is GPU-free, so the torch-backed modules are imported
lazily inside `cut` — importing them here would make `pip install focalcut` drag in the
whole CUDA stack for users who never touch the SAM engines.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from . import video as V
from .contactsheet import build_sheet
from .timeline import FORMATS as _FORMATS


def _build_engine(a):
    """Construct the tracking engine once (reusable across many clips)."""
    if a.engine == "video":
        from .track import Sam3Tracker
        return ("video", Sam3Tracker(model_id=a.model, device=a.device))
    from .detect import Sam3ImageDetector
    return ("image", Sam3ImageDetector(model_id=a.model, device=a.device))


def _cut_one(a, path: str, engine, single: bool) -> int:
    from . import timeline as T
    from .detect import associate
    from .export import write_dancer_clips, write_proof_montage
    from .segment import segment_focal

    info = V.probe(path)
    size = V.target_size(info, a.long_side)
    fps = a.fps if a.fps is not None else (4.0 if a.engine == "video" else 1.0)
    print(f"\n[focalcut] {os.path.basename(path)}  {info.width}x{info.height} "
          f"{info.fps:.0f}fps {info.duration:.0f}s -> {a.engine} @ {fps}fps {size[0]}x{size[1]}",
          file=sys.stderr)

    t = time.time()
    if a.engine == "video":
        frames = V.decode(path, fps, size, start=a.start, duration=a.duration)
        times = [i / fps + (a.start or 0.0) for i in range(len(frames))]
    else:
        frames, times = V.decode_keyframes(path, size, target_fps=fps)
    dec = time.time() - t

    kind, eng = engine
    t = time.time()
    if kind == "video":
        per_frame = eng.track(frames, prompt=a.prompt, min_score=a.min_score)
    else:
        per_frame = associate(eng.detect(frames, prompt=a.prompt, min_score=a.min_score))
    try:
        import torch
        vram = f"  peakVRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB" if a.device == "cuda" else ""
    except Exception:
        vram = ""
    print(f"[focalcut] decoded {len(frames)} + {kind} {len(per_frame)} frames "
          f"in {dec:.1f}+{time.time()-t:.1f}s{vram}", file=sys.stderr)

    segs = segment_focal(per_frame, frames, times, size, area_frac=a.area_frac,
                         min_score=a.min_score, sim_thresh=a.sim_thresh,
                         min_dwell=a.min_dwell, min_segment=a.min_segment)

    print(f"[focalcut] {len(segs)} focal segments:")
    for i, s in enumerate(segs):
        print(f"  {i+1:2d}  {s.start:7.1f}s -> {s.end:7.1f}s  ({s.duration:5.1f}s)  ids={s.obj_ids}")

    stem = os.path.splitext(path)[0]
    base = (a.output if (single and a.output) else stem + ".focalcut.xml")
    record = [{"name": os.path.basename(path),
               "segments": [[s.start, s.end] for s in segs]}]
    tl = T.build_timeline(os.path.dirname(os.path.abspath(path)), record,
                          os.path.basename(stem), windows_paths=a.windows_paths)
    T.write_timeline(tl, base, "fcp7")
    print(f"[focalcut] timeline -> {base}")

    if a.dump_json and single:
        import json
        with open(a.dump_json, "w") as fh:
            json.dump({"video": path, "segments": [{"start": round(s.start, 3),
                       "end": round(s.end, 3), "obj_ids": s.obj_ids} for s in segs]},
                      fh, indent=2)
        print(f"[focalcut] segments json -> {a.dump_json}")

    if a.clips:
        clips_dir = (a.clips_dir if (single and a.clips_dir)
                     else os.path.join(os.path.dirname(path), os.path.basename(stem) + "_dancers"))
        written = write_dancer_clips(segs, path, clips_dir, reencode=not a.copy)
        print(f"[focalcut] {len(written)} dancer clips -> {clips_dir}/")

    if a.proof:
        proof = os.path.splitext(base)[0] + ".proof.jpg"
        if write_proof_montage(segs, path, proof):
            print(f"[focalcut] proof montage -> {proof}")
    return len(segs)


def _expand(inputs: list[str]) -> list[str]:
    """Expand directories/globs into a sorted list of video files."""
    exts = (".mp4", ".mov", ".m4v", ".mkv")
    out = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fn in sorted(os.listdir(inp)):
                if fn.lower().endswith(exts):
                    out.append(os.path.join(inp, fn))
        elif any(ch in inp for ch in "*?["):
            out.extend(sorted(glob.glob(inp)))
        else:
            out.append(inp)
    return out


def cmd_cut(a: argparse.Namespace) -> int:
    files = _expand(a.inputs)
    if not files:
        print("[focalcut] no input videos found", file=sys.stderr)
        return 1
    single = len(files) == 1
    t0 = time.time()
    engine = _build_engine(a)
    print(f"[focalcut] {a.engine} engine loaded in {time.time()-t0:.1f}s; {len(files)} clip(s)",
          file=sys.stderr)
    total = 0
    for _i, path in enumerate(files):
        try:
            total += _cut_one(a, path, engine, single)
        except Exception as e:
            print(f"[focalcut] ERROR on {path}: {e}", file=sys.stderr)
    print(f"\n[focalcut] done: {len(files)} clip(s), {total} segments, "
          f"{time.time()-t0:.1f}s total ({(time.time()-t0)/len(files):.1f}s/clip)", file=sys.stderr)
    return 0


def cmd_library(a: argparse.Namespace) -> int:
    """Coarse pass for a folder: build a contact sheet per clip + a sheet-jobs file.

    The vision judge (Claude, fanned out one agent per sheet) then reads the jobs file
    and returns cut points; see README for the coarse->fine->ultra pipeline. This step
    is the deterministic, GPU-free part — no cutting.
    """
    import time as _t

    from . import library as lib
    libd = lib.lib_dir(a.folder)
    out_dir = a.sheets_dir or os.path.join(libd, "sheets")

    def prog(i, n, j):
        ok = "ok" if j.get("sheet") else f"ERROR {j.get('error','')}"
        print(f"  [{i:>3}/{n}] {j['name']:<22} {ok}", file=sys.stderr)

    t0 = _t.time()
    jobs = lib.build_sheets(a.folder, out_dir, interval=a.interval, cols=a.cols,
                            portrait_only=a.portrait_only, on_progress=prog)
    ok = [j for j in jobs if j.get("sheet")]
    jobs_path = os.path.join(libd, "sheets.json")
    json_dump(jobs_path, {"folder": a.folder, "interval": a.interval, "sheets": ok})
    # skeleton manifest (cuts to be filled by the judge)
    clips = [{"name": j["name"], "duration": j["duration"], "sheet": j["sheet"],
              "cuts": [], "segments": [[0.0, j["duration"]]], "needs_review": True}
             for j in ok]
    man = lib.save(a.folder, clips, {"interval_s": a.interval, "status": "sheets_built"})
    print(f"\n[focalcut] {len(ok)}/{len(jobs)} sheets -> {out_dir}/  ({_t.time()-t0:.0f}s)")
    print(f"[focalcut] sheet jobs -> {jobs_path}")
    print(f"[focalcut] skeleton manifest -> {man}")
    print("[focalcut] next: fan out a vision agent per sheet to fill in cut points "
          "(see README: coarse->fine->ultra).")
    return 0


def json_dump(path, obj):
    import json
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


IMPORT_HINT = {
    "fcp7": "Premiere/Resolve: File > Import, select the XML.",
    "aaf": "Media Composer: File > Input > Import.",
    "edl": "Any NLE: import as an EDL, then relink to the source media.",
    "otio": "Open with any OTIO-aware tool, or convert with otioconvert.",
}


def cmd_export(a: argparse.Namespace) -> int:
    """Export the library's focal segments as an editor timeline in any OTIO format."""
    from . import library as lib
    from . import timeline as T
    man = lib.load(a.folder)
    if not man.get("clips"):
        print("[focalcut] no library.json found — run `focalcut library` first", file=sys.stderr)
        return 1
    clips = [c for c in man["clips"]
             if c.get("kind") != "non-couple" and c.get("n_couples") != 0 and c.get("segments")]
    name = a.name or (os.path.basename(a.folder.rstrip("/")) + " selects")
    out = a.output or T.default_output(a.folder, a.format, lib.lib_dir(a.folder))
    try:
        tl = T.build_timeline(a.folder, clips, name, windows_paths=a.windows_paths)
        T.write_timeline(tl, out, a.format)
    except T.TimelineError as e:
        print(f"[focalcut] {e}", file=sys.stderr)
        return 1
    nseg = sum(len(c["segments"]) for c in clips)
    print(f"[focalcut] {len(clips)} clips, {nseg} segments -> {out}")
    print(f"[focalcut] {IMPORT_HINT[a.format]}")
    return 0


def cmd_formats(a: argparse.Namespace) -> int:
    """List the timeline formats whose OTIO adapter is installed."""
    from . import timeline as T
    try:
        have = set(T.available_formats())
    except T.TimelineError as e:
        print(f"[focalcut] {e}", file=sys.stderr)
        return 1
    for fmt, (adapter, suffix) in T.FORMATS.items():
        mark = "yes" if fmt in have else "no "
        print(f"  {mark}  {fmt:8s} {suffix:8s} ({adapter})")
    if len(have) < len(T.FORMATS):
        print("\n  missing adapters: pip install opentimelineio-plugins")
    return 0


def cmd_sheet(a: argparse.Namespace) -> int:
    """Coarse pass: build a thumbnail-grid contact sheet for a vision judge to read."""
    out = a.output or os.path.splitext(a.input)[0] + ".sheet.jpg"
    path = build_sheet(a.input, interval=a.interval, cols=a.cols,
                       start=a.start or 0.0, duration=a.duration, out_path=out)
    print(path)
    return 0


def cmd_reel_init(a: argparse.Namespace) -> int:
    """Write <folder>/focalcut/reel/reel.json (merging into an existing one)."""
    from .reel.project import Project
    pr = Project(a.folder)
    cfg = pr.config
    cfg["song"] = os.path.abspath(a.song)
    if a.start is not None:
        cfg["start"] = a.start if a.start == "auto" else float(a.start)
    cfg.setdefault("start", "auto")
    if a.length is not None:
        cfg["length"] = a.length
    cfg.setdefault("length", 180)
    st = cfg.setdefault("stingers", {})
    if a.stinger_landscape:
        st["landscape"] = os.path.abspath(a.stinger_landscape)
    if a.stinger_portrait:
        st["portrait"] = os.path.abspath(a.stinger_portrait)
    print(pr.save_config())
    return 0


def cmd_reel_serve(a: argparse.Namespace) -> int:
    from .reel import serve
    from .reel.project import Project
    if not a.folder and not a.root:
        raise SystemExit("give one or more project folders, or --root")
    serve.run([Project(f) for f in a.folder], port=a.port, root=a.root)
    return 0


def cmd_reel_run(a: argparse.Namespace) -> int:
    from . import reel
    steps = a.steps.split(",") if a.steps else None
    bad = [s for s in steps or [] if s not in reel.STEPS]
    if bad:
        print(f"unknown step(s): {bad}; choose from {reel.STEPS}", file=sys.stderr)
        return 2
    reel.run(a.folder, steps=steps, formats=a.formats.split(",") if a.formats else None, workers=a.workers,
             draft=a.draft, cut=a.cut, clean=a.clean)
    return 0


def cmd_fx(a: argparse.Namespace) -> int:
    from .fx.__main__ import main as fx_main
    fx_main(a.args)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="focalcut")
    sub = p.add_subparsers(dest="cmd", required=True)

    rl = sub.add_parser("reel", help="beat-synced recap reels from a curated export folder")
    rsub = rl.add_subparsers(dest="reel_cmd", required=True)
    ri = rsub.add_parser("init", help="write the reel config (song, start, length, stingers)")
    ri.add_argument("folder")
    ri.add_argument("--song", required=True, help="the track to cut to")
    ri.add_argument("--start", help='"auto" (first bar after the intro, default) or seconds')
    ri.add_argument("--length", type=float, help="total reel length in seconds, stinger included (default 180)")
    ri.add_argument("--stinger-landscape", help="16:9 end stinger")
    ri.add_argument("--stinger-portrait", help="9:16 end stinger")
    ri.set_defaults(func=cmd_reel_init)
    rr = rsub.add_parser("run", help="run the pipeline (or chosen steps)")
    rr.add_argument("folder")
    rr.add_argument("--steps", help="comma list of steps: units,score,shake,focus,subject,semantic,moments,faces,expression,people,persons,role,windows,music,plan,render,xml,review "
                    "(default: all)")
    rr.add_argument("--formats", help="comma list (default: landscape,portrait, or the cut's formats)")
    rr.add_argument("--cut", help='one of reel.json\'s "cuts": another reel from the same footage, e.g. night')
    rr.add_argument("--workers", type=int, default=2, help="scoring processes (2 suits a 12 GB GPU)")
    rr.add_argument("--draft", action="store_true", help="render without effects or transitions (fast check)")
    rr.add_argument("--clean", action="store_true", help="render the same cut clean, to <name>_clean.mp4: the footage on the same "
                    "cut points with the music -- no effects, speed changes, transitions, lift, logos, overlay or stinger")
    rr.set_defaults(func=cmd_reel_run)
    rv = rsub.add_parser("serve", help="serve the review pages of one or more projects; they save picks.json as you "
                                       "make picks")
    rv.add_argument("folder", nargs="*", help="project folders; each is served under /p/<slug>/")
    rv.add_argument("--root", help="also serve every <root>/*/focalcut/reel/reel.json, including ones added later")
    rv.add_argument("--port", type=int, default=8765)
    rv.set_defaults(func=cmd_reel_serve)

    fxp = sub.add_parser("fx", help="apply effect modules to a clip (list | apply | gallery)")
    fxp.add_argument("args", nargs=argparse.REMAINDER)
    fxp.set_defaults(func=cmd_fx)

    lb = sub.add_parser("library", help="coarse pass: build a contact sheet per clip in a folder (no cutting)")
    lb.add_argument("folder")
    lb.add_argument("--interval", type=float, default=8.0, help="seconds between thumbnails")
    lb.add_argument("--cols", type=int, default=6)
    lb.add_argument("--portrait-only", action="store_true", help="only process portrait clips")
    lb.add_argument("--sheets-dir", help="where to write sheets (default <folder>/.focalcut_sheets)")
    lb.set_defaults(func=cmd_library)

    ep = sub.add_parser("export", help="export a selects timeline from a library (FCP7/AAF/EDL/OTIO)")
    ep.add_argument("folder")
    ep.add_argument("-o", "--output", help="output file (default <folder>/focalcut/selects.<ext>)")
    ep.add_argument("-f", "--format", default="fcp7", choices=list(_FORMATS),
                    help="timeline format (default fcp7: Premiere, Resolve, Final Cut)")
    ep.add_argument("--name", help="sequence name")
    ep.add_argument("--windows-paths", action="store_true", default=True,
                    help="map /mnt/<d> -> <D>: for Premiere on Windows (default on)")
    ep.add_argument("--no-windows-paths", dest="windows_paths", action="store_false")
    ep.set_defaults(func=cmd_export)

    fm = sub.add_parser("formats", help="list timeline formats and whether their adapter is installed")
    fm.set_defaults(func=cmd_formats)

    sh = sub.add_parser("sheet", help="build a contact sheet (coarse pass)")
    sh.add_argument("input")
    sh.add_argument("-o", "--output", help="output JPEG (default <input>.sheet.jpg)")
    sh.add_argument("--interval", type=float, default=10.0, help="seconds between thumbnails")
    sh.add_argument("--cols", type=int, default=4)
    sh.add_argument("--start", type=float, default=None)
    sh.add_argument("--duration", type=float, default=None)
    sh.set_defaults(func=cmd_sheet)

    c = sub.add_parser("cut", help="split video(s) by focal subject")
    c.add_argument("inputs", nargs="+", help="video file(s), a folder, or a glob")
    c.add_argument("-o", "--output", help="output XML path (single input only)")
    c.add_argument("--engine", choices=["image", "video"], default="image",
                   help="image = fast sparse detection (default); video = full memory-bank tracking")
    c.add_argument("--prompt", default="person", help="SAM 3 concept prompt")
    c.add_argument("--fps", type=float, default=None,
                   help="sample rate (default 0.5 for image, 4 for video)")
    c.add_argument("--long-side", type=int, default=512,
                   help="downscale long side (no speed effect; SAM 3 resizes internally)")
    c.add_argument("--start", type=float, default=None, help="start second")
    c.add_argument("--duration", type=float, default=None, help="seconds to process")
    c.add_argument("--device", default="cuda")
    c.add_argument("--model", default="facebook/sam3")
    c.add_argument("--min-score", type=float, default=0.5)
    c.add_argument("--area-frac", type=float, default=0.02, help="min focal area fraction")
    c.add_argument("--sim-thresh", type=float, default=0.45,
                   help="chroma-hist similarity above which it's the same subject")
    c.add_argument("--min-dwell", type=float, default=3.0, help="seconds a new subject must persist")
    c.add_argument("--min-segment", type=float, default=4.0, help="merge shorter segments")
    c.add_argument("--windows-paths", action="store_true", help="map /mnt/<d> -> <D>: in XML")
    c.add_argument("--clips", action="store_true", help="extract each dancer to its own file")
    c.add_argument("--clips-dir", help="subfolder for clips (single input; default <input>_dancers)")
    c.add_argument("--copy", action="store_true", help="stream-copy clips (fast, keyframe-snapped)")
    c.add_argument("--proof", action="store_true", help="render a thumbnail montage of segments")
    c.add_argument("--dump-json", help="write segments JSON (single input; for benchmarking)")
    c.set_defaults(func=cmd_cut)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
