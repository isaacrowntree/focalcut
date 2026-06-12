"""focalcut CLI — split video into focal-subject segments and export clips / a timeline.

Pass one file or many (a whole folder of clips). The SAM 3 model is loaded once and
reused across every clip, so batch runs amortise the load cost.
"""
from __future__ import annotations
import argparse, glob, os, sys, time
from . import video as V
from .track import Sam3Tracker
from .detect import Sam3ImageDetector, associate
from .segment import segment_focal
from .export import to_fcp7_xml, write_proof_montage, write_dancer_clips
from .contactsheet import build_sheet


def _build_engine(a):
    """Construct the tracking engine once (reusable across many clips)."""
    if a.engine == "video":
        return ("video", Sam3Tracker(model_id=a.model, device=a.device))
    return ("image", Sam3ImageDetector(model_id=a.model, device=a.device))


def _cut_one(a, path: str, engine, single: bool) -> int:
    info = V.probe(path)
    size = V.target_size(info, a.long_side)
    fps = a.fps if a.fps is not None else (4.0 if a.engine == "video" else 0.5)
    print(f"\n[focalcut] {os.path.basename(path)}  {info.width}x{info.height} "
          f"{info.fps:.0f}fps {info.duration:.0f}s -> {a.engine} @ {fps}fps {size[0]}x{size[1]}",
          file=sys.stderr)

    t = time.time()
    frames = V.decode(path, fps, size, start=a.start, duration=a.duration)
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

    segs = segment_focal(per_frame, frames, fps, size, area_frac=a.area_frac,
                         min_score=a.min_score, sim_thresh=a.sim_thresh,
                         min_dwell=a.min_dwell, min_segment=a.min_segment,
                         t_offset=(a.start or 0.0))

    print(f"[focalcut] {len(segs)} focal segments:")
    for i, s in enumerate(segs):
        print(f"  {i+1:2d}  {s.start:7.1f}s -> {s.end:7.1f}s  ({s.duration:5.1f}s)  ids={s.obj_ids}")

    stem = os.path.splitext(path)[0]
    base = (a.output if (single and a.output) else stem + ".focalcut.xml")
    with open(base, "w") as f:
        f.write(to_fcp7_xml(segs, path, info.fps, info.width, info.height,
                            sequence_name=os.path.basename(stem), windows_paths=a.windows_paths))
    print(f"[focalcut] timeline -> {base}")

    if a.dump_json and single:
        import json
        json.dump({"video": path, "segments": [{"start": round(s.start, 3),
                   "end": round(s.end, 3), "obj_ids": s.obj_ids} for s in segs]},
                  open(a.dump_json, "w"), indent=2)
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
    for i, path in enumerate(files):
        try:
            total += _cut_one(a, path, engine, single)
        except Exception as e:
            print(f"[focalcut] ERROR on {path}: {e}", file=sys.stderr)
    print(f"\n[focalcut] done: {len(files)} clip(s), {total} segments, "
          f"{time.time()-t0:.1f}s total ({(time.time()-t0)/len(files):.1f}s/clip)", file=sys.stderr)
    return 0


def cmd_sheet(a: argparse.Namespace) -> int:
    """Coarse pass: build a thumbnail-grid contact sheet for a vision judge to read."""
    out = a.output or os.path.splitext(a.input)[0] + ".sheet.jpg"
    path = build_sheet(a.input, interval=a.interval, cols=a.cols,
                       start=a.start or 0.0, duration=a.duration, out_path=out)
    print(path)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="focalcut")
    sub = p.add_subparsers(dest="cmd", required=True)

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
