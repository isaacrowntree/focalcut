"""Apply effect modules to any clip.

  python -m focalcut.fx list   (or: focalcut fx list)
  python -m focalcut.fx apply in.mp4 -e hero,neon [--peak 1.2] [--beat 0.604] [--cap 1.5] [--closeup] [-o out.mp4]
      --peak: where the move lands (s); a velocity curve (hero, bullet, montage) arrives slow there and holds
      --cap: top speed (dancing looks sped up past 1.5x); --closeup: hold deeper and longer (a hair flip)
  python -m focalcut.fx gallery in.mp4 [--peak 1.2] [-o gallery.mp4]      every effect side by side, labelled
"""
import argparse
import json
import subprocess

import cv2
import numpy as np

from . import EFFECTS, TRANSITIONS, Context


def read(path, max_w=None):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height,r_frame_rate:stream_side_data=rotation", "-of", "json", path]))
    s = j["streams"][0]
    rot = abs(int(next((d.get("rotation", 0) for d in s.get("side_data_list", [])), 0)))
    w, h = (s["height"], s["width"]) if rot in (90, 270) else (s["width"], s["height"])
    num, den = map(int, s["r_frame_rate"].split("/"))
    if max_w and w > max_w:
        w, h = max_w, int(round(h * max_w / w / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)), num / den


def write(frames, path, fps):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="focalcut fx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("apply", "gallery"):
        a = sub.add_parser(name); a.add_argument("input")
        a.add_argument("-e", "--effects", default="")
        a.add_argument("--peak", type=float); a.add_argument("--beat", type=float, default=0.604)
        a.add_argument("--cap", type=float, default=1.5); a.add_argument("--closeup", action="store_true")
        a.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("effects:    ", " ".join(sorted(EFFECTS)))
        print("transitions:", " ".join(sorted(TRANSITIONS)))
        return
    frames, fps = read(a.input, max_w=None if a.cmd == "apply" else 540)
    ctx = Context(fps=round(fps), beat=a.beat, peak=a.peak, params={"cap": a.cap, "closeup": a.closeup})
    if a.cmd == "apply":
        for n in [x for x in a.effects.split(",") if x]:
            frames = EFFECTS[n]().apply(frames, ctx)
        write(frames, a.out or a.input.rsplit(".", 1)[0] + f"_{a.effects.replace(',', '+')}.mp4", fps)
        return
    # gallery: original + every effect, tiled, each with its name
    names = ["original"] + sorted(EFFECTS)
    tiles = []
    for n in names:
        print("  ", n, flush=True)
        out = frames if n == "original" else EFFECTS[n]().apply(list(frames), ctx)
        out = [cv2.putText(np.ascontiguousarray(f.copy()), n, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
               for f in out[:len(frames)]]
        tiles.append(out + [out[-1]] * (len(frames) - len(out)))
    cols = 5 if frames[0].shape[1] < frames[0].shape[0] else 4
    rows = (len(tiles) + cols - 1) // cols
    blank = np.zeros_like(frames[0])
    grid = []
    for k in range(len(frames)):
        rws = [np.concatenate([tiles[r * cols + c][k] if r * cols + c < len(tiles) else blank for c in range(cols)], 1)
               for r in range(rows)]
        grid.append(np.concatenate(rws, 0))
    write(grid, a.out or "gallery.mp4", fps)


if __name__ == "__main__":
    main()
