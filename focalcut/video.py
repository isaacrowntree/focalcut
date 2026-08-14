"""Video probing and frame decoding via ffmpeg (no decord/pyav dependency)."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int          # display (rotation-corrected) width
    height: int
    fps: float
    duration: float     # seconds


def probe(path: str) -> VideoInfo:
    """Read stream metadata, applying container rotation so width/height are upright."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    st = data["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    # rotation can live in tags.rotate or in a display-matrix side-data entry
    rot = 0
    if "tags" in st and "rotate" in st["tags"]:
        rot = int(st["tags"]["rotate"])
    for sd in st.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = ([*st.get("r_frame_rate", "0/1").split("/"), "1"])[:2]
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    dur = float(data.get("format", {}).get("duration") or st.get("duration") or 0.0)
    return VideoInfo(path, w, h, fps, dur)


def target_size(info: VideoInfo, long_side: int) -> tuple[int, int]:
    """Scale so the long side == long_side, preserving aspect, dims even."""
    if info.width >= info.height:
        w = long_side
        h = round(info.height * long_side / info.width)
    else:
        h = long_side
        w = round(info.width * long_side / info.height)
    return (w - w % 2, h - h % 2)


def decode(path: str, fps: float, size: tuple[int, int],
           start: float | None = None, duration: float | None = None) -> np.ndarray:
    """Decode to an (N, H, W, 3) uint8 RGB array at `fps`, scaled to `size`, upright.

    ffmpeg auto-applies rotation metadata, so output frames are display-oriented.
    """
    w, h = size
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start}"]
    if duration is not None:
        cmd += ["-t", f"{duration}"]
    cmd += ["-i", path, "-vf", f"fps={fps},scale={w}:{h}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def decode_keyframes(path: str, size: tuple[int, int], target_fps: float | None = None):
    """Decode only keyframes (much faster than full decode) with their real timestamps.

    Keyframes are typically every ~0.5-1s, dense enough to locate focal handoffs, and
    decoding only them avoids the cost of decoding every inter-frame. Returns
    (frames (N,H,W,3) uint8, times list[float] seconds). `showinfo` reports each
    decoded frame's pts_time so timestamps are exact, not assumed.
    """
    w, h = size
    cmd = ["ffmpeg", "-v", "info", "-skip_frame", "nokey", "-i", path,
           "-vf", f"scale={w}:{h},showinfo", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    frames = np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", p.stderr.decode("utf-8", "ignore"))]
    n = min(len(frames), len(times))
    frames, times = frames[:n], times[:n]
    if target_fps and len(times) > 1:
        # subsample keyframes down toward target_fps (keeps timestamps honest)
        span = times[-1] - times[0]
        want = max(1, round(span * target_fps))
        if want < n:
            idx = np.linspace(0, n - 1, want).round().astype(int)
            idx = sorted(set(idx.tolist()))
            frames, times = frames[idx], [times[i] for i in idx]
    return frames, times
