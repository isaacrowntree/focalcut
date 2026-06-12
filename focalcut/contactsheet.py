"""Contact-sheet (thumbnail grid) builder — the coarse pass.

A frame every N seconds, labelled with its timestamp, tiled into a grid. Trivial and
fast (pure ffmpeg), and a *vision* judge reading the grid can tell couples apart far
more reliably than any box/embedding metric under uniform stage lighting. Sheets are
scratch: built into a temp dir and cleaned up after, unless `keep=True`.
"""
from __future__ import annotations
import os, subprocess, tempfile
from . import video as V


def build_sheet(video_path: str, interval: float = 10.0, cols: int = 4,
                thumb_w: int = 300, out_path: str | None = None,
                start: float = 0.0, duration: float | None = None) -> str:
    """Build a labelled thumbnail grid, one frame every `interval` seconds.

    Returns the path to the JPEG. If `out_path` is None it lands in a temp dir
    (caller owns cleanup — see clean_sheet()).
    """
    info = V.probe(video_path)
    dur = duration if duration is not None else max(0.0, info.duration - start)
    n = max(1, int(dur // interval))
    times = [start + interval / 2 + k * interval for k in range(n)]

    if out_path is None:
        fd, out_path = tempfile.mkstemp(prefix="focalcut_sheet_", suffix=".jpg")
        os.close(fd)

    with tempfile.TemporaryDirectory() as td:
        tiles = []
        for i, t in enumerate(times):
            mm, ss = divmod(int(t), 60)
            tp = os.path.join(td, f"t{i:03d}.jpg")
            r = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", video_path, "-frames:v", "1",
                 "-vf", f"scale={thumb_w}:-2,drawtext=text='{mm:02d}\\:{ss:02d}':"
                        "x=8:y=8:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.7",
                 "-y", tp], capture_output=True)
            if r.returncode == 0 and os.path.exists(tp):
                tiles.append(tp)
        if not tiles:
            raise RuntimeError(f"no frames extracted from {video_path}")
        rows = (len(tiles) + cols - 1) // cols
        # tile() works on frames of ONE stream, so feed the thumbnails as a glob
        # sequence (sequentially named t000.jpg, t001.jpg, ...), not as N -i inputs.
        subprocess.run(["ffmpeg", "-v", "error", "-framerate", "1", "-pattern_type", "glob",
                        "-i", os.path.join(td, "t*.jpg"), "-frames:v", "1", "-filter_complex",
                        f"tile={cols}x{rows}:padding=6:color=white", "-y", out_path],
                       capture_output=True, check=True)
    return out_path


def clean_sheet(path: str) -> None:
    """Remove a scratch contact sheet."""
    try:
        os.remove(path)
    except OSError:
        pass
