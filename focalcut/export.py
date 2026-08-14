"""Render focal segments to media: per-subject clip files and a proof montage.

Timeline export lives in timeline.py (OpenTimelineIO); this module is only the
ffmpeg-backed rendering that turns segments into things you can watch.
"""
from __future__ import annotations

import functools
import operator
import os

from .segment import Segment


def write_dancer_clips(segments: list[Segment], source_path: str, out_dir: str,
                       reencode: bool = True, crf: int = 20) -> list[str]:
    """Cut each focal segment out of the source into its own file under `out_dir`.

    reencode=True gives frame-accurate boundaries (H.264, broadly shareable);
    reencode=False stream-copies (fast, but snaps to keyframes).
    Returns the list of written clip paths.
    """
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i, s in enumerate(segments):
        name = f"dancer_{i+1:02d}_{s.start:.0f}s-{s.end:.0f}s.mp4"
        dst = os.path.join(out_dir, name)
        if reencode:
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{s.start}", "-i", source_path,
                   "-t", f"{s.duration}", "-c:v", "libx264", "-crf", str(crf),
                   "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", dst]
        else:
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{s.start}", "-i", source_path,
                   "-t", f"{s.duration}", "-c", "copy", dst]
        subprocess.run(cmd, check=False)
        if os.path.exists(dst):
            written.append(dst)
    return written


def write_proof_montage(segments: list[Segment], source_path: str, out_path: str) -> str | None:
    """Tile one frame from the middle of each segment so a human can eyeball that
    each segment really is a different dancer/couple. Returns path or None."""
    import subprocess
    import tempfile
    if not segments:
        return None
    tiles = []
    with tempfile.TemporaryDirectory() as td:
        for i, s in enumerate(segments):
            mid = (s.start + s.end) / 2
            mm, ss = divmod(int(s.start), 60)
            label = f"{i+1}  {mm:02d}\\:{ss:02d}"   # static label (escaped colon)
            tp = os.path.join(td, f"t{i:03d}.jpg")
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{mid}", "-i", source_path,
                            "-frames:v", "1", "-vf",
                            f"scale=320:-2,drawtext=text='{label}':"
                            "x=8:y=8:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6",
                            "-y", tp], capture_output=True)
            if os.path.exists(tp):
                tiles.append(tp)
        if not tiles:
            return None
        cols = min(5, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        inputs = functools.reduce(operator.iadd, [["-i", t] for t in tiles], [])
        # pad the grid to cols*rows so the tile filter always has enough inputs
        subprocess.run(["ffmpeg", "-v", "error", *inputs, "-filter_complex",
                        f"tile={cols}x{rows}:padding=4:color=black", "-y", out_path],
                       capture_output=True)
    return out_path if os.path.exists(out_path) else None
