"""Export focal segments to an editor timeline.

Primary format is FCP7 XML (xmeml v5) — the lingua franca that Premiere, DaVinci
Resolve, and Final Cut all import, and an open spec (no third-party adapter, no
ButterCut code). Each segment becomes one clip on V1, laid end to end, referencing
the original media by source in/out so the editor can trim, reorder, or delete.
"""
from __future__ import annotations
import os
from xml.sax.saxutils import escape
from .segment import Segment


def _pathurl(path: str, windows_paths: bool) -> str:
    """file:// URL for the media. Optionally map WSL /mnt/<d>/... -> <D>:/... ."""
    p = os.path.abspath(path)
    if windows_paths and p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        drive = p[5].upper()
        p = f"{drive}:{p[6:]}"
        return "file://localhost/" + p.replace("\\", "/").lstrip("/").replace(" ", "%20").replace(":", "%3A", 0)
    from urllib.request import pathname2url
    return "file://localhost" + pathname2url(p)


def to_fcp7_xml(segments: list[Segment], source_path: str, src_fps: float,
                width: int, height: int, sequence_name: str = "focalcut",
                windows_paths: bool = False) -> str:
    tb = int(round(src_fps))
    ntsc = "TRUE" if abs(src_fps - tb) > 1e-3 else "FALSE"
    fname = os.path.basename(source_path)
    url = _pathurl(source_path, windows_paths)

    def rate():
        return f"<rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>"

    items, tl = [], 0
    for i, s in enumerate(segments):
        src_in = int(round(s.start * src_fps))
        src_out = int(round(s.end * src_fps))
        n = src_out - src_in
        tl_start, tl_end = tl, tl + n
        tl = tl_end
        label = escape(f"{os.path.splitext(fname)[0]} · dancer {i+1}")
        file_el = (f'<file id="src"><name>{escape(fname)}</name>'
                   f'<pathurl>{escape(url)}</pathurl>{rate()}'
                   f'<media><video><samplecharacteristics><width>{width}</width>'
                   f'<height>{height}</height></samplecharacteristics></video></media></file>'
                   ) if i == 0 else '<file id="src"/>'
        items.append(
            f'<clipitem id="clip{i+1}"><name>{label}</name>{rate()}'
            f'<start>{tl_start}</start><end>{tl_end}</end>'
            f'<in>{src_in}</in><out>{src_out}</out>{file_el}</clipitem>')

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="5">\n'
        f'<sequence id="seq"><name>{escape(sequence_name)}</name>'
        f'<duration>{tl}</duration>{rate()}'
        f'<media><video><format><samplecharacteristics>{rate()}'
        f'<width>{width}</width><height>{height}</height></samplecharacteristics></format>'
        f'<track>{"".join(items)}</track></video></media></sequence>\n</xmeml>\n')


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
    import subprocess, tempfile
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
        inputs = sum([["-i", t] for t in tiles], [])
        # pad the grid to cols*rows so the tile filter always has enough inputs
        subprocess.run(["ffmpeg", "-v", "error", *inputs, "-filter_complex",
                        f"tile={cols}x{rows}:padding=4:color=black", "-y", out_path],
                       capture_output=True)
    return out_path if os.path.exists(out_path) else None
