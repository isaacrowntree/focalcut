"""The hardware a render can use, found once per run: all optional, plain ffmpeg on the CPU is the fallback.

Encoding: NVENC (h264_nvenc) when the GPU has it, at a quality a 1080p intermediate keeps (constant quality 16),
with SPS/PPS in every keyframe so shots from different ffmpeg builds join by stream copy. FOCALCUT_ENCODER=x264
forces libx264.

Decoding: camera files whose codec the Linux side can only decode on the CPU (e.g. Sony XAVC HS, 10-bit 4:2:2
HEVC: NVDEC before Blackwell has no 4:2:2, and Mesa's D3D12 VA-API bridge in WSL maps no 4:2:2 decode profile) can
go through a Windows ffmpeg build on WSL, which decodes them on the Intel iGPU with Quick Sync at a fraction of the
CPU time. Set FOCALCUT_WIN_FFMPEG to its ffmpeg.exe, or put a build under C:/tools/ffmpeg*/bin/; FOCALCUT_WIN_FFMPEG=
"" turns it off. Only files on a Windows drive (/mnt/<letter>/) go that way.

Parallel shots: render.DECODERS (FOCALCUT_DECODERS="qsv=5,cpu=2"). See docs/hardware.md for the measurements and
the WSL setup.
"""
from __future__ import annotations

import functools
import glob
import os
import subprocess

FPS = 50
COLOUR = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "14"]
NVENC = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "16", "-b:v", "0",
         "-profile:v", "high", "-bsf:v", "dump_extra=freq=keyframe"]


@functools.cache
def nvenc() -> bool:
    if os.environ.get("FOCALCUT_ENCODER", "").lower() == "x264":
        return False
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1", "-c:v",
                        "h264_nvenc", "-f", "null", "-"], check=True, capture_output=True, timeout=30)
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def enc() -> list[str]:
    """Video encoder arguments for every intermediate and final picture (1080p, FPS, BT.709)."""
    return [*(NVENC if nvenc() else X264), "-pix_fmt", "yuv420p", "-r", str(FPS), *COLOUR]


def enc_tag() -> str:
    """Part of each shot's cache key: shots joined by stream copy must all come from one encoder."""
    return "nvenc" if nvenc() else "x264"


@functools.cache
def win_ffmpeg() -> str | None:
    """A Windows ffmpeg.exe that decodes with Quick Sync (checked once), or None."""
    path = os.environ.get("FOCALCUT_WIN_FFMPEG")
    if path is None:
        found = sorted(glob.glob("/mnt/c/tools/ffmpeg*/bin/ffmpeg.exe"))
        path = found[-1] if found else ""
    if not path or not os.path.exists(path):
        return None
    try:
        subprocess.run([path, "-v", "error", "-init_hw_device", "qsv=hw", "-f", "lavfi", "-i", "nullsrc", "-t", "0.04",
                        "-f", "null", "-"], check=True, capture_output=True, timeout=60)
        return path
    except (subprocess.SubprocessError, OSError):
        return None


def decoders(default: dict) -> dict:
    """Shots rendered at once per decoder: FOCALCUT_DECODERS="qsv=5,cpu=2" overrides the default."""
    env = os.environ.get("FOCALCUT_DECODERS")
    if not env:
        return dict(default)
    return {k.strip(): int(v) for k, v in (kv.split("=") for kv in env.split(",") if "=" in kv)}


def winpath(p: str) -> str:
    return subprocess.check_output(["wslpath", "-w", os.path.abspath(p)], text=True).strip()


def on_windows_drive(p: str) -> bool:
    parts = os.path.abspath(p).split("/")
    return len(parts) > 3 and parts[1] == "mnt" and len(parts[2]) == 1


# what Quick Sync hands back for each camera format (its surface, copied to system memory with hwdownload)
QSV_SURFACE = {"yuv422p10le": "y210le", "yuv420p10le": "p010le", "yuv420p": "nv12", "yuvj420p": "nv12",
               "yuv422p": "yuyv422"}


@functools.cache
def pix_fmt(src: str) -> str:
    try:
        return subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=pix_fmt", "-of", "csv=p=0", src], text=True).strip()
    except subprocess.SubprocessError:
        return ""


def ffmpeg(src: str, qsv: bool):
    """(executable, decode arguments, path converter, filter prefix) for reading src: the Windows build decoding on
    the Intel iGPU with Quick Sync (frames copied back to memory, so the usual CPU filters follow) when qsv and it
    can, else the Linux ffmpeg on the CPU. (ffmpeg's generic D3D11 decode has no 4:2:2 HEVC: it falls back to the
    CPU without a word, which is why this goes through hevc_qsv. Its default async depth labels each frame three
    frames late, so -ss lands three frames early: depth 1 keeps the cuts frame-exact.)"""
    exe = win_ffmpeg() if qsv and on_windows_drive(src) else None
    surface = QSV_SURFACE.get(pix_fmt(src)) if exe else None
    if surface:
        return exe, ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv", "-async_depth", "1"], winpath, f"hwdownload,format={surface},"
    return "ffmpeg", [], (lambda p: p), ""
