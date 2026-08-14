"""Build an editor timeline from a focalcut library and write it in any NLE format.

focalcut describes a cut as data — an ordered list of (source file, in, out) selects —
and hands the format problem to OpenTimelineIO. OTIO is the ASWF/Pixar interchange
library the post industry already uses, so the fiddly parts we would otherwise own
forever (drop-frame timecode, NTSC 1000/1001 rates, mixed-rate source conform, per-NLE
quirks) are solved upstream and stay solved.

Formats come from whichever OTIO adapters are installed:

    otio      always                   round-trips losslessly, focalcut's native form
    fcp7      opentimelineio-plugins   Premiere / Resolve / Final Cut 7 xmeml
    aaf       opentimelineio-plugins   Avid Media Composer
    edl       opentimelineio-plugins   CMX 3600, universal conform

Timing is kept in RationalTime at the source frame rate throughout, so a 23.976 or
29.97 source never accumulates the rounding drift you get from float seconds.
"""
from __future__ import annotations

import os

from . import video as V

#: focalcut format name -> (OTIO adapter name, file suffix)
FORMATS: dict[str, tuple[str, str]] = {
    "otio": ("otio_json", ".otio"),
    "fcp7": ("fcp_xml", ".xml"),
    "aaf": ("AAF", ".aaf"),
    "edl": ("cmx_3600", ".edl"),
}


class TimelineError(RuntimeError):
    """Raised when a timeline cannot be built or written."""


def _otio():
    """Import OTIO, turning the ImportError into an actionable message."""
    try:
        import opentimelineio as otio
    except ImportError as e:  # pragma: no cover - exercised via the CLI
        raise TimelineError(
            "OpenTimelineIO is required for timeline export.\n"
            "  pip install 'focalcut[otio]'"
        ) from e
    return otio


def available_formats() -> list[str]:
    """Format names whose adapter is actually installed, in FORMATS order."""
    otio = _otio()
    installed = {a.name for a in otio.plugins.ActiveManifest().adapters}
    return [k for k, (adapter, _) in FORMATS.items() if adapter in installed]


def _media_reference(path: str, fps: float, duration: float):
    """An ExternalReference covering the whole source file."""
    otio = _otio()
    return otio.schema.ExternalReference(
        target_url=_file_url(path),
        available_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, fps),
            duration=otio.opentime.RationalTime(round(duration * fps), fps),
        ),
    )


def _file_url(path: str) -> str:
    """A file:// URL for `path` (absolute, percent-encoded)."""
    from urllib.request import pathname2url

    return "file://" + pathname2url(os.path.abspath(path))


def to_windows_path(path: str) -> str:
    """Map a WSL mount to its Windows drive: /mnt/d/foo -> D:/foo.

    Premiere on Windows cannot open /mnt/... paths, so a timeline cut in WSL needs
    its media paths translated. Non-mount paths are returned unchanged.
    """
    p = os.path.abspath(path)
    parts = p.split("/")
    if len(parts) > 3 and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():
        return f"{parts[2].upper()}:/" + "/".join(parts[3:])
    return p


def build_timeline(folder: str, clips: list[dict], sequence_name: str = "focalcut selects",
                   windows_paths: bool = False):
    """Lay every clip's segments end to end on one video track.

    `clips` are library records: {"name": str, "segments": [[start_s, end_s], ...]}.
    Each segment becomes a Clip referencing the source's in/out, which is what lets an
    editor trim, reorder or delete a select without re-rendering anything.

    The timeline's rate is taken from the first clip; per-clip source rates are kept on
    each Clip, and OTIO conforms them on write.
    """
    otio = _otio()
    if not clips:
        raise TimelineError("no clips to export")

    timeline = otio.schema.Timeline(name=sequence_name)
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)

    probed: dict[str, V.VideoInfo] = {}
    for record in clips:
        name = record["name"]
        path = os.path.join(folder, name)
        if name not in probed:
            if not os.path.exists(path):
                raise TimelineError(f"media missing: {path}")
            probed[name] = V.probe(path)
        info = probed[name]
        if not info.fps:
            raise TimelineError(f"could not read a frame rate from {name}")

        url_path = to_windows_path(path) if windows_paths else path
        stem = os.path.splitext(name)[0]

        for index, (start, end) in enumerate(record.get("segments") or [], start=1):
            if end <= start:
                continue
            source_range = otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(round(start * info.fps), info.fps),
                duration=otio.opentime.RationalTime(round((end - start) * info.fps), info.fps),
            )
            track.append(otio.schema.Clip(
                name=f"{stem}-{index:02d}",
                media_reference=_media_reference(url_path, info.fps, info.duration),
                source_range=source_range,
            ))

    if not len(track):
        raise TimelineError("no segments to export — has the judge filled in the manifest?")
    return timeline


def write_timeline(timeline, out_path: str, fmt: str = "fcp7") -> str:
    """Write `timeline` to `out_path` using the adapter for `fmt`."""
    otio = _otio()
    if fmt not in FORMATS:
        raise TimelineError(f"unknown format {fmt!r} (choose from {', '.join(FORMATS)})")
    adapter, _suffix = FORMATS[fmt]
    installed = {a.name for a in otio.plugins.ActiveManifest().adapters}
    if adapter not in installed:
        raise TimelineError(
            f"the {fmt!r} adapter is not installed — pip install opentimelineio-plugins")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    # AAF normally expects Avid MobIDs, which only exist on media ingested through
    # Media Composer. focalcut cuts camera-original files, so we let the adapter mint
    # them; without this it refuses to write at all.
    kwargs = {"use_empty_mob_ids": True} if fmt == "aaf" else {}
    otio.adapters.write_to_file(timeline, out_path, adapter_name=adapter, **kwargs)
    return out_path


def default_output(folder: str, fmt: str, lib_dir: str) -> str:
    """Where an export lands when the user does not name a file."""
    _adapter, suffix = FORMATS[fmt]
    return os.path.join(lib_dir, f"selects{suffix}")
