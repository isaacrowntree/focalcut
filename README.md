# focalcut

**Split a video into per-subject segments by following whoever the camera is focused on.**

[![CI](https://github.com/isaacrowntree/focalcut/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacrowntree/focalcut/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Point focalcut at dance footage — or any video where the camera moves between subjects —
and it finds the cuts where focus changes from one person or couple to the next, then
exports per-subject clips and a timeline you can open in Premiere, DaVinci Resolve,
Final Cut or Avid.

The recommended pipeline is **GPU-free**: ffmpeg contact sheets read by a vision model.

## Why this is hard (and how it works)

The core difficulty is **identity under bad conditions**: stage lighting washes every
dancer the same colour, lens flare swallows the subject, and dancers spin away from the
camera. Naive signals fail — frame-differencing over-cuts on every spin, and colour or
even CLIP-embedding similarity cannot tell two magenta-lit couples apart. (A SAM 3
detector plus appearance tracker was tried and under-segmented crowded floors; it
remains as the optional `cut --engine image|video` path.)

**What actually works is a contact sheet read by a vision model.** One thumbnail every
N seconds is the most legible identity signal there is — a vision model reading the grid
reasons about clothing, hair, faces and recurrence the way no box metric can. The
library pipeline is coarse → fine → ultra, all built from cheap ffmpeg contact sheets:

1. **Coarse** — one thumbnail every ~8 s per clip; a judge reads each and returns
   candidate cut points.
2. **Refine** — a ±8 s sheet at 2 s spacing around each candidate verifies it is a real
   couple change (not flare, a spin, or partner rotation) and gives a ~2 s boundary.
3. **Ultra** — a ±3 s sheet at 0.5 s spacing pins each boundary to ~0.5 s.
4. **Miss-sweep** — a skeptical second read of the single-segment clips catches any
   handoff the coarse pass missed.

The judge is a vision model fanned out one agent per sheet (Claude, via the scripts in
`workflows/`). On a 50-clip portrait dance library this ran in **~7 minutes total** and
pinned 26 cuts to 0.5 s.

## Install

Requires Python ≥3.10 and `ffmpeg`/`ffprobe` on `PATH`.

```bash
pip install focalcut
```

That is the full GPU-free pipeline — numpy, OpenTimelineIO and ffmpeg, nothing else.

<details>
<summary>Optional: the SAM 3 engines</summary>

`focalcut cut` uses SAM 3 (`facebook/sam3`), a **gated** model — accept its licence on
Hugging Face and `hf auth login` first. A CUDA GPU is recommended. This pulls torch and
transformers (~2.5 GB), which is why it is opt-in:

```bash
pip install "focalcut[sam]"
```

</details>

## Usage

### 1. Build contact sheets for a folder of clips

```bash
focalcut library "/path/to/footage" --interval 8 --portrait-only
```

Writes `focalcut/sheets/*.jpg`, a `sheets.json` job list, and a skeleton
`library.json` manifest with the cuts left blank.

### 2. Let a vision judge fill in the cuts

Fan out an agent per sheet using the four passes in `workflows/`
(`01-coarse-judge` → `02-refine-cuts` → `03-miss-sweep` → `04-ultra-boundary`). The
finished `library.json` holds, per clip, `cuts` (seconds), `segments`, and `notes`
describing each couple so a human can verify them.

### 3. Export a timeline

```bash
focalcut export "/path/to/footage"                 # FCP7 XML (default)
focalcut export "/path/to/footage" -f edl          # CMX 3600
focalcut export "/path/to/footage" -f aaf          # Avid Media Composer
focalcut export "/path/to/footage" -f otio         # OpenTimelineIO
focalcut formats                                   # what your install can write
```

| format | extension | opens in |
| ------ | --------- | -------- |
| `fcp7` | `.xml`    | Premiere Pro, DaVinci Resolve, Final Cut 7 |
| `aaf`  | `.aaf`    | Avid Media Composer |
| `edl`  | `.edl`    | anything (relink media after import) |
| `otio` | `.otio`   | any OTIO-aware tool; lossless round-trip |

Timelines are built as [OpenTimelineIO](https://opentimelineio.readthedocs.io) objects
and written through its adapters, so drop-frame timecode, NTSC 1000/1001 rates and
mixed-rate source conform are handled by the interchange library the post industry
already maintains — not by a bespoke XML writer here.

`--windows-paths` (on by default) maps WSL `/mnt/d/...` to `D:/...` so a cut made in WSL
opens in Premiere on Windows. Turn it off with `--no-windows-paths`.

### One-off contact sheet

```bash
focalcut sheet "clip.mp4" --interval 10
```

### Direct cut (optional SAM 3 engines)

```bash
focalcut cut "clip.mp4" --clips          # -> clip_dancers/ + clip.focalcut.xml
focalcut cut "/folder" --engine image --clips --copy
```

## Benchmark

`bench/` scores predicted cuts against hand-labelled ground truth — boundary
precision/recall, over/under-segmentation, and a `manual_corrections_per_min`
"how much re-cutting is left" number. See `bench/data/README.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs get a reply within a week.

## Licence

MIT — see [LICENSE](LICENSE). SAM 3 model weights are under Meta's separate SAM
Licence; review it before using the `[sam]` engines.
