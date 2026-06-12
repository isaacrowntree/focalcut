# focalcut

**Split a video into per-subject segments by following whoever the camera is focused on.**

Point focalcut at dance footage (or any video where the camera moves between subjects) and it finds the cuts where the focus changes from one person/couple to the next, then exports per-subject clips and an editor timeline (Premiere / Resolve / Final Cut).

MIT-licensed. The recommended pipeline is **GPU-free** (ffmpeg + a vision model).

## Why this is hard (and how it works)

The core difficulty is **identity under bad conditions**: stage lighting washes every dancer the same colour, lens flare swallows the subject, and dancers spin away from the camera. Naive signals fail — frame-differencing over-cuts on every spin, and colour or even CLIP-embedding similarity can't tell two magenta-lit couples apart. (A SAM 3 detector + appearance tracker was tried and under-segmented crowded floors; it remains as the optional `cut --engine image|video` path.)

**What actually works is a contact sheet read by a vision model.** One thumbnail every N seconds is the most legible identity signal there is — a vision model reading the grid reasons about clothing, hair, faces, and recurrence the way no box metric can. The library pipeline is coarse → fine → ultra, all built from cheap ffmpeg contact sheets:

1. **Coarse** — one thumbnail every ~8 s per clip; a judge reads each and returns candidate cut points.
2. **Refine** — a ±8 s sheet at 2 s spacing around each candidate verifies it's a real couple change (not flare/spin/partner-rotation) and gives a ~2 s boundary.
3. **Ultra** — a ±3 s sheet at 0.5 s spacing pins each boundary to ~0.5 s.
4. **Miss-sweep** — a skeptical second read of the single-segment clips catches any handoff the coarse pass missed.

The judge is a vision model fanned out one agent per sheet (Claude, via the scripts in `workflows/`). On a 50-clip portrait dance library this ran in **~7 minutes total** and pinned 26 cuts to 0.5 s — see `workflows/` for the four parallel passes.

## Install

Requires Python ≥3.10 and `ffmpeg`/`ffprobe` on PATH. The library pipeline needs no GPU.
(The optional `cut --engine` paths use SAM 3 `facebook/sam3`, a **gated** model — accept its
license on Hugging Face and `hf auth login` first; a CUDA GPU is recommended for those.)

```bash
pip install -e .
```

## Usage

### Build a cut-point library for a folder (recommended)

```bash
# Coarse pass: one contact sheet per clip (portrait-only optional). GPU-free.
focalcut library "/path/to/folder" --interval 8 --portrait-only
#  -> <folder>/.focalcut_sheets/*.jpg
#  -> <folder>/focalcut_sheets.json   (sheet jobs for the judge)
#  -> <folder>/focalcut_library.json  (skeleton manifest; cuts filled by the judge)
```

Then fan out a vision agent per sheet to fill in the cut points, refine, and pin
boundaries — the four passes in `workflows/` (`01-coarse-judge` → `02-refine-cuts` →
`03-miss-sweep` → `04-ultra-boundary`). The final `focalcut_library.json` holds, per clip,
`cuts` (seconds), `segments`, and `notes` describing each couple for human verification.

### One-off contact sheet

```bash
focalcut sheet "clip.mp4" --interval 10
```

### Direct cut/export (optional SAM 3 engines)

```bash
focalcut cut "clip.mp4" --clips            # -> clip_dancers/ + clip.focalcut.xml
focalcut cut "/folder" --engine image --clips --copy
```

Exports are FCP7 XML (`xmeml`), which Premiere, Resolve, and Final Cut all import.
`--windows-paths` maps WSL `/mnt/d` → `D:` for Windows NLEs.

## Benchmark

`bench/` scores predicted cuts against hand-labelled ground truth — boundary precision/recall,
over/under-segmentation, and a `manual_corrections_per_min` "how much re-cutting is left" number.
See `bench/data/README.md`.

## License

MIT (this code). SAM 3 model weights are under Meta's separate SAM License — review before use.
