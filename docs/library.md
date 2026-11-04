# Cutting a library by subject

## Why this is hard

The hard part is **identity under bad conditions**. Stage lighting washes every dancer the same colour, lens flare
swallows the subject, and dancers spin away from the camera. Naive signals fail. Frame differencing over-cuts on
every spin, and colour or even CLIP-embedding similarity can't tell two magenta-lit couples apart. A SAM 3
detector with an appearance tracker was tried and under-segmented crowded floors. It remains as the optional
`cut --engine image|video` path.

**What works is a contact sheet read by a vision model.** One thumbnail every few seconds is the most legible
identity signal there is: a vision model reading the grid reasons about clothing, hair, faces and recurrence in a
way no box metric can.

## The passes

All four are built from cheap ffmpeg contact sheets:

1. **Coarse:** one thumbnail every ~8 s per clip. A judge reads each sheet and returns candidate cut points.
2. **Refine:** a ±8 s sheet at 2 s spacing around each candidate checks that it's a real couple change (not flare,
   a spin or a partner rotation) and gives a ~2 s boundary.
3. **Ultra:** a ±3 s sheet at 0.5 s spacing pins each boundary to ~0.5 s.
4. **Miss-sweep:** a sceptical second read of the single-segment clips catches any hand-over the coarse pass missed.

The judge is a vision model fanned out one agent per sheet (Claude, via the scripts in `workflows/`). On a
50-clip portrait dance library this took **about 7 minutes in total** and pinned 26 cuts to 0.5 s.

## 1. Build contact sheets

```bash
focalcut library "/path/to/footage" --interval 8 --portrait-only
```

This writes `focalcut/sheets/*.jpg`, a `sheets.json` job list and a skeleton `library.json` manifest with the cuts
left blank.

## 2. Let a vision judge fill in the cuts

Fan out an agent per sheet using the four passes in `workflows/` (`01-coarse-judge` → `02-refine-cuts` →
`03-miss-sweep` → `04-ultra-boundary`). The finished `library.json` holds, per clip, `cuts` (seconds), `segments`
and `notes` describing each couple, so a person can check them.

## 3. Export a timeline

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

Timelines are built as [OpenTimelineIO](https://opentimelineio.readthedocs.io) objects and written through its
adapters. Drop-frame timecode, NTSC 1000/1001 rates and mixed-rate conform are handled by the interchange library
the post industry already maintains, not by a bespoke XML writer.

`--windows-paths` (on by default) maps WSL `/mnt/d/...` to `D:/...`, so a cut made in WSL opens in Premiere on
Windows. Turn it off with `--no-windows-paths`.

## One-off contact sheet

```bash
focalcut sheet "clip.mp4" --interval 10
```

## Direct cut (the SAM 3 engines)

```bash
focalcut cut "clip.mp4" --clips          # -> clip_dancers/ + clip.focalcut.xml
focalcut cut "/folder" --engine image --clips --copy
```

## Benchmark

`bench/` scores predicted cuts against hand-labelled ground truth: boundary precision and recall,
over- and under-segmentation, and a `manual_corrections_per_min` number for how much re-cutting is left. See
`bench/data/README.md`.
