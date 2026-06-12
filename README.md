# focalcut

**Split a video into per-subject segments by following whoever the camera is focused on.**

Point focalcut at dance footage (or any video where the camera moves between subjects) and it finds the cuts where the focus changes from one person/couple to the next, then exports per-subject clips and an editor timeline (Premiere / Resolve / Final Cut).

Built on Meta's [SAM 3](https://ai.meta.com/research/sam3/) for detection and tracking. MIT-licensed, runs locally on a single consumer GPU.

> Status: early. Works well on clips with a clear focal subject; crowded, uniformly-lit floors are hard (see *How it works*).

## Why this is hard (and how it works)

The core difficulty is **identity under bad conditions**: stage lighting washes every dancer the same colour, lens flare swallows the subject, and dancers spin away from the camera. Naive signals fail — frame-differencing over-cuts on every spin, and colour or even CLIP-embedding similarity can't tell two magenta-lit couples apart.

focalcut offers two engines and a coarse-to-fine workflow:

- **`image` engine (fast, default).** Samples a frame every ~2 s, runs SAM 3's image detector (~3.4 GB VRAM, no growing memory bank), and tracks the focal subject by appearance, *coasting* through flared/occluded frames rather than cutting on them. A few seconds per clip. Great when there's one dominant subject; can miss handoffs on a crowded floor.
- **`video` engine (slow, robust).** SAM 3's video tracker with a real learned memory bank — holds a subject's identity through flare, occlusion, and spins. Minutes per clip and heavier on VRAM, but correct on hard footage.
- **Coarse-to-fine (recommended for crowded footage).** A **contact sheet** — one thumbnail every N seconds — is the most legible identity signal there is, because a vision model reading the grid can reason about clothing, faces, and recurrence the way no box metric can. The intended workflow:
  1. `focalcut sheet clip.mp4` → thumbnail grid (scratch; cleaned up after use).
  2. A vision judge (Claude, or you) reads the grid and flags which buckets are a *different* subject.
  3. SAM 3 refines each flagged ~N-second window to pin the exact cut frame.

  This is a toolkit Claude can drive end to end, in the spirit of a contact-sheet edit.

## Install

Requires Python ≥3.10, `ffmpeg`/`ffprobe` on PATH, and a CUDA GPU for the SAM 3 engines.

```bash
pip install -e .
```

SAM 3 (`facebook/sam3`) is a **gated** model: accept its license on Hugging Face and log in (`hf auth login`) before first use.

## Usage

```bash
# Contact sheet (coarse pass) — one thumbnail every 10s
focalcut sheet "clip.mp4" --interval 10

# Cut a single clip into per-subject segments + extract clips to a subfolder
focalcut cut "clip.mp4" --clips            # -> clip_dancers/ + clip.focalcut.xml

# Whole folder, fast engine, model loaded once
focalcut cut "/path/to/folder" --clips --copy

# Hard clip: use the robust video tracker
focalcut cut "crowded.mp4" --engine video

# Proof montage of the detected segments (sanity check)
focalcut cut "clip.mp4" --proof
```

Key flags: `--engine image|video`, `--fps` (sample rate), `--clips` / `--clips-dir` / `--copy`,
`--prompt` (SAM 3 concept, default `person`), `--windows-paths` (map WSL `/mnt/d` → `D:` for
Windows NLEs), `--proof`, `--dump-json`.

Exports are FCP7 XML (`xmeml`), which Premiere, Resolve, and Final Cut all import.

## Benchmark

`bench/` scores predicted cuts against hand-labelled ground truth — boundary precision/recall,
over/under-segmentation, and a `manual_corrections_per_min` "how much re-cutting is left" number.
See `bench/data/README.md`.

## License

MIT (this code). SAM 3 model weights are under Meta's separate SAM License — review before use.
