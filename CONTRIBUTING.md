# Contributing to focalcut

Contributions are welcome — bug reports, footage that breaks the cutter, and pull
requests alike.

## What I promise you

- **A reply within a week.** If a PR needs changes, you will hear what they are in one
  message, not discovered one at a time over months. If I go quiet, ping the PR — that
  is a failure on my side, not rudeness on yours.
- **A decision, not a limbo.** If something is not going to be merged, I will say so and
  say why, while the branch is still fresh in your head.
- **Credit.** Merged work is credited in `CHANGELOG.md` under your GitHub handle.
- **Small PRs get taken over, not bounced.** If a change is nearly right and I have the
  context to finish it, I will finish it and merge rather than send you round again.

## Before you open a PR

- **One change per PR.** A bugfix and a feature in one branch is two PRs. This is about
  review, not bureaucracy: a focused branch is one I can read and merge in an evening.
- **Run the suite.** `pytest` must pass, `ruff check .` must be clean. Both run in CI on
  Python 3.10–3.13.
- **Add a test for behaviour you change.** Especially anything touching frame timing —
  a cut landing one frame early is invisible in review and obvious in an NLE.
- **Describe it in your own words.** A couple of honest sentences about what broke and
  what you did beats a generated changelog. Use whatever tools you like to write the
  *code* — just make sure the description is something you could defend out loud.

## Development setup

```bash
git clone https://github.com/isaacrowntree/focalcut
cd focalcut
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # GPU-free; add ".[dev,sam]" only for the SAM engines
pytest
```

You need `ffmpeg` and `ffprobe` on `PATH`. The suite synthesises its own fixture clips
with ffmpeg, so there is no media to download and no GPU, network, or Hugging Face
access required — if a test needs any of those, it is a broken test.

## Project shape

| module           | role                                                     |
| ---------------- | -------------------------------------------------------- |
| `video.py`       | ffprobe/ffmpeg probing and decoding, rotation-aware      |
| `contactsheet.py`| thumbnail grids — the coarse signal the vision judge reads |
| `library.py`     | folder scan, manifest read/write, segment derivation     |
| `timeline.py`    | OpenTimelineIO timeline construction and NLE export      |
| `export.py`      | ffmpeg rendering: per-subject clips, proof montages       |
| `segment.py`     | turning per-frame tracks into segments                   |
| `detect.py`, `track.py` | the optional SAM 3 engines (needs `[sam]`)         |
| `workflows/`     | the vision-judge passes (coarse → refine → sweep → ultra) |

## A note on timeline formats

focalcut does not hand-write NLE formats. Timelines are built as OpenTimelineIO objects
and written through OTIO adapters, so drop-frame timecode, NTSC 1000/1001 rates and
mixed-rate conform are handled upstream by the library the post industry already
maintains. If you want focalcut to support another editor, the answer is almost always
an OTIO adapter rather than a new writer in this repo.

## Licence

focalcut is MIT. By contributing you agree your work ships under that licence.

Please do not paste code from a source with an incompatible licence — in particular,
anything under a non-commercial or source-available licence cannot go into an MIT
project, however small the snippet. Facts about a file format (frame rates, field
names, how a rotation flag is signed) are not copyrightable and are fine; someone
else's implementation of them is not.
