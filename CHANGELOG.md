# Changelog

All notable changes to focalcut are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Timelines export to Premiere, Resolve, Final Cut, Avid and any EDL-capable editor.**
  `focalcut export -f {fcp7,aaf,edl,otio}` writes through
  [OpenTimelineIO](https://opentimelineio.readthedocs.io), so timecode handling comes
  from the interchange library the post industry maintains rather than from a writer
  in this repo. `focalcut formats` lists what your install can produce.
- **A test suite** — 66 tests covering timing, manifests, rotation handling and the
  CLI, all synthesising their own ffmpeg fixtures. No GPU, network or model access.
- **CI** on Python 3.10–3.13, which also asserts the GPU-free install stays free of
  torch.
- `CONTRIBUTING.md` and issue/PR templates.

### Changed

- **`pip install focalcut` no longer pulls the CUDA stack.** torch, transformers and
  accelerate moved to an opt-in `[sam]` extra; the recommended contact-sheet pipeline
  needs only numpy and ffmpeg. Install `focalcut[sam]` for the SAM 3 engines.
- `focalcut export-premiere` is now `focalcut export --format fcp7` (the default).

### Removed

- The hand-written FCP7 XML generators (`premiere.py`, `export.to_fcp7_xml`), replaced
  by the OTIO path above.

## [0.1.0]

Initial release: contact-sheet library pipeline with a vision judge
(coarse → refine → miss-sweep → ultra), optional SAM 3 engines, and per-subject clip
extraction.
