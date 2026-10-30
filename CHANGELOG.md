# Changelog

All notable changes to focalcut are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Recap reels** — `focalcut reel init|run` turns a curated export folder into beat-synced landscape and
  portrait reels plus Premiere (FCP7 XML) timelines: pose/flow/expression moment detection, coverage of every
  clip, frame-exact beat cuts and SAM 3 effects. Optional `reel` extra. See `docs/recipes/recap-reel.md`.
  - **Subject detection** (`subject` step): SAM 3 person masks scored by BiRefNet saliency and Depth Anything V2
    nearness pick who a shot is about, with edge contact so cut-off heads are marked down. `focus` (per-person
    sharpness) is the fallback; sharpness alone picks still bystanders over dancers in motion blur.
  - **Reframing**: every shot is cropped around its subject (up to 1.6× from 4K) with a smoothed path.
  - **What a shot shows** (`semantic` step): SigLIP 2 zero-shot on the subject's crop for hips / butt work and
    expressive faces; adds the `shake` moment kind and lifts faces. The opening of each unit is marked down.
  - **A story in acts**: teaser, arrival, shows, night, climax, finale on the song's bars, with a signature
    transition at each change of act and transitions picked per cut by content (no one type carries an act).
  - Beat accents in `music.json`; `proxies` in `reel.json` to decode a heavy master from a smaller file.
  - **Editing rules from music videos**:
    - Shot lengths are varied 8-beat phrase patterns that follow the music's energy, with holds and a build into
      the climax.
    - Move peaks land on beats, and cuts come 2 frames ahead of the beat.
    - Shots from the same clips are dealt like a card shuffle, never back to back.
    - Cuts carry the movement on (turn to turn, same direction of travel; new `flow.py`).
    - Every clip gets a fair share and appears at least once, with several different scenes from each.
  - **Velocity edits** (`velocity.py`, `fx/speed.py`): hero / bullet / montage run fast through the move, ease down
    to land slow on the dancer's own landing (`settle`: a burst, then a rest), hold, and ease back up. The landing sits
    on the strongest beat that leaves room for the hold. Dancing is capped at 1.5x, and close-ups hold deeper.
    `focalcut fx apply` takes `--cap` and `--closeup`.
  - **Transition grammar** (`grammar.py`): mostly hard cuts. Each act has its own short vocabulary with caps, and a
    transition goes in only with a reason: turn into turn, the same travel carried across, a push in to a face, a big
    hit (accent threshold) or the DJ. Never back to back. Replaces the per-act transition `share` top-up.
  - **Shake** (`shake` step, `shake.py`): camera jitter per unit. Shots too shaky for their crop give way to steadier
    parts of the clip, and mildly shaky shots are stabilised at render (vidstab). Knob: `shake`.
  - **Portrait framing**: a 9:16 strip follows one dancer of a split couple (`frame.follow`). Moments rank on how the
    crop holds the subject (present, wholly inside, a face to the camera; `plan.assess`), so backs on the social
    floor drop out.
  - **Band inserts** (`members`, `member_clips`, `band_in`) and a chosen **ending** shot (`ending`).
  - **Cohesive effects**: one effect at a time; one subject effect, on the climax's final bullet-time move. The
    punch-ins and orbs are gone.
  - **Any cut, not just the reels**: `plan.place()` / `plan.shot()` time, frame and transition a hand-built
    sequence. `render` / `xml_export` / `review` take `cut=`, writing `out/<name>_<cut>.mp4`.
  - **`skills/focalcut`**: a Claude Code skill listing every concept and the code behind it, with an example
    teaser script.
  - **Opening**: band / crowd B-roll with the event logo (`logo` in `reel.json`) centred across the cut.
    Daytime B-roll is held long and woven with the demos. The night weaves band shots between its dancing.
  - **Tracking**: between subject keyframes the crop follows the subject's pose boxes at 5 fps. A portrait strip
    cut from landscape footage follows faster.
  - **Effect masks**: BiRefNet matte refinement inside the SAM 3 mask, blended in-between frames and temporal
    smoothing, for clean edges.
  - The climax no longer reuses a moment it has reserved for a later slow-motion shot.
  - **Activity gate**: moments are marked down unless the subject dances, and stays in frame, for the whole shot.
  - **Editor picks**: ban or prefer ranges per export file, in `picks.json` or under reel.json's `picks`. Units
    carry `export_t0`, so the times are the export's own.
  - **Review step**: a local HTML page per reel, showing every shot with its export file, time range and a
    copyable picks line.
- **Effect modules** — `focalcut fx list|apply|gallery`: speed curves (montage, hero, bullet, jumpcut,
  flashin, flashout), SAM 3 subject effects (spotlight, freeze, neon, particles, velocity, echo, strobe, orbs,
  aura), beat effects (punchin, shake, glitch) and transitions (flash, fade, dipwhite, fadein, zoom, swipe —
  following the motion's direction —, spin, lightleak, glitch). Subject effects keep the salient couple, or the
  whole line on stage.
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
