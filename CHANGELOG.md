# Changelog

All notable changes to focalcut are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Frame portrait**, a third view of the source browser (`sources.html#frame=<reel>`): the reel's portrait cut
  shot by shot, the 9:16 crop over the picture and the cropped result beside it. Drag the frame to pan the crop
  (one pan for the whole shot, or pans at the playhead that it glides between), click a dancer to follow them,
  Backspace for the planner's framing again. The shot's state (planner's, your pan, following, or "no face found
  where you clicked") shows in the header, in the shot list and in a toast. The cut's own crops come with the
  page, so the frame shows at once and on a page opened as a file. Review pages of a portrait cut link each shot
  to it.
- **Two pages on one project no longer overwrite each other.** `api/picks` answers with an ETag; a page saves
  with `If-Match` and, when the file changed meanwhile, merges value by value and saves again. A page also takes
  the file as it is when it gets the focus. The server keeps a copy of `picks.json` every ten minutes of editing
  in `picks_history/`.
- **Undo is per change, with redo** (`Ctrl+Z`, `Ctrl+Shift+Z`, and buttons), on the review pages too. An undo
  after another page's edit was merged leaves that edit alone.
- **Picks mode** (reel.json `"mode": "picks"`, at the top or inside a cut; `plan.pick_story`): the reel is cut from
  the editor's picks. Every pick plays once, whole, in the story's order with the clips woven through each other;
  when the picks overrun the reel they give up beats in proportion, the least starred first, none under two beats,
  and only then do picks go (named in the log); when they leave room, B-roll the editor left alone fills it; the
  opening, `open_with`, `end_with`, `ending` and `close` are honoured; every format plays the same picks in the
  same order at the same lengths; no coverage, people, band, quick-cut or recast pass runs. The plan log says how
  many beats the picks ask for against what the reel has. `pick_hold`, `pick_grow`, `pick_share`,
  `pick_clip_share`, `pick_by`, `pick_whole`, `pick_like`, `reuse` and `open_broll` are accepted and not needed.
- **A pick's portrait strip goes to the faces first** (`plan.face_first`): on a pick, the strip re-centres on the
  faces by it (a face cut by its edge, a kiss at its side), and where it shows no face for a second on end (a back,
  hair, the gap between people) it goes to a face shot close instead, staying with a person through a turn of the
  head and panning once between two friends who face the camera in turn. A far-off face never takes the strip
  from a dancer whose own is hidden. Focal points and Follow clicks still override it;
  `portrait_rules: {"pick_faces": false}` turns it off.
- `plan.<fmt>.ending`'s crop centre takes `"follow"`: a single performer is framed like a pick, where a number holds
  the scene centred.
- **4K reels.** reel.json `"resolution"`: `"1080p"` (default), `"1440p"`, `"4k"` or `[width, height]`, for the
  project, per format (`{"landscape": "4k", "portrait": "1080p"}`) or per cut. Shots, effects, transitions, the
  overlay, the stinger and the XML's sequence follow it; shots are cached per size. MilkDrop and the built-in
  overlay effects keep their 1080p canvas and are scaled up, so a 4K reel has the same show.
- **A much faster render** ([Rendering](docs/render.md) has the review and the timings). The overlay is composited
  on the GPU (`composite.py`: the footage stays in 4:2:0 and the show is applied as `footage * a + b` on its
  planes; `FOCALCUT_OVERLAY=cpu` keeps the numpy path), with the decode and the encode on threads of their own;
  the MilkDrop layer renders beside the shots; stabilised shots have their shake measured many at once
  (`FOCALCUT_STAB_JOBS`) and then render beside the steady and the effect shots; transitions stream their shots;
  no joined copies of the cut are written. The render prints each stage's time and saves `out/<cut>_timings.json`.

- **Picks per reel.** One `picks.json` serves the project's reel and its `cuts` (a 90 s version, say): a clip's
  `only` gives a selection, ban, prefer, opening, ending or `more` to some reels only. The source page has a Reel
  switch (each reel's shots in the lanes, other reels' picks faded; picks, bans, opening and ending made or removed
  for the reel being picked for) and a tick per reel on every pick; a cut's review page edits for its own reel; the
  planner takes each reel's own. The source page also shows the cuts' shots, and a cut's review step refreshes it.
- A clip whose picks are all for other reels gives a reel nothing of it (not the planner's choice either), unless
  the reel's settings name it (an opening, ending, close, `open_with` / `end_with`); `end_with` puts picks just
  before the ending, and picks moved by `open_with` / `end_with` play whole.
- **Stars.** Rate a pick 1-5 (keys 1-5, or the stars on it; picks `"stars": {range: n}`): when a reel has more
  picks than room, the most starred are lengthened first and dropped last.
- **Follow a person.** In portrait framing, `G` (or Follow person) then a click on a dancer: the crop tracks that
  person's face through the shot (picks `"follow"`; `plan.follow_override`), or holds on the click where no face is
  found.
- **Fitting many picks into a short reel** (`plan.<fmt>` keys): `pick_grow` (every pick at `pick_hold` beats, the
  room left lengthening the best to `pick_grow`), `pick_by: "clarity"` (drop the picks whose subject is least
  clearly seen first, B-roll ranked among B-roll), `pick_whole` (clips whose picks play whole: the reel's focus),
  `pick_like` (a format plays the picks another kept, e.g. portrait the landscape cut's), `open_broll` (B-roll
  alone in the opening acts: people shots swap with later B-roll, and the window recast keeps to B-roll there).

### Changed

- **The night lift is a tracked window on the faces** (`window.py`), as a colourist would grade it. Each lit night
  shot is read as it plays (rotated and cropped), about ten times a second; YuNet finds the faces, and their own
  level decides the lift: toward `target`, at most `stops` (default 0.6) brighter, nothing for faces already bright,
  lost in black or seen in under a third of the shot (silhouettes, B-roll, the logo opening). A feathered ellipse
  from head to chest follows each face, and inside it a shadow lift raises the face (the deepest shadows and white
  stay put) with its colour (held back on skin already coloured by stage light); outside it the room is exactly as
  shot. Composited in the shot's own ffmpeg pass: a few seconds per lit shot. reel.json `exposure.lift`:
  `{"target": 0.4, "stops": 0.6}`; `"window": false` keeps the old curve over the whole frame, which lifted the
  background as much as the people and the darkest faces least.
- **No stabilisation by default; shaky footage is avoided instead.** The planner's own shots (fillers, B-roll under
  the logo, people it brought in) move along their clip to its steadiest stretch (`plan.steady_shots`: clear of
  bans and the clip's other shots, on passing windows, a move for a move, framing as well in portrait), and a pick
  trimmed to fit plays the steadiest part of what was picked (`best_part`), the same part in every format. The
  editor's picks play as shot. reel.json `"stabilise": true` brings vidstab back; a render ignores an older plan's
  stabilised shots unless it is set.
- **The review pages' front end is files** (`focalcut/reel/web/`: `ui.css`, `picks.js`, `app.js`, a page's own
  HTML / CSS / JS), put together inline by `webui.document()`; no build step, and a page still works opened as a
  file. The picks model (`picks.js`) is tested under Node and the pages in a headless Chromium when one is there
  (`tests/test_web.py`, `tests/ui/pages.mjs`).
- **The source browser is laid out as a list beside the picture** in every view, so the picture and its tools
  stay on screen while the list scrolls; what an action did shows as a toast, the save state stays in the header,
  and `?` lists every key. `focalcut reel serve` makes a sources page from another focalcut version again when it
  starts.

### Fixed

- A selection moved or trimmed in Clips kept neither its stars nor its opening / ending role when it was dragged
  past its own length (the role stayed behind on the old range, where no page showed it). It is the same pick
  now: its reels, stars and role move with it. Removing a selection clears the role it held.
- Removing a pick while working on one reel says which reels it stays in.
- Making a pick the opening (or ending) of one reel when another reel opens on a different pick of the same clip
  says that the other reel's is dropped (a clip holds one).
- Ranges written by the review pages are to the hundredth of a second, as the source browser writes them (a ban
  made on one page and on the other no longer end up as two entries).
- A page whose video cannot play (opened as a file, a codec the browser lacks) keeps its playhead: shots open at
  their in-point and marks land where the strip says.
- The pick editor no longer scrolls off the top of the page when stepping through the cut with the arrow keys.
- Thumbnails re-made by a re-plan show at once (files are revalidated instead of cached for an hour); a crop the
  server cannot work out says why instead of "needs focalcut reel serve".

### Docs

- **Docs site** at https://isaacrowntree.com/focalcut/ (mdBook from `docs/`, `book.toml`, rebuilt by
  `.github/workflows/pages.yml` when the docs change): install, cutting a library by subject, recap reels, the
  editor's picks, the editing concepts, hardware, security, contributing and this changelog.
- `docs/picks.md`: the review pages, `picks.json` (with portrait focal points), how picks are kept and condensed,
  review rounds (`reject`, `refill`) and reel.json profiles. The recipe's reel.json table gains `exposure`, `reject`,
  `refill`, `cuts`, `profiles` / `use` and the pick keys.
- `SECURITY.md` with the threat model (ffmpeg on untrusted media, the localhost-only review server, BiRefNet's
  remote code); an issue-template config pointing at private advisories and the docs; package metadata
  (description, keywords, classifiers, documentation, issues and changelog URLs); a README hero image.

### Fixed

- A pick on a clip the moments step found nothing in was dropped without a word; it plays now.
- A cut that sets its own `ending` or `end_with` no longer inherits the project's `close`, which took over its
  last shot unless the cut set `"close": null`.
- A flash or dip to white in the cut is a full white frame under the overlay: its dark edges and MilkDrop burn out
  with the footage (it was a white oval inside a black border).
- The zoom and spin transitions blur with enough copies for the frame's size; five copies showed as ghost outlines.
- The overlay dithers its output, the MilkDrop layer is kept 4:4:4 until it is keyed, and the delivered file has a
  keyframe every second (it was every five) and spatial AQ; blended frames are rounded, not truncated.
- An encode that finds every NVENC session taken (another render running) waits and retries instead of failing.

- Portrait strips follow the subject's face when the body track misses it (`plan.face_track`): on close crowd shots
  the pose boxes lost the person or settled on the one beside them, and the strip showed a shoulder, a back or the
  gap between people. Only the shot's main people's faces count, and only one near the strip already, so a bigger
  face beside a masked or turned subject never takes the shot.

- Effect masks keep what the matte joins to the subject. BiRefNet's edge was clipped to SAM 3's "person" mask, which
  stops at the arm, so a skirt flung out, a fan or a serpentine dancer's wing on its wand was cut out of the glow,
  outline and aura. Matte pieces joined to the subject now stay in, within the crop around it; pieces apart from it
  still stay out.

- Renders carry BT.709 primaries, transfer and matrix in the H.264 stream. Raw-RGB encodes (transitions, effects,
  the visualiser) were untagged, and Android read them as BT.2020 at 4K and oversaturated them.
- People coverage no longer rates footage that is banned anyway (a picked clip's other footage): a portrait plan
  with most clips picked went from over 30 minutes to about 2.

### Changed

- **MilkDrop renders natively** with projectM 4 (`reel/milkdrop/mdrender.cpp`, off-screen EGL, a fixed frame time
  and the song's own samples per frame), about 6x faster than Butterchurn in headless Chromium and sharper: frames
  are read back raw, not as JPEG. `milkdrop/build.sh` builds projectM and the renderer once into
  `~/.cache/focalcut/bin`; presets are the same `.milk` sources Butterchurn's packs were converted from, downloaded on
  first use. The Chromium path (`render.mjs`, `cdp.mjs`, `FOCALCUT_CHROME`) is gone, and with it the need for node.

### Added

- reel.json `plan.<fmt>.pick_clip_share` (default 1.5): how far above an even share one clip's condensed picks may
  go. Raise it when the editor's picks bunch in a few clips (Electric Nation 2026: 90 to 127 of 132 picks kept).
- reel.json `"refill"`: re-cut by keeping every shot of an earlier plan and moving only those on footage banned since.
- reel.json `"exposure"` `"lift"`: night subjects that sit dark get a small midtone lift on luma, black and white held.

- **Cuts** (reel.json `"cuts"`, `focalcut reel run --cut <name>`): other reels from the same footage and analysis,
  e.g. a portrait-only night reel of everything shot vertically plus the band (`"footage": ["@portrait", "band"]`)
  and a day reel of the rest (`["@landscape", "!band"]`). Each cut has its own formats, footage, picks (those of its
  own footage) and setting overrides (`"plan"` merged per format); out as `<name>_<cut>_<fmt>`. Footage matches
  (`plan.fits`) take `@portrait` / `@landscape` for the way a clip was shot.
- MilkDrop overlay `"edge_fill"` (default 0.35: how much of a corner MilkDrop's unkeyed ground may cover, the rest
  keeps the darkened footage) and `"bare"` (default 0.5: how much deeper the fade gets where a preset lights little of
  the border).
- **Style libraries** (`focalcut/fx/looks.py`): glitch, carnival and edm, each a family of looks for a whole film;
  reel.json `"look"` names one (names are unique across libraries). A look can carry its overlay (MilkDrop presets,
  flickers, strength), used as the default under reel.json's own `"overlay"`. New **carnival** library
  (`fx/carnival.py`): fiesta glow, sequins, confetti pop, serpentina and pop-art freeze on the dancer; whip pan, confetti
  cannon, colour flash, serpentina wipe and sun iris transitions; zoom punch, colour strobe, sun flare, mirror ball and
  halftone accents; looks `carnaval`, `tropical` and `disco` with fireworks, flowers, disco and fire MilkDrop presets.
  New **edm** library (`fx/edm.py`): `mainstage` (MilkDrop + white/dark flickers) and `lasershow` (the built-in light
  show). `focalcut fx lookbook --library carnival`.
- **Picks cut to their best part** (reel.json `plan.<fmt>.pick_hold`, beats): each selection longer than that plays
  only its best window, scored on the moment curve (moves, quality, framing in that format, activity, face); a clip
  with many selections is condensed to its best ones, and the picks fill about half the reel (`pick_share`), a long
  one earning a second piece when there is room. The picks then take turns with the rest of the footage (the rest of
  a picked clip included), no act more than ~55% picks, no filler over 4 beats or under 2 outside the climax. The
  pick check lists what was condensed out. `moments.curve` exposes the per-frame moment score.
- **MilkDrop edges fade cleanly:** only MilkDrop's vivid light fills the edges (its dull ground goes to black), and
  where a preset lights little of the border the fade to black comes in deeper and wider, so line-art and sparse
  presets no longer leave a grey haze in the corners. reel.json `overlay.milkdrop_edge` forces it per preset.
- **Opening and ending as picks:** the source browser's pick editor marks one selection as the reel's opening
  (under the logo) and one as its ending (picks.json `"open"` / `"end"` on the clip); the plan uses them over
  reel.json and the pick check reports one that isn't first or last.
- **Heavy picks fill the reel:** every selection plays its full length; an act they overfill passes its last shots
  on to the acts after it while those have room, and only then shortens or drops the planner's shots; an act left
  short (or a long filler in the finale) gets quick cuts from the picks. An act the picks leave empty no longer
  stops the plan.
- **`"keep": true`** re-cuts the clips the editor left alone from the last cut rendered (render saves
  `plan_<fmt>_rendered.json`), so there's no plan file to copy by hand.
- **Glitch looks** (`focalcut/fx/glitch.py`): 7 subject effects on the dancer through the SAM 3 masks (RGB outline,
  datamosh, pixel sort, ghost trails, a dissolve that reassembles, slit-scan, block damage), 8 transitions (blocks,
  datamosh cut, slices, RGB tear, VHS roll, signal lost, sort wipe, stutter) and 9 accents, grouped into four looks.
  reel.json `"look"` makes the planner say every transition reason and subject effect in that look and put its
  accents on the hardest hits. `focalcut fx lookbook <folder>` renders every style and look on the reel's own
  footage as one local page.
- **Framing follows the action:** the crop is on whoever is doing the move (pose activity about the shoulders),
  not whoever stands together; a couple shares the frame with the follow featured (new `role` step: SigLIP 2 reads
  each dancer, woman or man), the follow kept whole when they don't fit; group routines stay whole.
- **Moments that don't play are switched:** the subject walking or standing, a crowd crossing the crop, or someone
  blocking the camera ranks a moment well down, for planned shots and the people-coverage shots alike.
- **Band shots show the band:** the `role` step scores band footage for a player performing, in focus; the plan
  cuts to the band only there, never the crowd, a back or a blur between the real shots.
- **Hardware rendering** (`focalcut/reel/hw.py`, [docs/hardware.md](docs/hardware.md)): on WSL, camera files only
  the CPU could decode (Sony 10-bit 4:2:2 HEVC) decode on the Intel iGPU with Quick Sync through a Windows ffmpeg
  build, frame-exact (`-async_depth 1`) and about 6x less CPU; shots encode with NVENC and render several at once
  (`FOCALCUT_DECODERS`, `FOCALCUT_WIN_FFMPEG`, `FOCALCUT_ENCODER`). A 140-shot reel's plain shots: about 30
  minutes down to about 4.5.
- **Picks keep their length and bans hold:** a pick takes the beats it needs from the weakest plain shots in its
  act; a ban keeps any shot window out of the banned stretch, not just moments peaking inside it. Picks take
  precedence: each plays once, never sped up (slow motion stays; no speed curve for long ones), and no other shot
  reuses its footage.
- **Pick check after every plan** (`plan.pick_report`): the plan step lists any pick that is missing, repeated,
  shorter than marked, sped up or overlapped by another shot, and any ban that did not hold (or that one of the
  editor's own picks overrides). Whole-plan regression tests cover the cases that broke on a real reel.
- **Bulk drop in the source browser:** tick shots in Cut & picks (or drag along the reel lanes in Clips, ✕ on one
  shot) and drop them all at once; drags snap to the playhead.
- **close_only** (reel.json): footage that only works close up keeps just the moments with a big face.
- **DJ-style song hand-overs** (`songs`): the outgoing song rides its tempo to the next one's over 8 bars
  (Rubber Band through ffmpeg, pitch kept), the two play beat on beat for 4 bars with the highs crossfaded and the
  bass swapped on the downbeat, and the songs are level-matched. Replaces the one-bar crossfade and the hard cut
  between songs of different tempos (only gaps over 15% still cut). reel.json `mix` tunes it.
- **One review server for many projects**: `focalcut reel serve <folder> [<folder> ...] [--root <dir>]` serves
  each project under `/p/<slug>/` with its own `api/picks` and `api/crop`, an index of every project's pages at
  `/`, and `POST /api/projects` (or `--root` scanning) to add projects without a restart. One folder still works
  as before. The pages call the API relative to their project, and the review pages now save `picks.json` too.
- **Source browser** (`review/sources.html`, served by `focalcut reel serve`): every export on a zoomable
  timeline; in-out picks, bans and the current cut shot by shot, saved straight to `picks.json`; portrait focal
  points (picks `"portrait"`) that move the derived crop, shown beside it.
- **Overlay flickers**: whole-frame white or dark flickers into each drop and on the hardest hits, or at reel
  times from reel.json overlay `"flicker"`.

- **Show the landing**: every plain dance shot keeps the end of its move and about a beat of the dancer at rest
  after it lands (`velocity.landed`, `plan.lands`; `land_hold`), borrowing a beat from a neighbour when it must.
- **Every export file in the reel**: coverage is per unit (export file), not per camera clip, so all of a clip's
  segments appear; long shots are split to make room. Band inserts and extra shots never take a unit's only shot.
- **Faces from YuNet**: portrait framing and ranking read the `people` step's face scan when there is one;
  ViTPose keypoints mark backs of heads as faces on the social floor.
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
