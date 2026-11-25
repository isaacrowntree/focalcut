# Changelog

All notable changes to focalcut are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A Premiere package to hand on** (`masters.py`, `xml_export.package`; reel.json `"xml_package": true`,
  `"premiere_dir"`, `"masters"`): one folder to upload as it is. `footage/` holds whole-clip editing masters
  (the camera's LUT and finish, vid.stab two-pass where a clip shook -- its correction capped so the zoom that hides
  it stays near 1.1x -- a constant frame rate, H.264 at a set bitrate, the camera's sound kept), `effects/` the baked
  shots, `captions/` the captions as PNGs, `audio/` the song and the ambience bed as the film mixed it, `film/` the
  finished films, and a README. The XML lays plain shots on V1 from the masters at their in-points, effect shots
  baked, captions on V2 with cross dissolves (snapped to a cut they overhang by a few frames), the song on A1 with
  Audio Levels keyframes tracing the film's envelope and ducking, each cheer on A2 from its master with fades, the
  ambience on A3 -- stereo in Premiere's own layout (linked channel pairs, `premiereTrackType="Stereo"`), files at
  their real frame rate, gains capped at +12 dB (the README says what needs more), written indented with Windows
  line endings.
- **Captions** (reel.json `"captions": [{"at", "to", "text"}]`, `cards.caption`, `render.add_captions`): documentary
  lower-thirds faded in and out over the finished film; `"burn_captions": false` keeps the film clean and the
  captions in the Premiere package only.
- **Spatial photos** (`fx/spatial.py`, `focalcut fx spatial`; a `photo` slot): a still made 3D like iOS's spatial
  wallpaper -- the people facing the camera matted (SAM 3, pymatting foreground), the background behind them filled
  by big-LaMa, an orbit about the subject's depth; the photo's own colour kept (its ICC profile honoured). A
  `photo_grid` slot plays several as a staggered grid.
- **An ambience bed** (reel.json `"ambience": {"source", "from", "to", "lufs"}`): a stretch of a camera's own sound
  looped under the whole film, levelled.
- **Delivery copies** (reel.json `"deliver"`, `render.deliver`): e.g. 1080p at 30 fps for a projector, frame rate
  changed by blending neighbours.

- **A song re-arranged to the film** (`music.arrange`, `bar_profile`, `jumps`, `render_path`; reel.json `"arrange":
  {"bars": n, "quiet": [first, last], "intro": n, "outro": n}`): a beam search through the song's own bars -- playing
  on, or cutting at a bar line to a bar that sounds the same -- to the film's length, its quiet break inside the
  window (film bars) and nowhere else, no bar thrice, no chopped stretches; cuts crossfaded and nudged to meet. The
  music step writes `music_arranged.wav`. A drumless intro is kept: the beat grid carries back to the song's start
  when `start` asks for it.
- **The music built to the film** (reel.json `"music_shape"`, `render.shape_envelope` / `shape_song`): a little under
  full at the opening, down and muffled (a gentle low-pass) under the talking scenes, then from the first dunk a
  build to full and bright by the last, each rise lifted by a bar-long swell. The final level is one gain and a peak
  limiter (loudnorm's linear mode squashed a real crescendo).
- **Documentary structure:** `"section_starts"` pins a section (the first dunk) to a time in the song, counted after
  dropped slots; `"event_s"` sets a moment's length in seconds, whatever the tempo; `"montage"` sections play the
  editor's picks only, each its own length in beats; `"last_pick"` closes the film on an editor's pick; `"grow":
  null` lets a film with nothing to fill come out shorter, the card running to the song's end (up to 22 s).
- **Dissolve transition** (`fx/transitions.Dissolve`): both shots keep moving through a two-beat cross-dissolve; the
  cut into a double exposure uses it.

- **The speaker is whoever holds the mic** (new `holding.py`, reel.json `"holding": {"speakers": "microphone"}`):
  SAM 3 finds the prop by name, the holder is the person whose wrist is at it; speaker slots play only stretches with
  it held, and panels and grid cells frame the holder instead of the biggest face.
- **The event plays in the order it happened** (`template.in_order`, `recap_in_order`): the first dunk opens the
  event; no ground shot from it shows a moment after the next dunk still to play; after the last, the look back runs
  in time order.
- **The camera rule is hard and sees what it missed** (`template.Steady.faults`, new `aim.py`). Every pick, dunk,
  stutter and split-screen panel plays clear of camera faults, at every 0.1 s, never relaxed: a swing, the camera
  lowered or dropped at the end of a take, a twist, shake beyond what the steadying crop takes out, frames the tracker
  lost (a whip), the picture going soft against the clip's own sharpness (a focus hunt), and frames of nothing --
  SigLIP 2 reading each frame as of something or of nothing (the ground, feet, the sky, a hand on the lens). Before,
  a lost track read as a still camera, handheld shake went unmeasured, and the camera rule was relaxed when a slot
  ran short. A slot whose footage is used up now widens its search (any category, then all the footage) instead.
- **`"stabilise_always"`**: clip patterns or categories whose every shot renders through the steadying crop.
- **Exposure matching spares the faces** (`exposure.gain_filter` `protect`): a scene brought down a stop (the ISO
  raised for darker faces) is brought down above a luma point only -- skin and shadows stay as shot, the water and sky
  take the full gain, never a tone inversion (`protect_span`). The tracked face window, no longer needed there, had
  lifted the water around each head too: a halo. The window's lift itself now has a luma qualifier (`window.knee`).
- **Per-stretch camera looks** (`render.lut_cube`, reel.json `"lut_segments"`): a clip whose camera changed part way
  (an ISO step, the fade drifting) takes the cube of the stretch its shot's middle falls in -- from color-grade-ai's
  `match_camera.py science`, which matches each camera to a model of the hero camera's colour across the shoot.
- **Integration tests against the truth** (`tests/integration/test_truth.py`): inputs whose right answer is known by
  construction (ffmpeg-synthesised moves, cuts, splits, titles, dust, exposure) or checked by eye on real footage.

- **DJ shots show the decks** (`lead.py`, `validate.py`, `plan.py`). In portrait a DJ shot is never zoomed past
  z=1; when the deck work is the action (hands down on the gear, decks a real part of the frame, nobody hyping,
  the strip not mostly empty) the strip centres on the hands, mixer and near deck and the DJ's head may sit on
  its edge or be partly cut -- the focal point of the moment, not every body part -- and the validator lets that
  head pass on those strips only. Hype moments frame the people. Hands-on-decks moments rank up, arms-up or
  deck-less ones down. Editor focal points untouched; portrait bench unchanged. Tests: `tests/test_dj_framing.py`.
- **A band or DJ clip plays at most twice** in a reel (`validate` `player_max` 2, was 4); the editor's own picks
  of it always stay.
- **Story mode chooses the footage on the music's stops and entries** (`plan.motion_cast`).
- **No shot shorter than the eye can read** (`plan.<fmt>` `"min_shot"`, seconds; default 0: off, so earlier
  edits plan as they did). Fiesta del Sol's Instagram cut at 131 BPM had 36 of 80 shots one beat long (0.46 s):
  quick cuts nobody can see. With `"min_shot": 0.9` every shot plays at least that long, in whole beats
  (`least_beats`: 2 beats at 131 BPM), handled where the short shots come from: the 8-beat phrase patterns with a
  shorter shot are not used (none left -- the quick phrases -- their short shots join a neighbour inside the
  phrase, so it still sums to 8: `regroup`; the climax's slots and the arrival's dancing rhythm likewise); quick cuts
  of the editor's picks are that long each, fewer of them for the same beats; an editor's pick is never fitted,
  settled or trimmed under it, and one whose own in-out can't fill it leaves its slot (named in the plan log) --
  never sped up, never played past its out-point; and when the picks overfill the reel, the weakest go rather than
  play a beat each. Last, after the music has moved the cuts, `plan.hold_min` holds any shot still short: a pick,
  the logo, the opening, a title, a band insert grows from a neighbour where its footage has room; any other short
  shot (a quick cut, a planner's shot, an effect shot) joins a neighbour -- the planner's own shot first, the cuts
  on the beat grid, the reel's length unchanged. Snap-window halving and people splits keep both halves at least
  that long.

- **The picture's movement follows the song's** (`reel/music.py` `movement`, `reel/motion.py`, `reel/sync.py`,
  `plan.stage_holds` / `motion_order` / `follow_motion`; on by default, `plan.<fmt>` `"follow_motion": false` turns
  it off). The music is read for its movement, beat by beat: a **stop** (everything pauses or closes down: the
  level falls away from a driving passage, or the kick drops out), a **close** (a phrase resolving in a calm
  passage, a sustain), a **burst** (the music coming back in), rise, pulse, glide -- and its **texture**, soft and
  flowing to hard and struck (`beat_texture`). The picture is read for its own: a new `motion` step measures each
  clip's directogram (Davis & Agrawala, "Visual Rhythm and Beat", 2018: optical flow by direction, inside the
  people's boxes, the camera's motion taken out) at 25 fps, from which come the speed of the movement, its impact
  envelope (deceleration: their visual beats), where a movement settles or takes off, and how much it punches
  rather than flows. `sync.fit` scores a stretch of footage against the music it would play under; the planner
  then (1) holds a shot through each stop (no cut between the last beat that sounded and the beat the music comes
  back on: shots nearby give a beat each), (2) in picks mode trades picks of the same act and kind between places
  so the ones that rest sit on the stops, the ones that take off on the entries, soft with soft and hard with
  hard, (3) slides in-points -- any part of a trimmed pick inside the editor's own in-out, a planner's own shot up
  to 2 s along its clip -- for a clearly better fit, never onto shakier, less clear or worse-framed footage, and
  (4) lands velocity edits and plain landings on stops as it did on sustains. Cuts stay on the beat grid; nothing
  is sped up; only structure and texture are matched, never every beat (M2Beats, IJCAI 2024). The plan log gains
  a `motion:` line (`plan["motion"]`): the mean fit, the stops and entries met, and the correlations of texture and
  energy, shot by shot. Without the `motion` step the score step's 10 fps subject motion stands in; without
  `beat_texture` (an older music.json) the drive and the kick do.
- **Story-mode reels choose which footage sits on the music's stops and entries** (`plan.motion_cast`, between
  the steadier and `follow_motion`; `"follow_motion": false` turns it off too). A planner's own plain shot on a
  stop, a close or an entry plays another moment of its own unit, a moment of a unit of the same casting the reel
  has no shot of yet (only when its own unit keeps another shot), or trades footage with another of the planner's
  shots of the same casting (same act, role, kind of footage, home act, day or night, crowd for crowd; the same
  category under song sets or in a `climax_from` climax), at most 16 shots away, for a fit at least 0.12 better
  (a trade: between the two, the slot on the music's event gaining half). Each moved shot must pass every rule the
  validator holds (`validate.judge`), stay on passing windows where the old one was, clear of bans, kisses and
  the unit's other shots, no shakier, framing as well in portrait, no B-roll clip over its share, never beside
  its own clip. The editor's picks, effect and velocity shots, and a shot whose transition comes from its
  footage (a whip, a turn, a push in) stay; the cuts stay on the grid. Deterministic: stops first, then by place.
- **`"no_kissing": true`** (reel.json, or a cut's own): no kiss plays in the reel. `people.kiss_spans` times every
  kiss (SigLIP 2 at 3 frames a second, told apart from a close dance hold, a hug and a whisper as well; runs with
  a frame at 0.97 or more, half a second clear either side; the frames' embeddings are cached) into
  `kisses.json`; each is banned footage for the reel, and an editor's pick that plays more than 0.15 s of one is
  left out and named in the plan log. The planner times the kisses itself when the people step hasn't.

- **The editor's rules are checked on every shot the planner chose, and a shot that breaks one is replaced**
  (`reel/validate.py`, the last pass of `plan.build`, in story and picks mode). Each shot is measured over exactly
  the stretch it plays, as the strip shows it: the subject in focus (an absolute floor -- a sharp foreground stand,
  speaker or phone counts for nothing), no back (performers too; nor the nearer of a couple turned away with the
  partner hidden behind them), the subject and a couple's partner in the strip and not cut by the frame, nobody
  halved at the strip's edge, no head cut by the frame's top, a couple close enough to fill the strip, the shake
  limit, the editor's bans, day footage by day, and a B-roll clip at most twice in a reel. A failing shot takes the
  nearest passing stretch of its own clip, else the best passing moment of another unit of its kind (for couples
  the featured ones first: picks `"more"`, then units the editor picked from; then the least used), else it stays
  and the plan log says `rule fail: <rule> at <time>`. Every replacement is logged with the rule it broke. The
  editor's own shots are never touched: what they break is listed, as their choice. `plan.<fmt>.validate`: `false`
  turns it off, a dict overrides a threshold (`validate.RULES`). A clip shot sideways is judged too (its pose
  turned with the picture).

- **The picture follows the music.** The music step now says what the music is doing on each beat
  (`music.features`, read through `music.feel`): `beat_kick` (the low end against the song's loud level),
  `beat_drive`, `beat_section` (intro / groove / break / build / drop / outro), `beat_sustain` and `sustains`
  (the music lands on a beat and holds). `beat_accent` and `beat_energy` could not tell a vocal breakdown from a
  drop: a sung syllable is an accent, and a pad is loud. A `music.json` without the new fields reads as before;
  re-run `--steps music` to get them.
  - **Hits are real.** A hit is an accent on a beat the rhythm section plays. The cut's flash (and the flash an act
    change falls back to), a look's phrase glitch and hit accents, band inserts, a velocity edit's landing beat, the
    overlay's flickers, kick blink, zoom punch, RGB split and drop flash all go by it. Where the music is calm
    nothing flashes, blinks, whips or pushes: an act changes on a dissolve there, and the flickers up to a drop
    play only on the beats of a build that sound (a drop out of a vocal break gets its own flash and no more).
  - **Pace by section.** A calm phrase is cut long (`PHRASES["calm"]`) and a drop quick, whatever the act's pace;
    shots that start in a calm stretch take beats from those that start where the music drives
    (`plan.pace_through`; in picks mode `pick_story`, once the order is settled). `follow_music: false` turns the
    story's pass off. The climax's big effect goes on a shot that starts on the drop, not over the break before it.
  - **Holds on sustains.** No cut on a beat the music holds (`plan.hold_sustains`): it moves onto the hit or the
    next beat that sounds. A velocity edit lands on the sustain's first beat, and a plain shot's landing is slid
    onto it (`velocity.on_sustain`).
  - The plan carries what the music does under the cut (`plan["music"]`: sections and sustains in reel seconds, and
    whether each sustain is held through), and the plan log prints it.
- **Picks mode: B-roll gives before people.** When the picks overrun the reel, B-roll picks give up their beats
  first, and go first when picks must go; among equal stars the least clearly seen picks are trimmed first, so a
  clear shot of people keeps its length; the log counts the trims by kind. In portrait a B-roll pick goes down to
  one beat and then out (named in the log) before a pick of people is trimmed.

- **Portrait strips stay on one person and keep them in frame while they move** (`reel/lead.py`). Every pick's
  9:16 strip is aimed at one subject for the whole shot -- the person the shot opens on unless someone is clearly
  more its subject (nearest, dancing, in focus; a still face beside a dancer no longer takes the strip, and it is
  never handed to whoever turns to the camera later) -- and follows their head and body from the pose track, face
  showing or not (masks, costumes, heads down), a point every 0.1 s. Two people together share the strip; a floor
  move is framed on the torso; a DJ's hands with no head in frame are framed at the fingertips. In portrait a
  trimmed pick plays the part of the pick that frames best (people together, nobody halved, no head cut off by the
  camera). `bench/portrait_eval.py` scores the automatic strip against the editor's own focal points and follow
  clicks with them hidden.
- **Focus editor (Frame portrait)**: everyone tracked in the shot has a ring, the bright one being who the crop is
  on, and the state line says what the planner's crop is on and why. Click a person to follow them through the shot
  whether or not they show a face; `H` keeps the person the crop is on for the whole shot; `Shift`+click puts a
  focus point at the playhead (hands, a prop) and more at other times make the crop glide; drag still pans.
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

- **A faster analysis and render, the same picture** (measured on a 4 min 4K film; `docs/render.md`, findings
  11-15, and `docs/hardware.md`).
  - *Speed curves* (`fx/speed.py`, `render.fx_shot`): a stretch at exactly 1x plays the source's own frames
    (`snap_steps` nudges the ramp before it by half a source frame at most, so it starts on a whole one; before,
    every frame after a ramp was an optical-flow in-between of two real frames). A chain that is one speed curve
    streams in the 4:2:0 the encoder takes (`Speed.stream`, `Yuv`): a few frames held, not the shot as RGB
    (8.6 GB for 7 s of 4K), decode, curve and encode each on a thread, three such shots at once
    (`FOCALCUT_FX_JOBS`) in the plain shots' slots. The in-between arithmetic keeps its pixel grid and builds its
    maps with `cv2.scaleAdd`. 13 hero shots at 4K: about 390 s to 68 s.
  - *Cache keys* (`render.key`, `render_shots`): a shot's key is made of what shapes its pixels and nothing else
    (not the planner's scores of the crop; no peak, role or beat for a plain shot), its file is `shot_<key>.mp4`
    rather than numbered by its place in the plan, and LUTs and the dust map are identified by content
    (`render.file_id`, `dust.version`). A shot added early in a plan, or a re-plan that only re-scores, re-renders
    nothing else. **Every shot renders once more after this change**; `work/shot_NNN_*.mp4` are no longer read.
  - *Score* (`score.py`, `hw.readers`) reads each source's proxy when reel.json has one, on NVDEC (Quick Sync
    for a 4:2:2 camera file without one), frames thinned on the GPU and scaled on the CPU: about 3.5x real time a
    worker, from 1.4x. Every frame is decoded on every decoder (the CPU path no longer skips non-reference
    frames), so **new scores differ from old ones** on 50p camera files: 15-20% more subject motion (the old
    sampling was uneven), sharpness about 10% higher from a proxy. Existing scores are kept as they are; scoring
    a few sources again moved 3 of 18 and 8 of 20 moments by more than 0.3 s on the two tried.
  - *Shake* (`shake.py`) prints each source as it finishes and decodes through `hw.readers` (the same numbers;
    a 4:2:2 camera file 3.4x quicker).
  - *Reference* (`reference.py`): the film's frames in one pass a size (`frames_at`: the same pixels as 270
    seeks), its shots' moves read several at once, unit frames from keyframes alone, six units at once, each
    stamped with its own time (they were up to a sample step early; `reference_frames.npz` `store` 2), and
    `run()` keeps the embeddings it already has. 5.2 min to 2.3. One of 90 shots changed category (a 0.001 tie)
    with the re-timed frames.
- **Calm passages glide (landscape)** (Isaac: "in the calm sections for landscape gentle slow mo"). Where the music is
  calm (music.feel: an intro, a breakdown, a build), every shot of two beats or more with no effect of its own plays
  at 0.8x throughout (fx "glide", optical-flow in-betweens, the face guard keeping faces sharp); picks included,
  never the ending, the close, titles, the band or a reaction. `plan.<fmt>.glide`: true / false (default: landscape
  only). Slow-motion in-betweens find their optical flow at 960 px and scale it up, so a 4K slow-motion shot costs
  what a 1080p one did.
- **Steadying is a crop, not a warp** (`render.steady_crop`, reel.json `"stabilise"`). vidstab followed the dancers on
  SSMC's animation shots and came out 4-5x shakier (single frames lurching a third of the strip). Now the camera's
  motion is measured on the very frames the shot decodes (same decoder: Quick Sync and the CPU land a seek two
  frames apart), at 960 px, smoothed over 0.25 s, and the crop moves with the jitter by at most 1.2% of the frame
  (3% zoom for room) -- and only where moving the shot's own frames by that correction measurably steadies them
  (15% or more; else the shot plays as shot). SSMC animation shot 38: shake 2.39 -> 1.76; shots where it wouldn't
  help are left alone. `"stabilise": "all"` also steadies sources stabilised beforehand. No vidstab pass any more,
  so shaky shots decode on Quick Sync like the rest. A filter graph too long for the Windows command line (a
  long steadied crop path) goes through a file (`-/filter_complex`).
- **Effects go on the editor's picks too** (Isaac: "effects are allowed on picks"). The look's subject effects and
  accents take picks like any shot, and when the picks have taken the climax's slow-motion slots, `drop_effect`
  puts the one big subject effect ("bullet") on the strongest dancing shot of the climax's first four with room for
  it -- a pick as readily as not. Picks are still never sped up. `plan.<fmt>.drop_effect: false` turns it off.
- **The night lift is a tracked window on the faces** (`window.py`), as a colourist would grade it. Each lit night
  shot is read as it plays (rotated and cropped), about ten times a second; YuNet finds the faces, and their own
  level decides the lift: toward `target`, at most `stops` (default 0.6) brighter, nothing for faces already bright,
  lost in black or seen in under a third of the shot (silhouettes, B-roll, the logo opening). A feathered ellipse
  from head to chest follows each face, with a gentler one over the body below it (shoulders to hips, about half
  the lift), and inside them a shadow lift raises the face (the deepest shadows and white
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

- A documentary scene shows only its own footage, and a drone slot only the drone's: a slot with nothing left is
  dropped, never filled from another scene out of its time. A section set to 0 bars is gone (a phrase snap no longer
  grows it back).
- Drone b-roll never goes back in time (`AIR_BEHIND`: at most 2 min before where the day has got to), never turns
  (yaw over `steady.air_turn` rad/s), and away from the event only the editor's picks or what `aim.prefer` asks for.
- A side-by-side dunk's drone panel is the same second exactly, or the dunk plays full frame.
- A grid cell can be framed by hand (`grid_extra`: `[clip, t, x, y]`).
- A split screen's canvas is labelled BT.709 like the footage, so panels no longer shift colour.
- A drone shot can no longer land on the take-off or landing after its stretch is clamped to the clip.
- The reference's opening title is read even when its ground is plain enough to pass for a card.

- **An effect shot could come from a stale cache** (`render.key`): the key used the plan's beat where the shot
  renders with its own, and left out `group`, `add` and the exposure match's `gain_protect`.
- **Speed-curve shots were a code value darker than the shots around them** (4:2:0 to RGB and back through
  swscale, truncating twice); they now stay in 4:2:0 and match a plain cut of the same frames.
- **The same inputs give the same plan.** `cover_people` chose among equally good spots in a set's order, which
  changes with the interpreter's hash seed: two runs of one cut differed in most shots. Spots are now taken in
  order (unit, then time).
- **A look's phrase transition needs a hit.** The glitch looks put a transition on the cut nearest every 4-bar
  phrase whatever the music did there; on SSMC 2026's night cut 13 of those 21 sat on beats softer than the ones
  around them and eight were off the phrase's first beat (27 transitions in three minutes). It now goes on the cut
  by the phrase that the music hits hardest (accent 1.2 or more, `plan.PULSE_HIT`, a look's `pulse_hit`), and
  nowhere when none does.
- **Transitions only where the cut can carry them.** A dissolve joins daytime scenes only (it went into the
  night's crowd shots as "one daytime scene to the next"); no transition but a flash goes across a one-beat shot;
  and the change of act into the climax is a velocity whip only with a ramp or travel to whip between: across an
  editor's picks it is a flash when the music hits there, else the plain cut (it was a bare swipe).
- **Nothing shown twice.** Quick cuts made from picks no longer replay footage the reel shows elsewhere
  (`plan.replays`; Electric Nation's main reel had 4 in landscape, 8 in portrait); `reuse: true` allows them.
- **The picks are no longer condensed out in favour of two unpicked clips.** With `pick_hold` and no `pick_share`,
  the picks' share grows when little footage was left alone (`plan.rest_share`), a clip left alone gives up the
  shots it has above its share to the picks' own in-outs (`plan.thin_clips`), and in a cut with no night act the
  day's picks may play in the climax. SSMC 2026's day cut: 30 picks of 46 in landscape (it was 22), 37 in
  portrait (27); its two unpicked show clips 10 shots of 46 (it was 18 of 47), 8 of 49 in portrait (22 of 53).
- **Cuts land on the beat again.** The planner timed cuts as so many beats of the song's nominal period, and for
  a song whose beat grid was kept as tracked (the tempo drifting, a DJ mix, an intro the tracker wandered through)
  that period is the tracker's hop-quantised median: 1% out, a whole beat after a hundred. On Electric Nation and
  SSMC 2026 half of every reel's cuts were more than a quarter beat off, and none of the SSMC day cut's were within
  two frames, while the overlay's blinks (which read the grid itself) sat on the beat. `plan.clock` now follows
  the grid's own beat times and `plan.beats_in` counts its beats; `music.fit_beats` finds the one steady grid of a
  produced track whose intro the tracker lost (`steady_grid`, from the song's first moment) and smooths the
  tracker's 23 ms steps out of a drifting one. Re-run `--steps plan,render,xml,review` (and `music` first for the
  exact grid).
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
