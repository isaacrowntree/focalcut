# Recipe: recap reels

`focalcut reel` turns a curated footage folder into beat-synced highlight reels — one landscape, one portrait —
and a Premiere timeline for each. It was built on a 2.6-hour salsa convention shoot (60 camera files, 97 curated
clips) and produces two 3:00 reels from it unattended.

```bash
pip install "focalcut[reel] @ git+https://github.com/isaacrowntree/focalcut"
focalcut reel init "/footage/Show 2026" --song track.mp3 \
    --stinger-landscape end_16x9.mov --stinger-portrait end_9x16.mov
focalcut reel run "/footage/Show 2026"
```

## What it expects

```
Show 2026/
  C2020.MP4 …                       camera files (the reel cuts from these, at full quality)
  export/
    segments/C2020-03_couple.mp4    focal-subject segments (from `focalcut library` + `export`)
    broll/C2006_crowd.mp4           B-roll: used as transitions between dance runs
    matinee/C1990.mp4               any other folder: whole clips, treated as "shows"
  focalcut/library.json             segment and B-roll time ranges
```

The export folder is your curation: anything not in it is left out. The exported files themselves are not
used — each one is mapped back to its source file and time range, so logo intros and stingers baked into the
exports never reach the reel.

Everything the recipe writes goes to `focalcut/reel/`: `reel.json` (settings), `units.json`, `scores/`,
`moments.json`, `thumbs/`, `music.json`, `plan_<fmt>.json`, `work/` and `out/`.

### reel.json

`focalcut reel init` writes the basics; add the rest by hand when a shoot needs it.

| Key | Meaning |
|---|---|
| `song`, `start`, `length` | the track; `"auto"` (first bar after the intro) or seconds; total length incl. stinger (default 180) |
| `songs` | several tracks instead of `song`, one after another (e.g. one per DJ of a festival lineup): `[{"song": path, "bars": 16, "pre": 8, "acts": {"teaser": 2, "arrival": null}}, …]`. Each song plays its `bars` around its strongest drop (`pre` bars of build first); the last may have `"bars": null` to fill `length`. Each hand-over is mixed like a DJ set: the outgoing song rides its tempo to the next one's over 8 bars (time-stretched, pitch kept), the two play beat on beat for 4 bars with the highs crossfaded and the bass swapped on the downbeat, and the songs are level-matched; tempos more than 15% apart cut on the downbeat. `mix` changes this. `acts` puts the acts on the songs (bars; `null` = the rest of the song). The step writes `music_mix.wav` |
| `mix` | how songs hand over (with `songs`): `{"bars": 4, "ramp": 8, "max_ramp": 0.15, "split": 150}`: overlap bars, tempo-ramp bars, the largest tempo gap to ramp across, and the bass split in Hz; `false`: cut on the downbeat |
| `start_hint` | pick the bar nearest this time instead of detecting the intro |
| `stingers` | `{"landscape": path, "portrait": path}` — closes each reel; the music fades under its first 1.2 s |
| `roles` | `{"<category>": "dance" \| "show" \| "transition"}` (defaults: `segments` dance, `broll` transition) |
| `sources` | `{"C2037": "C2037_stabilized.mov"}` — cut a clip from a replacement file |
| `rotate` | `{"C2035": "transpose=2"}` — an ffmpeg filter for a clip shot sideways |
| `spans` | `{"C2036": [0, 7]}` — the usable part of a clip (`[6.5, null]`: from 6.5 s to the end) |
| `proxies` | `{"C2037_stabilized.mov": ["export/animations/C2037.mp4", []]}` — decode a heavy master (e.g. DNxHD) from a smaller file that starts on the same frame; `[[53.28, -0.28]]` shifts times after a dropped join |
| `picks` | the editor's calls, by export file name, times in that file's own timeline: `{"C2025": {"ban": ["0:00-0:20"], "prefer": ["0:41-0:46"], "add": ["1:12.5"], "more": 2}, "C1984-05_couple": {"ban": ["all"]}}` (`add`: extra shots the reel must include; `more`: more shots from that clip). Merged over `focalcut/reel/picks.json`, which is the easier place to keep them |
| `name` | output file prefix (default: the folder name) |
| `hips` | `"movement"`: the semantic step scores hips on how someone dances (hips moving to the music, the whole body in it), with no view-from-behind or body close-up prompts — for an audience filmed at a festival, not performers. `false` leaves it out (scored 0) |
| `overlay` | `true` (or `{"strength": {act: 0..1}, "opacity": 0..1, "presets": [[[effects], palette], ...], "beat": [...]}`): a MilkDrop-style light show over the cut, driven by the mix. A new preset every 8 bars, each two or three effects in one palette: geometric (`tunnel`, `mandala`, `kaleido`, `lasers`, `grid`), particles (`starfield`, `embers`, `confetti`, `bokeh`, `sparks`), light (`plasma`, `leaks`, `flare`, `flame`); palettes `neon`, `synthwave`, `vaporwave`, `acid`, `halloween`, `ice`, `fire`, `rainbow`. Beat layer: a blink and zoom punch every beat, RGB split on bars, and on drops a white flash, kaleidoscope flood, duotone recolour and confetti and spark bursts. Calmer presets and less strength in the day acts; eased back under the logo; light goes into the dark parts of the frame. `"engine": "milkdrop"` swaps the hand-built presets for real MilkDrop ones (projectM 4 on the GPU, driven frame by frame from the mix; `milkdrop` / `milkdrop_calm` name the presets by their Butterchurn pack names), keyed by brightness and saturation so only their light shows; the renderer builds itself once (`focalcut/reel/milkdrop/build.sh`: cmake, g++, EGL / GL headers; `FOCALCUT_MDRENDER` points at another build). Every effect fades out towards the centre of the frame, where the subject is (`"clear"`: the clear radius, default 0.3 of the half-frame). Not in the Premiere XML |
| `look` | a look for the whole film from a style library (fx/looks.py): glitch `clean`, `tape`, `mosh`, `neon`; carnival `carnaval`, `tropical`, `disco`; edm `mainstage`, `lasershow` (see `focalcut fx lookbook --library glitch\|carnival`): every transition reason and subject effect said in its styles, its accents on the hardest hits, and its MilkDrop presets and flickers as the default under `overlay` |
| `logo` | the event's logo (a PNG with transparency) — laid big and centred over the opening shots |
| `plan` | per-format overrides of the planner: `acts` (`[["teaser", 0, 2], ["arrival", 2, 12], …]` in bars — put them on your song's section changes), `split` (the default act proportions), `opening` (B-roll tags the reel opens on, best first), `day` (B-roll tag keywords for the arrival act), `day_shows` / `night_shows` (show categories danced by day / at night), `night_pace` (`"medium"` or `"fast"`), `heroes`, `grammar` / `caps` / `flash_at` (each act's transitions, how many of each, how hard a beat must hit for a flash), `members` / `member_clips` (B-roll tags and clips of individual band players), `band_in` (player inserts per act), `ending` (`["C2023", 126.6, 133.4, 0.45]`: the last shot, export file and its own seconds, optional crop centre), `shake` (`[0.0012, 0.006]`: stabilise above the first, avoid above the second), `late` (categories or B-roll tags kept for the climax and finale, e.g. the headliner's set), `land_hold` (beats each dance shot stays on the dancer at rest after a move lands; default 1) |

## The steps

`focalcut reel run <folder> --steps …` runs any subset; each step reads the previous step's files.

1. **units** — one unit per exported clip: source file, time range, role, orientation.
2. **score** — every source at 640 px: optical flow at 10 fps (camera motion vs. subject motion), people
   (RT-DETR) and 17-point poses (ViTPose) at 5 fps, sharpness and exposure of the main subject.
3. **shake** — camera shake per unit, from its export file (GPU-decoded H.264) or the source: corners tracked frame
   to frame, a RANSAC similarity fit for the camera's own motion, and what's left after a ~0.1 s smooth as an RMS over
   1 s. A pan or a follow drops out; handheld wobble doesn't. The planner divides it by the crop's width (a 9:16 strip
   shows the shake three times over): too shaky and another part of the clip is used; mildly shaky and the shot is
   stabilised at render (vidstab).
4. **focus** — per-person sharpness on the camera's keyframes at 1280 px: the fallback subject picker.
5. **subject** — who each shot is about, on keyframes (~1 a second) inside the units: SAM 3 masks every person,
   [BiRefNet](https://arxiv.org/abs/2401.03407) says what stands out as the foreground subject, and Depth
   Anything V2 who is nearest the camera. The subject is the salient, near, large person plus anyone nearly as
   strong (the couple, or a line on stage), with whether their mask touches the frame edge. Focus alone gets
   this wrong on stage footage: dancers in motion blur read as soft, still bystanders as sharp.
6. **semantic** — SigLIP 2 zero-shot on the subject's crop: hips / butt working (shakes, rolls, the view from
   behind that animations and shines are about) and expressive faces, against wide, empty or blurry shots.
7. **moments** — the best non-overlapping moments in each unit, counting only the subject's moves:
   - *spin*: the face is seen, then the back of the head, then the face again, within ~1.5 s
   - *dip*: the torso tilts far from vertical (also catches cartwheels and floor work)
   - *trick*: a confidently seen ankle above the hips (kicks), or one dancer's hips above the other's shoulders (lifts)
   - *shake*: the semantic hips score, backed by the hips wobbling under a steady torso
   - *turn pattern*: several turns inside ~4 s with arms passing overhead
   - *close-up*: a face-forward person whose shoulders span much of the frame
   - *travelling shot*: the camera moving steadily while the subject stays framed (not marked down as shake)
   - *energy* and *style* (arms overhead) fill the gaps; B-roll ranks on steadiness and image quality

   A moment is marked down unless the subject is dancing for the whole stretch a shot plays (about −0.8 to
   +1.2 s around its peak) and stays in frame throughout. A close-up may hold still, because the face carries it.
   Someone waiting or walking no longer outranks real dancing. (A VideoMAE Kinetics "dancing" classifier was
   tried and rejected: it scored a man waiting at 0.76 and real dancing at 0.28.)

   Each moment is also scored per format on how well it frames after reframing (centred, a good size, not cut
   off — a head at the frame's top edge scores near zero), and the first quarter of each unit is marked down:
   routines rarely show their best in the walk-on.
8. **faces** — a full-resolution frame per moment, face crops from the pose keypoints, and a SigLIP 2 zero-shot
   expression score (joy, laughter, intensity vs. neutral or turned away). Cached, so re-planning is instant.
9. **people** — everyone in the footage, for events where people asked to be filmed and should all be in it. Each
   unit's export is sampled twice a second through OpenCV's YuNet face detector and SFace recogniser (built into
   cv2; MIT / Apache 2.0 models). Faces that move smoothly from one sample to the next are one person whatever the
   lighting does; tracks that look alike join, inside a unit and then across units. A person counts once seen
   clearly twice (a confident detection of a usable size). Kisses are found by SigLIP 2 on a frame a second. The
   plan then makes sure every kiss and as many people as it can are in a shot (`cover_people`): the moment showing
   the most people still missing goes in first, in place of the weakest plain shot in its act, or by splitting a
   long shot when nothing can be replaced -- never a shot that is someone's only appearance, never a unit dropped.
   `review/people.jpg` shows one face per person, numbered, to check nobody was missed.
10. **music** — librosa beat tracking. Salsa tracks at half-time, so one detected bar is an 8-count, which is the
   cut grid. The window starts on a downbeat and ends on a phrase boundary that keeps the reel inside `length`.
   Each beat's accent (how hard it hits against its neighbours) marks horn stabs and breaks.
11. **plan** — a story in acts on the song's bars:

   | Act | What | Into it |
   |---|---|---|
   | teaser | the festival: band / crowd B-roll (tags in `opening`) with the `logo` held centred across the cut | — |
   | arrival | daytime B-roll (tags matching `day`), each held long, woven with the daytime shows (`day_shows`, e.g. demos) | dip to white |
   | shows | the stage units, their routines shuffled through each other; a few moves held in velocity edits, one spotlight | zoom |
   | night | the night's dancing (`night_shows`, e.g. animations, and the social floor) broken up by band / crowd / DJ B-roll; one neon glow (portrait) | light leak |
   | climax | phrases of short cuts landing on slowed-down moves, crowd reactions, the best move in bullet time with the reel's one fire aura | velocity whip |
   | finale | a face, a slow-motion move, the packed floor, a hair flip in deep slow motion, then the `ending` shot (or one last slow move) into the stinger | dip to white |

   Inside an act a cut gets a transition only when it has a reason, from that act's own vocabulary and within its
   caps (`grammar.py`): a spin cut between two turns, a swipe when the same travel carries across, a zoom pushing
   in to a face, a flash on a big hit, a glitch at the DJ. Everything else, and every crowd reaction or band insert,
   is a hard cut on the beat — about 15 transitions in a 3-minute reel. Band players cut in on the hardest hits
   (`members`, `band_in`). Every shot gets a crop that keeps its
   subject centred: one zoom per shot, the centre gliding with the dancers like a camera operator. Between the
   subject step's keyframes (~1 s apart) the crop follows the subject's own pose boxes at 5 fps, and a portrait
   strip cut from landscape footage (a third of the frame wide) follows twice as fast.
12. **render** — 1080p 50 fps (Instagram's ceiling), effects and transitions applied, music mixed to −14 LUFS.
13. **xml** — an FCP7 XML per reel for Premiere: plain shots link the original camera files (scaled to fit, with
   handles); shots with effects, transitions, reframing or rotation use their rendered files (copied to
   `out/<fmt>_media/`), with the untouched source on a disabled track above. The song sits on A1 from the
   reel's start point, the stinger's sound on A2. Animated position keyframes are avoided — they don't survive
   FCP XML import reliably.
14. **review** — `review/<fmt>.html`: a frame per shot, from the rendered reel or else its source. Each shot shows
   its export file, the time range it uses in that file, and a copyable picks line. Paste the line into
   `focalcut/reel/picks.json` and leave it as `ban`, or change it to `prefer` with a better range. Then re-run
   `--steps plan,render,xml,review`.

   - **ban:** removes moments in that range. `"all"` removes the whole clip.
   - **Scrub source** on any shot opens the export it came from. The timeline shows that shot, the reel's other
     shots from the same video and every moment the planner found. From there you can:
     - mark extra shots anywhere (`A`)
     - swap the shot to the playhead (`S`)
     - ban it
     - ask for more from that video

     The page builds the picks.json for you (Copy or Download) and remembers unsaved changes in the browser.
   - **add:** extra shots at those times; each replaces the weakest plain shot in its act.
   - **more:** a number; the clip's fair-share discount is divided by 1 + more, and its moments rank up.
   - **prefer:** gives moments in that range +1.0. If nothing was detected there, one is made at the middle of the
     range, with its kind and framing taken from the nearest detected moment.

## Serving the review pages

```bash
focalcut reel serve "/footage/Show 2026"                          # one project: http://localhost:8765/
focalcut reel serve "/footage/Show 2026" "/footage/Fest 2026"     # several on one port
focalcut reel serve --root /footage                               # every /footage/*/focalcut/reel/reel.json
```

The server makes the pages save `focalcut/reel/picks.json` as the editor works (the previous file is kept as
`picks.json.bak`) and answers the source browser's portrait-crop questions. Each project is served under
`/p/<slug>/`, the slug from its folder name (`"SSMC 2026"` → `/p/ssmc-2026/`), and `/` lists every project's
sources and review pages. Add a project to a running server with
`curl -X POST -H 'Content-Type: application/json' -d '{"folder": "/footage/New 2026"}' localhost:8765/api/projects`,
or let `--root` find it. With a single folder and no `--root`, the project is also served at the top level as
before. Every page has a menu: the project's sources and reel pages, and All projects (the index at `/`).

It only answers requests addressed to localhost, and only serves files inside each project's own folder. A page
opened as a plain file still works: changes stay in the browser, with Copy and Download for `picks.json`.

   Times are in the export file's own timeline. Units record where each export starts in its source
   (`export_t0`), so the picks land in the right place.

### How it cuts

Each of these is also a standalone concept you can reuse for other clips; the
[focalcut skill](../../skills/focalcut/SKILL.md) lists them with the module behind each.

These are the editing rules the planner follows. They come from the editor the recipe was built with, checked
against how music videos are cut:

- **Every clip, several scenes from each.** Each extra shot from a clip has to beat a shot from another clip by a
  margin (`SHARE`), a unit's shots sit at least `APART` (6 s) apart, and a final pass makes sure every clip in the
  export appears.
- **Card shuffle.** An act's shots are dealt so a clip's shots spread through the act and interleave with others
  of the same style: never one clip twice in a row, never two B-roll shots side by side. Static B-roll is held as
  one long shot instead of several similar ones.
- **Cut on the action.** A turn cuts to a turn, and travel keeps its direction across the cut. Direction comes
  from optical flow inside the subject's box minus the background's (`flow.py`, cached in `flow_cache.json`), so a
  camera panning with a dancer still reads as the dancer moving. A movement carries two or three cuts, then
  changes.
- **Rhythm in phrases, not one length.** Shot lengths come in 8-beat phrases (one salsa 8-count) drawn from
  varied patterns. Quick patterns go where the music is busier, a long hold falls every fourth quieter phrase,
  and the night builds into the climax. Moves that need time (turns, travel, patterns) take the long slots, and
  shakes and hits take the quick ones. Effects go to shots whose slot suits them.
- **Hits on the beat.** Each move's peak lands on a beat inside its shot (on the cut itself for a one-beat hit),
  and every cut comes 2 frames (`ANTICIPATE`, 40 ms) ahead of its beat.
- **The opening is the festival:** band and crowd footage with the event's logo front and centre.
- **Velocity edits, not slow-motion blobs.** A speed ramp runs fast through the move (never above 1.5x on dancing),
  eases down to arrive slow exactly as the dancer lands, holds a beat so hair and eyes play out (two in bullet time,
  more on a close-up hair flip), and eases back up. The landing comes from the subject's own motion (a burst, then a
  rest: `velocity.settle`) and goes on the strongest beat that leaves room for the hold.
- **Transitions with a reason.** Mostly hard cuts on the beat. Each act has its own short vocabulary and caps
  (`grammar.py`): a spin only turn into turn, a swipe only when the same travel carries across, a zoom only to push in
  to a face, a flash only on a big hit, a glitch only at the DJ. Never two in a row; reactions and band inserts cut
  straight in. One big subject effect, on the drop.
- **Portrait framing.** A 9:16 strip from a 16:9 frame follows one dancer of a couple dancing apart (the one showing
  their face). Shots rank on how the crop holds the subject over the shot (present, wholly inside, a face to the
  camera); on the social floor a back ranks well down.
- **Shake.** Shots too shaky for their crop give way to a steadier part of the same clip; mildly shaky ones are
  stabilised.
- **The band.** Individual players cut in through the reel on the music's hardest hits.
- **The ending.** `ending` names the reel's last shot (e.g. the whole room's hands in the middle); its stretch is kept
  for it and it runs out exactly at its end time.

Measured against music videos, the first rhythm planner was too even: 33 of 42 show shots were exactly 2 beats,
with runs of up to 26 equal lengths, and no move peak landed on a beat. Now no run of equal lengths is longer than
5, every peak is on a beat, and every 8-count starts with a cut.

The research behind these rules:

- [Is there a "language of music-video clips"?](https://arxiv.org/abs/2108.00970) (548 official music videos,
  plus interviews with music-video editors and directors).
  - In dance videos the most common shot lasts about one beat (64.5% of clips), with strong bar-level
    co-occurrence.
  - Editors place cuts with anticipation, just before the downbeat.
  - Editing accelerates near climaxes, and bridges are slower.
- [Larry Jordan on cutting montages to music](https://nofilmschool.com/2014/09/larry-jordan-teaches-us-how-create-video-montage-set-music).
  Clips of one length make the rhythm predictable; cut to secondary beats as well.
- Cutting-rhythm practice: alternate cuts every 2 bars, every 4 bars, and long holds, rather than a single
  rhythm.

**Cuts are beat-exact.** Each cut sits on the frame nearest its beat counted from the reel's start. Rounding
each shot to whole frames on its own drifts the cuts by most of a second over three minutes.

## Effect modules

`focalcut fx list` shows them; each works on any clip (`focalcut fx apply clip.mp4 -e hero,aura --peak 1.2`),
and `focalcut fx gallery clip.mp4` renders them all side by side.

| Kind | Modules |
|---|---|
| Speed curves | `montage`, `hero`, `bullet` (velocity: fast → ease down → hold on the landing → ease up; `cap` param, default 1.5x), `jumpcut`, `flashin`, `flashout` — slow stretches get optical-flow in-betweens, fast ones motion blur |
| Subject (SAM 3) | `spotlight`, `freeze`, `neon`, `particles`, `velocity`, `echo`, `strobe`, `orbs`, `aura` |
| Beat / camera | `punchin`, `shake`, `glitch` |
| Transitions | `flash`, `fade`, `dipwhite`, `fadein`, `zoom`, `swipe` (follows the motion's direction), `spin`, `lightleak`, `glitch` |

Subject effects track every "person" with SAM 3 video and keep the subject by the same rule as the subject step
(salient, near, large): the couple on a social floor, or the whole line on stage. SAM 3's masks are coarse once
scaled up to 4K, so the edge comes from BiRefNet's matte, run on a crop around the subject and kept inside SAM's
slightly grown mask. Frames between the tracked ones are blended rather than held, and a little memory stops the
edge shimmering.

## Hardware notes

Full notes and measurements: [docs/hardware.md](../hardware.md).

- **Decoding is the bottleneck.** NVDEC before Blackwell can't decode Sony 4K 10-bit 4:2:2 HEVC. On WSL with an
  Intel iGPU, the render decodes it with Quick Sync through a Windows ffmpeg build (about 6x less CPU) and encodes
  with NVENC, several shots at once (`FOCALCUT_DECODERS`); otherwise it decodes on the CPU. Scoring skips
  non-reference frames and deblocking, which is about 20% faster and fine for analysis.
- **Two scoring workers on a 12 GB GPU.** More overflows VRAM; on WSL that pages to system RAM and everything
  slows to a crawl. Each worker also caps its own threads so they don't starve the decoders.
- **Cap SAM 3 in crowds.** Tracking every person in a packed room floods the GPU; the masks module tracks at most
  6 confident people at 960 px and scales the masks back up.
- **The subject step** runs SAM 3 + BiRefNet + Depth Anything on ~1 keyframe a second, about 0.4 s each on a
  12 GB card: roughly an hour for a 2.6-hour shoot. Point `proxies` at H.264 exports of DNxHD / ProRes masters to
  keep decoding off the critical path.
- Set `FOCALCUT_DISABLE_CUDNN=1` if your torch build has broken cuDNN engines.
