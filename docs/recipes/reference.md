# Recipe: cut it like last year's film

A client hands over last year's film and asks for "the same again" with this year's footage. `focalcut reel
reference` reads that film and writes a template; reel.json `"mode": "reference"` then cuts this project's
footage on it, shot for shot.

```bash
focalcut reel run <folder> --steps units,score,shake,focus,subject,semantic,moments,faces,windows,music
focalcut reel reference <folder> "<folder>/last year.mp4"     # writes reference.json, looks/*.cube; sets "mode"
focalcut reel run <folder> --steps plan,render,xml,review --formats landscape
```

Run `units` before `reference` (it matches against the units' footage), and the `moments` step after setting
`"immersion"` (below) so the dunks are found.

## What it learns (reference.py)

| | How | Used for |
|---|---|---|
| cuts and lengths | ffmpeg scene score; a flash frame joins its shot | every slot's place on the music |
| scale | SigLIP 2 zero-shot: aerial / wide / medium / close | the look groups (aerial vs ground) |
| split screens | seams that run the whole height or width (guillotine panels) | `split` slots: 2-up, 3-up, stacked |
| stutter | three or more shots under 0.35 s in a row | one slot of jump-cuts |
| title | sharp strokes appearing on smooth ground in the first shot, and their fade | the title's box and fade |
| end card | a plain ground with something small on it (detail and ink box), held | the end card's logo box |
| music end | the last silence after which sound returns | shots after it (a tag) left out unless `"tag"` |
| category | the shot's SigLIP 2 embedding against frames of every unit; a Viterbi path that keeps to the order the categories happened (wall clock) | which footage fills each slot |
| look | Lab statistics of its ground and aerial shots | `looks/<camera>.cube` |

## How it cuts (template.py)

- The cuts land on the beat nearest the reference's own (with the same song, where its editor put them).
- Each slot takes the moment that looks most like the reference's shot, ranks well, and sits at the time of day
  its place in the reference's run of that category stands for.
- `reference_min` guarantees categories slots (`{"arrival": 5}`), taken where the order of the day allows.
- The shoot's event (`events.py`, reel.json `"events"`: e.g. a baptism's dunk, from a pose dip and the cheer): every one once, in order, on the reference's long shots in the water, with a speed
  ramp (`reference_ramp`, default `hero`: fast in, slowing to the rise, held, back to 1x).
- Split screens in the water show one dunk from every camera that saw it (the drone at the same wall-clock moment),
  else more dunks.
- Drone B-roll (`aerial` patterns, default `DJI_*`) prefers the moves (orbits, dives) and, near a dunk, the drone
  over that dunk; a held horizon plays as a time-lapse (`drone_speed`: `[held, moving]`, default `[4, 1.25]`).
- `reference_last`: the category of the last shot before the end card (the group photo, arms up).
- The end card: the logo where the reference had its logo, over a calm aerial blurred into a ground, fading out.
- Nat sound (`"nat"`, default on): the crowd's cheer from the camera file under each dunk's rise, the song ducked.

## Judged by the move or by the subject

Footage is chosen either for what the camera does (a drone's spin over the action, a dive, a pull-out reveal, a
held horizon for a time-lapse: `moves.py` labels each stretch held / spin / push / pull / pan / tilt / fly /
drift / wobble) or for what the people in it are doing (events, faces, the moment's rank). Aerial clips go by
their move and everything else -- handheld, gimbal, tripod -- by its subject, unless reel.json `"shot_by"` says
otherwise: `{"C23*": "move", "drone-dawn": "subject"}`. Footage judged by its move matches the move the
reference's shot made, prefers `"moves_for": {"<category>": ["spin", "push"]}`, and plays a held stretch sped
up (`drone_speed`). The `moves` step measures only `"moves_in"` (clip patterns or categories) when set.

## Every pick holds up start to finish (template.Steady)

- **Camera (a hard rule, never relaxed):** every 0.1 s of a pick is clear of the clip's camera faults
  (`Steady.faults`, each padded 0.3 s): on a tripod (`"tripod"`: categories or clip patterns) any pan, tilt or zoom
  while the operator finds framing; handheld or gimbal a swing, the camera lowered or raised (the end of a take,
  the camera dropped to the ground), a twist, shake beyond what the steadying crop takes out; for every camera the
  tracker losing the picture (a smear), the picture going soft against the clip's own sharpness (a focus hunt), and
  frames of nothing (`aim.py`: SigLIP 2 reads each frame as of something -- people, the scene -- or of nothing: the
  ground, feet, the sky, a hand on the lens, a bag; reel.json `"aim"` sets the prompts and the limit). Motion is
  tracked on the background only (moves.py masks the people), so a splash or a hug isn't read as the camera moving.
  Limits: `"steady": {"still": 0.09, "zoom": 0.09, "swing": 0.3, "drop": 0.15, "twist": 0.12, "jitter": 0.012,
  "jitter_steadied": 0.025, "blur": 0.35, "pad": 0.3}`. The rule holds for every shot, dunk, stutter and split-screen
  panel, and again where a shot lands after it is clamped to its clip and moved off a ban; a slot whose footage is
  used up looks wider (any category, then all the footage) rather than relax it. Only the editor's own picks pass.
- **Subject:** the people stay centred (`centre`) and inside the frame for the whole pick, and whoever it opens on is
  still there at its end.
- **Aerial:** the drone is flying -- each frame scored "high above the beach and sea" against "low over the sand"
  (SigLIP 2, stored as reference_frames "high"; `air_high`), clear of each flight's first and last `air_trim` s.
- A pick that fails slides up to 2 s to find a stretch that passes; a slot with none takes the best with the soft
  rules (the subject's) relaxed -- the camera and aerial rules never are -- and is listed in the plan's `relaxed`.
- `"stabilise_always": ["C2*", "C0*"]`: every shot of those cameras renders through the steadying crop
  (render.steady_crop), kept only where it measurably steadies; their shake limit is then `jitter_steadied`.

## The speaker is whoever holds the mic (holding.py)

`"holding": {"speakers": "microphone"}`: the `holding` step finds the mic in each speaker clip every half second (SAM
3's image detector, by name) and its holder -- the person whose wrist is at it (a face behind the mic in the picture
is not proof; a nose counts only when no hand is near). A slot of that category plays only a stretch where the mic is
held (80% of it: a hard rule), its panels and grid cells are framed on the holder (not the biggest face -- the pastor
standing beside them), and a clip where nobody ever holds it is no speaker. Any category, any prop: `{"band":
"guitar"}`.

## Everyone, side by side

`"every_unit": {"speakers": {"columns": 3, "grid": true}}`: every clip of a category on screen -- its slots become
side-by-side split screens of up to 3, as many panels as clips, in the order they were shot; with `grid`, the
category's longest slot shows all of them at once (a Brady Bunch grid, cells sized near square, a spare cell for
the logo, filling in cell by cell). Split screens of one moment from several cameras (the ground and the drone on
one dunk) are always side by side.

Split screens sit in a border (`"split_style": {"gutter": 0.0062, "colour": "0xF2EFEA", "enter": 0.45,
"stagger": 0.12}`: 24 px at 4K) and arrive with an entrance, rotating `"split_styles": ["slide", "stagger",
"push"]` -- slide: each panel from its own side; stagger: up from below one after another; push: the row in from the
right; cascade: a grid filling in.

## Dust on the lens (dust.py)

`"dust": {"clips": ["baptisms"]}`: specks found by stacking the clips with people masked out (dark in the same pixel
in most clips: on the lens), each taken out by dividing by its transmission -- measured in each shot's own frames,
so bright water and dark water alike -- in the camera's own 10-bit Y'CbCr (lut2), every other pixel untouched. The
dust step writes review/dust/index.html: before, after and the specks circled, the worst at 2x.

## Exposure across a scene

`"exposure": {"match": {"clips": ["baptisms"]}}`: one scene shot at different exposures (the ISO raised a stop to see
darker faces) brought to one level -- each shot's background scaled to the group's. The gain spares the faces:
`"protect": [0.25, 0.6]` (legal-range luma) -- the gained picture is laid over the shot as shot, keyed by its own
luma, so skin and shadows below 0.25 stay exactly as shot (the ISO was raised for them) and the water and sky above
0.6 take the full gain, widened where a strong gain would otherwise invert tones (`exposure.protect_span`). No window,
so no halo; `"protect": false` scales the whole picture and lifts the faces back through the tracked window.
`"lift": {"day": true}` lifts dark faces in a bright frame through that window, with a luma qualifier
(`window.knee`): only levels up to about the face's own take the lift, so bright water or sky around a head inside
the window stays as shot. `"highlights"` pulls bright daylight frames down from their top end, faces in the
midtones untouched.

## Matching the cameras (the look)

Every camera is matched to the hero camera first (`look_hero`, a camera name or clip pattern, default the camera
with the most footage), on frames of the same kind of scene, then the whole film moves `look_strength` (default 0.5;
0 leaves the hero camera as shot, with no LUT) of the way to the reference's colour. Each match is checked on frames
it wasn't fitted on (Lab distance to the hero, printed before -> after). `"match_per_clip": ["<camera>"]` fits a
camera whose clips vary too much for one LUT clip by clip; `"match_max"` keeps out a clip whose match stays poor.
`"look_method": "regions"` matches by the same things in both cameras instead of whole-frame statistics (SAM 3
regions, frames paired by `clock`; `regions.py`, which runs color-grade-ai's `match_camera.py` -- `"match_camera"`
is its path, `"look_flat": ["drone"]` the cameras shooting a log profile, `"look_classes"` the regions a camera is
matched on). It leaves the hero as shot and writes a before/after page to `review/colour/`.
`"cameras"` names each camera by clip pattern (a second shooter's card in `"source_dirs"`), `"camera_max"` and
`"camera_penalty"` keep a weaker camera to its best few shots. Cameras are told apart by clip-name prefix. `clock` sets each camera's offset so the
categories, the drone sync and the order of the day line up.

## Keys

`mode`, `reference`, `events`, `reference_event`, `shot_by`, `moves_for`, `moves_in`, `moves`, `reference_min`, `reference_last`, `reference_ramp`, `reference_dunk_categories`, `aerial`,
`drone_speed`, `immersion`, `clock`, `look_hero`, `look_strength`, `lut`, `title`, `endcard`, `tag`, `nat`,
`proxies_render`, `tripod`, `steady`, `aim`, `holding`, `stabilise_always`, `every_unit`, `split_style`, `split_styles`, `angles_layout`, `dust`,
`cameras`, `source_dirs`, `camera_max`, `camera_penalty`, `match_per_clip`, `match_max`, `shake_max`, `shake_in`,
`stabilise` (a list of clip patterns), `title_on`, and `exposure.lift.day` (lift faces that sit dark in a bright frame).
