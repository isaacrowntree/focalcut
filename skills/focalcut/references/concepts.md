# focalcut editing concepts

These are the rules the recipe's editor cuts dance recaps by, checked against how music videos and aftermovies are made. Each
section gives the rule, the reason, how focalcut measures or enforces it (module and function under
`focalcut/` in the repo), the numbers, and the reel.json knob where there is one. Use them together for a full
reel, or pick out the ones a new clip needs.

---

## 1. Opening: the festival first

**Rule:** open on B-roll of the band or the packed room, with the event's logo big and centred over it. Never open
on a dance shot.
**Why:** the first second says where we are. Dancing makes sense once the place is set.
**Code:**
- `plan.story` teaser act: two different clips ranked by `opening` tags (`band-wide`, `band-crowd`, `band`, `crowd`).
- The logo goes in on the first shot and holds across the cut (`render.cut_shot`, `logo`).
- With no festival B-roll, the opening falls back to the two best moves in slow motion.

## 2. A story in acts

**Rule:** tell the day.

| Act | Content | Effects |
|---|---|---|
| teaser | the festival | logo |
| arrival | daytime B-roll (classes, set-up, panels), each held long, woven with daytime demos | — |
| shows | the stage routines, shuffled | — |
| night | animations, the social floor, broken up by band, DJ and crowd | — |
| climax | short cuts landing on slow-motion moves, crowd reactions | bullet time to close |
| finale | faces, the floor | the ending shot |

**Why:** a recap that is only highlights has no shape. The acts give it a beginning, a build and a peak.
**Code:**
- `plan.story` and `plan.default_acts`; a signature transition at each act change (`grammar.SIGNATURE`).
- Knobs: `acts` (bars; put the boundaries on the song's section changes), `split`, `day`, `day_shows`,
  `night_shows`.

## 3. Use everything: card shuffle and fair share

**Rule:** every file in the export appears in each format it belongs to, with several different scenes from each.
This is a hard requirement: use more, shorter cuts rather than leave a file out. A file is the unit, not the camera
clip: one clip split into five social segments is five files, and each one appears. Footage shot in portrait (the
social segments) is portrait-only; landscape never uses it.
- A clip's shots are spread through the reel and interleaved with others of the same style (the matinee routines
  weave through each other).
- Never play one clip as a block of consecutive shots, cut a clip to itself, or put B-roll next to B-roll.

**Why:** the people who are in the footage watch the reel. Leaning on a few strong clips leaves people out, and
blocks feel like separate videos.
**Code:**
- `Picker`: every extra shot from a clip must beat other clips by `SHARE` = 0.3 per shot it already has, and a
  unit's shots sit `APART` = 6 s apart.
- `cover` orders clips by their best moment as it plays in this format.
- `weave` deals the act like cards.
- `cover_all` puts every missing unit in: it takes the place of the weakest plain shot of a unit with several, in
  its home act (the social floor may also go in the climax), or splits a long plain shot to make room. The band
  inserts and the editor's extra shots never take a unit's only shot.
- With picks cut to pieces (`pick_hold`) and no `pick_share` named, the picks take half the reel, and more when
  the clips the editor left alone are too few to fill the rest fairly (`rest_share`: two shots a clip, one for
  B-roll, up to 90%); a clip left alone that still piles up gives its plain shots' beats back to the picks' own
  in-outs (`thin_clips`); in a cut with no night act the day's picks may play in its climax.
- Nothing is shown twice: a quick cut that would replay footage the reel shows elsewhere goes, its beats to the
  picks beside it (`replays`, `no_reuse`; `plan.<fmt>.reuse: true` allows it, `false` takes every quick cut out).

## 4. Cut on the action: movement flow

**Rule:** movement carries across the cut. A spin cuts to a spin, and travel to the right keeps going right. A
movement carries 2–3 cuts, then changes.
**Why:** matched action makes cuts invisible and keeps the energy going.
**Code:**
- `flow.py`: optical flow inside the subject's box minus the background, so a camera panning with a dancer still
  reads as the dancer moving. Cached in `flow_cache.json`.
- `plan.flow_link` scores a cut from −1 to 2; `weave` uses it.

## 5. No same-y runs

**Rule:** don't cut between near-identical shots. If footage is static, keep one longer shot; if it is fairly
static, break it up with other dancing.
**Code:** B-roll is held long (arrival: 3–5 beats), the same B-roll family is penalised within 3 cuts, and the
same move kind is penalised after 2 in a row (`weave`).

## 6. Music-video rhythm

**Rule:**
- Shot lengths come in 8-beat phrases (one salsa 8-count) drawn from varied patterns: quick cuts, bar-length shots,
  occasional long holds.
- Busier phrases of the music get quicker patterns, and the night builds into the climax.
- Never long runs of equal-length shots.
- Each move's peak lands on a beat, and cuts come 2 frames (40 ms) before the beat.

**Why:** in a study of 548 official music videos, dance-video shots mostly last about one beat. Editors cut with
anticipation, accelerate near climaxes and slow at bridges. One shot length makes the rhythm predictable (Larry
Jordan).
**Code:**
- `PHRASES`, `rhythm`, `phrase_energy`, `fit_slots` (moves that need time get the long slots), `in_point`,
  `ANTICIPATE`.
- Cut frames are counted from the reel's start (per-shot rounding drifts most of a second over 3 minutes).
- Cuts sit on the music's own beat grid (`plan.clock`), never on a count of nominal periods: a tracked grid
  (a live band, a DJ mix, an intro the tracker lost) is cut on its real beat times. `music.fit_beats` gives a
  steady song one exact grid from its first moment (`steady_grid`), and smooths a drifting one.
- Salsa tracks at half-time, so one detected bar is an 8-count.

## 7. Subject: who the shot is about

**Rule:** the subject is centred and fully in frame, with no head cut off. Effects target the subject, not
bystanders.
**Why:** sharpness alone gets stage footage wrong. Dancers in motion blur read as soft, and still bystanders as
sharp.
**Code:**
- `subject.py` / `subject_engine.py`: SAM 3 masks every person, BiRefNet saliency and Depth Anything nearness pick
  the salient, near, large person, plus anyone nearly as strong (the partner, or a line on stage).
- `frame.track` interpolates between keyframes, and `frame.framing` scores centring, size and edge contact.

## 8. Moments: the middle, where the dancing is

**Rule:** for routines and animations, use the middle sections, where the butt shakes and faces are; skip the
walk-on. The subject must be dancing for the whole stretch a shot plays.
**Code:**
- `moments.py` detectors:
  - spin: face, back of head, face, within about 1.5 s
  - dip: torso far from vertical
  - trick: ankle above hips, or a lift
  - shake: SigLIP hips score plus hip wobble under a steady torso
  - pattern, close-up, travel, energy, style
- The first quarter of each unit is marked down, and an activity gate drops people waiting or walking.

## 9. Velocity edits (speed ramps)

**Rule:**
- Fast through the move, then ease down to arrive slow exactly as the dancer lands (the end of a turn, a pose hit).
- Hold there so the hair and the eyes play out, then ease back up.
- Dancing never runs faster than 1.5x; any faster and people look sped up, not energetic.
- Super-slow close-ups of hair flips are great.

**Why:** this is how CapCut velocity edits and Premiere / After Effects time-remap ramps are built. Slowing mid-move
gets it backwards: the fast part looks slow and the hold has nothing to show.
**Code:**
- `fx/speed.py` `Speed.curve` uses cosine eases with no corners. Presets (fast×, slow×, hold beats, ease beats):

  | Preset | fast | slow | hold | ease | Use |
  |---|---|---|---|---|---|
  | `montage` | 1.5 | 0.5 | 0.75 | 0.25 | |
  | `hero` | 1.4 | 0.3 | 1.0 | 0.4 | |
  | `bullet` | 1.5 | 0.15 | 2.0 | 0.6 | the reel's peak |

  - A close-up slows a further ×0.6 and holds 0.5 beat longer.
  - `cap` param: 1.5 for dancing, 3.5 for B-roll.
- `velocity.settle(project, unit, t)` finds the landing: subject motion is a burst in the 0.6 s before and a rest
  from 0.1 to 0.6 s after, searched over −1.0 to +0.8 s.
- `velocity.landing()` puts the landing on the strongest accent beat that leaves room for the hold;
  `plan.place` applies both.
- One-off: `focalcut fx apply clip.mp4 -e hero --peak <landing s> [--closeup]`.

**Knob:** `heroes`, the slow-motion holds in the shows act.

### 9b. Show the landing (every dance shot)

**Rule:** when a dancer finishes a trick and comes to rest, the shot stays on them a little longer so the audience
sees them land gracefully. Never "dancer does a trick" without "dancer finishes the trick".
**Code:**
- `velocity.landed(project, unit, t)` finds where the move around the peak comes to rest (burst then rest, from
  0.3 s before the peak to 1.6 s after), or None.
- `plan.lands` marks each plain dance shot's landing and, when the shot is too short to show the end of the move
  (`velocity.PRE` = 0.5 s) and the rest, takes a beat from a plain neighbour in the same act.
- `velocity.hold_in` moves the in-point as little as possible so both fit; a one-beat hit too short for both lands
  40% in.

**Knob:** `land_hold`, beats of rest after the landing (default 1, at least 0.45 s).

## 10. Transition grammar: cohesive, motivated, mostly hard cuts

**Rule:** mostly hard cuts on the beat. Each act has its own short vocabulary and a cap on each transition, and
every transition needs a reason in the two shots it joins. Never two in a row; crowd reactions and band inserts cut
straight in. About 15 transitions in a 3-minute reel, not 45.

| Act | Vocabulary (preference order, cap) | Into the act |
|---|---|---|
| arrival | fade ×3 (only when B-roll is involved) | dip to white |
| shows | spin ×3, swipe ×3, zoom ×2 | zoom |
| night | swipe ×3, spin ×2, glitch ×2, flash ×2 (accent ≥ 1.8) | light leak |
| climax | flash ×4 (accent ≥ 1.6), spin ×2, swipe ×3 | velocity whip (flash-out → flash-in) |
| finale | zoom ×1 | dip to white |

Reasons:
- **spin:** turn or pattern into turn or pattern, from different clips.
- **swipe:** both shots travelling the same way, flow ≥ 0.12 each side.
- **zoom:** push in to a close-up that is at least as tight.
- **flash:** a big musical hit.
- **glitch:** in or out of the DJ.
- **fade:** between daytime scenes: both shots daytime footage, never into or out of the night's.

A transition other than a flash needs two beats of each shot (`grammar.MIN_BEATS`): across a one-beat shot the
dissolve or push is the whole shot. The change of act takes its signature only where the shots can carry it: the
velocity whip into the climax needs a ramp (the planner's own shots) or the same movement across the cut; between
an editor's picks, which play at real speed, the drop gets a flash when the music hits there and the plain cut
when it does not.

**Why:** music-video and aftermovie editors use hard cuts for rhythm and save transitions for moments that mean
something. A transition cycled from a menu reads as a template, and that random look is the main complaint about
CapCut-style edits.
**Code:** `grammar.py` (`GRAMMAR`, `CAPS`, `FLASH_AT`, `SIGNATURE`, `reasons()`), `plan.transitions`.
**Knobs:** `grammar`, `caps`, `flash_at`.

## 11. Cohesive effects, not gimmicks

**Rule:**
- Effects serve the story: one at a time, never on neighbouring shots.
- Each act has its own effects:
  - shows: three slow-motion holds and one spotlight
  - night: one neon glow (the club) and two holds
  - climax: holds building to one bullet-time move in fire (the only subject effect, on the drop)
  - finale: a face, a held move, a hair flip in deep slow motion
- No per-beat punch-ins, orbs or random overlays. No freeze-frame outline effects (they read as gimmicks).

**Code:** `plan.give_slot_fx` (skips a shot whose neighbour has an effect), the effect choices in `plan.story`, and
the subject effects in `fx/subject.py` (`aura`, `neon`, `spotlight`, …).

## 12. Portrait framing (9:16 from 16:9)

**Rule:**
- The strip is a third of the frame wide, so track the subject tightly.
- When a couple dances apart, follow one dancer (the one showing their face), not the empty floor between them.
- Strongly favour faces; on the social floor, avoid people's backs.
- Favour shots where the subject stays wholly in frame and isn't partly leaving it.

**Code:**
- `frame.follow(d, times, union, cw)`:
  - If the group is wider than 0.8 of the strip in ≥30% of frames, it tracks one dancer box to box (jumps under
    0.12) and holds the last position through gaps.
  - The dancer is scored by face ×2, coverage, √area and sharpness.
  - Otherwise it reports the face of the main (largest) member.
- `frame.face_seen`: with the unit's YuNet face scan (`scores/faces_<unit>.npz`, the `people` step), a person
  shows their face when a confident face (score ≥ 0.8, height ≥ 3% of the frame) sits in the head of their pose
  box within 0.3 s. ViTPose puts confident eyes on the backs of heads on the social floor, so the pose test
  `frame.faces(kp)` (nose and both eyes above 0.6, left eye right of right eye) is only the fallback. Run the
  `people` step (or at least its face scan) before planning a portrait cut of social dancing.
- `plan.face_track`: when the body track's strip doesn't hold the shot's face (a crowd shot up close, where the pose
  boxes lose the person or settle on the one beside them), the strip follows the face instead: the persons step's
  dense face scan, only faces in the head of one of the shot's main people, and only one near the strip already
  (the planner's subject, never a bigger face elsewhere). The face sits in the upper third; a second face that fits
  shares the strip; a face too big for it zooms out. It takes over only when it holds a face for clearly more of
  the shot (`FACE_GAIN` 0.25). On Electric Nation 2026's portrait reel it re-centred 45 of 142 shots.
- `plan.reframe` returns the crop plus `present` (≥60% of the subject inside), `inside` (≥92% inside and not cut
  by the camera's edge) and `face`, each as a fraction of the shot.
  - The crop centre is smoothed over about 0.3 s in a narrow strip and 0.6 s otherwise.
- `plan.assess` multiplies each moment's rank:
  - present < 0.6: ×0.04
  - inside:
    - social floor: (0.3 + 0.7·inside)
    - others: (0.45 + 0.55·inside)
  - portrait face:
    - social floor: (0.3 + 1.4·face), and a further ×0.25 if face < 0.3
    - others: (0.5 + 1.0·face)
  - landscape face: 0.85 + 0.3·face
  - The editor's picks are never dropped.

## 13. Shake

**Rule:** measure camera shake. If a stretch is too shaky, use another part of the same clip; if it is mildly
shaky, stabilise it. The worst offenders are handheld talking heads (panels) and animation clips.
**Code:**
- `shake.py` gets camera motion from corner tracking (LK) plus a RANSAC similarity fit, so dancers are outliers.
  `hf` is the motion minus its 5-tap (~0.1 s) smooth, as an RMS over 1 s, in frame widths per frame.
  - Pans and follows drop out.
  - It reads the GPU-decoded H.264 export when there is one.
- `plan.assess` divides `hf` by the crop's width (a 9:16 strip shows the shake three times over):
  - above `shake[1]` (0.006): rank ×0.03
  - above `shake[0]` (0.0012): up to ×0.65
- `place` sets `stab` when the shot's own jitter is above `shake[0]` (not on `_stabilized` sources).
- `render.stab` runs vidstab (detect shakiness 6; transform smoothing 15, optzoom 1).

**Reference levels:**

| Footage | hf |
|---|---|
| stage tripods | 0.3–1 e-3 |
| handheld socials | 1–6 e-3 |
| handheld talking head | p95 about 22 e-3 |

## 14. Band members interspersed

**Rule:** individual players (singer, violin, horns, percussion) cut in through the reel on the music's hardest
hits, one at a time.
**Why:** the live band is half the event, and a player on a horn stab ties the picture to the sound.
**Code:**
- `plan.band_in`: each act gets its count of short inserts.
  - Each replaces a plain dance shot of 2 beats or less on the highest-accent beat near evenly spaced positions.
  - Never next to other B-roll, an effect or another insert, and never with a transition.
- Knobs: `members` (tag substrings), `member_clips`, `band_in` (`shows=2, night=3, climax=3`).

## 15. The ending shot

**Rule:** end on a chosen shot that sums up the event. In the reference shoot (a 2026 salsa congress) that is the demo where everyone puts a hand in the
middle and bursts up (C2023 at 126.6–133.4 s).
**Code:**
- `plan.reserve_ending` keeps ±4 s around it out of other shots.
- `plan.ending_record` finds it; the finale gives it the remaining beats.
- `place` sets the in-point so the shot runs out exactly at its `to` time, with a centred, unzoomed crop at the
  optional x.
- **Knob:** `ending`: `[export file, from, to, crop x?]`, in that file's own seconds.

## 16. The song's start

**Rule:** the reel starts where the song proper starts, on a downbeat. Downloaded tracks often have an intro: the reference
shoot's track starts about 0:17, on the bar where the music returns after a break.
**Code:** `music.analyse(start="auto" | seconds, start_hint=…)` picks the bar nearest the hint, and the window ends
on a phrase boundary inside `max_len` (reel.json `length`).
**Knob:** top-level `start_hint`.

### 16b. Several songs, one per act (a festival's run of sets)

**Rule:** at a festival with a lineup, each act can play a song by the DJ whose set its footage comes from, in set
order, mixed into each other the way a DJ would, so the music never stops or lurches. Each song gives its window
around its strongest drop (a few bars of build, then the drop). Into each hand-over the outgoing song rides its
tempo to the next song's over 8 bars (time-stretched, pitch kept; Americano at 123 into Shooting Stars at 125 is
inaudible, 112 into 123 sounds like a build), then both play beat on beat for 4 bars centred on the hand-over: the
highs crossfade across it and the bass swaps in the half beat before its downbeat, so two kicks never stack. The
songs are level-matched. Only songs more than 15% apart in tempo still cut on the downbeat.
**Code:** `music.join(songs, max_len, mix=…)` (window per song by `drops`, the low end coming back after a build,
and `song_window`; each section's source->output beat map, `ramp`); `music.mix` renders `music_mix.wav` (`warp`
lays each song on its beat map with Rubber Band through ffmpeg). The joined grid keeps each song's real beat times, so
`plan.clock` places cuts on them across a tempo change, and each shot carries its own `beat`.
**Knob:** top-level `songs`: `[{"song": path, "bars": 16, "pre": 8, "acts": {"teaser": 2, "arrival": null}}, …]`; the
last song's `bars` may be `null` (whatever fills `length`). `acts` names the acts each song carries, in bars (`null`:
the rest of that song). Top-level `mix`: `{"bars": 4, "ramp": 8, "max_ramp": 0.15, "split": 150}` (overlap bars,
ramp bars, the largest tempo gap to ramp across, bass split in Hz); `false` cuts between songs.

### 16c. The headliner's set saved for the end

**Rule:** the last set's footage (the biggest crowd) stays out of the earlier acts, and the climax and finale cut from
it first.
**Code:** `plan.story` with `late`.
**Knob:** `plan.<fmt>.late`: categories or B-roll tags (substrings), e.g. `["headline"]`.

## 17. Clean effect masks

**Rule:** subject effects need clean edges.
**Code:** `fx/masks.py` uses SAM 3 video to track and BiRefNet's matte on a crop around the subject, kept inside
SAM's slightly grown mask. It blends between tracked frames and adds a little memory against shimmer. It caps at
6 people at 960 px in crowds.

## 18. Deliverables and the editor's loop

**Rule:** always deliver the rendered video plus an FCP7 XML, plus a local HTML review page. The editor answers
with picks: `ban` a range or a whole clip, `prefer` a range. If nothing was detected in a preferred range, a
moment is made there.
**Code:** `xml_export.py` (rendered shots on V1, sources disabled on V2, song on A1, stinger on A2; no animated
position keyframes), `review.py`, `plan.apply_picks`.

## 19. Pick rules: what never makes the cut

Isaac's rules for which stretch of footage may become a shot. They are for every agent, every event and every
format: landscape, portrait and the profile (portrait) segment videos alike. They also cover the planner's own picks, cover and people shots, recasts, band inserts and openings alike (only the
editor's own picks are exempt). The scrub (`windows.py`) checks each one over the whole span a shot plays, and
`plan.snap_windows` / `plan.window_report` keep every planner shot on passing windows. A rule the scrub cannot
check yet is a bug in the scrub, not a reason to let the shot through.

The editor's picks always win, in every format: no rule overrides a pick. A clip with picks is the editor's call
on its best part, so the planner takes nothing else from it, in either format (`apply_picks`, on by default;
`plan.<fmt>.only_picks: false` lets the rest back in). The planner may still change a pick itself: cut it to its
best beats, take a second piece of it, condense a clip's many picks (`pick_hold`, `pick_share`).

The face that counts is the subject's own: the person the shot is about, and for a couple, duo or group, every
person in the action. A face somewhere else in the frame (a cheering crowd behind, a friend at the edge) never
makes a back pass.

A shot is never:
- **A back.** No subject seen from behind for the shot (the back of a head, a back-turned dancer hiding the
  partner, a singer facing away). Slide along the clip to where the face or the partner shows. (`head`, `body`)
- **Turned away.** A face looking down at a phone, away from the action, or turning its back during the shot
  fails. Side-on alone is not a back: a profile engaged in the moment (singing, dancing, looking at a partner)
  passes (EN: C2107, C2091, C2103 kept). Take the stretch where they turn back toward the camera.
- **Behind a foreground back.** When the largest, nearest person in frame shows their back, the shot fails even
  if the people beyond them are smiling at the camera.
- **Blocked or at the edge.** A subject hidden behind their own arm or another person, or half out of the frame
  at the edge, with a stranger's head poking in. (`blocked`, `exit`)
- **A band from behind.** A band shot is about a player's face: the front man, the guitarist, the keyboard player
  facing the camera mid-song. A front man's bare back, even with arms up, never stands in for that while the
  set has stretches with his face in full view. Take the band's best faces first.
- **A group that isn't together.** When half the group has turned away or they've drifted apart instead of
  dancing with each other, it isn't a group shot. Pick the stretch where they're together and facing the camera.
- **The crowd.** A pan off the dancers onto an audience sitting or standing still, or a single bored onlooker as
  the only face, is never a subject shot. Someone has to be dancing or performing for the whole shot. (`idle`)
- **Blurry.** Haze, stage fog and a dark but readable face are fine (EN: C2130, C2125 kept). The subject's face
  must not be smeared, though, not just something in the frame: a sharp foreground arm in front
  of a soft dancer fails. Missed focus, a face smeared by camera motion, or a face pressed
  up against the lens all fail. The floor is absolute, not relative to the clip, and it applies after speed
  effects too: frame-blending on a fast stretch must not smear a close face. (`soft`, `blocked`)
- **Shaky, leaving, sliced or low.** Wild camera swings, the subject leaving the frame, a couple split by the
  strip edge, or the subject far down the frame all fail. (`shake`, `exit`, `split`, `low`, `lost`)
- **The gap between people.** When a pair or a group is wider than the portrait strip, frame one dancer or
  couple whole and centred and let the others go out of frame. Never centre the floor between two couples and cut
  both (SSMC: C1991's yellow couples). Groups take the dancer in the centre by default (`frame.GROUP` "centre";
  reel.json `"group": "pan"` asks for a pan). B-roll shot among the dancers up close, like a class floor, follows
  one dancer the same way (`plan.close_dancers`; SSMC: C1968). Someone the camera's own edge cuts in half is never
  one of a couple.
- **Bored, blinking or grimacing.** Engaged expressions only: smiling, eyes open, in the moment. This is about
  expression, never appearance. (`expr`)
- **A weaker moment of the same person.** If the same clip shows that person smiling and letting loose
  somewhere else, use that stretch. Rank a clip's passing windows and take its best, not the first that passes.
- **A clip's weak moments, repeated.** Several bad picks from one clip (the dinosaur riders: back, profile,
  blur) mean its best stretch was never found. Rank the clip's windows and use the top ones.
- **A repeat that's worse.** A person or group coming back for a second shot needs a moment as good as the
  first, from a different stretch.

Effects follow the same rule: a subject effect (glow, outline, matte) takes in every person in the action or
none of them. Never cut through a person, leaving them half in and half out of the effect.

Selection order: scrub every clip densely for windows that pass all of these, then choose moments and coverage
from those. Don't pick by energy and reject afterwards.
