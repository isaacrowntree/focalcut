# romp's timing ideas, read for edit timing

Source: github.com/isaacrowntree/romp at `7fae6b4` (0.15.0). romp is a TypeScript library for animation made in code:
Box2D motion baked once, character "acting" curves, a synthesised soundtrack and a motion check that runs in CI.
It has no edit or music-cutting code. What carries over is how it **times** things: durations come from what the
content needs, every accent has an anticipation and a settle, sound leads picture, and every rule is a check with
thresholds, exemptions and an allow-list.

Target: a timing engine for focalcut's new slow documentary mode (the Grace City beach baptisms: drone, tripod,
handheld, one song). The current planner (`reel/plan.py` `rhythm`, `pace_lengths`, `hold_sustains`) gives each
8-beat phrase a pattern of shot lengths. The complaint was "missing better beats". The ideas below are ordered from
most to least transferable. Each one gives where it lives in romp, what it does, and a proposed mapping to cutting.

---

## 1. Durations come from what the content needs, and the music only places them (`src/timing/index.ts`)

**In romp.** Scene lengths are never typed in. `readingTime` gives the seconds a viewer needs, and `timeline()`
makes each scene at least that long. It adds transition and padding frames, and leaves extra length (`seconds`,
"the footage, a physics beat") after the content:

```ts
readingTime(text) = max(min = 1.5 s, words * 60 / wpm(180))
captionTiming     = enter + type(chars / cps) + hold(readingTime) + exit
scene.frames      = max(need, asked, tIn + tOut + 1)
need              = tIn + lead(0.2 s) + Σ(captions + gap 0.15 s) + tail(0.2 s) + tOut
```

`timeline()` also returns `check: { covered, scenes, sections }`, the transition spans and cut times, which the
check uses. The comment says a film timed this way "passes `unfinished` and `unread` by construction".

**For edits.** Give every shot a **need**, in seconds, that does not depend on the music:

- `read`: the time to take in the frame. A drone wide or establishing shot needs more (≈ 3–5 s), and a face or
  close-up less (≈ 1.5–2.5 s). Scale it by frame complexity (people count, edge density) the way `readingTime`
  scales by word count, with a floor.
- `action`: the time for the shot's event to finish (walk into the water → dunk → emergence), taken from the
  pose/motion tracks.
- `lead` / `tail`: padding before and after the action (romp uses 0.2 s; for a contemplative film use about 0.5–1
  beat).

Then let the music **quantise** the need: shot length = the smallest phrase-aligned length ≥ need. The planner
should no longer pick a pattern and then look for footage to fill it. It should collect needs and snap them to the
grid. The same pass emits `covered` (dissolves) and `scenes` (cuts) for the check (§6).

## 2. Every key moment is a "take": anticipation, accent, hold, settle, release (`src/core/acting.ts` `takeCurve`, `takeLength`)

**In romp.** A reaction follows a fixed shape with frame counts taken from Williams and School of Motion (24 fps
counts scaled to 30):

```
antic 6f (goes −0.35 the other way) → pop 2f (hard: smear frame + 12% overshoot; soft: 9f ease)
→ hold 3f (hard) / 0 (soft) → settle 9f (bounces to 0.5) → keep 8f (held) → release 10f
takeLength = antic + pop + hold + settle + keep + release
```

A **hard** accent pops and bounces back. A **soft** accent eases in with no bounce. Heavier bodies get a longer
anticipation: `jump` uses `crouch = 5 + 3·max(0, weight−0.4)/0.6`.

**For edits.** Build each key moment (each dunk) as a **shot sequence shaped like a take**, in beats rather than
frames:

| take phase | baptism sequence | length (proposal) |
|---|---|---|
| anticipation | walking out, pastor's hand raised, held breath (often a wider or quieter shot: "a smaller move the other way") | ≈ 0.6 × accent strength × phrase |
| accent | the immersion, cut **on** the musical accent | short: 1–2 beats for hard, a slow dissolve for soft |
| hold | the emergence held, no cut | ≥ the `action` need |
| settle | reaction: crowd, hug, applause, a wider shot (half the intensity of the accent) | ≈ 1.5 × hold |
| keep / release | back to calm b-roll (drone, water) | to the phrase end |

The accent type is chosen per moment. A **hard** accent is a straight cut on a strong downbeat with a speed-ramp
"smear" (one quick in-between). A **soft** accent is a dissolve that lands on a swell. The anticipation length grows
with the moment's **weight** (importance). The last baptism, or the hero subject, gets a longer build. `takeLength`
gives the total length up front, so the planner can schedule a whole sequence into the song before it picks any
footage.

## 3. Lead: transients lead the picture by about a frame, and bigger events lead more (`src/audio/mix.ts` `LEAD`, `acting.ts` `talk`/`talkPose`/`gaze`)

**In romp.**
- `LEAD = 1/30`: "transients lead the picture by about a frame". A boing is scheduled at `hit.t - 1/30`.
- `talk()`: the mouth leads the sound by 2 frames ("the eye reads a mouth shape before the ear hears it"). The head
  nod in `talkPose` leads by 4 frames, and its small anticipation lift comes 5 frames ahead.
- `gaze()`: the body dips starting 4 frames **before** a big look, scaled by the look's size (`min(1, d)`).
- `follow()`/`gaze` head: secondary parts lag the primary by 1–2 frames.

**For edits.** Make the lead a parameter of each cut instead of always cutting exactly on the beat:

```
cut_time = accent_time − lead,   lead = lead0 + k · strength      (frames)
lead0 ≈ 1 frame, k ≈ 2–3 frames for the strongest accent
```

A picture change reads slightly before its sound, so a cut that sits a frame or two early lands on the beat for the
viewer. Strong accents (a drum hit into the chorus) take a longer lead. Weak ones sit at the beat. Secondary changes
inside a shot (a speed-ramp peak, a punch-in) **lag** the cut by 1–2 frames, the way `follow()` lags the body. Put
`lead` in music.json or the plan per cut so it can be tuned and tested.

## 4. Restraint: only the focal event of each beat gets an accent (README "rules worth keeping", `Score`)

**In romp.** "**Restraint** (only each beat's focal event gets a sound)", "**no identical repeats** (every voice
takes a seed)", "one family". Impacts are recorded with their **speed**, and every response is scaled by
`k = min(1.4, speed/speedRef)`. Side hits count at half (`hit.side ? 0.5 : 1`).

**For edits.** This is probably the biggest source of "missing better beats". In a slow film, cutting on every beat
or bar flattens it. Use this instead:
1. Score the musical accents in each phrase (focalcut already has `accents`, `sections`, `sustains`).
2. Give a **hard cut only to the phrase's focal accent**, the strongest one. A phrase with no clear accent is held
   or dissolved.
3. Match picture strength to musical strength. Detect "impacts" in the footage (sudden changes of velocity in the
   pose or flow tracks: the dunk, the splash, arms going up, a hug) with a **speed** each. Pair the strongest picture
   impacts with the strongest musical accents, and scale the emphasis (ramp depth, lead) by
   `k = min(1.4, picture_speed/ref) × accent_strength`.

## 5. Settling: springs, decays and "nothing is ever perfectly still" (`src/core/ease.ts`, `motion.ts`, `acting.ts`)

**In romp.**
- `spring(elapsed, {damping c, stiffness k, mass m})` is the closed-form damped spring:
  `ω0 = √(k/m)`, `ζ = c / (2√(km))`. When underdamped,
  `1 − e^{−ζω0t}(cos ω_d t + ζω0/ω_d sin ω_d t)`. When critically damped, `1 − e^{−ω0 t}(1 + ω0 t)`.
  Presets: snappy 14/160/0.6, smooth 20/90/1, bouncy 11/170/0.7.
- `follow(curve, frame, {w 0.55, zeta 0.6, delay 1})`: a stepped spring that chases a curve one frame late, with
  about 10% overshoot, settled in about 8 frames.
- `rock()`/`boing()`/`jiggle()`: after each impact, `e^{−decay·dt}·cos(2π f dt)`, summed over impacts.
- `alive()`/`blinks()`: idle that "breathes" (in slower than out), irregular blinks about 14/min at seeded
  moments, sometimes doubled. "A held pose has to read, so idle is mostly breath, not sway."
- `gaze()`: no drift "in the first 20 frames of a look". Drifts come every 1.2–2.5 s while a look holds.

**For edits.**
- **Camera settle time decides cut-out points.** For a drone or handheld move, fit the camera's velocity decay
  (stabiliser or flow data) and take the 2% settle time (critically damped: `t_s ≈ 5.8/ω0`; underdamped:
  `t_s ≈ 4/(ζω0)`). Don't cut while the move is still decelerating. Cut either mid-move at constant speed or after
  `t_s + keep`, where keep is about 8 frames of held composition (`takeCurve`'s `keep`). This is the edit version of
  "the spike check": a cut placed on the kink of a move.
- **Pacing changes through a spring.** Treat target shot length as a signal driven by music energy. Pass it through
  a critically damped `follow()`, so pace never jumps from calm to busy between phrases. It eases, overshoots
  slightly into the chorus (a `bouncy` preset) and settles. After a climax, lengths ring back out:
  `L_k = L_calm − (L_calm − L_hit)·e^{−k/τ}`, the shot-length version of `ringOut`.
- **Never perfectly still.** Long holds on tripod shots need slow internal motion. Prefer shots with water, wind
  or drone drift. Or add a slow push-in on `ease.sine` (romp's "gentle in-out for slow drifts"), sized like
  `alive()`: barely there.
- **Let a cut land.** After a cut, add no secondary change (punch-in, title, ramp) for about 20 frames, as `gaze`
  does.
- **Irregular calm cuts.** In calm passages, place cuts the way `blinks()` places blinks: about N per minute, each
  at a seeded offset inside its slot (`slot·SLOT + hash01(seed, slot)·0.6·SLOT`), snapped to the nearest beat. They
  are deterministic but don't tick like a metronome.

## 6. The motion check, mirrored as an edit check (`src/check/analyze.ts`, `text.ts`, `layout.ts`, `audio/gaps.ts`)

**In romp.** `analyze(frames, opts)` steps every frame's tracked samples and returns
`{frame, name, kind, detail}` issues (`romp check` exits 1). Three things in its design are worth copying:
**relative thresholds** (spike = `acc > 3·median(neighbours) + 4px`), **exemptions as spans** (`covered`,
`whips`, `scenes` cuts, impacts ±2 frames), and **allow-lists** per rule and per object
(`allow: {kind: ["name*"]}`). Thresholds scale with frame width (`/886`). Runs that only rise or only fall
(entering or leaving) aren't flagged.

The rules, with an edit version of each. Proposed for a `reel/timing_check.py`. focalcut's `validate.py` already
covers per-shot *picture* rules (soft, back, headcut…), so these are the *timing* rules:

| romp rule | what it flags in romp | edit-check version |
|---|---|---|
| `unread` | text whole for less than `readingTime` before it goes | shot shorter than its `need.read` (§1). Default floor 1.5 s in romp; doc mode ≈ 2.5–3 s |
| `unfinished` | caption cut before every character showed | cut lands **inside** a shot's action window (dunk cut before emergence, a hug cut mid-embrace) |
| `stale` | nothing tracked changes for > 3 s after the words were read | a shot held > `need + stale_s` with no internal motion (flow energy ≈ 0): "cut sooner, or give the eye something" |
| `spike` | acceleration far above neighbours': a kink where two moves meet | (a) a cut joining two moving shots whose screen velocities differ by > 3× (direction flip or speed mismatch); (b) a cut during a camera's deceleration (§5); (c) a **shot length** that jumps > 3× the local median without a musical section change, so the rhythm itself has a kink |
| `jump` | a one-frame snap between still frames | a jump cut: two static shots of similar framing and subject (cosine sim of embeddings / box IoU high) cut together |
| `pop` | appears at full size and opaque | a hard cut into a bright, big, sudden image inside a calm phrase. It should have been a dissolve (exempt when it's the phrase's focal accent, §4) |
| `strobe`/`teleport` | > 100 / 300 px per frame at 886 wide | pan speed in the output too fast for the frame rate (judder): handheld whip or drone yaw; exempt on `whips` (deliberate) |
| `lost` | subject leaves the frame and stays out > 1 s | subject out of the strip > 1 s inside a shot (`allowExit` for deliberate exits) |
| `blank` | nothing tracked visible > 0.5 s outside `covered` | > 0.5 s of no subject / near-black outside a dissolve or title |
| `empty` (heading) | a heading still typing 0.5 s after its scene starts | a title card still animating in well after its shot starts |
| `dropouts` | music bed 30 dB under its median for 0.1–1 s, with sound either side | the same, run on the rendered mix (song joins, ducking under live audio) |
| — (new, from §4) | — | `crowded`: more than one hard cut per phrase in calm sections; `missed`: a phrase's focal accent with no cut and no picture event within ±lead |

Copy the exemption model as well. Dissolves are `covered` spans and skip continuity rules. Deliberate whip pans
are `whips`. Declared cuts skip speed rules. The editor's own picks are reported, not fixed, as `validate.py` already
does. romp also has `--every N`, which runs only the single-frame rules for speed. The equivalent split is: plan
rules on the cut list (ms) versus picture rules that need decoded frames.

## 7. The soundtrack as a queryable score (`src/audio/score.ts` `listen`, `Score`)

**In romp.** `listen(schedule)` runs the soundtrack schedule against a silent AudioContext and records every event
`{t, end, voice, bus, vel, midi, shape: hit|swell|bed, peakAt}`. `Score` answers, as pure functions of t:
- `envelopeAt(t, {bus|voice, decay})`: the sum of the selected events' envelopes. A hit has a 5 ms attack and
  decays as `exp(−4·dt/ring)`. A swell is linear to `peakAt`. A bed is flat.
- `beatAt(t)`: `{index, phase 0→1, since, t0, t1}` on the onset grid (onsets < 30 ms apart merge).
- `onsetsBetween(t0, t1)`.

**For edits.** Wrap focalcut's music.json the same way: a `Score` with `envelope(t, band)`, `beat_at(t)` →
`(bar, beat, phase)`, `phrase_at(t)`, `swells()` and `sustains()`. Each planner pass and each check then asks it
questions instead of indexing arrays. The **swell** shape with `peakAt` matters most for this song. In a slow song
the "beat" that matters is often a crescendo peak, not a percussive onset. A soft accent (§2) should land its
dissolve mid-point on the swell's peak, and the emergence should be timed to it.

## 8. Phrasing as speech: declination, cadence and stress (`acting.ts` `syllables`, `prosody`)

**In romp.**
- `syllables`: each syllable is `(1/rate)·(0.75 + 0.5·hash01(seed,n))·(stress ? 1.25 : 1)`. There is a pause after
  punctuation (0.16 s) and a gap between words (0.03 s). `fit` stretches the whole line to an exact length.
- `prosody`: each phrase "starts high and drifts down" (declination `3·(1−u)`). The ending says what the phrase is:
  a statement falls (−4), a comma lifts (+2, "more to come"), a question rises (+5), an exclamation is higher and
  louder throughout.

**For edits.** Treat each musical phrase as a sentence of shots:
- **Declination.** Inside a phrase, shots start shorter or more active and lengthen toward the phrase end (the
  picture "drifts down" to rest). The pattern is reset at the next phrase.
- **Cadence → cut type at the phrase end.** A resolved cadence (end of verse or chorus, "statement") gets a held
  wide shot and a clean cut or dissolve into the next idea. A half cadence or mid-section boundary ("comma", more to
  come) stays in the same scene and subject, with a matched cut and no change of location. A build into a drop
  ("exclamation") gets the hard accent of §2.
- **Stress = length.** The shot on a stressed beat (bar downbeat, phrase start) is 1.25× longer.
- **Seeded jitter.** Multiply each length by `0.75 + 0.5·hash01(seed, i)` before quantising. Lengths vary without
  randomness and stay the same on every run. romp's `hash01` is a 32-bit integer mix.
- **`fit`.** Scale a whole act's lengths to land exactly on a section boundary. focalcut's `fit()` and `squeeze()`
  already do this.

## 9. Transitions never sit on content, and segments never split a transition (`timeline()`, `pipeline/segments.ts`)

**In romp.** "Transitions never land on a caption's reading window." Each transition overlaps the scene before it
(`start += frames − tOut`). `planSegments` puts render boundaries at scene starts and ends but "never strictly inside
a transition", and a stretch shared by two scenes becomes its own segment `"a→b"`.

**For edits.** A dissolve's span must not overlap either shot's `need` window (§1). Each shot is laid out as
`tIn + lead + need + tail + tOut`. If focalcut caches renders per shot, cache the dissolve overlaps as their own
segments, so re-rendering one shot doesn't re-render its neighbours.

## 10. Easing vocabulary and the asymmetry of entrances and exits (`src/core/ease.ts`)

**In romp.** "No linear interpolation should reach the screen without one of these on it."
- `out` bezier(0.16,1,0.3,1): entrances.
- `inOut` (quint): camera and long moves.
- `in`: exits only. "Exits are faster than entrances, ~0.6×".
- `sine`: slow drifts.
- `handoff()`: a 5-frame smoothstep to blend two poses, so "a hand-off never jumps".
- `stagger`: 2–4 frames between siblings.
- `jiggleParts`: each part lags the previous by 1.5 frames.

**For edits.** Speed ramps should be C¹-continuous: the speed curve uses `inOut`, and joins blend over at least 5
frames, so the time map has no kink. A kink is what the spike rule (§6) would catch. Fade-ins are slower than
fade-outs (out ≈ 0.6 × in), and title-cards follow the same rule. When several cameras cover one moment (drone →
tripod → handheld on the same dunk), stagger the angles so each takes over a little later in the action rather than
all re-showing the same instant. Use `ease.sine` for slow drone push-ins and Ken Burns.

---

## Architecture worth copying

- **Bake once, read as a pure function of time** (`Stage.bake` at a fixed 240 Hz; `poseAt(actor, frame)`). Every
  consumer (live viewer, renderer, scrubber, check) reads the same baked data. For focalcut: compute the timing plan
  once into a timeline structure (per shot: `need`, `lead`, `action`, accent, transition spans, cut times). The
  planner, the check, the review page and the XML export then all read from it, and nothing recomputes timing.
- **Determinism through seeded hashes, not RNG** (`hash01(a, b)`). Every variation is `hash01(seed, index)`, so the
  same inputs give the same cut, in any order. focalcut already promises "the same plan every run" (`validate.py`).
  A shared `hash01` would make the jitter of §5 and §8 safe.
- **Lint the plan before rendering** (`pipeline/lint.ts`). Layout rules run on plain data in milliseconds, using
  "the same names and thresholds" as the frame check. For focalcut: run the timing rules of §6 on the plan or cut
  list before decoding any video. Re-run the picture-dependent ones after render. One rule vocabulary for both.
- **The timeline emits its own check config** (`timeline().check = {covered, scenes, sections}`), so the check
  always knows where cuts and transitions are and a check can't drift from the plan.
- **Issue reports**: `frame  time  kind  name: detail`, sorted by frame, with `--json` and a non-zero exit. Each rule
  has thresholds as options plus an allow-list with prefix wildcards, and per-object `allow`. Tests plant a bug and
  assert that it's caught (the demo has "plant a bug" toggles).
- **Contact sheets per scene** (`romp sheet`): every Nth frame, labelled with timecode, frame and scene name. This
  fits Isaac's preference for local HTML previews. A per-phrase sheet of cut points would show the rhythm at a
  glance.
- **Cut detection on long takes** (`romp marks`: ffmpeg `select=gt(scene,0.3)` with a minimum gap of 0.5 s), for
  splitting long drone or tripod takes into candidate beats.
- **One continuous bed** (`crossfadeBed`): each section is stretched to the next one's start, written `fade/2` past
  both joins, and joined with an equal-power (sin/cos) crossfade. The `dropouts` check verifies the result.

## Not transferable

Box2D actors and collision layers, particles, the synth and voices, lip-sync visemes, Remotion, three.js, Theatre
and Lottie adapters, and the text-layout rules (offsafe, covers, overlap, legibility), except as a model for any
titles.
