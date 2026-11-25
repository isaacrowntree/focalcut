# Speed ramps: which ramp for which motion

Research for focalcut's speed ramps (`focalcut/fx/speed.py`: `RAMPS`, `Ramp`), prompted by the dunk shots in a
slow documentary film (a beach baptism: 122 BPM, bars of 1.97 s, shots of 2–4 bars; tripod 4K50, drone and a second
handheld camera 4K25, a 50 fps timeline). Three parts: what was wrong with the ramps we had (measured on the
rendered film), what editors say (web, sourced), and the table the code implements.

## How far to trust this

- **Written sources.** Most pages explain how to build a ramp in one editor and give few numbers. The numeric
  rules of thumb come mainly from two tool-vendor blogs (Narrative, Wideframe): conventions, not standards. Pages
  were read through a fetch tool that returns extracts; items marked *(snippet)* come from search-result text only.
- **Not found.** Nothing specific to baptisms or to editing water footage; nothing on turns, walks, hugs, orbits
  or hyperlapses; nothing on bars or sustained notes; no authoritative definition of CapCut's preset curves (the
  six names are confirmed, the shapes are described differently by different pages); no source for a "50–80%"
  optical-flow floor. Adobe's optical-flow help page, the Resolve manual and Motion Array's tutorial would not load.
- **Inferences** are marked *(inference)*. The dunk shape in particular is ours: sources give the principle (slow
  on the moment, on the beat, eased, held long enough to see), not the shape.
- **YouTube tutorials** are listed by title and channel (verified); what they show is from descriptions and search
  results, not from watching them.
- **Our measurements** are frame-to-frame differences of the rendered film (how far the picture moves each frame:
  twice the speed, twice the difference; a repeated frame, zero) and frame strips looked at by eye. They show
  speed and cadence. They do not show whether a ramp *feels* right: nobody has watched the new ramps in motion.

## 1. What was wrong with ours (measured)

The film: `grace_city_baptisms_2026_recap_landscape.mp4`, 13 dunk shots, each with the `hero` curve (a dance
move's: fast in, a hold on the landing) and the rise placed 0.62 of the way through the shot.

| # | Fault | Evidence |
|---|---|---|
| 1 | **The slow motion is a blip.** 0.3x for one beat: 0.5 s on screen, 0.15 s of the event (7 source frames). The face clears the water and the film is already back at real time. Sources ask for a hold of 1–3 s. | The curve: 1x until 2.5 s, 1.4x for 1.0 s, 0.3x from 3.65 s to 4.15 s, 1x from 4.4 s (shot 29, C2210, 5.9 s). |
| 2 | **The changes of speed are slams.** 1.4x to 0.3x in 0.4 beats (0.2 s, 10 frames), and back in another 10. Sources: 8–15 frames at 24–30 fps (0.25–0.5 s) at least, longer for a subtle ramp. | Frame differences fall from 8.4 to 1.6 in 9 frames. |
| 3 | **The fast run judders.** 1.4x on a shot with a face plays the nearest source frame (so the face is never blurred), which at 1.4x steps 1, 2, 1, 2, 1, 1, 2 source frames. | Frame differences alternate 4.9, 8.2, 4.9, 8.0, 5.1, 4.9, 8.0 for the whole second. |
| 4 | **The speed-up is too small to read** (1.4x saves 0.4 s) but its judder is visible: the cost of a ramp without the effect. | — |
| 5 | **0.3x from 50p is below what the footage carries**: 15 real frames a second, the rest optical-flow in-betweens of a splash, the case every source names as where optical flow fails. | Section 4. |
| 6 | **Nothing is on the music.** The rise lands 0.62 of the way through a three-bar shot: beat 7.4 of 12, between beats, half a beat short of the bar line; the speed changes fall wherever the curve puts them. | plan: `peak` 3.659 s, bar lines at 1.967 s and 3.934 s. |
| 7 | **A 25p source would stutter.** The decode conforms every source to 50 fps by repeating frames, so 25p is already 25 real frames a second at 1x; `hero` would take it to 7.5. (No dunk in this film is 25p; nothing stopped one.) | Marcos's C0682 at 1x: every second frame a repeat. |
| 8 | **Moving drone shots at 1.25x judder** (not a ramp, found on the way). 25p at 1.25x on a 50 fps timeline shows its frames for 2, 1, 2, 2, 1 timeline frames: an uneven cadence on an orbit. 4x (held shots) is clean. | Shot 3 (drone spin): 38% of frames are repeats in the pattern `0 0 9 9 0 9 0 9 9 0`. Shot 64 at 4x: none. |

## 2. What editors say

### Curve shapes

- A ramp is a "gradual acceleration and/or deceleration" of a clip, at least two keyframes
  ([StudioBinder](https://www.studiobinder.com/blog/how-to-speed-ramp-in-premiere/)).
- An unsplit speed keyframe in Premiere is an instant change; split, "the farther apart you drag the keyframes,
  the more gradual the transition" ([Adobe](https://www.adobe.com/creativecloud/video/hub/guides/premiere-pro-speed-ramp.html)).
  The split gives a linear ramp; bezier handles ease the rate of change itself
  ([Motion Array](https://motionarray.com/premiere-pro/premiere-pro-speed-ramping-tutorial/), *snippet*).
- Final Cut's "Speed Ramp to 0% / from 0%" is four stepped segments with editable transitions between them
  ([Apple](https://support.apple.com/guide/final-cut-pro/ver95783cbbc/mac)). Resolve's Retime Curve has bezier
  points ([Tuts+](https://photography.tutsplus.com/tutorials/how-to-create-slow-motion-in-davinci-resolve-with-optical-flow--cms-106752), *snippet*).
- Stepping 1x, 2x, 4x, 2x, 1x rather than jumping 1x to 4x
  ([Kapwing](https://www.kapwing.com/resources/how-to-use-the-speed-ramp-effect/)). A short ramp reads as drama,
  a long one as subtlety (same).
- CapCut's presets are Montage, Hero, Bullet, Jump Cut, Flash In, Flash Out
  ([Shotkit](https://shotkit.com/velocity-capcut/)); they are made for dance and velocity edits.

### Speeds and durations

| | Figure | Source |
|---|---|---|
| Slow side, usual | 30–50% | Wideframe |
| Slow side, "slow drama" | 0.15–0.3x; under 0.15x use a freeze | Narrative |
| Subtle B-roll slowdown | 80% | Wideframe |
| Fast approaches and exits | 2–4x | Narrative |
| Whip | 8–16x over 0.2–0.4 s | Narrative |
| Time-lapse lead-in | 500–1000%, ramping down to real time | Wideframe |
| Drone speed-up | about 16x (a 30 s flight in 2 s) | DroneXL |
| The transition | 8–15 frames (Wideframe); at least 6 (Narrative) | |
| The slow hold | 1–3 s (Wideframe); 8–20 frames, past 0.7 s a crawl loses viewers "unless the frame is beautiful" (Narrative) | |

[Wideframe](https://try.wideframe.com/blog/how-to-speed-ramp-videos-with-ai-timing/),
[Narrative](https://usenarrative.ai/blog/add-slow-motion-and-speed-ramps-to-a-video),
[DroneXL](https://dronexl.co/2024/06/14/if-beginner-drone-pilots-did-this/). The frame counts name no frame rate;
at 24–30 fps they are 0.25–0.5 s, 12–25 frames at 50 fps *(inference)*. At 122 BPM a beat is 0.49 s: one beat is
the long end of that range.

### When to ramp, and when not

- Ramps belong to action and sport, to "emphasize pivotal moments" (Adobe). "You don't just want to speed ramp
  everything": be "focused and targeted"
  ([PremiumBeat](https://www.premiumbeat.com/blog/alter-the-flow-of-time-with-speed-ramping/)).
- Use them "sparingly and always for a reason"; "cut the ramp if the scene works without it"
  ([FilmDaft](https://filmdaft.com/what-is-speed-ramping-and-how-to-create-it/)).
- "Speed ramps are kind of aggressive" and do not fit every scene (DroneXL). As the main motion device they give
  "repetitive visuals and missed narrative depth"
  ([Beverly Boy](https://beverlyboy.com/film-technology/speed-ramping-isnt-a-plan-better-motion-choices)).
- Three to five ramps a minute at most; "not every peak needs a ramp" (Wideframe).
- "Dialogue is never ramped"; never hold on a blurred frame (Narrative).
- Speed changes magnify shake: stabilise first (StudioBinder).
- Wedding editors are split: slow motion for highlights only, or a film "running at 50% speed" throughout; some
  see heavy slow motion as cover for weak footage
  ([DVinfo forum](https://www.dvinfo.net/forum/wedding-event-videography-techniques/240054-slow-motion.html)).

### By motion

- "The slow point goes on the hit" (contact, apex, a look to camera) and the ramps are built outward from that
  frame (Narrative). A jump: real time on the approach, slow at the peak, real time on the landing (Wideframe). A
  reveal: real time, slow as the subject appears, real time again (Wideframe).
- Drone: "speed up the duration of long tracking shots and then slow the clip down when landing on something of
  interest"
  ([Streaming Media](https://www.streamingmedia.com/Producer/Articles/Editorial/Featured-Articles/Five-Creative-Techniques-for-Editing-Drone-Video-116692.aspx)).
  A vertical "rocket" starts slow and accelerates ([Adorama](https://www.adorama.com/alc/cinematic-drone-shots)).
- Across a cut: speed up the end of one shot and the start of the next (Kapwing).

### On the music

- "The slow-motion peak should land exactly on the beat" and "the ramp transition should happen during the beat
  leading up to the peak" (Wideframe). Put the hit frame on the beat and build outward (Narrative).
- The TikTok velocity recipe (8x for three frames, then 0.5x, a frame before the beat) is the far end of the
  scale ([Filmora](https://filmora.wondershare.com/video-effects/tiktok-velocity-dance-tutorial.html), *snippet*).

### Frame rate and made-up frames

- Clean slow motion is timeline rate over source rate: 59.94p in a 23.976 timeline is 40%
  ([Frame.io](https://blog.frame.io/2019/10/17/mixed-frame-rates-part-3/)). 60 fps gives 2.5x, 120 gives 5x;
  24 fps slowed to 50% "looks soft and stuttery" (Wideframe).
- Nearest frame (dropping or repeating): "stuttering with fast-moving footage or using uneven speed changes"
  (Frame.io).
- Frame blending: ghosting and double outlines, worst "at very slow speeds (e.g., below 50% speed)"
  ([Aiarty](https://www.aiarty.com/knowledge-base/frame-blending-vs-optical-flow.htm)).
- Optical flow: "jelly" warping; it fails on "fast-moving hands, splashing water, hair, or reflections" (Aiarty);
  Frame.io shows a drummer's hand that "warps and tears". When it fails, "the soft blur of blending is almost
  always preferable". Sharper source frames (a faster shutter) interpolate better (Frame.io).
- Past 200%, sped-up footage wants motion blur (Wideframe); sharp frames strobe *(inference)*.

**What follows for a 50 fps timeline** *(inference, with our measurements)*: 50p footage has no clean slow motion
at all: 1x is native and anything slower repeats or invents frames. At 0.5x every other frame is real (25 real
frames a second, the motion of a 25p film) and each made-up frame sits exactly between two real ones, the easiest
case for optical flow; on the dunk's splash it comes out soft, not torn. At 0.3x two frames in three are made up.
So 0.5x is the floor for 50p, and a 25p source (already 25 real frames a second at 1x) is never slowed.

### Tutorials

| Title | Channel | Shows |
|---|---|---|
| [How to Speed Ramp in Premiere Pro, Resolve, Final Cut Pro, and After Effects](https://www.youtube.com/watch?v=97I4-0iBZ5M) | Chris Wilmshurst | the same ramp built in all four editors |
| [Beginner VS Pro: Speedramping Tutorial in Davinci Resolve!](https://www.youtube.com/watch?v=B-AbHbfZdbc) | Joel Van Beek | Retime Curve with optical flow or Speed Warp |
| [Speed Ramping in Premiere Pro (Smooth Tutorial)](https://www.youtube.com/watch?v=vZagYtk1cJ8) | VerticDesigns | eased keyframes in Premiere |
| [DaVinci Resolve Speed Ramp Tutorial: Fastest Way to Ramp on the Beat](https://www.youtube.com/watch?v=0RQKBcD9a7s) | Films By Roy | speed changes on beat markers |
| [How to Speed Ramp in DaVinci Resolve](https://www.youtube.com/watch?v=nO4gAUv8Zyo) | Sergio Mota Academy | retime controls and curves |
| [If Beginner Drone Pilots Did THIS, They'd Be 10x BETTER!](https://youtu.be/41CnbQXgndg) | Mihail Svilenov | drone ramps at about 16x |
| [How to Speed Ramp like a PRO](https://www.youtube.com/watch?v=X5u0qN9TR5I) | Future Design | — |

## 3. The table

Times in beats (a shot is cut on bar lines, so a change a whole number of beats from the landing is on the
music). "Eased" is a half cosine: the speed never jumps and has no corner at either end. Speeds multiply the
shot's pace (a drone's `drone_speed`).

| Motion | Ramp | Speed keyframes | Basis |
|---|---|---|---|
| **A fall into water and a lift out** (a dunk) | `dunk` | 1x while they stand. One bar before the landing: 1 beat easing to **2x** (the lean back), 1 beat at 2x (under), 1 beat easing to **0.5x** (the lift). Slow from 1 beat before the landing (the face clearing the water, on the bar line) to up to 3 beats after it; 1 beat easing to 1x; the last 0.5 s at 1x. | slow on the hit, on the beat, the transition in the beat before (Wideframe, Narrative); 2–4x approaches; hold 1–3 s; the shape is the owner's brief *(inference)* |
| **A rise, an emergence, a person revealed, a reaction** | `rise` | 1x, 2 beats easing to 0.5x, slow from 1 beat before the landing to up to 4 beats after, 2 beats easing to 1x. No speed-up. | reveal: real time, slow as the subject appears (Wideframe) |
| **A jump** | `rise`, the peak at the apex | as above | Wideframe |
| **A drone push-in, fly-over, reveal** | `reveal` | **2x** the pace from the cut, 4 beats easing down to the pace as the move arrives (the peak), then held. | speed the travel, slow on what it lands on (Streaming Media) |
| **A held drone horizon as a time-lapse** | `drift` | half the pace at the cut, 2 beats easing up to the pace (4x), 2 beats easing back down at the end. | time-lapse ramping down to real time (Wideframe); eased ends *(inference)* |
| **A turn or spin of a dancer** | `hero`, `bullet`, `montage` | fast through the move, slow on the landing (the dance curves, as before) | velocity edits |
| **A walk, a hug, people talking, a slow pan, an orbit** | `none` | constant speed. For emotion: the whole shot at 0.5x from 50p (`glide`), not a ramp. | never ramp dialogue (Narrative); cut the ramp if the scene works without it (FilmDaft); *(inference)* for hugs and orbits |

**When not to ramp:** anyone speaking; a shot under two bars (`dunk` drops its speed-up when the run-up does not
fit, and plays real time into the slow motion); shaky footage; more than three to five ramps a minute; a moment
that already works at real time. The baptism film has 13 dunks in two minutes, one every 6 s or so: over the
sourced density if each is counted as a ramp. `"reference_ramp": "rise"` (no speed-up) or `"none"` on some of them
is the conservative reading; that is the owner's call.

**Frame-rate limits** (`slow_floor`, `clean_speeds`):

| Source on a 50 fps timeline | Slowest | Constant speeds with an even cadence |
|---|---|---|
| 50p | 0.5x (25 real frames a second) | 1, 2, 3, 4x |
| 25p | 1x: never slowed | 1, 2, 4x (not 1.25x) |
| 100p or more | 0.5x (the decode drops it to 50 fps first) | as 50p |

The transition between speeds: one beat (0.49 s at 122 BPM, 25 frames) at the least; two for `rise` and four for
`reveal`.

## 4. What the code does

`focalcut/fx/speed.py`:

- `RAMPS`: the shapes above by name: `dunk`, `rise`, `reveal`, `drift`, `none`, each a registered effect.
- `Ramp.land()`: the action's place in the shot. The shot's peak (template `DUNK_AT`: 0.62 of the shot), moved to
  the bar line when one is within a beat, else to the nearest beat, when the shot keeps room for the slow motion.
  `source_peak()` puts the shot's moment there.
- `slow_floor(src_fps)`: no slow motion below 25 real frames a second. `render.fx_shot` passes the source's frame
  rate (`src_fps`).
- Through a change of speed a ramp reads evenly spaced positions (an optical-flow in-between where one falls
  between frames) instead of the nearest frame: no 1-2-1-2 judder. At exactly 2x every frame is a real one.
- `ramp_window()`: how much source a ramp reads before and after the moment (for a planner's steadiness check).
- `clean_speeds(src_fps)`: the constant speeds without judder.

`reel.json`:

```json
{
  "reference_ramp": "dunk",
  "ramps": {
    "fps_floor": 25,
    "synth": "flow",
    "dunk": {"fast": 2.0, "slow": 0.5, "up": 1, "run": 1, "down": 1, "lead": 1, "hold": 3, "out": 1, "snap": "bar"},
    "rise": {"slow": 0.5, "down": 2, "lead": 1, "hold": 4, "out": 2},
    "reveal": {"fast": 2.0, "down": 4},
    "drift": {"edge": 0.5, "up": 2, "out": 2}
  }
}
```

All of it optional; those are the defaults. `"reference_ramp"`: `dunk`, `rise`, `none`, or an old curve by name
(`hero`); without the key a dunk plays `dunk`. `"synth": "hold"` repeats a frame instead of making an in-between
(a 25p look for the slow motion; no softness on the splash). `"snap"`: `"bar"`, `"beat"` or `null`.

Measured on the same shot (C2210, the dunk at 120.8 s), old against new: the speed-up starts on the first bar
line as she leans back, the face clears the water on the second, slow motion runs 1.5 s (3.44–4.93 s) instead of
0.5 s, and the largest change in motion from one frame to the next is under 2x where the old curve alternated
5 and 10 for a second. The slow stretch alternates real and made-up frames (differences 2.9, 3.2, 2.9, 3.2): the
in-betweens are a little softer than the footage.

Not done: drone shots keep their constant speeds (`reveal` and `drift` exist but nothing assigns them), and the
1.25x judder on 25p drone moves is left for `"drone_speed": [4.0, 1.0]` or `[4.0, 2.0]` in reel.json.
