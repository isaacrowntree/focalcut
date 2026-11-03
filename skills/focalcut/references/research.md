# Research behind the concepts

## Rhythm and cutting to music
- Is there a "language of music-video clips"? arXiv 2108.00970 covers 548 official music videos plus interviews
  with editors:
  - In dance videos the most common shot lasts about one beat.
  - Editors cut with anticipation, just before the downbeat.
  - Editing speeds up near climaxes and slows at bridges.
- Larry Jordan on montages set to music (nofilmschool.com): clips of one length make the rhythm predictable; cut
  to secondary beats as well.

## Velocity edits and speed ramps
- capcut.com/resource/how-to-do-velocity-on-capcut and capcut.com/tools/speed-ramp: curve presets (Montage, Hero,
  Bullet, Jump Cut, Flash In, Flash Out) and custom curves.
- The nemovideo CapCut slow-motion blog, Nerdschalk and FlexClip velocity guides: speed up into the move, slow on
  the beat or pose, ease out.
- Creative COW time-remap thread and AEJuice: ease the ramps (no linear corners), and land the slow part on the
  action's resolution.

## Transitions
- Insta360, Nikon "Cut to the chase", Uppbeat, eventplanner (aftermovies), Flixier and Lightworks:
  - Hard cuts carry rhythm.
  - Motivated transitions (match action, whip on camera motion, zoom into a detail) beat decorative ones.
  - Keep one consistent style per piece.
- A beat-sync editing guide: flashes on big hits only; glitches suit electronic or DJ moments.

## Subject effects
- nemovideo neon template, MagicShot Energy Aura and Fire Flair: one strong subject effect on the drop, not
  throughout.
- capeditcut's list of CapCut mistakes: too many effects, random transitions and over-speeding people.

## Footage and models
- SAM 3 (facebook/sam3), BiRefNet (arXiv 2401.03407), Depth Anything V2, SigLIP 2, RT-DETR and ViTPose:
  subject, saliency, nearness, semantics, people and poses.
- vidstab (ffmpeg vidstabdetect / vidstabtransform) for stabilising.

## Glitch
- How each style is built: Boris FX Sapphire docs (S_DigitalDamage, DissolveDigitalDamage, DissolvePixelSort), Red Giant
  Universe VHS Transition, AEJuice (RGB split), School of Motion and Motion Array (slice shift, glitch transitions),
  Epidemic Sound and Born Digital (datamoshing), the `pixelsort` package (thresholds), the AV Artifact Atlas (VHS
  head-switching noise), crt-lottes (CRT curvature and bloom), CapCut (glitch filters, beat shake).
- Taste: keep it a look, not a gimmick. Keep 80–90 % hard cuts, fire a heavy glitch only where a musical event meets a
  motion event, never the same glitch on two cuts in a row, and leave a clean stretch after each heavy one. The
  research proposed four looks (`fx/glitch.py` LOOKS); `focalcut fx lookbook` renders them on the reel's own footage.

## Carnival
- What the genre's edits share (CapCut and Pippit carnival and Latin festival recap templates, PremiumBeat and FlexClip
  on whip pans and confetti, reggaeton editing briefs): beat-synced hard cuts, whip pans between dancers, confetti
  cannons and bursts on the big moments, saturated tropical colour, animated text over Latin-inspired graphics. 90s
  merenhouse videos (Proyecto Uno, Remezcla's merenhouse retrospective) add pop-art freezes, halftone print and
  mirror-ball club light.
- Taste: colour, not damage. Never white or black flashes (that is the EDM look); a colour flash or a strobe at most
  once a beat. Keep faces clean (light goes into the dark parts of the frame), save confetti for the climax's biggest
  move, and keep most cuts hard. `fx/carnival.py` has three looks (carnaval, tropical, disco), each with MilkDrop presets
  that suit it (fireworks, flowers, disco, glowsticks, fire); `focalcut fx lookbook <folder> --library carnival`.

## Style libraries
- `fx/looks.py` merges the libraries: glitch (digital damage), carnival (Latin street party / tropical disco), edm
  (MilkDrop and white/dark flickers, Electric Nation). reel.json "look" names one look; its "overlay" settings are the
  default under reel.json's own "overlay".
