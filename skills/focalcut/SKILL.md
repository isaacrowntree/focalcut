---
name: focalcut
description: Cut dance and event footage into beat-synced clips with focalcut, using the editing concepts it encodes (velocity edits, transition grammar, portrait framing that follows a dancer and favours faces, shake measurement and stabilising, band inserts, card-shuffle coverage, music-video rhythm, a story in acts, a chosen ending shot). Use it for any recap reel, teaser, highlight, promo or social clip cut from dance, festival, concert or event footage, and whenever someone asks for speed ramps, CapCut-style velocity, beat-synced cuts, 9:16 reframing of 16:9 footage, or an FCP7 / Premiere XML of an edit. It also covers splitting footage by focal subject (`focalcut library`, `cut`, `export`).
---

# focalcut

focalcut is MIT-licensed at github.com/isaacrowntree/focalcut; paths below are relative to a clone of it (e.g.
`~/focalcut`). Everything below is code in that repo.
Footage folders hold only data and settings: `export/`, camera files and `focalcut/reel/reel.json`. Never copy code
into a footage folder. If an event needs different behaviour, add a reel.json key and leave the code general.

```bash
cd <repo> && python -m venv .venv && .venv/bin/pip install -e ".[reel]"   # once
.venv/bin/focalcut --help
```

## What every deliverable includes

- **Rendered video:** 1080p 50 fps, music at −14 LUFS.
- **FCP7 XML for Premiere:** the editor finishes there. Rendered shots go on V1, untouched sources on a disabled V2.
- **Review page:** `review/<cut>.html` shows one frame per shot with a copyable picks line. Open it in a
  browser (on WSL, for example: `explorer.exe "$(wslpath -w review/<cut>.html)"`).
- **Editor's picks:** the review page's **Scrub source** opens each shot's export video. The editor marks extra
  shots (`A`), swaps a shot to the playhead (`S`), bans it, or asks for more from that video. The source browser
  (`review/sources.html`) shows every export and the whole cut, takes in-out picks and sets portrait focal points.
  `focalcut reel serve <folder> [<folder> ...]` serves them all from one port (each project at `/p/<slug>/`,
  an index at `/`) and the pages write `focalcut/reel/picks.json` directly; it takes `ban`, `prefer`, `add`,
  `more`, `portrait`, and `open` / `end` (the selection the reel opens on, under the logo, and ends on: the pick
  editor's Opening and Ending buttons), with times in each export file's own timeline. Picks win over the planner's
  shots, so a heavily picked reel is mostly picks; reel.json `"keep": true` keeps the last rendered cut's footage
  for clips the editor left alone. A reel the editor has picked outright (a short cut of their selections) takes
  reel.json `"mode": "picks"` (or the same inside a cut): every pick once, whole, in the story's order, trimmed only
  when they overrun the reel (the least starred first), B-roll they left alone filling the rest, the same picks in
  every format, no coverage passes (docs/picks.md). Otherwise, when the picks are too many or too long, reel.json
  `plan.<fmt>.pick_hold` (beats, e.g. 4 landscape, 3 portrait) plays each pick's best few beats instead, condenses
  a clip with many picks to its best ones, keeps the picks to about half the reel (`pick_share`) and puts them
  between shots from the rest of the footage, picked clips included. Re-run `--steps plan,render,xml,review` after
  any change.

Several reels from one shoot: reel.json `"cuts": {"night": {"formats": ["portrait"], "footage": ["@portrait", "band"],
"open": {}, "plan": {"portrait": {"acts": [...], "ending": null}}}, "day": {"footage": ["@landscape", "!band"]}}`, then
`focalcut reel run <folder> --cut night --steps plan,render,xml`. A cut reuses the project's analysis; acts it has no
footage for can be given zero bars.

## Three ways to make a clip

1. **A full recap reel** (landscape + portrait, a story in acts):
   `focalcut reel init <folder> --song x.mp3 …`, then `focalcut reel run <folder>`.
   Steps: units, score, shake, focus, subject, semantic, moments, faces, people, music, plan, render, xml, review.
   Each step caches, so `--steps plan,render,xml,review` re-cuts in minutes. Full guide:
   `docs/recipes/recap-reel.md`.
2. **Any other cut** (teaser, a player's highlight, a band promo, a 15 s story) from an analysed project:
   - Build the sequence by hand with `plan.shot()`, or with `plan.Picker` for fair sharing between clips.
   - Run it through `plan.place()`. It does beat timing, velocity landings, reframing, shake and stabilise flags,
     and transitions by grammar.
   - Save it as `plan_<cut>.json`.
   - Call `render.run` / `xml_export.run` / `review.run` with `cut=<cut>`. Output is `out/<name>_<cut>.mp4`.
   Worked example: `skills/focalcut/examples/teaser.py <folder> --bars 8` opens on a band player, cuts faces,
   hair flips and turns on the song's loudest phrase, and ends on the project's ending shot. Copy it and change
   PATTERN.
3. **One clip, one effect:**
   `focalcut fx apply clip.mp4 -e hero --peak 1.2 [--closeup] [--cap 1.5]`. Also `focalcut fx list` and
   `focalcut fx gallery clip.mp4`.

A different song for a new cut: `music.analyse(song, start_hint=…, max_len=…)` returns the dict `place()` takes.
Pass `song=` to `place()` and the render and XML use it.

## The concepts (each is its own piece of code)

Read `references/concepts.md` for the rule, the reason, the numbers and the code behind each. In brief:

| Concept | The rule | Code | reel.json `plan.<fmt>` key |
|---|---|---|---|
| Opening | festival B-roll (band, crowd) with the event logo centred, never dance first | `plan.story` teaser | `opening`, top-level `logo` |
| Story in acts | teaser, arrival (day), shows, night, climax, finale on the song's sections | `plan.story`, `default_acts` | `acts`, `split`, `day`, `day_shows`, `night_shows` |
| Use everything / card shuffle | every export file appears in each format it belongs to (hard requirement: more cuts, never a file dropped), several scenes each, spread out and interleaved, never a block | `Picker`, `cover`, `weave`, `cover_all` | — |
| Cut on the action | turn→turn, travel keeps direction, a movement carries 2–3 cuts | `flow.py`, `flow_link` | — |
| No same-y runs | static footage gets one long shot; never near-identical neighbours | `weave` | — |
| Music-video rhythm | 8-beat phrases of varied lengths, peaks on beats, cuts 2 frames early, beat-exact frames | `rhythm`, `in_point`, `ANTICIPATE` | `night_pace` |
| Subject | the salient, near, large dancer (+ partner/line), not just the sharpest | `subject.py`, `frame.track` | — |
| Moments | middle of routines; spins, dips, tricks, shakes, close-ups; dancing the whole shot | `moments.py` | — |
| Velocity edits | fast through the move (≤1.5x dancing) → ease → slow on the landing → hold → ease up | `velocity.settle/landing`, `fx/speed.py` | `heroes` |
| Show the landing | every dance shot keeps the end of the move and about a beat of rest after it lands, borrowing a beat from a neighbour if needed | `velocity.landed/hold_in`, `plan.lands` | `land_hold` |
| Transition grammar | mostly hard cuts; each act's own short vocabulary; a reason for every transition | `grammar.py`, `plan.transitions` | `grammar`, `caps`, `flash_at` |
| Cohesive effects | one effect at a time, one subject effect (on the drop), effects target the subject | `give_slot_fx`, `fx/subject.py` | — |
| Portrait framing | the crop is on the action (pose activity), never a bystander; a couple shares the frame, the follow featured (role step) and kept whole; group routines whole; faces (YuNet) strongly favoured; no backs | `frame.follow/couple/activity/face_seen`, `role.py`, `plan.reframe` | — |
| Moments that play | the subject walking or standing, a crowd crossing the crop, someone blocking the camera: switched for a better moment | `plan.plays`, `plan.assess`, `spot_rater` | — |
| Shake | measure; too shaky → another part of the clip; mildly shaky → stabilise | `shake.py`, `plan.assess`, `render.stab` | `shake` |
| Band members | individual players cut in on the hardest musical hits, only where a player is on screen and sharp | `plan.band_in`, `role.musician` | `members`, `member_clips`, `band_in` |
| Style look | one look for the film from a style library (glitch: digital damage; carnival: Latin carnival / tropical disco; edm: MilkDrop + white/dark flickers): its transitions, subject effects, accents and MilkDrop presets; chosen from the look book | `fx/looks.py` (`glitch.py`, `carnival.py`, `edm.py` LOOKS), `plan.glitch_look`, `fx/lookbook.py --library` | `look` (its `overlay` sits under reel.json's own) |
| Ending | a chosen last shot that runs out exactly at its end time; its stretch reserved | `plan.ending_record/reserve_ending` | `ending` |
| Song start | start on the bar nearest where the song proper begins | `music.analyse` | top-level `start_hint` |
| Several songs | one song per act (a festival's sets), each around its drop, mixed like a DJ: tempo ramp, 4 bars beat on beat, bass swap on the downbeat | `music.join/mix`, `plan.clock` | top-level `songs`, `mix` |
| Headliner last | the last set's footage kept for the climax and finale | `plan.story` | `late` |
| EDM light show | MilkDrop-style presets (tunnels, mandala, kaleidoscope, lasers, particles, plasma, flame) in changing palettes every 8 bars, beat blinks, drop bursts; ≤1 full flash per beat | `overlay.py` | top-level `overlay` |
| Everyone in it | every person found by face (YuNet + SFace, grouped across clips) and every kiss (SigLIP 2) gets a shot; the moment showing the most missing people goes in first, splitting long shots when needed | `people.py`, `plan.cover_people` | — (the `people` step) |
| Clean masks | BiRefNet matte inside the SAM 3 mask, never raw upscaled SAM | `fx/masks.py` | — |
| Pick rules | never a back, a split-up group, the crowd, a blurry or bored face, or a weaker moment of the same person; effects take in everyone or no one (references/concepts.md §19) | `windows.py`, `plan.snap_windows`, `window_report` | — |

## Before rendering, check the plan against the rules

The planner prints a summary per act: shots, transitions, effects. Look at `plan_<cut>.json`, or render with
`--draft` and open the review page, and check:

- The summary's **picks** line says "all in once, at full length, bans held". Anything it lists (a pick missing,
  repeated, short, sped up or overlapped; a ban not held) gets fixed before rendering (`plan.pick_report`).
- Every export file is present in each format it belongs to (the summary's `units` reads n/n); no clip cuts to
  itself; no B-roll sits next to B-roll.
- Dance shots with a landing show the rest after it (`land`, `land_fits` in the plan).
- Transitions are few (about 15 in 3 minutes) and each has a `why`; none are back to back.
- Speed effects never sit side by side; there is one subject effect; dancing is never above 1.5x.
- Portrait: `crop.face` is high on social-floor shots, there are no backs, and `inside` is near 1.
- `jitter` is below the second `shake` threshold except on B-roll that is shaky throughout; those shots get `stab`.

## Event-specific settings go in reel.json, never in code

Example from the shoot the recipe was built on (a 2026 salsa congress):
`"start_hint": 17.0`, and `plan.<fmt>`:
- `"members": ["singer", "violin"]`
- `"member_clips": ["C2003"]`
- `"ending": ["C2023", 126.6, 133.4, 0.45]`
- `"heroes": 3`
- `"day": ["class", "stage-setup", "panel"]`
- `"day_shows": ["demos"]`
- `"night_shows": ["animations"]`
- `acts` on the song's sections

Plus `proxies` for the DNxHD stabilised masters.

## Practical notes

- **Python environment:** ML runs from the repo's `.venv` with the `reel` extra. Set `FOCALCUT_DISABLE_CUDNN=1` if
  cuDNN engines are broken.
- **Decoding and encoding (docs/hardware.md):** Sony 10-bit 4:2:2 HEVC can't use NVDEC before Blackwell, and
  Mesa's WSL VA-API bridge has no 4:2:2 decode. On WSL the render decodes it on the Intel iGPU with Quick Sync via a
  Windows ffmpeg (`C:\tools\ffmpeg-*\bin\ffmpeg.exe` or `FOCALCUT_WIN_FFMPEG`; needs `-async_depth 1` or cuts land
  3 frames early), encodes with NVENC, and runs shots in parallel (`FOCALCUT_DECODERS="qsv=5,cpu=2"`). Analysis
  still decodes on the CPU; the shake pass decodes the H.264 exports on the GPU.
- **Faster analysis (option):** the score step decodes every 4K source frame on the CPU (about 1.6x real time).
  reel.json `proxies` can point each source at a 1080p H.264 copy that NVDEC decodes on the GPU, e.g.
  `{"C2150.MP4": ["export/dusk/C2150.mp4"]}`, adding `[[0, -t0]]` shifts for an export trimmed to start at t0. A proxy
  must cover the whole source, so a clip split across several exports needs one full-length 1080p copy of its own.
  Caveat: render also decodes from proxies, and a 9:16 strip from 1080p is only ~600 px wide (about 1,200 from 4K).
  Use proxies for the analysis steps, then remove the key before the render step.
- **GPU workers:** for example, two scoring workers suit a 12 GB GPU; more can overflow VRAM (on WSL it pages to RAM).
- **Long jobs:** run them in the background. Kill workers by PID, or with an anchored `pgrep -af '^/full/path'`;
  never an unanchored `pkill -f`, which matches the calling shell.
- **Tests:** `.venv/bin/python -m pytest -q` and `ruff check .` must stay clean.
