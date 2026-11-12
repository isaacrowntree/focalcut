# The editor's picks

A recap reel is a loop between the planner and an editor. The planner cuts a first reel from the analysis. The
editor watches it, marks the shots they want and the ones they don't, and the planner cuts again. **The editor's
picks always win.** No rule the planner follows (fair share, framing, rhythm, coverage) overrides a pick.

## The pages

```bash
focalcut reel serve "/footage/Show 2026"        # http://localhost:8765/
```

- **Sources** (`review/sources.html`) lists every export clip in a big player over a wide, zoomable timeline. The
  timeline shows a filmstrip, the moments the planner found and where each reel uses the clip. Mark as many in-out
  selections per clip as you like. In portrait it also shows the planner's crop and takes focal points you set. The
  Picks view plays the whole cut, landscape or portrait, with your picks in place.
- **Review** (`review/<fmt>.html`) shows one frame per shot of the rendered reel. Each frame has a copyable picks line
  and a **Scrub source** button that opens the clip it came from.

Served, the pages save `focalcut/reel/picks.json` on every change and keep the previous file as `picks.json.bak`.
Opened as plain files, they keep changes in the browser and offer Copy and Download.

## picks.json

The keys are export file names. Times are in that export file's own timeline.

```json
{
  "C2125": {
    "add": ["1:12.40-1:15.10", "5:32.9"],
    "ban": ["0:00-0:20"],
    "prefer": ["0:41-0:46"],
    "more": 2,
    "open": "1:18.6-1:22.8",
    "end": "6:01.0-6:05.5",
    "portrait": [[332.9, 0.50, 0.42], [335.3, 0.55, 0.42]]
  },
  "C1984-05_couple": {"ban": ["all"]},
  "_note": "keys starting with _ are notes"
}
```

| Key | What it does |
|---|---|
| `add` | An in-out range is one shot: from its in-point, for its length. A single time is an extra shot there. |
| `ban` | Keeps every shot out of the range; `"all"` keeps the clip out. |
| `prefer` | Moments in the range rank up by 1.0. If none was found there, one is made in the middle. |
| `more` | More shots from this clip: its fair-share discount is divided by 1 + `more`. |
| `open` / `end` | The in-out the reel opens on (under the event's logo) or ends on. |
| `portrait` | Portrait focal points, `[time, x, y]` with x and y from 0 to 1 of the frame. The strip centres on each point for 0.5 s either side. |

reel.json `"picks"` takes the same shape and merges over `picks.json`.

## How picks are kept

- **Once each, at real speed.** An in-out pick plays exactly once, at least as long as marked. It may run in slow
  motion but is never sped up, and no other shot uses its footage.
- **A picked clip gives only its picks.** The editor chose its best part, so the rest of that clip is banned.
  reel.json `plan.<fmt>.only_picks: false` keeps the rest in play between the picks.
- **A pick over a ban wins.** The plan log reports it as a conflict.
- **The pick check.** After every plan, `pick_report` lists any pick that is missing, short, sped up, doubled, or has
  another shot over its footage. A clean plan prints nothing.

### A reel cut from the picks: `"mode": "picks"`

When the editor has picked the reel (a 90-second cut made almost entirely of their selections, say), set
`"mode": "picks"` at the top of reel.json, or inside one of its `cuts`. The planner then cuts the reel from the
picks and nothing else needs setting:

- **Every pick plays once, whole** (to the nearest beat), and nothing is replayed, cut into pieces or made into
  quick cuts. A pick on a clip the moments step found nothing in still plays.
- **In the story's order.** The acts run in turn (day, the stage, the night, the headliner), and within an act the
  clips are woven through each other in shooting order, so no clip plays as a block; no clip cuts to itself, B-roll
  never sits beside B-roll, and runs of one-beat or equal-length shots are broken up.
- **More picks than reel:** the picks give up beats, each the same share of its length, the least starred first,
  and none goes under two beats (or its own length, if shorter). Stars keep a pick whole: an unstarred pick is
  trimmed before a starred one. A trimmed pick plays its clearest part. Only when every pick is down to two beats do
  picks go, the least starred, then the least clearly seen, and the plan log names each one. The log also says how
  many beats the picks ask for against how many the reel has, and roughly how many fewer picks would play whole.
- **More reel than picks:** B-roll from clips the editor left alone fills the rest, a few beats a shot. Never a clip
  they picked from or banned, and no people footage they didn't pick unless the B-roll runs out (the log says so).
- **The opening and ending are theirs.** It opens on the pick marked Opening (or reel.json `"open"`) under the logo,
  else on B-roll; `plan.<fmt>.open_with` names the picks that follow the logo and `end_with` those that lead into
  the ending; the last shot is `plan.<fmt>.ending` (or the pick marked Ending), or the pick named by `close`.
  `"ending": [file, from, to, "follow"]` frames a single performer the way a pick is framed, where a number (or
  nothing) holds the whole scene centred.
- **Every format plays the same picks,** in the same order at the same lengths: nothing in the cut looks at
  framing.
- **No coverage passes.** Nothing is brought in to cover a clip, a person or the band, and nothing is recast.

```json
"cuts": {"short": {"mode": "picks", "length": 97.2, "formats": ["landscape", "portrait"],
  "open": {"C2087_crowd-night": "0:24.00-0:28.00"},
  "plan": {"landscape": {"open_with": [["C2122", "0:30.62"]], "end_with": [["C2122", "2:37.23"]],
                         "ending": ["C2119_dj-topic-headline", 91.0, 93.9, "follow"]}}}}
```

The keys below (`pick_hold`, `pick_share`, `pick_clip_share`, and `pick_grow`, `pick_by`, `pick_whole`,
`pick_like`, `reuse`, `open_broll`) are for a reel the planner cuts with the picks mixed in. Picks mode accepts
them and does not need them.

A cut that says how it ends (`ending` or `end_with` in its own `plan.<fmt>`) no longer takes the project's `close`
as well.

### Portrait: a pick is framed on the faces

The editor picks a stretch for the faces in it. In a portrait strip cut from a landscape frame, the body tracker
can cut a face at the strip's edge, or sit on a back, a shoulder or the person beside them, so a pick's strip goes
to the faces first (`plan.face_first`, on the persons step's face scan):

- **The planner's subject keeps the shot while the strip shows a face.** The strip only re-centres, on the faces
  by it: their own face cut by its edge, a kiss at its side, the person right in front of them.
- **Only when it shows no face for a second on end** (the whole shot, if shorter) may it go to someone else, and
  only to a face shot close (13% of the frame's height, and a head's size for the person followed). A far-off face
  never pulls the strip off a dancer whose own is hidden behind hair or a mask.
- It stays with a person through a turn of the head, pans once between two friends who face the camera in turn,
  shares the strip with someone beside them who fits, and centres a face bigger than the strip.

The editor's focal points and Follow clicks still override it, shots the planner chose itself are framed as
before, and `plan.portrait.portrait_rules: {"pick_faces": false}` turns it off.

### When there are more picks than reel

An editor can pick more footage than the song holds. The planner then condenses the picks:

- **`plan.<fmt>.pick_hold`** (beats): a pick longer than this is cut down to its best `pick_hold` beats, scored on
  the scrub's windows. The editor chose the stretch; the planner chooses the best part of it.
- **`plan.<fmt>.pick_share`** (default 0.5): the share of the reel's beats the picks may take. The rest is the
  planner's own footage, for coverage and pacing.
- **`plan.<fmt>.pick_clip_share`** (default 1.5): how far above an even share one clip's picks may go. Raise it when
  the editor's picks bunch in a few clips; otherwise those clips lose picks to the cap.

A pick-heavy reel, as a profile:

```json
"profiles": {
  "picks-heavy": {"plan": {
    "landscape": {"pick_hold": 4, "pick_share": 0.9, "pick_clip_share": 4.0},
    "portrait":  {"pick_hold": 4, "pick_share": 0.9, "pick_clip_share": 4.0}}}
},
"use": ["picks-heavy"]
```

On Electric Nation 2026, with 132 picks bunched in a few long clips, this took the landscape reel from 90 to 127
picks kept.

## Review rounds

- **`reject`** (reel.json): stretches a review turned down, in picks syntax by export file. The planner treats them
  as bans, but they live apart from the editor's own `picks.json`.
- **`refill`** (reel.json, `true` = the last cut rendered, or a plan file name): re-cut by keeping every shot of an
  earlier plan and moving only those whose footage is banned now. Each moves along its own clip first, else onto
  another clip's footage that passes the rules. A review round then replaces only what was rejected, instead of
  reshuffling half the cut into shots nobody has looked at.

```bash
focalcut reel run "/footage/Show 2026" --steps plan,render,xml,review
```

## Profiles

reel.json `"profiles"` are named rule sets. The main reel, or any cut, stacks them with `"use"`, in order, then
applies its own keys. `"footage"` and `"ban"` add up across profiles, `"plan"` merges per format, and any other key
is taken from the last profile that sets it.

```json
"profiles": {
  "no-night": {"footage": ["!night"]},
  "short": {"length": 90}
},
"cuts": {
  "day": {"formats": ["landscape", "portrait"], "use": ["no-night", "short"]}
}
```

Keep an editor's standing rulings as profiles. The next event's reel can then reuse them by name.
