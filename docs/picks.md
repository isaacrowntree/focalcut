# The editor's picks

A recap reel is a loop between the planner and an editor. The planner cuts a first reel from the analysis. The
editor watches it, marks the shots they want and the ones they don't, and the planner cuts again. **The editor's
picks always win.** No rule the planner follows (fair share, framing, rhythm, coverage) overrides a pick.

## The pages

```bash
focalcut reel serve "/footage/Show 2026"        # http://localhost:8765/
```

- **Sources** (`review/sources.html`) has three views, each a list on the left and the picture with its tools on
  the right (`Tab` moves between them, `?` lists every key, and the **Reel** switch in the header chooses the reel
  you are working on):
  - **Clips** lists every export clip in a big player over a wide, zoomable timeline. The timeline shows a
    filmstrip, the moments the planner found and where the reel uses the clip. Mark as many in-out selections per
    clip as you like, star them, tick the reels they are for, ban stretches.
  - **Cut & picks** is the cut shot by shot, landscape or portrait, with your picks in place: trim, slip, star,
    make one the opening or the ending, drop shots one at a time or many at once.
  - **Frame portrait** is the reel's portrait cut shot by shot with the 9:16 crop over the picture and the cropped
    result beside it. **Drag** the frame to pan the crop; **click a dancer** to have the crop follow them through
    the shot; `Backspace` gives the shot back to the planner. The shot's state (the planner's framing, your pan,
    following) shows in the header and in the list. `sources.html#frame=<reel>` opens it on a reel.
- **Review** (`review/<cut>.html`) shows one frame per shot of the rendered reel, with **Ban**, **More** and a
  **Scrub** button that opens the clip it came from (portrait pages also link each shot to its framing).

Served, the pages save `focalcut/reel/picks.json` on every change. The previous file is kept as `picks.json.bak`
and a copy every ten minutes of editing in `picks_history/`. Several pages can be open on one project: a page
saves only onto the file it last saw, and merges with what another page changed meanwhile. `Ctrl+Z` undoes one
change at a time (`Ctrl+Shift+Z` redoes). Opened as plain files, the pages keep changes in the browser and offer
Copy and Download (in **More**).

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
