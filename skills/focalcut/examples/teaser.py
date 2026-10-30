"""A short teaser from any reel project, built from focalcut's concepts rather than the recap planner.

    python teaser.py <folder> [--bars 8] [--format portrait] [--cut teaser]

Needs the reel project's analysis (units, score, shake, subject, moments, faces, music). It opens on a band player,
then cuts faces, hair flips and turns on the song's most energetic phrase, with a hair flip in velocity slow motion
and the best turn in bullet time to close (or the project's `ending` shot). Writes plan_<cut>.json, renders
out/<name>_<cut>.mp4, a Premiere XML and a review page. Change PATTERN to make a different kind of clip.
"""
from __future__ import annotations

import argparse

import numpy as np

from focalcut.reel import plan as P
from focalcut.reel import render, review, xml_export
from focalcut.reel.flow import Flow
from focalcut.reel.project import Project

# (move kinds, beats, effects) per shot after the opening -- one 8-count phrase per line
PATTERN = [
    (("closeup",), 4, ["hero"]), (("spin",), 2, []), (("spin", "pattern"), 2, []),
    (("shake",), 1, []), (("dip", "trick"), 1, []), (("closeup",), 2, []), (("spin", "pattern"), 4, ["hero"]),
    (("energy", "shake"), 2, []), (("closeup",), 2, []), (("spin", "dip", "trick"), 4, []),
    (("closeup",), 2, []), (("spin",), 2, []), (("closeup",), 4, ["hero"]),
]


def strongest_phrase(music, bars):
    """Index of the bar where the loudest `bars`-bar stretch of the music window starts."""
    bt, e = np.array(music["beats"]), np.array(music["beat_energy"])
    first = int(np.argmin(np.abs(bt - music["start"])))
    last = int(np.argmin(np.abs(bt - (music["start"] + music["length"])))) - bars * 4
    starts = range(first, max(first + 1, last), 4)                  # on downbeats of the reel's own bar grid
    return max(starts, key=lambda i: e[i:i + bars * 4].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--format", default="portrait")
    ap.add_argument("--cut", default="teaser")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    pr, fmt = Project(a.folder), a.format
    cfg = P.config(pr, fmt)
    music, moments = pr.load("music.json"), pr.load("moments.json")
    recs = [r for r in moments if r["moments"] and (fmt == "portrait" or r["orient"] == "landscape")]
    P.reserve_ending(recs, cfg)
    P.assess(pr, recs, fmt, cfg)                                    # framing, faces and shake in every rank
    i = strongest_phrase(music, a.bars)
    music = dict(music, start=music["beats"][i])
    pk = P.Picker(fmt, recs)
    dance = [r for r in recs if r["role"] != "transition"]
    players = [r for r in recs if r["role"] == "transition" and (
        any(k in r["unit"].get("tag", "") for k in cfg["members"]) or r["unit"]["clip"] in cfg["member_clips"])]
    seq = [pk.take(*(pk.best(players) or pk.best(dance)), beats=4, act="climax")]   # open on the band
    seq[0]["band"] = True
    budget = a.bars * 4 - 4
    end = P.ending_record(recs, cfg)
    end_beats = 6 if end else 0
    for kinds, beats, fx in PATTERN:
        if sum(s["beats"] for s in seq) - 4 + beats > budget - end_beats:
            break
        used = {s["unit"]["clip"] for s in seq}                     # a different clip for every shot if possible
        x = next((c for c in pk.pool(dance, kinds, 0.4) if c[0]["unit"]["clip"] not in used), None) or \
            next((c for c in pk.pool(dance, kinds) if c[0]["unit"]["clip"] != seq[-1]["unit"]["clip"]), None) or \
            pk.best(dance)
        s = pk.take(*x, beats=beats, act="climax")
        s["fx"], s["effect"] = list(fx), (fx[0] if fx else None)
        seq.append(s)
    left = budget - (sum(s["beats"] for s in seq) - 4)
    if end:                                                         # the project's ending shot
        r, t = end
        near = min(r["moments"], key=lambda m: abs(m["t"] - t))
        s = pk.take(r, dict(near, t=round(t, 2), kind="crowd"), beats=left, act="climax")
        s["ending"] = True
    else:                                                           # the best turn in bullet time
        s = pk.take(*pk.best(dance, ("spin", "dip", "trick"), 0.45), beats=left, act="climax")
        s["fx"], s["effect"] = ["bullet"], "bullet"
    seq.append(s)
    fl = Flow(pr)
    plan = P.place(pr, fmt, seq, music, cfg, fl.direction)
    fl.save()
    pr.dump(f"plan_{a.cut}.json", plan)
    print(f"{a.cut}: {len(seq)} shots, {plan['video_length']:.1f}s from {music['start']:.1f}s in the song; "
          f"transitions {[s.get('transition') for s in seq if s.get('transition')]}, "
          f"effects {[f for s in seq for f in s['fx']]}")
    if not a.no_render:
        render.run(pr, fmt, cut=a.cut)
        xml_export.run(pr, fmt, cut=a.cut)
        review.run(pr, fmt, cut=a.cut)


if __name__ == "__main__":
    main()
