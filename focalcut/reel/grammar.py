"""Transition grammar: mostly hard cuts on the beat, and a transition only where the cut has a reason for it.

Music-video and aftermovie editors cut hard on the beat almost everywhere; a reel that cycles through a menu of
transitions reads as a template. So each part of the story (act) has its own short vocabulary, a cap on how often
each may appear, and every transition needs its reason in the two shots it joins:

  spin    a turn carried into the next turn (different clips)
  swipe   the same travel carried across the cut: both shots clearly moving the same way (flow direction)
  zoom    a push in to a face (the next shot a close-up, at least as tight)
  flash   a hit in the music: the beat's accent at least FLASH_AT for the act, on a beat the rhythm section plays
          (music.feel's "hit": a sung syllable or a sweep over a breakdown is an accent but not a hit)
  glitch  in or out of the DJ: the digital edge belongs to the booth
  fade    a dissolve between daytime B-roll scenes (both shots daytime footage: never into or out of the night)

The change of act gets its own signature transition (SIGNATURE). Never two transitions back to back, and none into
a crowd reaction or a band-member insert: those cut straight in. The picture follows the music: where it is calm
(no rhythm section: an intro, a breakdown) nothing flashes, whips or pushes (LOUD) -- an act changes on a dissolve
there, and a dissolve joins two held shots every CALM_EVERY beats or so. Everything here can be overridden per project in
reel.json's plan: {"grammar": ..., "caps": ..., "flash_at": ...}.
"""
from __future__ import annotations

# the change of act: each says something different about what comes next
SIGNATURE = {("teaser", "arrival"): "dipwhite", ("arrival", "shows"): "zoom", ("shows", "night"): "lightleak",
             ("night", "climax"): "velocity", ("climax", "finale"): "dipwhite"}
# each act's vocabulary, in order of preference: the day dissolves; the stage is hard cuts with turns and travel
# carried across; the night adds the DJ's glitch and the horn hits; the climax hits hardest; the finale goes quiet
GRAMMAR = dict(arrival=["fade"], shows=["spin", "swipe", "zoom"], night=["swipe", "spin", "glitch", "flash"],
               climax=["flash", "spin", "swipe"], finale=["zoom"])
CAPS = dict(arrival=dict(fade=3), shows=dict(spin=3, swipe=3, zoom=2), night=dict(swipe=3, spin=2, glitch=2, flash=2),
            climax=dict(flash=4, spin=2, swipe=3), finale=dict(zoom=1))
FLASH_AT = dict(night=1.8, climax=1.6)                   # how hard a beat must hit (accent) to earn a flash
# what the music must drive for: going to white, a rush, a whip, a push, a digital tear. Where it is calm (an intro,
# a breakdown: no rhythm section, music.feel) a cut that asks for one of these is a dissolve or stays a hard cut
LOUD = ("flash", "dipwhite", "lightleak", "velocity", "swipe", "spin", "zoom", "glitch")
CALM_EVERY = 16                                          # beats between the dissolves of a calm passage
TRAVEL = 0.12                                            # flow speed both sides of a cut need for a swipe
MIN_BEATS = 2                                            # both shots of a transition (a flash aside) run this long
# beats each side of the cut a transition takes (a shot keeps half a beat of its own); the looks' own too
HALF = {"flash": 0.25, "zoom": 0.25, "swipe": 0.25, "spin": 0.34, "fade": 0.5, "dipwhite": 0.75, "lightleak": 1.0,
        "glitch": 0.25, "g_tear": 0.375, "g_slices": 0.375, "g_blocks": 0.375, "g_signal": 0.25, "g_stutter": 0.25,
        "g_moshcut": 0.5, "g_vhsroll": 0.5, "g_sortwipe": 0.5,
        "c_whip": 0.25, "c_flash": 0.25, "c_stripes": 0.375, "c_sun": 0.375, "c_burst": 0.5}   # (fx/carnival.py)


def reasons(A, B, vocab, accent, flash_at, travel=None, daytime=None):
    """The transitions this cut has a reason for, in the act's order of preference: [(name, why), ...].
    A, B: the shots either side (kind, unit role/tag/clip, crop); travel(A, B) -> bool: the same movement carries
    across the cut; daytime(unit) -> bool: the footage was shot in the day (None: not known, every scene counts)."""
    a_k, b_k = A["kind"], B["kind"]
    broll = A["unit"]["role"] == "transition" or B["unit"]["role"] == "transition"
    za = (A.get("crop") or {}).get("z", 1); zb = (B.get("crop") or {}).get("z", 1)
    out = []
    for name in vocab:
        if name == "fade" and broll and (daytime is None or (daytime(A["unit"]) and daytime(B["unit"]))):
            out.append((name, "dissolve from one daytime scene to the next"))
        elif name == "spin" and a_k in ("spin", "pattern") and b_k in ("spin", "pattern") and \
                A["unit"]["clip"] != B["unit"]["clip"]:
            out.append((name, "a turn carried into the next turn"))
        elif name == "swipe" and not broll and travel is not None and travel(A, B):
            out.append((name, "the movement carried across the cut"))
        elif name == "zoom" and not broll and b_k == "closeup" and a_k != "closeup" and zb >= za:
            out.append((name, "push in to a face"))
        elif name == "flash" and accent >= flash_at:
            out.append((name, f"a hit in the music ({accent:.1f}x)"))
        elif name == "glitch" and "dj" in (A["unit"].get("tag", ""), B["unit"].get("tag", "")):
            out.append((name, "in or out of the DJ booth"))
    return out
