"""The picture's movement follows the song's: music.movement, motion.py, sync.py, the planner's passes, no_kissing."""
import json

import numpy as np

from focalcut.reel import motion as MO
from focalcut.reel import music as MU
from focalcut.reel import sync as SY
from focalcut.reel.plan import _starts, stage_holds
from focalcut.reel.project import Project


def song():
    """16 driving beats, everything stops for two, it comes back in; then a calm passage with a phrase closing."""
    level = [1.0] * 16 + [0.3, 0.3] + [1.0] * 8 + [0.45] * 8 + [0.15] + [0.45] * 3
    kick = [1.0] * 16 + [0.05, 0.05] + [1.0] * 8 + [0.1] * 12
    return np.array(level), np.array(kick)


def test_a_stop_is_where_the_level_falls_away_and_a_burst_where_it_comes_back():
    level, kick = song()
    mv = MU.movement(level, kick, calm=kick < 0.4)
    assert (16, 18, 1.0) in mv["stops"]                      # the drop-out from a driving passage: a stop
    assert (34, 35, 0.6) in mv["stops"]                      # a deep fall inside a calm one: a close
    assert mv["bursts"] == [18]
    assert mv["move"][16] == "stop" and mv["move"][18] == "burst" and mv["move"][34] == "close"
    assert (26, 30, 1.0) in mv["stops"]                      # the groove closing down into the calm passage, and
    assert mv["move"][4] == "pulse" and mv["move"][31] == "glide"   # past its four beats the music is simply calm


def test_one_soft_beat_in_a_groove_is_not_a_stop():
    level = np.array([1.0, 0.9, 1.0, 0.55] * 8)              # a bar's quiet fourth beat, over and over
    mv = MU.movement(level, np.array([1.0, 1.0, 1.0, 0.3] * 8), calm=np.zeros(32, bool))
    assert mv["stops"] == [] and mv["bursts"] == []


def test_feel_gives_holds_through_stops_and_falls_back_without_the_fields():
    level, kick = song()
    n = len(level)
    m = dict(beats=list(np.arange(n) * 0.5), start=0.0, end=n * 0.5, beat=0.5, beat_energy=list(level),
             beat_kick=list(kick), beat_drive=list(np.clip(kick, 0, 1)), beat_section=["groove"] * n,
             beat_sustain=[0] * n, beat_accent=[1.0] * n)
    F = MU.feel(m)
    assert (15, 18) in F["holds"]                            # from the last beat that sounded until it sounds again
    assert F["texture"][0] > 0.8 > 0.3 > F["texture"][30]    # (no beat_texture: from the drive and the kick)
    old = MU.feel(dict(beats=m["beats"], start=0.0, end=n * 0.5, beat=0.5))
    assert old["stops"] == [] and old["holds"] == [] and not old["known"]


def half(k):
    """A clock at 120 BPM: beat k in seconds."""
    return 0.5 * k


def series(v, rate=25):
    v = np.asarray(v, float)
    return dict(t=np.arange(len(v)) / rate, v=v, imp=np.clip(-np.diff(v, prepend=v[:1]), 0, None), rate=rate)


def test_impact_is_the_directograms_deceleration():
    dg = np.zeros((20, MO.BINS), np.float32)
    dg[:10, 2] = 0.01                                        # moving one way, then stopped dead
    v, imp = MO.impact(dg)
    assert v[5] > 0.2 and v[15] == 0
    assert int(np.argmax(imp)) in (9, 10, 11) and imp[5] == 0


def test_settle_and_burst_and_punch():
    land = series([0.3] * 50 + [0.01] * 50)                  # two seconds of movement, then rest
    assert MO.settle(land, 2.0) > 0.8 and MO.burst(land, 2.0) < -0.8
    assert MO.settle(land, 1.0) < 0.1
    flow = series(0.2 + 0.02 * np.sin(np.arange(100) / 8))   # a sway: it never stops
    hits = series(([0.5] * 3 + [0.0] * 3) * 17)              # pops: it stops six times a second
    assert MO.punch(flow, 0, 4) < 0.2 < 0.8 < MO.punch(hits, 0, 4)
    assert MO.punch(series([0.0] * 100), 0, 4) == 0.0         # stillness is soft


def test_fit_wants_the_landing_on_the_stop():
    level, kick = song()
    F = dict(stops=[(16, 18, 1.0)], bursts=[18], texture=np.clip(kick, 0, 1), level=level, drive=np.clip(kick, 0, 1))
    ctx = SY.context(F, 14, 4, half)                            # a shot over beats 14..18: the stop is a second in
    assert ctx["events"] == [(1.0, "rest", 1.0, 1.0)]
    M = series([0.3] * 100 + [0.01] * 50 + [0.3] * 100)      # the dancer lands at 4.0 s and rests until 6.0
    on, early, never = (SY.fit(M, a, 1.0, ctx) for a in (3.0, 4.5, 7.0))
    assert on["events"][0] > 0.9 and on["score"] > early["score"] and on["score"] > never["score"] + 0.3
    go = SY.context(F, 18, 2, half)                             # the shot on the entry starts moving
    assert go["events"][0][1] == "go"
    assert SY.fit(M, 6.0, 1.0, go)["events"][0] > SY.fit(M, 4.2, 1.0, go)["events"][0] + 0.5


def test_score_counts_the_stops_the_picture_rests_on():
    level, kick = song()
    F = dict(stops=[(16, 18, 1.0)], bursts=[18], texture=np.clip(kick, 0, 1), level=level, drive=np.clip(kick, 0, 1))
    M = series([0.3] * 100 + [0.01] * 50 + [0.3] * 100)
    shots = [dict(beats=14, unit=dict(id="a"), src=0.0), dict(beats=4, unit=dict(id="a"), src=3.0),
             dict(beats=4, unit=dict(id="a"), src=6.0)]
    sc = SY.score(shots, F, lambda k: 0.5 * k, lambda u: M, lambda s: (s["src"], s["beats"] * 0.5))
    assert sc["stops"] == [1.0, 1.0] and sc["entries"] == [1.0, 1.0]
    shots[1]["src"] = 7.0                                    # the same cut with the dancer still going on the stop
    assert SY.score(shots, F, lambda k: 0.5 * k, lambda u: M, lambda s: (s["src"], s["beats"] * 0.5))["stops"][0] == 0


def test_stage_holds_borrows_a_beat_from_further_off():
    L, lo, hi = [2, 2, 2, 2, 3, 2], [2, 2, 2, 2, 2, 1], [2, 4, 2, 2, 3, 2]
    out, n = stage_holds(L, lo, hi, [(3, 6)])                # a cut on beat 4, inside the hold; neighbours at their least
    assert n == 1 and out == [2, 4, 2, 2, 2, 1]              # the second shot plays on until the music sounds again:
    assert not any(3 < p < 6 for p in _starts(out)[1:])      # the two shots that can spare a beat give one each
    assert stage_holds(L, lo, [2] * 6, [(3, 6)]) == (L, 0)   # nobody can take it: as it was


def test_no_kissing_drops_the_picks_that_play_a_kiss_and_bans_every_kiss(tmp_path):
    d = tmp_path / "focalcut" / "reel"
    d.mkdir(parents=True)
    (d / "reel.json").write_text(json.dumps({"cuts": {"short": {}, "insta": {"picks_as": "short", "no_kissing": True}}}))
    (d / "units.json").write_text(json.dumps([dict(id="C1", export_t0=10.0), dict(id="C2", export_t0=0.0)]))
    (d / "picks.json").write_text(json.dumps({"C1": {"add": ["0:01.00-0:02.00", "0:05.00-0:06.00", "0:07.5"]}}))
    (d / "kisses.json").write_text(json.dumps({"spans": {"C1": [[14.8, 17.6, 0.99]], "C2": [[3.0, 4.0, 0.98]]}}))
    short, insta = Project(str(tmp_path), "short"), Project(str(tmp_path), "insta")
    assert short.picks()["C1"]["add"] == ["0:01.00-0:02.00", "0:05.00-0:06.00", "0:07.5"] and short.kiss_spans() == {}
    p = insta.picks()
    assert p["C1"]["add"] == ["0:01.00-0:02.00"]             # (source 14.8-17.6 is 4.8-7.6 of the export file)
    assert insta.kissed == [("C1", "0:05.00-0:06.00"), ("C1", "0:07.5")]
    assert p["C1"]["ban"] == ["4.80-7.60"] and p["C2"]["ban"] == ["3.00-4.00"]   # the planner's own shots stay out too


def test_no_kissing_without_kiss_spans_keeps_clear_of_each_kiss_moment(tmp_path):
    d = tmp_path / "focalcut" / "reel"
    d.mkdir(parents=True)
    (d / "reel.json").write_text(json.dumps({"no_kissing": True}))
    (d / "people.json").write_text(json.dumps({"people": [], "kisses": {"C1": [[20.0, 0.99], [40.0, 0.3]]}}))
    assert Project(str(tmp_path)).kiss_spans() == {"C1": [(18.5, 21.5)]}


def test_kiss_spans_join_runs_and_need_a_sure_frame():
    from focalcut.reel.people import spans_of
    t = np.arange(30) / 3 + 1 / 6
    q = np.zeros(30)
    q[3:9] = [0.6, 0.99, 0.9, 0.2, 0.2, 0.8]                 # one kiss with a gap of two frames in it
    q[20:22] = 0.9                                           # close dancing the model is not sure of
    sp = spans_of(t, q)
    assert len(sp) == 1 and sp[0][2] == 0.99 and sp[0][0] == 0.5 and sp[0][1] == 3.5


def _cast_reel(monkeypatch, a_moments=(), act_b="night"):
    """Five four-beat shots at 120 BPM; the music stops half a second into the last one (beats 17-19). A, in that
    slot, never stops moving (another moment of its clip, at 21 s, lands at 20.5 s); B (slot 1) lands at 10.5 s of
    its clip and rests; C, D, E keep moving."""
    from focalcut.reel import plan as PL
    from focalcut.reel import shake
    go = series([0.3] * 750)
    rest = {"A": series([0.3] * 512 + [0.005] * 238), "B": series([0.3] * 262 + [0.005] * 488)}
    monkeypatch.setattr(MO, "series", lambda project, u: rest.get(u["id"], go))
    monkeypatch.setattr(shake, "series", lambda project, name: None)
    n = 20
    F = dict(known=True, stops=[(17, 19, 1.0)], bursts=[], texture=np.full(n, 0.5), level=np.ones(n),
             drive=np.ones(n))
    monkeypatch.setattr(PL.MU, "feel", lambda music, whole=False: F)

    def unit(k):
        return dict(id=k, clip="clip" + k, role="dance", t0=0.0, t1=30.0, score=k, source=k + ".mp4", tag="",
                    category="floor")
    shots, recs = [], []
    for i, (k, t) in enumerate([("C", 5.0), ("B", 11.0), ("D", 5.0), ("E", 5.0), ("A", 5.0)]):
        shots.append(dict(unit=unit(k), t=t, kind="pattern", rank=0.5, face=0.5, framing=0.5, thumb=None, fx=[],
                          effect=None, act=act_b if k == "B" else "night", beats=4, beat=0.5, length=2.0,
                          src_in=t - 1.0, peak=1.0, rec_in=2.0 * i, crop=None, jitter=None))
        ms = [dict(t=t, kind="pattern", score=0.5)] + [dict(t=x, kind="pattern", score=0.5) for x in (a_moments if k == "A" else ())]
        recs.append(dict(id=k, unit=unit(k), role="dance", orient="landscape", moments=ms))
    cfg = dict(validate=False, shake=[1.0, 2.0], late=[], day=[], day_shows=[], night_shows=[])
    return PL, shots, recs, cfg, dict(beat=0.5, beats=[], start=0.0)


def test_motion_cast_trades_footage_so_the_picture_rests_on_the_stop(monkeypatch):
    PL, shots, recs, cfg, music = _cast_reel(monkeypatch)
    lines = PL.motion_cast(None, "landscape", shots, music, cfg, recs, W={})
    assert len(lines) == 1 and "traded" in lines[0]
    assert [s["unit"]["id"] for s in shots] == ["C", "A", "D", "E", "B"]
    assert abs(shots[4]["src_in"] + 0.5 - 10.5) <= 0.25     # B lands as the music stops, half a second in
    assert all(s["beats"] == 4 and s["length"] == 2.0 for s in shots)   # the cuts stay where they were


def test_motion_cast_is_deterministic_and_keeps_to_the_casting(monkeypatch):
    import copy
    PL, shots, recs, cfg, music = _cast_reel(monkeypatch)
    again = copy.deepcopy(shots)
    PL.motion_cast(None, "landscape", shots, music, cfg, recs, W={})
    PL.motion_cast(None, "landscape", again, music, cfg, recs, W={})
    assert shots == again
    PL, shots, recs, cfg, music = _cast_reel(monkeypatch, act_b="climax")   # B is another act's: no trade
    assert PL.motion_cast(None, "landscape", shots, music, cfg, recs, W={}) == []
    PL, shots, recs, cfg, music = _cast_reel(monkeypatch)
    shots[4]["add"], shots[4]["span"] = True, [4.0, 6.0]     # the editor's pick on the stop: as they chose it
    assert PL.motion_cast(None, "landscape", shots, music, cfg, recs, W={}) == []


def test_motion_cast_takes_another_moment_of_the_slots_own_unit(monkeypatch):
    PL, shots, recs, cfg, music = _cast_reel(monkeypatch, a_moments=(21.0,), act_b="climax")
    lines = PL.motion_cast(None, "landscape", shots, music, cfg, recs, W={})
    assert len(lines) == 1 and "its moment at 21.0 s" in lines[0]
    assert shots[4]["unit"]["id"] == "A" and abs(shots[4]["src_in"] + 0.5 - 20.5) <= 0.25   # it lands on the stop
    PL, shots, recs, cfg, music = _cast_reel(monkeypatch, a_moments=(21.0,), act_b="climax")
    shots[4]["unit"]["bans"] = [[19.0, 23.0]]                # ... never into the editor's ban
    assert PL.motion_cast(None, "landscape", shots, music, cfg, recs, W={}) == []


def test_motion_cast_brings_in_a_unit_only_where_no_unit_leaves(monkeypatch):
    import copy
    PL, shots, recs, cfg, music = _cast_reel(monkeypatch)
    shots[1] = dict(shots[1], unit=shots[3]["unit"], t=15.0, src_in=14.0, act="night")   # B is not in the reel
    one = copy.deepcopy(shots)
    assert PL.motion_cast(None, "landscape", one, music, cfg, recs, W={}) == []   # A's one shot: A stays
    shots[0] = dict(shots[0], unit=shots[4]["unit"], t=25.0, src_in=24.0)        # A has another shot
    lines = PL.motion_cast(None, "landscape", shots, music, cfg, recs, W={})
    assert len(lines) == 1 and "B (not in the reel yet)" in lines[0]
    assert shots[4]["unit"]["id"] == "B" and abs(shots[4]["src_in"] + 0.5 - 10.5) <= 0.25
