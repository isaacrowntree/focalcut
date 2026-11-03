"""Recap-reel recipe: units from an export folder, beat-exact planning, crop expressions, XML paths, CLI config.
No GPU, no models: scores and moments are synthesised."""
from __future__ import annotations

import copy
import json
import os
from collections import Counter
from itertools import pairwise

import numpy as np
import pytest

from focalcut import cli
from focalcut.reel import plan as P
from focalcut.reel import units as U
from focalcut.reel.music import window
from focalcut.reel.project import Project
from focalcut.reel.render import crop_expr
from focalcut.reel.xml_export import pathurl

from .conftest import _synth, needs_ffmpeg


@pytest.fixture
def project(tmp_path):
    """A footage folder with an export/ tree and a library.json, like a real shoot's layout."""
    folder = tmp_path / "Show 2026"
    folder.mkdir()
    for name, w, h in [("C100", 320, 180), ("C101", 180, 320), ("C102", 320, 180), ("C103", 320, 180),
                       ("C104", 180, 320)]:
        _synth(folder / f"{name}.MP4", 4, w, h, 50)
    _synth(folder / "C102_stabilized.mov", 4, 320, 180, 50)
    for cat, files in {"matinee": ["C100"], "segments": ["C101-01_couple", "C101-02_couple"],
                       "broll": ["C103_crowd", "C104-C100_floor"], "animations": ["C102"]}.items():
        (folder / "export" / cat).mkdir(parents=True)
        for f in files:
            (folder / "export" / cat / f"{f}.mp4").write_bytes(b"")
    (folder / "focalcut").mkdir()
    lib = {"clips": [{"name": "C101.MP4", "segments": [[0, 2.0], [2.0, 4.0]],
                      "labels": [{"subject": "couple"}, {"subject": "couple"}]}],
           "broll": [{"name": "C103.MP4", "file": "export/broll/C103_crowd.mp4", "span": [1.0, 3.0]},
                     {"name": "C104.MP4+C100.MP4", "file": "export/broll/C104-C100_floor.mp4"}]}
    (folder / "focalcut" / "library.json").write_text(json.dumps(lib))
    pr = Project(str(folder))
    pr.config = {"sources": {"C102": "C102_stabilized.mov"}, "rotate": {"C103": "transpose=2"},
                 "spans": {"C100": [0.5, 3.5]}}
    pr.save_config()
    return Project(str(folder))


@needs_ffmpeg
def test_units_map_export_folder_to_sources(project):
    units = {u["id"]: u for u in U.build(project)}
    assert set(units) == {"C100", "C101-01_couple", "C101-02_couple", "C103_crowd", "C104-C100_floor#1",
                          "C104-C100_floor#2", "C102"}
    seg = units["C101-02_couple"]
    assert (seg["role"], seg["source"], seg["t0"], seg["t1"], seg["orient"]) == ("dance", "C101.MP4", 2.0, 4.0, "portrait")
    assert units["C100"]["role"] == "show" and (units["C100"]["t0"], units["C100"]["t1"]) == (0.5, 3.5)   # span
    crowd = units["C103_crowd"]
    assert crowd["role"] == "transition" and crowd["tag"] == "crowd" and (crowd["t0"], crowd["t1"]) == (1.0, 3.0)
    assert crowd["rotate"] == "transpose=2" and crowd["orient"] == "portrait"        # rotated 90 degrees
    assert units["C104-C100_floor#1"]["clip"] == "C104" and units["C104-C100_floor#2"]["clip"] == "C100"
    assert units["C102"]["source"] == "C102_stabilized.mov" and units["C102"]["score"] == "C102"
    # where each export file starts in its source, for picks given in the export's own timeline
    assert seg["export_t0"] == 2.0 and crowd["export_t0"] == 1.0
    assert units["C100"]["export_t0"] == 0.0                     # a span trims the reel, not the export


def test_score_name_strips_stabilised_and_joined():
    assert Project.score_name("C1985_joined_stabilized.mov") == "C1985"
    assert Project.score_name("C2020.MP4") == "C2020"


def fake_reel(tmp_path, n_units=16, beat=0.6):
    """Moments + music for a plan with no score files (so no reframing): shows, socials and day / night B-roll."""
    pr = Project(str(tmp_path))
    moments = []
    tags = ["class", "crowd", "band", "panel"]
    for i in range(n_units):
        role = "transition" if i % 4 == 3 else "dance" if i % 4 == 2 else "show"
        cat = {"transition": "broll", "dance": "segments", "show": "matinee"}[role]
        u = dict(id=f"U{i:02d}", category=cat, role=role, tag=tags[(i // 4) % 4] if role == "transition" else cat,
                 clip=f"C{i:03d}", score=f"C{i:03d}", source=f"C{i:03d}.MP4", t0=0.0, t1=90.0, rotate=None,
                 orient="landscape", desc="")
        kinds = ["spin", "dip", "trick", "shake", "closeup", "energy", "pattern"]
        ms = [dict(t=5.0 + 6 * k, score=0.9 - 0.04 * k + 0.01 * i, kind="broll" if role == "transition" else kinds[k % 7],
                   quality=0.8, face=0.2, parts={}) for k in range(12)]
        moments.append(dict(id=u["id"], category=cat, role=role, orient="landscape",
                            interest=sum(m["score"] for m in ms[:5]), moments=ms, unit=u))
    beats = list(np.arange(0, 200, beat))
    music = dict(song="x.mp3", beats=beats, start=20.0, length=round(72 * 4 * beat, 3))
    return pr, music, moments


def test_plan_cuts_are_beat_exact_and_follow_the_acts(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    plan = P.build(pr, "landscape", music, moments, stinger_len=8.4)
    shots, beat = plan["shots"], plan["beat"]
    # cut frames follow the beat grid from the reel's start, so the total never drifts
    total_beats = sum(s["beats"] for s in shots)
    assert sum(s["frames"] for s in shots) == round(total_beats * beat * P.FPS)
    rec = 0
    for s in shots:
        assert s["frames"] == round((rec + s["beats"]) * beat * P.FPS) - round(rec * beat * P.FPS)
        rec += s["beats"]
    # the video stops where the stinger starts, which is just before the music ends
    assert plan["video_length"] == pytest.approx(music["length"] - round(P.STINGER_OVERLAP / beat) * beat, abs=1e-3)
    assert plan["stinger_at"] == plan["video_length"]
    # the acts come in story order, each a contiguous run, and every act change carries a transition
    acts = [s["act"] for s in shots]
    runs = [a for k, a in enumerate(acts) if k == 0 or acts[k - 1] != a]
    assert runs == list(P.ACT_NAMES)
    assert all(shots[k].get("transition") for k in range(1, len(shots)) if acts[k] != acts[k - 1])
    # it opens on the festival -- band / crowd B-roll from two clips -- with the logo held across the cut
    teaser = [s for s in shots if s["act"] == "teaser"]
    assert [s["logo"] for s in teaser] == ["in", "hold"] and not teaser[1].get("transition")
    assert all(s["unit"]["role"] == "transition" and s["unit"]["tag"] in ("band", "crowd") for s in teaser)
    assert teaser[0]["unit"]["clip"] != teaser[1]["unit"]["clip"]
    # every clip is in the reel, and no clip ever cuts to itself
    assert {s["unit"]["clip"] for s in shots} == {r["unit"]["clip"] for r in moments}
    assert all(shots[k]["unit"]["clip"] != shots[k - 1]["unit"]["clip"] for k in range(1, len(shots)))
    # cuts come ANTICIPATE ahead of the beat: the music starts that much earlier
    assert plan["music_start"] == pytest.approx(music["start"] - P.ANTICIPATE)
    # daytime B-roll opens the day; the arrival takes no night B-roll while day B-roll is left
    arrival = [s for s in shots if s["act"] == "arrival"]
    assert {s["unit"]["tag"] for s in arrival if s["unit"]["role"] == "transition"} <= {"class", "panel"}
    # with no daytime shows, stage dancing fills between the day's B-roll: never B-roll beside B-roll
    assert all(not (a["unit"]["role"] == b["unit"]["role"] == "transition") for a, b in pairwise(arrival))


def test_plan_without_stinger_fills_the_whole_music(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    plan = P.build(pr, "landscape", music, moments, stinger_len=0.0)
    assert plan["video_length"] == pytest.approx(music["length"], abs=1e-3)


def music_beat(music):
    return float(np.median(np.diff(music["beats"])))


def test_plan_speed_effects_put_the_peak_on_a_beat(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    shots = P.build(pr, "portrait", music, moments)["shots"]
    speed = [s for s in shots if any(f in P.SPEED for f in s["fx"])]
    assert speed
    b = music_beat(music)
    for s in speed:                               # unless the unit's edge got in the way
        u, L = s["unit"], s["length"]
        lead = s.get("lead_beats", s["beats"] // 2)
        if u["t0"] < s["t"] - lead * b < u["t1"] - L:
            assert s["peak"] == pytest.approx(lead * b, abs=1e-3)
        if "lead_beats" in s:                     # a velocity edit leaves room for its hold after the landing
            assert 1 <= lead < s["beats"]


def test_plan_acts_from_reel_json(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    acts = [["teaser", 0, 2], ["arrival", 2, 6], ["shows", 6, 40], ["night", 40, 50], ["climax", 50, 66],
            ["finale", 66, 72]]
    pr.config = {"plan": {"landscape": {"acts": acts}}}
    plan = P.build(pr, "landscape", music, moments)
    assert plan["acts"]["shows"] == pytest.approx([24 * plan["beat"], 160 * plan["beat"]])
    assert P.CFG["landscape"]["acts"] is None                   # defaults untouched


def test_default_acts_split_the_song_in_proportion():
    acts = P.default_acts(P.CFG["landscape"], 72)
    assert acts == [("teaser", 0, 2), ("arrival", 2, 12), ("shows", 12, 36), ("night", 36, 53),
                    ("climax", 53, 66), ("finale", 66, 72)]
    short = P.default_acts(P.CFG["portrait"], 40)
    assert [a[0] for a in short] == list(P.ACT_NAMES) and short[-1][2] == 40
    assert all(a[1] < a[2] for a in short)


def shot(kind, unit, act="shows", beats=2, role="show", tag="matinee"):
    return dict(kind=kind, unit=dict(id=unit, clip=unit, tag=tag, role=role), act=act, beats=beats, fx=[], t=10.0)


def test_transitions_are_hard_cuts_unless_the_cut_has_a_reason():
    seq = [shot("shake", f"U{i}") for i in range(30)]
    music = dict(beats=list(np.arange(0, 100, 0.6)), start=0.0)
    P.transitions(seq, music, P.CFG["landscape"])           # shakes with no travel between: nothing to carry
    assert not any(s.get("transition") for s in seq)
    P.transitions(seq, music, P.CFG["landscape"], direction=lambda u, t: (0.3, 0.3))   # all travelling right
    names = [s.get("transition") for s in seq]
    assert names.count("swipe") == P.G.CAPS["shows"]["swipe"]   # carried on, up to the act's cap
    for i in range(1, len(seq)):                            # never two transitions back to back
        assert not (names[i] and names[i - 1])


def test_transitions_flash_only_on_a_hard_hit_and_glitch_only_at_the_dj():
    seq = [shot("energy", f"U{i}", act="night") for i in range(8)]
    seq[5]["unit"]["tag"] = "dj"; seq[5]["unit"]["role"] = "transition"
    acc = np.ones(200); acc[6] = 2.2                        # one big hit, on the cut into shot 3 (beat 6)
    music = dict(beats=list(np.arange(0, 120, 0.6)), start=0.0, beat_accent=list(acc))
    P.transitions(seq, music, P.CFG["landscape"])
    assert [s.get("transition") for s in seq] == [None, None, None, "flash", None, "glitch", None, None]


def test_transitions_spin_cut_between_turns_and_hard_cut_to_a_reaction():
    seq = [shot("spin", "A"), shot("spin", "B"), shot("energy", "C"), shot("broll", "D")]
    seq[3]["reaction"] = True
    P.transitions(seq, dict(beats=list(np.arange(0, 100, 0.6)), start=0.0), P.CFG["landscape"])
    assert seq[1]["transition"] == "spin" and seq[1]["why"] == "a turn carried into the next turn"
    assert not seq[3].get("transition")


def test_crop_expr_is_piecewise_linear_ffmpeg_expression():
    path = [[0, 0.4, 0.5], [0.3, 0.5, 0.55], [0.6, 0.45, 0.5]]
    x, y = crop_expr(path, 1), crop_expr(path, 2)
    assert "if(lt(t\\,0.3)\\,(0.4+(0.5-0.4)*(t-0)/0.3)" in x
    assert "(0.5+(0.55-0.5)*(t-0)/0.3)" in y
    assert x.count("(") == x.count(")")


def test_proxy_times_round_trip(tmp_path):
    pr = Project(str(tmp_path))
    pr.config = {"proxies": {"M.mov": ["export/M.mp4", [[50.0, -0.28]]]}}
    f, t = pr.decode_from("M.mov", 60.0)
    assert f.endswith("export/M.mp4") and t == pytest.approx(59.72)
    assert pr.source_time("M.mov", t) == pytest.approx(60.0)
    assert pr.decode_from("M.mov", 10.0)[1] == 10.0
    assert pr.decode_from("C1.MP4", 3.0) == (os.path.join(pr.folder, "C1.MP4"), 3.0)


def test_beat_accents_mark_the_hits():
    from focalcut.reel.music import accents
    beats = np.arange(0, 20, 0.5)
    ot = np.arange(0, 20, 0.01)
    onset = np.ones_like(ot)
    onset[np.abs(ot - 10.0) < 0.02] = 5.0                   # a horn stab on the beat at 10 s
    a = accents(onset, ot, beats)
    assert a[20] > 3 and np.median(a) == pytest.approx(1.0)


def test_music_window_prefers_an_8_bar_phrase():
    bars = np.arange(0, 400, 2.4)
    i = window(bars, 0, 180.0)
    assert (i % 8) == 0 and bars[i] <= 180.0 and bars[i] > 172


def test_pathurl_maps_wsl_mounts_to_windows_drives():
    assert pathurl("/mnt/k/Videos/Show 2026/C2020.MP4") == "file://localhost/K:/Videos/Show%202026/C2020.MP4"
    assert pathurl("/home/u/a b.mov") == "file://localhost/home/u/a%20b.mov"


def test_cli_reel_init_writes_config(tmp_path):
    song = tmp_path / "song.mp3"; song.write_bytes(b"")
    assert cli.main(["reel", "init", str(tmp_path), "--song", str(song), "--start", "20.9",
                     "--stinger-portrait", str(song)]) == 0
    cfg = json.load(open(os.path.join(tmp_path, "focalcut", "reel", "reel.json")))
    assert cfg["song"] == str(song) and cfg["start"] == 20.9 and cfg["length"] == 180
    assert cfg["stingers"] == {"portrait": str(song)}


def test_cli_reel_run_rejects_unknown_steps(tmp_path, capsys):
    assert cli.main(["reel", "run", str(tmp_path), "--steps", "units,bogus"]) == 2


def test_fit_beats_removes_hop_quantisation_drift():
    from focalcut.reel.music import fit_beats
    hop = 512 / 22050                                   # librosa's frame step
    true = 0.25 + 0.6 * np.arange(300)                  # exactly 100 BPM
    quantised = np.round(true / hop) * hop              # what the tracker reports
    fitted, period = fit_beats(quantised)
    assert abs(period - 0.6) < 1e-4
    assert np.max(np.abs(fitted - true)) < 0.012        # within the quantisation step, no drift
    # the median interval of the quantised beats is off by a whole hop -> about a second over 300 beats
    assert abs(np.median(np.diff(quantised)) * 300 - 180) > 0.5


def test_fit_beats_keeps_real_tempo_changes():
    from focalcut.reel.music import fit_beats
    bt = np.cumsum(np.r_[0.6 * np.ones(100), 0.5 * np.ones(100)])
    fitted, _ = fit_beats(bt)
    assert np.allclose(fitted, bt)


def test_framing_marks_down_a_head_cut_off_by_the_camera():
    from focalcut.reel import frame as FR
    union = np.array([[0.4, 0.2, 0.6, 0.9], [0.4, 0.0, 0.6, 0.9]])
    whole = FR.framing(union, 3840, 2160, 1920, 1080, touch=np.array([[0, 0, 0, 0], [0, 0, 0, 0]], bool))[0]
    cut = FR.framing(union, 3840, 2160, 1920, 1080, touch=np.array([[0, 0, 0, 0], [0, 1, 0, 0]], bool))[0]
    assert whole[0] > 0.6 and cut[1] < 0.25 * whole[1]


def test_framing_centres_and_zooms_onto_a_small_subject():
    from focalcut.reel import frame as FR
    union = np.array([[0.15, 0.45, 0.25, 0.8], [np.nan] * 4])
    sc, z, cx, _ = FR.framing(union, 3840, 2160, 1920, 1080)
    assert z[0] > 1.3 and cx[0] < 0.5 and sc[0] > 0.3             # a tighter crop, moved towards the subject
    assert sc[1] == 0 and z[1] == 1                                # nobody there
    # portrait out of landscape: a full-height strip, centred on the dancer
    sc, z, cx, _ = FR.framing(np.array([[0.7, 0.2, 0.8, 0.9]]), 3840, 2160, 1080, 1920)
    assert abs(cx[0] - 0.75) < 0.01 and z[0] <= FR.ZMAX


def test_track_interpolates_the_subject_and_drops_gaps():
    from focalcut.reel import frame as FR
    d = dict(sub_t=np.array([0.0, 1.0, 5.0]), sub_box=np.array([[0.1, 0.1, 0.3, 0.9], [0.3, 0.1, 0.5, 0.9],
                                                                 [0.5, 0.1, 0.7, 0.9]], np.float32),
             sub_touch=np.zeros((3, 4), bool))
    union, _, src = FR.track(d, np.array([0.5, 3.0]))
    assert src == "sam" and union[0, 0] == pytest.approx(0.2) and np.isnan(union[1, 0])   # 2 s from any keyframe


def person(h, w, x0, x1, y0=10, y1=60):
    m = np.zeros((h, w), bool); m[y0:y1, x0:x1] = True
    return m


def test_subject_is_the_salient_near_couple_not_the_bystander():
    from focalcut.reel import subject_engine as E
    h, w = 80, 160
    masks = np.stack([person(h, w, 40, 60), person(h, w, 62, 82), person(h, w, 130, 140, 30, 50)])
    sal = np.zeros((h, w)); sal[:, 30:90] = 1.0                    # the couple stands out from the stage
    dep = np.full((h, w), 0.3); dep[:, 30:90] = 0.9                # and is nearer the camera
    sc = E.score(masks, sal, dep)
    assert sorted(E.choose(sc)) == [0, 1]
    r = E.describe(masks, sc)
    assert r["n"] == 2 and r["box"][0] == pytest.approx(40 / w) and r["box"][2] == pytest.approx(82 / w)
    assert not any(r["touch"])
    edge = E.describe(np.stack([person(h, w, 40, 60, 0, 60)]), np.ones(1))
    assert edge["touch"][1]                                        # the head at the frame's top edge


def test_effect_masks_keep_the_couple_or_the_whole_line():
    from focalcut.reel.subject_engine import keep_subject
    avg = {1: 1.0, 2: 0.6, 3: 0.35, 4: 0.25, 5: 0.1}
    assert keep_subject(avg, group=False) == [1, 2]
    assert keep_subject(avg, group=True) == [1, 2, 3, 4]
    assert keep_subject({}, group=True) == []


# --- the editing rules and the music-video rhythm ------------------------------------------------------------------

def test_rhythm_fills_the_act_in_varied_8_beat_phrases():
    energy = [1.0, 1.4, 0.9, 0.8, 1.5, 1.0, 0.7, 1.2]
    for pace in ("medium", "fast"):
        out = P.rhythm(64, pace, energy)
        assert sum(out) == 64 and all(n >= 1 for n in out)
        starts, rec = set(), 0                                  # every phrase (8-count) starts on a cut
        for n in out:
            starts.add(rec); rec += n
        assert all(8 * k in starts for k in range(8))
        assert len(set(out)) >= 3                               # never one length over and over
        best = run = 1
        for k in range(1, len(out)):
            run = run + 1 if out[k] == out[k - 1] else 1; best = max(best, run)
        assert best <= 5
    held = P.rhythm(32, "medium", [1.0, 1.0, 1.0, 0.5])          # the 4th phrase is quiet: held long
    assert held[-1] >= 4 or held[-2:] in ([4, 4], [6, 2])
    built = P.rhythm(36, "medium", [1.0] * 4, build=True)        # a build into the next act, then the remainder
    assert sum(built) == 36 and built[-8:-2] == P.PHRASES["build"][0]


def test_squeeze_merges_short_pairs_without_making_runs():
    src = [2, 1, 1, 2, 3, 1, 1, 2, 2, 1, 1, 1, 1, 2]
    out = P.squeeze(src, 10)
    assert len(out) == 10 and sum(out) == sum(src)
    assert out[-3:] == src[-3:]                                 # the closing phrase (a build) is spared
    assert not any(out[k] == out[k - 1] == out[k - 2] for k in range(2, len(out)))
    assert P.squeeze([2, 2], 5) == [2, 2]


def wshot(clip, kind="shake", role="show", tag="matinee", t=10.0):
    return dict(kind=kind, t=t, unit=dict(id=clip, clip=clip, role=role, tag=tag), act="shows", beats=2, fx=[],
                rank=0.5)


def test_weave_shuffles_clips_and_spaces_the_broll():
    shots = [wshot(c, t=10.0 + k) for c in "ABC" for k in range(4)] + \
            [wshot(f"B{k}", "broll", "transition", "band") for k in range(4)]
    out = P.weave(shots)
    assert sorted(id(s) for s in out) == sorted(id(s) for s in shots)
    assert all(out[k]["unit"]["clip"] != out[k - 1]["unit"]["clip"] for k in range(1, len(out)))
    roles = [s["unit"]["role"] for s in out]
    assert not any(roles[k] == roles[k - 1] == "transition" for k in range(1, len(roles)))
    # a clip's shots spread through the act rather than bunching at one end
    a = [k for k, s in enumerate(out) if s["unit"]["clip"] == "A"]
    assert a[-1] - a[0] >= len(out) // 2


def test_weave_cuts_on_the_action():
    prev = wshot("X", "spin")
    shots = [wshot("A", "shake"), wshot("B", "spin"), wshot("C", "dip")]
    assert P.weave(shots, prev)[0]["unit"]["clip"] == "B"       # a turn cuts to a turn
    right = {"X": (0.0, 0.1), "L": (-0.1, 0.0), "R": (0.1, 0.0)}

    def d(u, t):
        return right.get(u["clip"], (0.0, 0.0))
    shots = [wshot("L", "energy"), wshot("R", "energy")]
    assert P.weave(shots, wshot("X", "energy"), direction=d)[0]["unit"]["clip"] == "R"   # travel keeps its direction


def test_broll_slots_never_touch():
    lengths = [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1]
    slots = P.broll_slots(lengths, len(lengths), 5)
    assert len(slots) == 5 and 0 not in slots and len(lengths) - 1 not in slots
    assert all(abs(a - b) > 1 for a in slots for b in slots if a != b)


def test_fit_slots_gives_long_slots_to_moves_that_need_time():
    shots = [wshot("A", "shake"), wshot("B", "spin"), wshot("C", "shake"), wshot("D", "shake")]
    P.fit_slots(shots, [4, 1, 1, 2])
    assert [s["beats"] for s in shots] == [4, 1, 1, 2] and shots[0]["kind"] == "spin"


def test_fair_share_and_every_clip_in_the_reel(tmp_path):
    pr, music, moments = fake_reel(tmp_path, n_units=24)
    shots = P.build(pr, "portrait", music, moments)["shots"]
    count = {}
    for s in shots:
        count[s["unit"]["clip"]] = count.get(s["unit"]["clip"], 0) + 1
    assert set(count) == {r["unit"]["clip"] for r in moments}
    assert max(count.values()) <= 3 * len(shots) / len(count) + 1   # no clip hogs the reel
    by_unit = {}
    for s in shots:                                             # a unit's shots are different scenes
        by_unit.setdefault(s["unit"]["id"], []).append(s["t"])
    assert all(abs(a - b) >= P.APART for ts in by_unit.values() for i, a in enumerate(ts) for b in ts[i + 1:])


def test_plain_shots_land_the_move_peak_on_a_beat(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    plan = P.build(pr, "landscape", music, moments)
    beat = plan["beat"]
    for s in plan["shots"]:
        u = s["unit"]
        if not any(f in P.SPEED for f in s["fx"]) and u["t0"] < s["src_in"] < u["t1"] - s["length"]:
            assert s["peak"] / beat == pytest.approx(round(s["peak"] / beat), abs=1e-2)


def test_refine_follows_the_subjects_pose_boxes_between_keyframes():
    from focalcut.reel import frame as FR
    aw, ah = 640, 360
    boxes = np.full((3, 2, 5), np.nan)
    boxes[:, 0] = [[100, 50, 180, 300, 0.9], [140, 50, 220, 300, 0.9], [180, 50, 260, 300, 0.9]]   # the dancer moves
    boxes[:, 1] = [500, 50, 560, 300, 0.9]                                                      # a bystander
    d = dict(pose_t=np.array([0.0, 0.2, 0.4]), boxes=boxes, aw=aw, ah=ah)
    union = np.tile([140 / aw, 50 / ah, 220 / aw, 300 / ah], (3, 1))                         # keyframe: held still
    out = FR.refine(d, np.array([0.0, 0.2, 0.4]), union)
    cx = (out[:, 0] + out[:, 2]) / 2 * aw
    assert cx == pytest.approx([140, 180, 220], abs=1)
    assert (out[:, 2] - out[:, 0]) == pytest.approx(union[:, 2] - union[:, 0])   # same size, re-centred


def test_effect_mask_edges_come_from_the_matte_inside_the_sam_mask():
    pytest.importorskip("torch")
    from focalcut.fx.masks import refine
    h, w = 120, 160
    coarse = np.zeros((h, w), np.float32); coarse[30:90, 50:90] = 1     # SAM: a blocky box
    frame = np.zeros((h, w, 3), np.uint8)

    def fake(crop):                                                      # a matte that keeps everything it sees
        return np.ones(crop.shape[:2], np.float32)
    a = refine(frame, coarse, matte=fake)
    assert a[60, 70] > 0.95                                              # the subject
    assert a[5, 5] < 0.05 and a[60, 150] < 0.05                          # the rest of the frame stays out


# --- the editor's picks, activity, the review page ------------------------------------------------------------------

def picks_rec(export_t0=10.0):
    u = dict(id="C200-01_couple", clip="C200", t0=10.0, t1=70.0, export_t0=export_t0)
    ms = [dict(t=t, score=0.5, score_l=0.5, score_p=0.4, kind="spin") for t in (15.0, 25.0, 40.0, 60.0)]
    return dict(id=u["id"], moments=ms, unit=u)


def test_picks_ban_ranges_and_whole_clips_in_the_exports_timeline():
    r = picks_rec()
    P.apply_picks([r], {"C200-01_couple": {"ban": ["0:04-0:16"]}, "_how": "notes are ignored"})
    assert [m["t"] for m in r["moments"]] == [40.0, 60.0]       # export 4-16 s = source 14-26 s
    r = picks_rec()
    P.apply_picks([r], {"C200-01_couple": {"ban": ["all"]}})
    assert r["moments"] == []


def test_picks_prefer_boosts_or_makes_a_moment():
    r = picks_rec()
    P.apply_picks([r], {"C200-01_couple": {"prefer": [[28, 32]]}})   # export 28-32 s = source 38-42 s
    m = next(m for m in r["moments"] if m["t"] == 40.0)
    assert m["pick"] and m["score_l"] == pytest.approx(1.5) and m["score_p"] == pytest.approx(1.4)
    r = picks_rec(export_t0=0.0)
    P.apply_picks([r], {"C200-01_couple": {"prefer": ["0:50-0:52"]}})   # nothing detected there: one is made
    made = [m for m in r["moments"] if m.get("pick")]
    assert [m["t"] for m in made] == [51.0] and made[0]["kind"] == "spin" and made[0]["score_l"] == pytest.approx(1.5)
    assert P._secs("1:23.5") == pytest.approx(83.5) and P._secs(7) == 7.0


def test_picks_reach_the_plan(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    banned = moments[0]["unit"]["id"]
    os.makedirs(pr.reel_dir, exist_ok=True)
    with open(pr.path("picks.json"), "w") as f:
        json.dump({banned: {"ban": ["all"]}}, f)
    plan = P.build(pr, "landscape", music, moments)
    assert banned not in {s["unit"]["id"] for s in plan["shots"]}
    assert moments[0]["moments"]                                 # the caller's moments are left alone


def test_picks_add_ranges_are_the_editors_in_out_one_shot_each():
    r = picks_rec()
    P.apply_picks([r], {"C200-01_couple": {"add": ["0:20-0:22.5", "0:40-0:46", "0:05"]}})
    spans = sorted(m["span"] for m in r["moments"] if m.get("span"))
    assert spans == [[30.0, 32.5], [50.0, 56.0]]                 # a long pick stays one shot; export + 10 s = source
    assert any(m.get("add") and not m.get("span") and m["t"] == 15.0 for m in r["moments"])   # a point stays a point
    s = dict(unit=r["unit"], t=31.0, beats=4, fx=[], span=[30.0, 32.5])
    assert P.in_point(s, 0.5) == 30.0                                # starts at the editor's in-point


def test_picks_only_categories_keep_just_the_editors_adds():
    r, other = picks_rec(), picks_rec()
    r["unit"]["category"], other["unit"]["category"] = "lesson", "night"
    P.apply_picks([r], {"C200-01_couple": {"add": ["0:20-0:22.5"]}})
    unpicked = dict(picks_rec(), id="C201", unit=dict(picks_rec()["unit"], id="C201", category="lesson"))
    P.picks_only([r, other, unpicked], {"lesson"})
    assert [m["span"] for m in r["moments"]] == [[30.0, 32.5]]       # only the add survives
    assert len(other["moments"]) == 4 and unpicked["moments"] == []    # other categories untouched; no pick, no shots


def test_act_logos_title_the_acts_first_plain_shot_long_enough_to_read():
    u = dict(id="u", role="show")
    seq = [dict(unit=u, act=act, beats=b, fx=[]) for act, b in
           (("night", 4), ("climax", 4), ("climax", 2), ("climax", 6), ("climax", 8))]
    seq[1]["fx"] = ["hero"]                                              # an effect shot is never titled
    P.act_logos(seq, {"climax": "mj.png", "finale": "x.png"})
    assert [s.get("logo_image") for s in seq] == [None, None, None, "mj.png", None]
    assert seq[3]["logo"] == "in"


def test_act_logos_match_puts_the_title_on_the_acts_own_footage():
    day, star = dict(id="d", category="day", tag=""), dict(id="s", category="broll", tag="band-star-headline")
    seq = [dict(unit=u, act="climax", beats=4, fx=[]) for u in (day, star, day)]
    P.act_logos(seq, {"climax": {"image": "star.png", "match": ["star"]}})
    assert [s.get("logo_image") for s in seq] == [None, "star.png", None]
    seq = [dict(unit=day, act="climax", beats=4, fx=[])]                 # nothing matches: the first plain shot
    P.act_logos(seq, {"climax": {"image": "star.png", "match": ["star"]}})
    assert seq[0]["logo_image"] == "star.png"


def test_focal_points_steer_the_crop_centre_and_keep_its_size():
    unit = dict(id="C200-01_couple", export_t0=10.0)
    keys = P.focus_keys({"C200-01_couple": {"portrait": [["0:21", 0.2, 0.5], ["0:23", 0.95, 0.4]]}}, unit, "portrait")
    assert keys == [(31.0, 0.2, 0.5), (33.0, 0.95, 0.4)]                 # export time + 10 s = source time
    crop = dict(w=0.3, h=1.0, z=1.0, path=[[0, 0.6, 0.5]], face=0.4)
    c = P.focus_override(crop, keys, 31.0, 2.0)
    xs = [x for _, x, _ in c["path"]]
    assert c["focus"] and c["w"] == 0.3 and c["face"] == 0.4
    assert xs[0] == 0.2 and xs[-1] == 0.85 and xs[5] == pytest.approx(0.575)   # glides; held inside the frame
    assert {y for *_, y in c["path"]} == {0.5}                           # the crop is the full height: y can't move
    assert P.focus_override(crop, keys, 40.0, 2.0) is crop                # no point near this shot: as derived
    assert P.focus_override(None, keys, 31.0, 2.0) is None                # nothing to crop


def test_fit_pick_gives_a_selection_its_length_and_keeps_the_grid():
    u = lambda i: dict(id=f"U{i}", clip=f"C{i}", role="dance")      # noqa: E731
    seq = [dict(unit=u(k % 3), act="night", beats=2, fx=[]) for k in range(8)]
    seq[3].update(span=[0.0, 4.5], add=True)                         # 4.5 s at 0.5 s a beat: 9 beats, one shot
    total = sum(x["beats"] for x in seq)
    P.fit_pick(seq, 3, 0.5)
    assert next(x for x in seq if x.get("span"))["beats"] == 9 and sum(x["beats"] for x in seq) == total


def test_source_browser_page_and_server_save_picks(tmp_path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from focalcut.reel import browse, serve
    clip = dict(id="C1", cat="night", tag="", clock="21:00", sort="1", dur=10.0, video="v.mp4", strip="sources/C1.jpg",
                step=1.0, n=10, moments=[[2.0, "spin"]], L=[[1.0, 2.0, 5]], P=[])
    html_ = browse.page([clip], {"C1": {"add": ["0:01-0:02"]}})
    assert '"id": "C1"' in html_ and "Play all" in html_ and 'D.root+"api/picks"' in html_ and "api/crop" in html_
    assert '"/api/' not in html_ and "`/api/" not in html_                   # relative: works under any prefix
    pr, _, _ = fake_reel(tmp_path)
    os.makedirs(pr.path("review"), exist_ok=True)
    with open(pr.path("review", "sources.html"), "w") as f:
        f.write(html_)
    with open(os.path.join(pr.folder, "v.bin"), "wb") as f:
        f.write(bytes(range(256)))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.handler(pr))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/api/picks", data=b'{"C1": {"add": ["0:03-0:04"]}}', method="PUT")
        assert urllib.request.urlopen(req).status == 200
        assert json.load(open(pr.path("picks.json"))) == {"C1": {"add": ["0:03-0:04"]}}
        assert json.load(urllib.request.urlopen(base + "/api/picks")) == {"C1": {"add": ["0:03-0:04"]}}
        part = urllib.request.urlopen(urllib.request.Request(base + "/v.bin", headers={"Range": "bytes=10-19"}))
        assert part.status == 206 and part.read() == bytes(range(10, 20))
        bad = urllib.request.Request(base + "/api/crop?id=nope&a=1&b=2")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        srv.shutdown()


def reel_folder(path, clips=("C1",)):
    """A project folder as the server sees it: reel.json, a sources page and an export per clip."""
    os.makedirs(os.path.join(path, "focalcut", "reel", "review"), exist_ok=True)
    with open(os.path.join(path, "focalcut", "reel", "reel.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(path, "focalcut", "reel", "review", "sources.html"), "w") as f:
        f.write("<p>sources of " + os.path.basename(path))
    with open(os.path.join(path, "focalcut", "reel", "review", "landscape.html"), "w") as f:
        f.write("<p>landscape")
    for c in clips:
        with open(os.path.join(path, c + ".bin"), "wb") as f:
            f.write(bytes(range(256)))
    return Project(str(path))


def served(site):
    import threading
    from http.server import ThreadingHTTPServer

    from focalcut.reel import serve
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.handler(site))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def http(url, data=None, method=None, headers=None):
    """(status, body) of a request, errors included."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_one_server_holds_several_projects_each_with_its_own_picks(tmp_path):
    from focalcut.reel import browse, serve
    a, b = reel_folder(tmp_path / "SSMC 2026"), reel_folder(tmp_path / "Electric Nation 2026", clips=("C2",))
    site = serve.Site([a, b])
    assert set(site.projects) == {"ssmc-2026", "electric-nation-2026"} and site.single is None
    srv, base = served(site)
    try:
        for s, picks in (("ssmc-2026", b'{"C1": {"ban": ["all"]}}'), ("electric-nation-2026", b'{"C2": {"more": 2}}')):
            assert http(f"{base}/p/{s}/api/picks", picks, "PUT")[0] == 200
        assert json.load(open(a.path("picks.json"))) == {"C1": {"ban": ["all"]}}       # each its own file
        assert json.load(open(b.path("picks.json"))) == {"C2": {"more": 2}}
        assert json.loads(http(f"{base}/p/electric-nation-2026/api/picks")[1]) == {"C2": {"more": 2}}
        http(f"{base}/p/ssmc-2026/api/picks", b'{"C1": {}}', "PUT")
        assert json.load(open(a.path("picks.json") + ".bak")) == {"C1": {"ban": ["all"]}}   # .bak on every write
        st, body = http(f"{base}/p/ssmc-2026/C1.bin", headers={"Range": "bytes=10-19"})
        assert st == 206 and body == bytes(range(10, 20))
        st, body = http(f"{base}/p/ssmc-2026/focalcut/reel/review/sources.html")
        assert st == 200 and b"SSMC 2026" in body
        idx = http(base + "/")[1].decode()                                    # the index lists both, with pages
        assert "/p/ssmc-2026/focalcut/reel/review/sources.html" in idx and "Electric Nation 2026" in idx
        assert "/p/electric-nation-2026/focalcut/reel/review/landscape.html" in idx
        assert http(base + "/api/picks")[0] == 404                            # no top level with several
        assert http(base + "/p/nope/api/picks")[0] == 404
        assert http(f"{base}/p/ssmc-2026/api/picks", b'{"x": 1}', "PUT", {"Host": "evil.example"})[0] == 403
    finally:
        srv.shutdown()
    assert browse.api_root(a) == "../../../"                                  # the pages' API base
    os.makedirs(a.path(), exist_ok=True)
    with open(a.path("plan_landscape.json"), "w") as f:
        f.write("{}")
    nav = browse.menu(a, "landscape")                                        # every page's menu: relative links
    assert 'href="sources.html"' in nav and 'href="landscape.html" class=cur' in nav and 'href="/" id=allp hidden' in nav


def test_server_refuses_paths_out_of_a_project_even_into_another(tmp_path):
    from focalcut.reel import serve
    reel_folder(tmp_path / "A", clips=("secret",))
    a, b = Project(str(tmp_path / "A")), reel_folder(tmp_path / "B")
    with open(tmp_path / "outside.txt", "w") as f:
        f.write("no")
    srv, base = served(serve.Site([a, b]))
    try:
        assert http(f"{base}/p/a/secret.bin")[0] == 200
        for bad in ("/p/b/../a/secret.bin", "/p/b/%2e%2e/a/secret.bin", "/p/b/..%2fA%2fsecret.bin",
                    "/p/b/%2e%2e%2f%2e%2e%2foutside.txt", "/p/a/../../outside.txt"):
            assert http(base + bad)[0] == 404, bad
    finally:
        srv.shutdown()


def test_a_running_server_takes_new_projects_and_one_folder_works_as_before(tmp_path):
    from focalcut.reel import serve
    one = reel_folder(tmp_path / "One")
    srv, base = served(serve.Site([one]))
    try:                                                                     # a single project: the top level too
        assert http(base + "/api/picks", b'{"C1": {"more": 1}}', "PUT")[0] == 200
        assert json.load(open(one.path("picks.json"))) == {"C1": {"more": 1}}
        assert json.loads(http(base + "/p/one/api/picks")[1]) == {"C1": {"more": 1}}
        assert http(base + "/C1.bin")[0] == 200
        assert b"/p/one/focalcut/reel/review/sources.html" in http(base + "/")[1]   # / is still the index
    finally:
        srv.shutdown()
    site = serve.Site([one])
    srv, base = served(site)
    try:
        two = reel_folder(tmp_path / "Two")
        assert http(base + "/api/projects", json.dumps({"folder": two.folder}).encode(), "POST")[0] == 415
        st, body = http(base + "/api/projects", json.dumps({"folder": two.folder}).encode(), "POST",
                        {"Content-Type": "application/json"})
        assert st == 200 and json.loads(body)["slug"] == "two" and http(base + "/p/two/C1.bin")[0] == 200
        assert http(base + "/api/projects", b'{"folder": "/nonexistent"}', "POST",
                    {"Content-Type": "application/json"})[0] == 400
    finally:
        srv.shutdown()
    reel_folder(tmp_path / "Three")                                          # --root: found as it appears
    site = serve.Site(root=str(tmp_path))
    assert {"one", "two", "three"} <= set(site.projects) and site.single is None
    reel_folder(tmp_path / "Four")
    assert site.get("four") is not None


def test_activity_gate_marks_down_someone_standing_still():
    from focalcut.reel.moments import activity
    pt = np.arange(0, 10, 0.2)
    t10 = np.arange(0, 10, 0.1)
    subj = np.where(t10 < 5, 0.2, 3.0)                            # waiting, then dancing
    present = np.ones(len(pt), bool)
    act = activity(pt, t10, subj, present)
    assert act[pt < 3.5].max() < 0.05 and act[(pt > 6) & (pt < 8.5)].min() > 0.9
    gone = activity(pt, t10, np.full(len(t10), 3.0), pt < 2)      # dancing, but out of the frame
    assert gone[(pt > 4) & (pt < 8)].max() == 0


def test_review_page_lists_every_shot_with_a_picks_line():
    from focalcut.reel import review
    unit = dict(id="C300_band", export_t0=12.0)
    shots = [dict(unit=unit, src_in=75.5, length=1.2, act="night", kind="broll", fx=[], transition="flash"),
             dict(unit=dict(id="C301"), src_in=3.0, length=2.4, act="shows", kind="spin", fx=["hero"])]
    html_ = review.page(dict(shots=shots), "portrait", ["portrait/001.jpg", None])
    assert html_.count("<figure data-i=") == 2 and 'src="portrait/001.jpg"' in html_
    assert "Mark extra shot (M)" in html_ and '"id": "C300_band", "a": 63.5' in html_   # the scrubber's data
    assert review.pick_line(shots[0]) == '"C300_band": {"ban": ["1:03.5-1:04.7"]}'   # export time = source - 12 s
    assert "&quot;C301&quot;: {&quot;ban&quot;: [&quot;0:03.0-0:05.4&quot;]}" in html_
    assert "prefers-color-scheme" in html_


def test_velocity_landing_on_the_strongest_beat_that_leaves_room_for_the_hold():
    from focalcut.reel import velocity as V
    assert V.landing(4, "hero", False, [1, 1.2, 2.0, 3.0]) == 2      # beat 3 would leave no room for the hold
    assert V.landing(6, "bullet", False, [1, 1, 1, 2.0, 1, 1]) == 3  # bullet's 2-beat hold still fits after beat 3
    assert V.landing(6, "bullet", False, [1, 1, 1, 1, 3.0, 1]) == 1  # beat 4 doesn't: the first of the rest
    assert V.landing(1, "hero", False, [5.0]) == 1                   # no room at all: as far in as it goes


def test_settle_finds_where_the_move_lands(tmp_path):
    from focalcut.reel import velocity as V
    pr = Project(str(tmp_path))
    os.makedirs(pr.path("scores"))
    t = np.arange(0, 20, 0.1)
    subj = np.where(t < 10.4, 4.0, 0.5)                     # a burst of motion, then a rest from 10.4 s
    np.savez(pr.path("scores", "C1.npz"), t=t, subj=subj, size=np.array([3840, 2160]),
             boxes=np.zeros((1, 1, 5)), pose_t=np.zeros(1))
    u = dict(score="C1", t0=0.0, t1=20.0)
    assert V.settle(pr, u, 10.0) == pytest.approx(10.4, abs=0.15)
    assert V.settle(pr, dict(u, score="missing"), 10.0) == 10.0


def test_assess_drops_shaky_and_lost_subjects_and_favours_faces_in_portrait(tmp_path):
    pr, _, moments = fake_reel(tmp_path)
    os.makedirs(pr.path("scores"), exist_ok=True)
    t = np.arange(0, 90, 0.04)
    np.savez(pr.path("scores", "C002_shake.npz"), t=t, hf=np.where(t < 45, 0.0005, 0.02).astype(np.float32))
    social = next(r for r in moments if r["unit"]["clip"] == "C002")
    cache = {}
    for m in social["moments"]:                                 # [present, inside, face, crop width]
        cache[f"portrait|U02|{m['t']}|c2"] = [1.0, 1.0, 0.9 if m["t"] == 5.0 else 0.0, 1.0, None, 0.0, 0.0, 0.0, 1.0]
    json.dump(cache, open(pr.path("hold_cache.json"), "w"))
    before = {m["t"]: m.get("score_p", m["score"]) for m in social["moments"]}
    P.assess(pr, [social], "portrait", P.config(pr, "portrait"))
    after = {m["t"]: m["score_p"] for m in social["moments"]}
    assert after[5.0] > before[5.0]                             # a face on the social floor ranks up
    assert after[11.0] < 0.25 * before[11.0]                    # a back ranks well down
    shaky = [m for m in social["moments"] if m["t"] > 46]
    assert all(m["jitter"] > 0.006 and m["score_p"] < 0.05 * before[m["t"]] for m in shaky)


def test_portrait_strip_never_cuts_through_the_partner_and_body_shots_rank_over_head_shots():
    # a couple too far apart for a 9:16 strip (each 0.12 wide, 0.1 apart): the strip takes one whole and leaves the
    # other out whole, never half of each
    F, aw, ah = 10, 640.0, 360.0
    boxes = np.full((F, 3, 5), np.nan, np.float32)
    boxes[:, 0, :4] = [0.30 * aw, 0.2 * ah, 0.42 * aw, 0.95 * ah]          # the subject
    boxes[:, 1, :4] = [0.52 * aw, 0.2 * ah, 0.64 * aw, 0.95 * ah]          # the partner, the same size
    d = dict(pose_t=np.arange(F) * 0.2, boxes=boxes, kp=np.zeros((F, 3, 17, 3), np.float32), aw=aw, ah=ah)
    union = np.tile([0.30, 0.2, 0.64, 0.95], (F, 1))
    cw0 = (9 / 16) / (16 / 9)
    sx = np.full(F, 0.47)                                                  # the old centre: the floor between them
    z, c = P.clean_centre(d, d["pose_t"], union, sx, 1.2, cw0, 1.0)
    for x in c:
        for b in ((0.30, 0.42), (0.52, 0.64)):
            vis = max(0.0, min(b[1], x + cw0 / z / 2) - max(b[0], x - cw0 / z / 2)) / (b[1] - b[0])
            assert vis < 0.15 or vis > 0.85
    assert P.plays(dict(w=0.3, body=1.0), "dance") > 2 * P.plays(dict(w=0.3, body=0.0), "dance")
    assert P.plays(dict(w=0.3, split=1.0), "dance") < 0.5 * P.plays(dict(w=0.3, split=0.0), "dance")
    assert P.plays(dict(w=1.0, body=0.0, split=1.0), "dance") == 1.0           # landscape: unchanged


def test_portrait_leaves_out_a_clip_that_never_frames_unless_it_was_picked():
    r = dict(role="dance", moments=[dict(t=1.0, plays_p=0.05), dict(t=5.0, plays_p=0.1)])
    assert not P.framable(r, "portrait") and P.framable(r, "landscape")
    assert P.framable(dict(r, moments=[*r["moments"], dict(t=9.0, plays_p=0.6)]), "portrait")
    assert P.framable(dict(r, moments=[*r["moments"], dict(t=9.0, plays_p=0.01, pick=True)]), "portrait")
    assert P.framable(dict(r, role="transition"), "portrait")


def test_band_members_cut_in_and_the_reel_ends_on_the_chosen_shot(tmp_path):
    pr, music, moments = fake_reel(tmp_path, n_units=20)
    moments[19]["unit"]["tag"] = "violin"                       # a band player's B-roll
    moments[18]["unit"]["clip"] = moments[18]["unit"]["id"] = moments[18]["id"] = "END"
    pr.config = {"plan": {"landscape": {"ending": ["END", 40.0, 46.0]}}}
    shots = P.build(pr, "landscape", music, moments, assess_shots=False)["shots"]
    band = [k for k, s in enumerate(shots) if s.get("band")]
    assert band and all(shots[k]["unit"]["tag"] == "violin" for k in band)
    assert all(not shots[k].get("transition") for k in band)    # the band cuts straight in
    last = shots[-1]
    assert last.get("ending") and last["unit"]["id"] == "END"
    assert last["src_in"] + last["length"] == pytest.approx(46.0, abs=1e-3)   # runs out at its "to" time
    other = [s for s in shots[:-1] if s["unit"]["id"] == "END"]
    assert all(not 36 <= s["t"] <= 50 for s in other)           # its stretch is kept for the ending


def test_a_hand_built_clip_is_placed_on_the_beat_and_named_as_its_own_cut(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    r = {x["unit"]["clip"]: x for x in moments}
    seq = [P.shot(r["C000"], r["C000"]["moments"][0], "portrait", 4, act="climax", fx=["hero"]),
           P.shot(r["C001"], r["C001"]["moments"][0], "portrait", 2, act="climax"),
           P.shot(r["C004"], r["C004"]["moments"][1], "portrait", 2, act="climax")]
    plan = P.place(pr, "portrait", seq, music, song="other.mp3")
    b = plan["beat"]
    assert [s["rec_in"] for s in plan["shots"]] == pytest.approx([0, 4 * b, 6 * b])
    assert plan["video_length"] == pytest.approx(8 * b) and plan["song"] == "other.mp3" and not plan["stinger"]
    assert "lead_beats" in plan["shots"][0]                 # the velocity edit has its landing beat
    assert pr.out_stem("teaser30", "portrait").endswith("_teaser30")
    assert pr.out_stem("portrait", "portrait").endswith("_recap_portrait")


def test_picks_add_extra_shots_and_ask_for_more_from_a_clip(tmp_path):
    pr, music, moments = fake_reel(tmp_path, n_units=20)
    picks = {"U05": {"add": ["1:20.0"]}, "U06": {"more": 3}}
    shots = P.build(pr, "landscape", music, moments, picks=picks, assess_shots=False)["shots"]
    assert any(s["unit"]["id"] == "U05" and abs(s["t"] - 80.0) < 1.5 for s in shots)      # the marked extra shot
    base = P.build(pr, "landscape", music, moments, picks={}, assess_shots=False)["shots"]
    def n(ss, u):
        return sum(s["unit"]["id"] == u for s in ss)
    assert n(shots, "U06") > n(base, "U06")                                                  # more from that clip


def test_assess_drops_a_subject_who_keeps_leaving_the_crop(tmp_path):
    pr, _, moments = fake_reel(tmp_path)
    social = next(r for r in moments if r["unit"]["clip"] == "C002")
    os.makedirs(pr.reel_dir, exist_ok=True)
    json.dump({f"portrait|U02|{m['t']}|c2": [0.3 if m["t"] == 5.0 else 1.0, 1.0, 1.0, 1.0, None, 0.0, 0.0, 0.0, 1.0]
               for m in social["moments"]},
              open(pr.path("hold_cache.json"), "w"))
    before = {m["t"]: m["score"] for m in social["moments"]}
    P.assess(pr, [social], "portrait", P.config(pr, "portrait"))
    after = {m["t"]: m["score_p"] for m in social["moments"]}
    assert after[5.0] < 0.1 * before[5.0] and after[11.0] > before[11.0]


def _couple(n=10, faces=(True, False), focus=(1.0, 1.0), apart=0.5, empty_slot=True):
    """Score data for two dancers apart in a 640x360 analysis frame, plus an empty pose slot."""
    k = 3 if empty_slot else 2
    boxes = np.full((n, k, 5), np.nan)
    kp = np.zeros((n, k, 17, 3))
    foc = np.full((n, k), np.nan)
    for j, x in enumerate((0.25, 0.25 + apart)):
        boxes[:, j] = [(x - 0.05) * 640, 0.2 * 360, (x + 0.05) * 640, 0.9 * 360, 0.9]
        if faces[j]:                                    # nose, left eye (image right), right eye (image left)
            kp[:, j, 0] = [x * 640, 80, 0.9]
            kp[:, j, 1] = [x * 640 + 4, 76, 0.9]
            kp[:, j, 2] = [x * 640 - 4, 76, 0.9]
        foc[:, j] = focus[j]
    d = dict(pose_t=np.arange(n) * 0.2, boxes=boxes, kp=kp, focus=foc, aw=640, ah=360)
    union = np.tile([0.2, 0.2, 0.30 + apart, 0.9], (n, 1)).astype(float)
    return d, union


def test_faces_needs_nose_and_both_eyes_seen_from_the_front():
    from focalcut.reel import frame as FR
    d, _ = _couple(faces=(True, False))
    f = FR.faces(d["kp"][0])
    assert f[0] and not f[1]
    kp = d["kp"][0].copy()
    kp[0, [1, 2]] = kp[0, [2, 1]]                       # eyes swapped: the back of a head
    assert not FR.faces(kp)[0]


def test_follow_one_dancer_of_a_split_couple_preferring_a_face_then_the_sharper():
    from focalcut.reel import frame as FR
    d, union = _couple(faces=(False, True))
    box, face, one = FR.follow(d, d["pose_t"], union, cw=0.32)
    assert one and face.all() and abs((box[0, 0] + box[0, 2]) / 2 - 0.75) < 0.01    # the one facing the camera
    d, union = _couple(faces=(False, False), focus=(1.0, 10.0))
    box, _, _ = FR.follow(d, d["pose_t"], union, cw=0.32)                             # neither: the sharper one,
    assert abs((box[0, 0] + box[0, 2]) / 2 - 0.75) < 0.01                              # past the empty pose slot
    d["boxes"][3:5, 1] = np.nan                                                        # lost for two frames:
    box, _, _ = FR.follow(d, d["pose_t"], union, cw=0.32)                             # held where they were
    assert not np.isnan(box).any() and np.allclose(box[3], box[2])


def test_a_couple_shares_the_frame_with_the_follow_featured():
    from focalcut.reel import frame as FR
    d, union = _couple(faces=(True, False), apart=0.12)          # dancing together: x 0.2-0.3 and 0.32-0.42
    box, face, one = FR.follow(d, d["pose_t"], union, cw=0.32)
    assert one and face.all()
    assert box[0, 0] <= 0.2 and box[0, 2] >= 0.42                        # both in, room on one side
    assert box[0, 2] - box[0, 0] == pytest.approx(0.245)
    n = len(d["pose_t"])
    d["role"] = dict(t=np.repeat(d["pose_t"], 2).astype(np.float32), p=np.tile([0.1, 0.9], n).astype(np.float32),
                     box=np.tile([[0.2, 0.2, 0.3, 0.9], [0.32, 0.2, 0.42, 0.9]], (n, 1)).astype(np.float32))
    box, _, _ = FR.follow(d, d["pose_t"], union, cw=0.32)                # the right one reads as a woman: the
    assert box[0, 0] == pytest.approx(0.2) and box[0, 2] > 0.42 + 0.02   # room goes on her side
    d["role"]["p"] = np.tile([0.9, 0.1], n).astype(np.float32)
    box2, _, _ = FR.follow(d, d["pose_t"], union, cw=0.32)               # and the other way round
    assert box2[0, 0] < 0.2 - 0.02 and box2[0, 2] == pytest.approx(0.42)
    d["role"]["p"] = np.tile([0.1, 0.9], n).astype(np.float32)
    d2, _ = _couple(faces=(True, False), apart=0.12)                     # no role scan: the shorter one
    d2["boxes"][:, 0, 1] = 0.1 * 360
    box2, _, _ = FR.follow(d2, d2["pose_t"], union, cw=0.32)
    assert box2[0, 2] > 0.42 + 0.02
    box, _, _ = FR.follow(d, d["pose_t"], np.tile([0.2, 0.2, 0.3, 0.9], (n, 1)), cw=0.32)
    assert box[0, 2] > 0.42                                              # the subject pass caught only the lead
    box, _, _ = FR.follow(d, d["pose_t"], union, cw=0.2)                 # wider than the crop: the follow whole,
    assert box[0, 0] <= 0.32 and box[0, 2] >= 0.42 and box[0, 2] - box[0, 0] == pytest.approx(0.18)
    assert box[0, 0] < 0.3                                               # and part of the lead


def test_shake_hf_ignores_pans_and_keeps_one_value_per_frame():
    from focalcut.reel.shake import hf
    pan = np.tile([0.004, 0.0], (60, 1))
    jolt = np.tile([[0.004, 0.0], [-0.004, 0.0]], (30, 1))
    assert hf(pan).max() < 1e-9 and hf(jolt).mean() > 1e-3
    assert len(hf(pan[:7])) == 7 and len(hf(np.zeros((0, 2)))) == 0
    assert hf(jolt)[0] == pytest.approx(hf(jolt)[30], rel=0.3)    # no zero-padding dip at the ends


def test_stabilise_runs_after_rotation_before_the_crop_and_changes_the_cache_key(tmp_path, monkeypatch):
    from focalcut.reel import render
    monkeypatch.setattr(render, "stab", lambda path, t, dur, rot: "/tmp/x: y.trf")
    pr = Project(str(tmp_path))
    s = dict(unit=dict(id="C1", source="C1.MP4", rotate="transpose=2", role="dance"), src_in=1.0, length=2.0,
             frames=100, fx=[], crop=dict(w=0.5, h=1.0, z=1.0, path=[[0, 0.5, 0.5]]), peak=1.0, kind="spin", stab=True)
    f = render.shot_filter(s, "portrait", pr)
    assert f.index("transpose") < f.index("vidstabtransform") < f.index("crop=") and "x\\: y.trf" in f
    assert render.key(s, 0.6) != render.key(dict(s, stab=False), 0.6) != render.key(dict(s, stab=False), 0.5)


def test_ending_never_starts_before_its_clip_and_band_inserts_never_touch_broll(tmp_path):
    pr, music, moments = fake_reel(tmp_path, n_units=20)
    moments[19]["unit"]["tag"] = "violin"
    moments[18]["unit"]["clip"] = moments[18]["unit"]["id"] = moments[18]["id"] = "END"
    pr.config = {"plan": {"landscape": {"ending": ["END", 0.0, 3.0]}}}
    shots = P.build(pr, "landscape", music, moments, assess_shots=False)["shots"]
    assert shots[-1].get("ending") and shots[-1]["src_in"] >= shots[-1]["unit"]["t0"]
    for k, s in enumerate(shots):
        if s.get("band"):
            assert all(shots[j]["unit"]["role"] != "transition" or shots[j].get("band") for j in (k - 1, k + 1))


def test_picks_take_single_times_and_hours():
    rec = dict(id="U1", unit=dict(id="U1", t0=0.0, t1=4000.0),
               moments=[dict(t=20.0, score=1.0), dict(t=3725.0, score=1.0)])
    P.apply_picks([rec], {"U1": {"ban": ["0:20"], "prefer": ["1:02:05"]}})
    assert [m["t"] for m in rec["moments"]] == [3725.0] and rec["moments"][0]["pick"]


def test_plan_on_joined_songs_cuts_on_their_real_beats(tmp_path):
    # two songs at different tempos (music.join): cuts land on each song's own beats, the acts sit on the songs
    pr, _, moments = fake_reel(tmp_path)
    a, b = list(np.arange(0, 32 * 4 * 0.6, 0.6)), list(32 * 4 * 0.6 + np.arange(0, 40 * 4 * 0.5, 0.5))
    beats = [0.5 + x for x in a + b]
    music = dict(song="mix.wav", beats=beats, start=0.5, length=beats[-1] + 0.5 - 0.5, beat=0.55,
                 sections=[dict(song="a"), dict(song="b")],
                 acts=[["teaser", 0, 2], ["arrival", 2, 16], ["shows", 16, 32], ["night", 32, 48], ["climax", 48, 62],
                       ["finale", 62, 72]])
    plan = P.build(pr, "landscape", music, moments, stinger_len=0.0)
    grid = [x - beats[0] for x in beats] + [beats[-1] + 0.5 - beats[0]]
    rec = 0
    for s in plan["shots"]:
        assert s["rec_in"] == pytest.approx(grid[rec], abs=1e-3)
        assert s["frames"] == round(grid[rec + s["beats"]] * P.FPS) - round(grid[rec] * P.FPS)
        assert s["beat"] == pytest.approx((grid[rec + s["beats"]] - grid[rec]) / s["beats"], abs=1e-4)
        rec += s["beats"]
    assert rec == len(beats)
    assert plan["acts"]["night"][0] == pytest.approx(grid[32 * 4], abs=1e-3)   # the second song's first beat


def test_join_lets_an_act_run_on_into_the_next_song(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    from focalcut.reel.music import join
    sr = 22050
    for name, bpm in (("a", 120), ("b", 100)):                         # far apart in tempo: a clean cut between them
        t = np.arange(60 * sr) / sr
        click = np.exp(-60 * (t % (60 / bpm))) * np.sin(2 * np.pi * 1000 * t)
        sf.write(tmp_path / f"{name}.wav", (0.6 * click).astype(np.float32), sr)
    songs = [dict(song=str(tmp_path / "a.wav"), bars=8, acts={"teaser": 2, "night": None}),
             dict(song=str(tmp_path / "b.wav"), bars=8, acts={"night": None})]
    music, _ = join(songs, max_len=60)
    assert [a[0] for a in music["acts"]] == ["teaser", "night"]       # one night act, over both songs
    assert music["acts"][1][1:] == [2, 16]


def test_drops_find_the_bass_coming_back_after_a_build():
    pytest.importorskip("librosa")
    from focalcut.reel.music import drops, song_window
    sr, bar = 22050, 2.0
    t = np.arange(int(48 * bar * sr)) / sr
    hats = 0.2 * np.sin(2 * np.pi * 6000 * t)
    bass = 0.8 * np.sin(2 * np.pi * 55 * t) * ((t >= 16 * bar) & (t < 24 * bar) | (t >= 32 * bar))
    bars = np.arange(0, 48 * bar + 1e-9, bar)
    d = drops((hats + bass).astype(np.float32), sr, bars)
    assert int(np.argmax(d)) in (16, 32)
    s, e = song_window(dict(bars=bars.tolist()), d, 24)
    assert e - s == 24 and s <= int(np.argmax(d)) < e


def test_mix_hands_over_the_dj_way(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")

    from focalcut.reel.music import mix
    sr = 48000
    for name, f in (("a", 220), ("b", 330)):
        sf.write(tmp_path / f"{name}.wav", 0.5 * np.sin(2 * np.pi * f * np.arange(20 * sr) / sr), sr)

    def beats(src, out, n):
        return [[src + 0.5 * i, out + 0.5 * i] for i in range(n + 1)]
    secs = [dict(song=str(tmp_path / "a.wav"), src_start=2.0, src_end=10.0, at=0.5, out="mix", bar=2.0,
                 map=[[1.5, 0.0], *beats(2.0, 0.5, 20)], fade_out=[6.5, 8.5, 10.5]),
            dict(song=str(tmp_path / "b.wav"), src_start=4.0, src_end=12.0, at=8.5, out="end", bar=2.0,
                 map=[*beats(2.0, 6.5, 20), [13.0, 17.5]], fade_in=[6.5, 8.5, 10.5])]
    y, _ = sf.read(mix(secs, str(tmp_path / "m.wav")))

    def rms(a, b):
        return float(np.sqrt((y[int(a * sr):int(b * sr), 0] ** 2).mean()))
    assert rms(0.6, 6.0) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)          # a alone
    assert rms(11.0, 16.0) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)        # b alone
    assert min(rms(t, t + 0.25) for t in np.arange(6.5, 10.5, 0.25)) > 0.3     # no dip through the overlap
    assert len(y) == pytest.approx(17.5 * sr, abs=sr * 0.01)


@needs_ffmpeg
def test_warp_lays_beats_on_their_output_times():
    pytest.importorskip("librosa")
    from focalcut.reel.music import ramp, warp
    sr = 48000
    y = np.zeros((10 * sr, 2), np.float32)
    for t in np.arange(0.5, 9.5, 0.5):
        y[int(t * sr):int(t * sr) + 100] = 0.8
    src = np.arange(0.5, 9.01, 0.5)
    dur = np.full(len(src) - 1, 0.5)
    dur[-8:] = ramp(dur[-8:], 0.45, 8)                                         # speeds up into the next song
    bmap = np.column_stack([src, 0.5 + np.concatenate([[0], np.cumsum(dur)])])
    out = warp(y[int(0.5 * sr):], sr, bmap)
    assert len(out) == pytest.approx((bmap[-1, 1] - 0.5) * sr, abs=2)
    for o in bmap[1:-1, 1]:                                                    # each click lands on its new time
        k = int((o - 0.5) * sr)
        hit = np.argmax(np.abs(out[k - 480:k + 480, 0]) > 0.3)
        assert abs(hit - 480) < 0.004 * sr


def test_late_categories_wait_for_the_climax(tmp_path):
    pr, music, moments = fake_reel(tmp_path, n_units=20)
    for r in moments[16:]:
        r["category"] = r["unit"]["category"] = "headline"
    pr.config = {"plan": {"landscape": {"late": ["headline"]}}}
    plan = P.build(pr, "landscape", music, moments, stinger_len=0.0)
    late = {r["unit"]["clip"] for r in moments[16:]}
    early_acts = {s["act"] for s in plan["shots"] if s["unit"]["clip"] in late}
    assert early_acts and early_acts <= {"climax", "finale"}


@needs_ffmpeg
def test_overlay_pulses_on_the_beat_and_flashes_on_drops(tmp_path):
    sf = pytest.importorskip("soundfile")

    from focalcut.reel import overlay
    from focalcut.reel.render import ENC, FPS, read_frames
    sr, beat = 48000, 0.5
    t = np.arange(4 * sr) / sr
    y = 0.3 * np.sin(2 * np.pi * 60 * t) * np.exp(-(t % beat) * 12) + 0.05 * np.sin(2 * np.pi * 2000 * t)
    sf.write(tmp_path / "song.wav", y, sr)
    src = tmp_path / "v.mp4"
    _synth(src, 3, 320, 180, FPS, source="color")
    beats = list(np.arange(0, 4, beat))
    music = dict(beats=beats, bars=beats[::4], beat_energy=[1.0] * len(beats),
                 sections=[dict(at=0.0), dict(at=2.0)])               # a song join at 2 s: a drop
    plan = dict(music_start=0.0, beat=beat, shots=[dict(act="night", frames=75), dict(act="climax", frames=75)])
    out = tmp_path / "o.mp4"
    beat_only = overlay.options({"presets": [[[], "neon"]]})
    overlay.apply(str(src), str(out), plan, music, str(tmp_path / "song.wav"), beat_only, ENC, FPS)
    lum = np.array([f.mean() for f in read_frames(str(out))])
    assert len(lum) == 150
    assert lum[100] > lum[110] + 40                                   # the drop's white flash, then gone
    assert lum[50] > lum[60]                                          # every beat blinks
    assert overlay.options(None) is None and overlay.options({"strength": {"night": 0}})["strength"]["climax"] == 1
    every = sorted({e for p in overlay.PRESETS + overlay.CALM for e in p[0]})       # every effect and palette runs
    show = overlay.Show(320, 180, FPS, beat, np.random.default_rng(0))
    f = np.full((180, 320, 3), 40, np.uint8)
    for pal in [*overlay.HEX, "rainbow"]:
        show.palette(pal)
        for k in range(3):
            c = dict(t=k / FPS, bass=0.8, bands=np.full(overlay.NB, 0.7), kick=1.0, since=0.0, down=True, drop=0,
                     bar=k, beat_now=True, frame=f)
            for e in every:
                assert getattr(show, e)(c).shape[:2] == (90, 160), e


def test_flickers_build_into_drops_hit_the_loudest_beats_and_take_the_editors_times():
    pytest.importorskip("cv2")
    from focalcut.reel import overlay as O
    beats = np.arange(0, 60, 0.5)
    e = np.ones(len(beats)); e[[40, 44, 90]] = 3.0                    # hard hits at 20 s, 22 s (too close) and 45 s
    G = dict(beats=beats, energy=e, drops=[20, 22, 70])               # drops at 10 s, 11 s (straight after) and 35 s
    S = np.full(60 * 25, 1.0, np.float32)
    f = O.flickers(G, S, O.options({"flicker": {"at": [["0:50.1", "dark", 2]]}}), 25)
    W, D = O.WHITE, O.DARK
    assert [f.get(i) for i in range(16, 20)] == [W, W, W, D] and 21 not in f   # one build, dark just before the drop
    assert f[66] == W and f[69] == D
    assert f[40] == W and 44 not in f and f[90] == D                  # hits alternate, 4 bars apart at least
    assert f[100] == f[101] == D and 102 not in f                     # the editor's: nearest beat, 2 in a row
    assert O.flickers(G, S, O.options({"flicker": False}), 25) == {}
    x = np.full((4, 4, 3), 100.0)
    assert O.flicker(x, W, 0.0, 0.5, 25).min() > 240 and O.flicker(x, D, 0.0, 0.5, 25).max() < 10
    assert np.allclose(O.flicker(x, W, 0.25, 0.5, 25), x, atol=1)      # gone within half a beat


def test_clarity_favours_defined_subjects_over_murky_ones():
    from focalcut.reel.moments import clarity
    c = lambda luma, contrast, ring: float(clarity(np.array(luma), np.array(contrast), np.array(ring)))  # noqa: E731
    lit, silhouette, murky, blown = c(120, 50, 90), c(30, 60, 150), c(40, 18, 45), c(245, 10, 200)
    assert lit == silhouette == 1.0
    assert murky < 0.25 and blown < 0.2
    assert c(40, 35, 45) > 0.8 and c(35, 20, 80) > 0.6               # dim but defined: detail, or set apart


def test_everyone_and_every_kiss_gets_a_shot(tmp_path):
    pr, music, moments = fake_reel(tmp_path, n_units=20)
    people, kisses = [], {"U05": [[40.0, 0.98], [70.0, 0.5]]}          # the second kiss is unsure: not required
    for i in range(30):                                                # 30 people, each seen in one unit only
        u = f"U{i % 20:02d}"
        t = 10.0 + 2.5 * (i // 20) + 30 * (i % 3)
        apps = [dict(unit=u, t=t + d, cx=0.5, cy=0.4, h=0.12, q=0.8) for d in (0.0, 0.5)]
        people.append(dict(id=i + 1, units=[u], best=apps[0], appearances=apps))
    os.makedirs(os.path.dirname(pr.path("people.json")), exist_ok=True)
    with open(pr.path("people.json"), "w") as f:
        json.dump(dict(people=people, kisses=kisses), f)
    plan = P.build(pr, "landscape", music, moments, stinger_len=0.0)
    shots, beat = plan["shots"], plan["beat"]

    def shown(p):
        return any(s["unit"]["id"] == a["unit"] and abs(s["t"] - a["t"]) <= s["beats"] * beat / 2 + 0.25
                   for s in shots for a in p["appearances"])
    assert sum(map(shown, people)) >= 27                               # (a few may have no free slot)
    assert any(s["unit"]["id"] == "U05" and abs(s["t"] - 40.0) <= s["beats"] * beat / 2 + 0.25 for s in shots)
    assert len({s["unit"]["id"] for s in shots}) == 20                 # still every unit
    assert sum(s["beats"] for s in shots) == sum(s["beats"] for s in P.build(
        pr, "landscape", music, copy.deepcopy(moments), stinger_len=0.0)["shots"])


def test_faces_are_tracked_through_lighting_changes():
    pytest.importorskip("scipy")
    from focalcut.reel.people import tracks_in
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=128), rng.normal(size=128)
    emb = []
    for k in range(6):                                                 # person A drifts in embedding, steady in place
        e = a + rng.normal(size=128) * (0.4 * k)
        emb.append(e / np.linalg.norm(e))
    emb.append(b / np.linalg.norm(b))                                  # someone else, across the frame
    d = dict(t=np.array([0, 0.5, 1, 1.5, 2, 2.5, 1.0]), cx=np.array([0.3] * 6 + [0.8]),
             cy=np.full(7, 0.4), h=np.full(7, 0.15), emb=np.array(emb, np.float32))
    groups = sorted(sorted(g.tolist()) for g in tracks_in(d))
    assert groups == [[0, 1, 2, 3, 4, 5], [6]]


def test_effects_clear_the_centre_of_the_frame():
    pytest.importorskip("cv2")
    from focalcut.reel.overlay import edge_mask
    m = edge_mask(320, 180, 0.3)
    assert m[90, 160, 0] == 0 and m[0, 0, 0] == 1 and 0 < m[90, 250, 0] < 1
    p = edge_mask(180, 320, 0.42)                                      # portrait: clear higher up, where faces sit
    assert p[110, 90, 0] == 0 and p[319, 0, 0] == 1


def test_landed_and_hold_in_show_the_finish_and_the_rest(tmp_path):
    from focalcut.reel import velocity as V
    pr = Project(str(tmp_path))
    os.makedirs(pr.path("scores"))
    t = np.arange(0, 20, 0.1)
    subj = np.where((t > 9.0) & (t < 10.8), 4.0, 0.5)       # a trick from 9 s that lands at 10.8 s
    np.savez(pr.path("scores", "C1.npz"), t=t, subj=subj, size=np.array([3840, 2160]),
             boxes=np.zeros((1, 1, 5)), pose_t=np.zeros(1))
    u = dict(score="C1", t0=0.0, t1=20.0)
    land = V.landed(pr, u, 10.0)
    assert land == pytest.approx(10.8, abs=0.15)
    assert V.landed(pr, dict(u, score="missing"), 10.0) is None
    s_in, fits = V.hold_in(9.4, 1.2, land, 0.6)              # 1.2 s from 9.4 would cut 0.2 s after landing
    assert fits and s_in + 1.2 >= land + 0.6 - 1e-6 and s_in <= land - V.PRE + 1e-6
    s_in, fits = V.hold_in(10.0, 0.6, land, 0.6)             # one beat: too short for both, lands 40% in
    assert not fits and s_in == pytest.approx(land - 0.24)


def test_lands_borrows_a_beat_for_a_landing_from_a_plain_neighbour(tmp_path, monkeypatch):
    from focalcut.reel import velocity as V
    monkeypatch.setattr(V, "landed", lambda project, unit, t: t + 0.3)
    seq = [shot("spin", "A", beats=2), shot("spin", "B", beats=1), shot("shake", "C", beats=3)]
    seq[0]["fx"] = ["hero"]                                   # (a speed effect lands itself: no donor, no landing)
    P.lands(None, seq, 0.6, 0.6)
    assert [s["beats"] for s in seq] == [2, 2, 2]
    assert seq[0].get("land") is None and seq[1]["land"] == pytest.approx(10.3)


def test_every_export_file_is_in_the_reel_not_just_every_camera_clip(tmp_path):
    pr, music, moments = fake_reel(tmp_path, n_units=24)
    extra = []
    for r in [r for r in moments if r["role"] == "dance"]:   # each social clip split into four segment files
        for k in range(1, 4):
            x = copy.deepcopy(r)
            x["id"] = x["unit"]["id"] = f"{r['id']}-{k:02d}"
            extra.append(x)
    moments += extra
    shots = P.build(pr, "portrait", music, moments)["shots"]
    assert {s["unit"]["id"] for s in shots} == {r["id"] for r in moments}
    assert all(a["unit"]["clip"] != b["unit"]["clip"] for a, b in pairwise(shots))


def test_face_seen_trusts_the_face_scan_over_the_pose():
    from focalcut.reel import frame as FR
    d, _ = _couple(faces=(True, True))                       # the pose says both face the camera
    d["yf"] = dict(t=np.array([0.0, 0.0]), cx=np.array([0.75, 0.25]), cy=np.array([0.24, 0.6]),
                   h=np.array([0.08, 0.08]), score=np.array([0.95, 0.95]))   # one face, on dancer 2's head
    f = FR.face_seen(d, d["pose_t"][:2], d["boxes"][:2], d["kp"][:2])
    assert not f[0, 0] and f[0, 1]                          # (the other detection is at dancer 1's waist)
    assert not f[1].any() or f[1, 1]
    del d["yf"]
    assert FR.face_seen(d, d["pose_t"][:1], d["boxes"][:1], d["kp"][:1])[0, :2].all()


def test_close_only_keeps_matching_files_close_shots(tmp_path):
    pr = Project(str(tmp_path))
    os.makedirs(pr.path("scores"))
    np.savez(pr.path("scores", "faces_C1_crowd-day.npz"), t=np.array([10.0, 20.0, 30.0]), h=np.array([0.3, 0.05, 0.25]),
             score=np.array([0.95, 0.95, 0.5]))
    rec = dict(id="C1_crowd-day", unit=dict(id="C1_crowd-day", category="broll", tag="crowd-day"),
               moments=[dict(t=10.2), dict(t=20.0), dict(t=30.0), dict(t=40.0, add=True)])
    night = dict(id="C2", unit=dict(id="C2", category="night", tag="night"), moments=[dict(t=5.0)])
    P.close_only(pr, [rec, night], {"match": ["day"], "face": 0.2})
    assert [m["t"] for m in rec["moments"]] == [10.2, 40.0]     # a big confident face, and the editor's add
    assert len(night["moments"]) == 1                            # other footage untouched


def test_bans_keep_shot_windows_out_and_picks_keep_their_length():
    u = dict(t0=0.0, t1=60.0, bans=[[10.0, 12.0]])
    assert P.clear_of_bans(9.0, 1.5, u) == 8.5                     # would cover the ban: ends where it starts
    assert P.clear_of_bans(11.0, 1.5, u) == 12.0                   # nearer its end: starts where it ends
    assert P.clear_of_bans(8.0, 1.5, u) == 8.0                     # clear already
    seq = [dict(unit=dict(id=f"U{i % 3}", role="dance"), act="night", beats=b, rank=r, fx=[], t=0.0)
           for i, (b, r) in enumerate([(2, 0.9), (2, 0.1), (2, 0.5), (2, 0.8), (2, 0.2)])]
    seq[2].update(span=[0.0, 2.0], add=True, rank=0.5)           # a 2 s pick at 0.5 s a beat wants 4 beats
    P.fit_pick(seq, 2, 0.5)
    assert seq[2]["beats"] == 4 and sum(s["beats"] for s in seq) == 10
    assert [s["beats"] for s in seq if s is not seq[2]] == [2, 1, 2, 1]   # the two weakest gave a beat each


def test_a_pick_plays_once_never_sped_up():
    pick = dict(unit=dict(id="U1"), act="night", beats=3, fx=["flashin", "bullet"], effect="flashin", span=[1.0, 2.5])
    seq = [dict(unit=dict(id="U0"), act="night", beats=2, fx=[]), pick, dict(unit=dict(id="U2"), act="night", beats=2, fx=[]),
           dict(pick, fx=[], beats=2)]
    P.picks_once(seq)
    assert [s["beats"] for s in seq] == [2, 3, 4] and seq[1]["fx"] == ["bullet"]          # slow motion stays
    end = dict(pick, fx=[], beats=3, act="finale", ending=True)    # the pick is also the configured ending: it
    seq = [dict(pick, fx=[]), dict(unit=dict(id="U0"), act="night", beats=2, fx=[]), end]   # stays last, early copy goes
    P.picks_once(seq)
    assert seq[-1] is end and [s["beats"] for s in seq] == [5, 3]


def test_picks_survive_a_whole_plan(tmp_path):
    # the cases that broke on a real reel: a pick on a moment the planner wants anyway, a pick next to a ban, a long
    # pick, two picks in one unit, and a ban on a stretch whose moment peaks just outside it
    pr, music, moments = fake_reel(tmp_path)
    picks = {"U02": {"add": ["0:04.5-0:06.0", "0:40-0:42.4"]},          # 5.0 is U02's best moment
             "U06": {"add": ["0:11-0:17"]},                             # 6 s: plays whole, never sped up
             "U01": {"add": ["0:28.5-0:30"], "ban": ["0:22.6-0:24.2"]},  # a ban whose moment peaks at 23.0... and 17
             "U05": {"ban": ["0:16.2-0:18"]}}
    for fmt in ("landscape", "portrait"):
        plan = P.build(pr, fmt, music, moments, stinger_len=0.0, picks=picks, assess_shots=False)
        assert P.pick_report(plan["shots"], picks) == []


def test_heavy_picks_fill_the_reel_and_mark_its_ends(tmp_path):
    # the SSMC case: three selections in every show and social clip -- more than their acts hold, so the acts must
    # pass them on (none cut to a beat, the climax not left empty) -- with the opening and the ending marked as picks
    pr, music, moments = fake_reel(tmp_path)
    picks = {f"U{i:02d}": {"add": [f"0:{t:02d}-0:{t + 2}.40" for t in (10, 34, 58)]} for i in range(16) if i % 4 != 3}
    picks["U00"]["open"] = "0:10-0:12.40"                         # also one of its selections
    picks["U01"]["end"] = "0:58-1:00.40"
    for fmt in ("landscape", "portrait"):
        plan = P.build(pr, fmt, music, moments, stinger_len=0.0, picks=picks, assess_shots=False)
        shots = plan["shots"]
        assert P.pick_report(shots, picks) == []
        assert shots[0]["unit"]["id"] == "U00" and shots[0]["span"] == [10.0, 12.4] and shots[0]["logo"] == "in"
        assert shots[-1]["unit"]["id"] == "U01" and shots[-1].get("ending") and shots[-1]["span"] == [58.0, 60.4]
        acts = Counter(s["act"] for s in shots)
        assert all(acts[a] for a in P.ACT_NAMES), acts            # every act has shots
        assert max(s["beats"] for s in shots if not s.get("span")) <= 8   # no filler held to plug a gap
        assert sum(s["beats"] for s in shots) == round(plan["video_length"] / plan["beat"])


def test_pick_hold_cuts_picks_to_pieces_between_the_rest_of_the_footage(tmp_path):
    # "pick_hold": long picks play only their best few beats, a clip with many is condensed, and the picks take turns
    # with the rest of the footage (the rest of a picked clip included) instead of filling the acts
    pr, music, moments = fake_reel(tmp_path)
    picks = {f"U{i:02d}": {"add": [f"0:{t:02d}-0:{t + 6}" for t in (10, 30, 50)]} for i in range(12)}
    picks["U00"]["add"] += ["0:01-0:04", "0:05-0:08", "0:20-0:23", "0:24-0:27", "0:36-0:39", "0:40-0:43",
                            "0:44-0:47"]                          # one clip with far more than its share
    picks["U00"]["open"] = "0:10-0:12.40"
    picks["U01"]["end"] = "0:58-1:00.40"
    for fmt in ("landscape", "portrait"):
        pr.config["plan"] = {fmt: {"pick_hold": 3}}
        plan = P.build(pr, fmt, music, moments, stinger_len=0.0, picks=picks, assess_shots=False)
        shots, beat = plan["shots"], plan["beat"]
        assert P.pick_report(shots, picks, plan["picks_condensed"], only=False) == []
        assert plan["picks_condensed"] and all(c[0] == "U00" for c in plan["picks_condensed"])
        mine = [s for s in shots if s.get("whole")]
        assert mine and all(s["span"][1] - s["span"][0] <= 3 * beat + 1e-6 for s in mine)
        assert all(s["whole"][0] <= s["span"][0] and s["span"][1] <= s["whole"][1] for s in mine)
        assert sum(s["length"] for s in shots if s.get("span")) <= 0.6 * plan["video_length"]
        assert shots[0]["span"] == [10.0, 12.4] and shots[-1].get("ending") and shots[-1]["span"] == [58.0, 60.4]
        assert any(not s.get("span") and s["unit"]["id"] in picks for s in shots)   # the rest of a picked clip
        assert sum(s["beats"] for s in shots) == round(plan["video_length"] / plan["beat"])


def test_marking_a_pick_as_the_ending_overrides_reel_json(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    assert P.pick_ending({"_how": "x", "U02": {"end": "1:01.5-1:03", "add": ["1:01.5-1:03"]}}) == ("U02", 61.5, 63.0)
    assert P.pick_ending({"U02": {"end": "1:01.5-1:03", "ban": ["all"]}}) is None
    pr.config["plan"] = {"landscape": {"ending": ["U05", 40.0, 42.0, 0.3]}}
    plan = P.build(pr, "landscape", music, moments, stinger_len=0.0, picks={"U02": {"end": "0:22-0:24"}},
                   assess_shots=False)
    assert plan["shots"][-1]["unit"]["id"] == "U02" and plan["shots"][-1]["span"] == [22.0, 24.0]


def test_a_clip_with_selections_gives_only_those():
    u = dict(id="U1", t0=0.0, t1=60.0, export_t0=0.0)
    r = dict(id="U1", unit=u, moments=[dict(t=t, score=1.0, kind="dance") for t in (5.0, 20.0, 30.5, 50.0)])
    P.apply_picks([r], {"U1": {"add": ["0:30-0:32", "0:10"]}})
    assert sorted(m["t"] for m in r["moments"]) == [10.0, 31.0]       # the pick sits mid-range
    assert u["bans"][-3:] == [[-1.0, 8.5], [11.5, 30.0], [32.0, 61.0]]
    shots = [dict(unit=u, span=[30.0, 32.0], src_in=30.0, length=2.0, fx=[]), dict(unit=u, src_in=45.0, length=1.0, fx=[])]
    assert any("outside your selections" in x for x in P.pick_report(shots, {"U1": {"add": ["0:30-0:32"]}}))


def test_untouched_clips_keep_the_earlier_cuts_shots():
    def rec(uid):
        return dict(id=uid, unit=dict(id=uid), moments=[dict(t=t, score=1.0) for t in (5.0, 12.0, 30.0)])
    a, b, c = rec("A"), rec("B"), rec("C")
    shots = [dict(unit=dict(id="A"), t=12.0), dict(unit=dict(id="A"), t=20.0), dict(unit=dict(id="B"), t=5.0)]
    P.keep_previous([a, b, c], shots, {"B": {"ban": ["0:00-0:02"]}, "C": {"portrait": []}})
    assert [(m["t"], m["score"]) for m in a["moments"]] == [(12.0, 1.5), (20.0, 1.5)]
    assert len(b["moments"]) == 3 and len(c["moments"]) == 3     # the editor's clip; one the cut didn't use


def test_the_finale_reuses_picks_as_quick_cuts():
    def shot(uid, act, beats, span=None, role="dance", **kw):
        return dict(unit=dict(id=uid, clip=uid, role=role), act=act, beats=beats, t=0.0, fx=[], effect=None,
                    **({"span": span} if span else {}), **kw)
    seq = [shot("O", "teaser", 4, [0.0, 3.0], "transition", logo="in", open=True),   # the logo opening
           shot("A", "arrival", 2, [10.0, 13.0]), shot("N", "night", 2, [40.0, 42.0]),
           shot("K", "climax", 2, [20.0, 22.0]),
           shot("F", "finale", 2, [5.0, 8.0], "transition"), shot("G", "finale", 1, [30.0, 30.5]),
           shot("X", "finale", 9)]                                                      # 12 finale beats
    P.finale_reuse(seq, 0.5)
    fin = [s for s in seq if s["act"] == "finale"]
    assert sum(s["beats"] for s in fin) == 12
    assert fin[-1]["unit"]["id"] == "F" and fin[-1]["beats"] == 6 and not fin[-1].get("reuse")   # held, all of it
    cuts = [s for s in fin if s.get("reuse")]
    assert cuts and all(c["beats"] <= 2 for c in cuts)
    assert not any(c.get("logo") or c.get("open") for c in cuts)                     # no logo from the opening
    assert not any(c["unit"]["id"] == "K" and c["span"][0] == 20.0 for c in cuts)    # nothing seen moments ago


def test_slow_motion_comes_off_a_shot_too_short_for_it(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    shots = P.build(pr, "landscape", music, moments, stinger_len=0.0)["shots"]
    assert all(s["beats"] >= 3 for s in shots if set(s["fx"]) & {"hero", "bullet", "montage", "aura"})


def test_no_clip_plays_twice_in_a_row():
    seq = [dict(unit=dict(clip=c), act="climax") for c in "XAABBCY"]
    P.unclump_clips(seq)
    clips = [s["unit"]["clip"] for s in seq]
    assert clips[0] == "X" and clips[-1] == "Y" and all(a != b for a, b in __import__("itertools").pairwise(clips))


def test_a_whole_clip_ban_beats_its_picks():
    u = dict(id="U1", t0=0.0, t1=60.0, export_t0=0.0)
    r = dict(id="U1", unit=u, moments=[dict(t=t, score=1.0, kind="dance") for t in (5.0, 20.0)])
    picks = {"U1": {"ban": ["all"], "add": ["0:04-0:06"]}}
    P.apply_picks([r], picks)
    assert r["moments"] == [] and P.pick_report([], picks) == []


def test_settling_keeps_each_act_on_its_room_without_stretching_picks():
    def pick(uid, act, a, b, role="dance"):
        return dict(unit=dict(id=uid, clip=uid, role=role), act=act, beats=1, t=a, fx=[], effect=None, span=[a, b])
    # night: 3 picks of 2 s (4 beats each at 0.5 s) for 10 beats of room; climax: one 1 s pick for 6 beats of room
    seq = [pick("A", "night", 0, 2), pick("B", "night", 0, 2, "transition"), pick("C", "night", 0, 2),
           pick("D", "climax", 10, 11), dict(unit=dict(id="E", clip="E", role="dance"), act="climax", beats=1, t=0,
                                             fx=["hero"], effect="hero")]
    P.settle_picks(seq, 0.5, {"night": 10, "climax": 6}, 0.5)
    night = [s for s in seq if s["act"] == "night"]
    climax = [s for s in seq if s["act"] == "climax"]
    assert sum(s["beats"] for s in night) == 10 and sum(s["beats"] for s in climax) == 6
    assert next(s for s in climax if s["unit"]["id"] == "D")["beats"] == 2          # never past its out-point
    assert sum(s.get("reuse", False) for s in climax) >= 1                          # the rest: quick cuts
    assert all(s["act"] == "climax" for s in climax)


def test_pick_report_names_each_problem():
    u = dict(id="U1", export_t0=0.0)
    pick = dict(unit=u, span=[10.0, 12.0], src_in=10.0, length=2.0, fx=[])
    shots = [pick, dict(pick), dict(unit=u, src_in=11.0, length=1.0, fx=[]),
             dict(unit=u, span=[20.0, 23.0], src_in=20.0, length=1.4, fx=[]),
             dict(unit=u, span=[30.0, 31.0], src_in=30.0, length=1.0, fx=["flashin"]),
             dict(unit=u, src_in=40.0, length=1.0, fx=[])]
    rep = P.pick_report(shots, {"U1": {"add": ["0:10-0:12", "0:20-0:23", "0:30-0:31", "0:50-0:51"],
                                       "ban": ["0:39.5-0:41"]}})
    assert any("2 times" in r for r in rep) and any("other shot" in r for r in rep)
    assert any("plays 1.40 s of 3.00" in r for r in rep) and any("sped up" in r for r in rep)
    assert any("0:50-0:51: missing" in r for r in rep) and any("still used" in r for r in rep)


def test_empty_broll_still_has_moments():
    """B-roll with nobody in it (no person box, so no measured quality) still gets moments, ranked on steadiness."""
    from focalcut.reel import moments as Mo
    t = np.arange(0, 20, 1 / Mo.HZ)
    zero = np.zeros_like(t)
    S = dict(t=t, quality=zero, steady=np.full_like(t, 0.9), parts={k: zero for k in Mo.W})
    u = dict(id="C1_venue", category="broll", role="transition", orient="landscape", t0=0.0, t1=20.0)
    r = Mo.unit_moments(u, S)
    assert r["moments"] and all(m["kind"] == "broll" for m in r["moments"])


def test_act_logos_match_lengthens_a_short_title_shot_from_the_next():
    day, star = dict(id="d", category="day", tag=""), dict(id="s", category="headline", tag="")
    seq = [dict(unit=u, act="climax", beats=b, fx=[]) for u, b in ((day, 4), (star, 1), (day, 2), (day, 3))]
    P.act_logos(seq, {"climax": {"image": "star.png", "match": ["headline"]}})
    assert [s["beats"] for s in seq] == [4, 4, 1, 1] and seq[1]["logo_image"] == "star.png"


def test_act_logos_holds_a_title_over_short_shots():
    day, star = dict(id="d", category="day", tag=""), dict(id="s", category="headline", tag="")
    seq = [dict(unit=u, act="climax", beats=b, fx=[]) for u, b in ((star, 2), (day, 1), (day, 1), (day, 1), (day, 2))]
    P.act_logos(seq, {"climax": {"image": "star.png", "match": ["headline"]}})
    assert [s.get("logo") for s in seq] == ["in", "hold", "hold", None, None]
    assert {s.get("logo_image") for s in seq[:3]} == {"star.png"}


def test_framing_follows_the_action_not_a_bystander_beside_it():
    from focalcut.reel import frame as FR
    n = 12
    boxes = np.full((n, 3, 5), np.nan)
    kp = np.zeros((n, 3, 17, 3))
    for j, x in enumerate((0.3, 0.42, 0.7)):                   # a dancer, someone standing by, a dancer far off
        boxes[:, j] = [(x - 0.05) * 640, 0.2 * 360, (x + 0.05) * 640, 0.9 * 360, 0.9]
        kp[:, j, :, 0], kp[:, j, :, 1], kp[:, j, :, 2] = x * 640, np.linspace(80, 300, 17), 0.9
        kp[:, j, 5:7, 1] = 120                                  # shoulders level
    kp[:, 0, 9:11, 0] += np.where(np.arange(n) % 2, 30, -30)[:, None]    # the first one's arms are working
    d = dict(pose_t=np.arange(n) * 0.2, boxes=boxes, kp=kp, focus=np.full((n, 3), np.nan), aw=640, ah=360)
    act = FR.activity(d)
    assert act[5, 0] > 5 * act[5, 1]
    union = np.tile([0.25, 0.2, 0.47, 0.9], (n, 1)).astype(float)        # the subject pass: both of them
    box, _, one = FR.follow(d, d["pose_t"], union, cw=0.32)
    assert not one and box[5, 2] == pytest.approx(0.35)                   # just the dancer


def _set_shot(uid, cat, tag="", role="show", beats=2, clip=None):
    u = dict(id=uid, category=cat, tag=tag, role=role, clip=clip or uid)
    return dict(unit=u, act="night", beats=beats, fx=[], effect=None, t=1.0)


def test_confine_trades_strays_into_their_own_song():
    sets = [dict(b0=0, b1=8, footage=["day"]), dict(b0=8, b1=16, footage=["night"])]
    seq = [_set_shot("a", "day"), _set_shot("n1", "night"), _set_shot("b", "day", beats=4),
           _set_shot("d1", "day"), _set_shot("n2", "night", beats=4), _set_shot("n3", "night")]
    P.confine(P.Picker("landscape", []), seq, [], sets)
    at = P.starts(seq)
    assert all(P.fits(s["unit"], P.set_at(sets, at[i])["footage"]) for i, s in enumerate(seq))
    assert [s["beats"] for s in seq] == [2, 2, 4, 2, 4, 2]          # the slots keep their beats


def test_relocate_moves_a_files_only_shot_to_its_song():
    sets = [dict(b0=0, b1=8, footage=["day"]), dict(b0=8, b1=16, footage=["night"])]
    seq = [_set_shot("a", "day", beats=6), _set_shot("b", "day", beats=2), _set_shot("n1", "night", beats=4),
           _set_shot("lost", "day", beats=2), _set_shot("n2", "night", beats=2)]
    P.relocate(seq, sets)
    assert sum(s["beats"] for s in seq) == 16
    at = P.starts(seq)
    assert all(P.fits(s["unit"], P.set_at(sets, at[i])["footage"]) for i, s in enumerate(seq))
    assert [s["unit"]["id"] for s in seq].count("lost") == 1


def test_set_titles_put_each_artist_on_their_own_footage():
    sets = [dict(b0=0, b1=8, footage=["day", "anna"], titles=[dict(image="anna.png", match=["anna"])]),
            dict(b0=8, b1=16, footage=["night", "ben"], titles=[dict(image="ben.png", match=["ben"])])]
    seq = [_set_shot("d1", "day", beats=4), _set_shot("anna", "broll", "singer-anna", "transition", 4),
           _set_shot("n1", "night", beats=1), _set_shot("ben", "broll", "singer-ben", "transition", 1),
           _set_shot("n2", "night", beats=1), _set_shot("n3", "night", beats=5)]
    P.set_titles(P.Picker("landscape", []), seq, [], sets)
    assert seq[1]["logo_image"] == "anna.png" and seq[3]["logo_image"] == "ben.png"
    assert seq[3]["logo"] == "in" and seq[4]["logo"] == "hold" and seq[5]["logo"] == "hold"   # held to 4 beats


def test_fits_with_exclusions():
    u = dict(category="broll", tag="singer-steffania-day")
    assert P.fits(u, ["day"]) and not P.fits(u, ["day", "!steffania"]) and P.fits(u, []) and not P.fits(u, ["night"])


def test_band_footage_counts_only_where_a_player_is_in_shot_and_sharp(tmp_path):
    from focalcut.reel import role
    pr = Project(str(tmp_path))
    os.makedirs(pr.path("scores"), exist_ok=True)
    np.savez(pr.path("scores", "band_C1_band.npz"), t=np.arange(0, 10, 0.5, dtype=np.float32),
             p=np.r_[np.full(10, 0.95), np.full(10, 0.1)].astype(np.float32),                  # player, then crowd
             sharp=np.r_[np.full(5, 900.0), np.full(5, 30.0), np.full(10, 900.0)].astype(np.float32))  # a blur
    u = dict(id="C1_band", role="transition", tag="band", clip="C1")
    assert role.band_footage(pr, u) and not role.band_footage(pr, dict(u, tag="crowd"))
    p, sh = role.musician(pr, u, 0.0, 2.0)
    assert p >= role.BAND_P and sh >= role.BAND_SHARP                     # the player, in focus
    assert role.musician(pr, u, 2.6, 4.4)[1] < role.BAND_SHARP           # the player, blurred
    assert role.musician(pr, u, 6.0, 8.0)[0] < role.BAND_P               # the crowd
    assert role.musician(pr, dict(u, id="C2_band"), 0, 2) is None         # not scanned


def test_assess_ranks_down_a_walker_a_crowd_crossing_and_a_blocked_camera(tmp_path):
    pr, _, moments = fake_reel(tmp_path)
    social = next(r for r in moments if r["unit"]["clip"] == "C002")
    os.makedirs(pr.reel_dir, exist_ok=True)
    seed = {5.0: [1.0, 1.0, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0],    # walking across: arms still about the shoulders
            11.0: [1.0, 1.0, 1.0, 1.0, 1.2, 0.7, 0.0, 0.0, 1.0],   # people crossing through the crop one after another
            17.0: [1.0, 1.0, 1.0, 1.0, 1.2, 0.0, 0.5, 0.0, 1.0]}   # someone's back right up against the camera
    json.dump({f"portrait|U02|{m['t']}|c2": seed.get(m["t"], [1.0, 1.0, 1.0, 1.0, 1.2, 0.0, 0.0, 0.0, 1.0])
               for m in social["moments"]},
              open(pr.path("hold_cache.json"), "w"))
    P.assess(pr, [social], "portrait", P.config(pr, "portrait"))
    sc = {m["t"]: m["score_p"] for m in social["moments"]}
    best = max(v for t, v in sc.items() if t not in seed)
    assert sc[5.0] < 0.2 * best and sc[11.0] < 0.25 * best and sc[17.0] < 0.2 * best


def test_cover_people_skips_a_spot_that_plays_badly(tmp_path, monkeypatch):
    pr, music, moments = fake_reel(tmp_path, n_units=20)
    people = [dict(id=i + 1, units=[f"U{i:02d}"], appearances=[dict(unit=f"U{i:02d}", t=40.0 + d, cx=0.5, cy=0.4,
                                                                    h=0.12, q=0.8) for d in (0.0, 0.5)])
              for i in range(4)]
    os.makedirs(os.path.dirname(pr.path("people.json")), exist_ok=True)
    json.dump(dict(people=people, kisses={}), open(pr.path("people.json"), "w"))
    monkeypatch.setattr(P, "spot_rater", lambda project, fmt: (lambda r, t: 0.05 if r["id"] == "U01" else 1.0))
    shots = P.build(pr, "landscape", music, moments, stinger_len=0.0)["shots"]
    added = {s["unit"]["id"] for s in shots if s.get("person")}
    assert "U01" not in added and added                  # a crowd crossing the crop: not that spot


def test_a_glitch_look_speaks_every_transition_and_effect_in_its_own_styles(tmp_path):
    from focalcut.fx.glitch import LOOKS
    pr, music, moments = fake_reel(tmp_path)
    pr.config["look"] = "clean"
    shots = P.build(pr, "landscape", music, moments, stinger_len=0.0)["shots"]
    look = LOOKS["clean"]
    trs = {s["transition"] for s in shots if s.get("transition")}
    assert trs and trs <= set(look["transitions"])                     # every reason, said in the look
    fx = [f for s in shots for f in s["fx"] if f not in P.SPEED]
    assert fx and set(fx) <= set(look["subject"]) | set(look["accents"])
    for i in range(1, len(shots) - 1):                                 # still one effect at a time
        if shots[i]["fx"] and not set(shots[i]["fx"]) & set(P.SPEED):
            assert not (set(shots[i - 1]["fx"]) | set(shots[i + 1]["fx"])) - set(P.SPEED) or shots[i].get("effect")


def test_the_editor_can_choose_the_opening_shot_and_a_short_ending(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    end_unit = next(r for r in moments if r["role"] == "dance")["unit"]["id"]
    pr.config["plan"] = {"landscape": {"ending": [end_unit, 20.0, 21.5]}}
    pr.config["open"] = {"U03": "0:30.00-0:33.00"}
    shots = P.build(pr, "landscape", music, moments, stinger_len=0.0)["shots"]
    first, last = shots[0], shots[-1]
    assert first["unit"]["id"] == "U03" and first.get("logo") == "in" and first["act"] == "teaser"
    assert first["span"] == [30.0, 33.0] and first["length"] >= 3.0 - 1e-6        # at its full length
    assert sum(s["unit"]["id"] == "U03" and s.get("span") == [30.0, 33.0] for s in shots) == 1
    assert last.get("ending") and last["beats"] == 3                               # 1.5 s at 0.6 s a beat
    assert abs(last["src_in"] - 20.0) < 0.05                                       # all of it, from its start
