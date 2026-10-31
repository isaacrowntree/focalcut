"""Recap-reel recipe: units from an export folder, beat-exact planning, crop expressions, XML paths, CLI config.
No GPU, no models: scores and moments are synthesised."""
from __future__ import annotations

import json
import os
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
    assert x.startswith("if(lt(t\\,0.3)\\,(0.4+(0.5-0.4)*(t-0)/0.3)")
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
        cache[f"portrait|U02|{m['t']}"] = [1.0, 1.0, 0.9 if m["t"] == 5.0 else 0.0, 1.0]
    json.dump(cache, open(pr.path("hold_cache.json"), "w"))
    before = {m["t"]: m.get("score_p", m["score"]) for m in social["moments"]}
    P.assess(pr, [social], "portrait", P.config(pr, "portrait"))
    after = {m["t"]: m["score_p"] for m in social["moments"]}
    assert after[5.0] > before[5.0]                             # a face on the social floor ranks up
    assert after[11.0] < 0.25 * before[11.0]                    # a back ranks well down
    shaky = [m for m in social["moments"] if m["t"] > 46]
    assert all(m["jitter"] > 0.006 and m["score_p"] < 0.05 * before[m["t"]] for m in shaky)


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
    json.dump({f"portrait|U02|{m['t']}": [0.3 if m["t"] == 5.0 else 1.0, 1.0, 1.0, 1.0] for m in social["moments"]},
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
    d, union = _couple(faces=(True, False), apart=0.12)                               # close together: the group
    box, face, one = FR.follow(d, d["pose_t"], union, cw=0.32)
    assert not one and np.allclose(box, union)


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


def test_drops_find_the_bass_coming_back_after_a_build():
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


def test_mix_crossfades_matching_tempos_and_cuts_the_rest(tmp_path):
    import soundfile as sf

    from focalcut.reel.music import mix
    sr = 48000
    for name, f in (("a", 220), ("b", 330)):
        sf.write(tmp_path / f"{name}.wav", 0.5 * np.sin(2 * np.pi * f * np.arange(20 * sr) / sr), sr)
    secs = [dict(song=str(tmp_path / "a.wav"), src_start=2.0, src_end=10.0, at=0.5, out="xfade", bar=2.0),
            dict(song=str(tmp_path / "b.wav"), src_start=4.0, src_end=12.0, at=8.5, out="end", bar=2.0)]
    y, _ = sf.read(mix(secs, str(tmp_path / "m.wav"), tail=1.0))

    def rms(a, b):
        return float(np.sqrt((y[int(a * sr):int(b * sr), 0] ** 2).mean()))
    assert rms(0.6, 7.0) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)          # a alone
    assert rms(10.0, 16.0) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)        # b alone
    assert rms(8.3, 8.7) > 0.2                                                 # no dip through the crossfade
    assert len(y) == pytest.approx((8.5 + 8.0 + 1.0) * sr, abs=sr * 0.01)


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
    import soundfile as sf

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
