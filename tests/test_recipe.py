"""reel.json "recipe": a house style kept outside focalcut, layered under a project's own settings."""
import json

from focalcut.reel.project import Project


def test_a_recipe_sits_under_the_project_and_stays_out_of_its_reel_json(tmp_path, monkeypatch):
    rec = tmp_path / "recipes"
    rec.mkdir()
    (rec / "house.json").write_text(json.dumps({"resolution": "4k", "steady": {"still": 0.09, "swing": 0.3},
                                                "_about": "a note, not a setting"}))
    (rec / "film.json").write_text(json.dumps({"recipe": "house", "mode": "documentary",
                                               "steady": {"swing": 0.25}, "documentary": {"lead": 0.09}}))
    monkeypatch.setenv("FOCALCUT_RECIPES", str(rec))
    reel = tmp_path / "shoot" / "focalcut" / "reel"
    reel.mkdir(parents=True)
    (reel / "reel.json").write_text(json.dumps({"recipe": "film", "song": "x.wav", "documentary": {"lead": 0.07}}))
    pr = Project(str(tmp_path / "shoot"))
    c = pr.config
    assert c["mode"] == "documentary" and c["resolution"] == "4k" and "_about" not in c
    assert c["steady"] == {"still": 0.09, "swing": 0.25}           # merged all the way down: the film's over the
    assert c["documentary"] == {"lead": 0.07} and c["song"] == "x.wav"   # house's, the project's over both
    pr.config["name"] = "shoot"
    pr.save_config()
    saved = json.load(open(reel / "reel.json"))
    assert saved["recipe"] == "film" and saved["name"] == "shoot" and "resolution" not in saved and "mode" not in saved
