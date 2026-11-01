from focalcut.reel import hw


def test_decoders_default_and_env(monkeypatch):
    monkeypatch.delenv("FOCALCUT_DECODERS", raising=False)
    assert hw.decoders({"qsv": 5, "cpu": 2}) == {"qsv": 5, "cpu": 2}
    monkeypatch.setenv("FOCALCUT_DECODERS", "qsv=1, cpu=4")
    assert hw.decoders({"qsv": 5, "cpu": 2}) == {"qsv": 1, "cpu": 4}


def test_only_windows_drives_go_to_the_windows_ffmpeg():
    assert hw.on_windows_drive("/mnt/j/Videos/x.MP4") and not hw.on_windows_drive("/home/u/x.MP4")
    assert not hw.on_windows_drive("/mnt/wsl/x.MP4")
