from focalcut.reel import hw


def test_decoders_default_and_env(monkeypatch):
    monkeypatch.delenv("FOCALCUT_DECODERS", raising=False)
    assert hw.decoders({"qsv": 5, "cpu": 2}) == {"qsv": 5, "cpu": 2}
    monkeypatch.setenv("FOCALCUT_DECODERS", "qsv=1, cpu=4")
    assert hw.decoders({"qsv": 5, "cpu": 2}) == {"qsv": 1, "cpu": 4}


def test_only_windows_drives_go_to_the_windows_ffmpeg():
    assert hw.on_windows_drive("/mnt/j/Videos/x.MP4") and not hw.on_windows_drive("/home/u/x.MP4")
    assert not hw.on_windows_drive("/mnt/wsl/x.MP4")


def test_raw_rgb_encodes_carry_bt709_in_the_stream(tmp_path):
    """Frames piped in as raw RGB come out tagged BT.709 throughout, not "unspecified" for players to guess."""
    import shutil
    import subprocess

    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    out = str(tmp_path / "t.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x256", "-r", "50",
                    "-i", "-", *hw.X264, "-pix_fmt", "yuv420p", out], input=bytes(256 * 256 * 3 * 5), check=True)
    tags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=color_space,color_primaries,color_transfer", "-of", "csv=p=0", out],
                          capture_output=True, text=True).stdout.strip()
    assert tags == "bt709,bt709,bt709"
    hw.retag(out, str(tmp_path / "r.mp4"))
