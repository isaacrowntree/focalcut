import numpy as np

from focalcut.reel import overlay


def test_grid_takes_one_song_energy_one_per_interval():
    beats = list(np.arange(0.5, 40.5, 0.5))
    music = dict(beats=beats, bars=beats[::4], beat_energy=[1.0] * (len(beats) - 1))   # music.analyse: per interval
    G = overlay.grid(music, dict(music_start=0.5), 50 * 39, 50)
    assert len(G["energy"]) == len(beats)
