import tempfile
from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from audiotools import AudioSignal
from audiotools.core import fif_index


def _write_fif(path, n_times, fmt="single", buffer_size_sec=1.0, seed=0):
    rng = np.random.RandomState(seed)
    info = mne.create_info(["a", "b", "stim", "c"], 256, ["eeg", "eeg", "stim", "eeg"])
    raw = mne.io.RawArray(rng.randn(4, n_times) * 1e-5, info, verbose=False)
    raw.save(path, fmt=fmt, buffer_size_sec=buffer_size_sec, overwrite=True, verbose=False)


@pytest.mark.parametrize(
    "n_times,fmt,buffer_size_sec",
    [(256 * 20 + 77, "single", 1.0), (1000, "double", 0.3), (100, "single", 1.0)],
)
def test_read_excerpt_matches_mne(n_times, fmt, buffer_size_sec):
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "x_raw.fif")
        _write_fif(path, n_times, fmt, buffer_size_sec)
        entry = fif_index.index_fif(path)
        raw = mne.io.read_raw_fif(path, preload=False, verbose=False)

        assert entry["n_times"] == raw.n_times == n_times
        assert entry["sfreq"] == raw.info["sfreq"]
        assert list(entry["eeg_picks"]) == [0, 1, 3]

        rng = np.random.RandomState(0)
        spans = [(0, n_times), (n_times - 1, n_times), (0, 1)]
        for _ in range(50):
            start = rng.randint(0, n_times)
            spans.append((start, rng.randint(start + 1, n_times + 1)))
        for start, stop in spans:
            picks = [int(rng.choice(4))]
            expected = raw.get_data(picks=picks, start=start, stop=stop)
            actual = fif_index.read_excerpt(path, entry, start, stop, picks)
            assert np.array_equal(actual, expected)
        stop = min(700, n_times)
        all_channels = fif_index.read_excerpt(path, entry, 3, stop, [0, 1, 2, 3])
        assert np.array_equal(all_channels, raw.get_data(start=3, stop=stop))


def test_excerpt_with_index_matches_excerpt_without():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "x_raw.fif")
        _write_fif(path, 256 * 30)
        entry = fif_index.load_index([path], cache_dir=d)[path]
        for seed in range(20):
            a = AudioSignal.excerpt(path, duration=5.0, state=seed, EEG=True)
            b = AudioSignal.excerpt(
                path, duration=5.0, state=seed, EEG=True, fif_entry=entry
            )
            assert torch.equal(a.audio_data, b.audio_data)
            assert a.metadata == b.metadata


def test_load_index_caches_and_refreshes():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "x_raw.fif")
        _write_fif(path, 2000)
        index = fif_index.load_index([path], cache_dir=d, num_workers=1)
        assert index[path]["n_times"] == 2000
        assert fif_index._cache_file([path], d).exists()

        # A file that changed on disk is re-indexed.
        _write_fif(path, 3000)
        index = fif_index.load_index([path], cache_dir=d, num_workers=1)
        assert index[path]["n_times"] == 3000

        # Files that cannot be indexed are left out instead of failing.
        bad = str(Path(d) / "bad_raw.fif")
        Path(bad).write_bytes(b"not a fif file")
        with pytest.warns(UserWarning, match="could not be indexed"):
            index = fif_index.load_index([path, bad], cache_dir=d, num_workers=1)
        assert list(index) == [path]
