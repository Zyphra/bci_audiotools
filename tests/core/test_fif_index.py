import pickle
import tempfile
from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from audiotools import AudioSignal
from audiotools.core import fif_index
from audiotools.data.datasets_eeg import AudioDataset, AudioLoader


def _write_fif(path, n_times, fmt="single", buffer_size_sec=1.0, seed=0, bads=()):
    rng = np.random.RandomState(seed)
    info = mne.create_info(["a", "b", "stim", "c"], 256, ["eeg", "eeg", "stim", "eeg"])
    raw = mne.io.RawArray(rng.randn(4, n_times) * 1e-5, info, verbose=False)
    raw.set_montage(mne.channels.make_dig_montage(
        ch_pos={"a": [0.01, 0.02, 0.03], "b": [0.04, 0.05, 0.06], "c": [0.07, 0.08, 0.09]},
        coord_frame="head",
    ))
    raw.info["bads"] = list(bads)
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
        raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
        for seed in range(20):
            a = AudioSignal.excerpt(path, duration=5.0, state=seed, EEG=True)
            b = AudioSignal.excerpt(
                path, duration=5.0, state=seed, EEG=True, fif_entry=entry
            )
            assert torch.equal(a.audio_data, b.audio_data)
            assert a.metadata == b.metadata
            state = np.random.RandomState(seed)
            start = int(state.uniform(0, 25) * 256)
            channel = int(state.choice([0, 1, 3]))
            expected = raw.get_data(picks=[channel], start=start, stop=start + 1280)
            assert torch.equal(a.audio_data[0], torch.from_numpy(expected).float())
        raw.close()


@pytest.mark.parametrize("missing", ["channel_xyz", "channel_names"])
def test_load_index_caches_and_refreshes(missing):
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "x_raw.fif")
        _write_fif(path, 2000)
        index = fif_index.load_index([path], cache_dir=d, num_workers=1)
        assert index[path]["n_times"] == 2000
        assert fif_index._cache_file([path], d).exists()
        # Older indexes must gain coordinates even when the FIF is unchanged.
        del index[path][missing]
        with fif_index._cache_file([path], d).open("wb") as f:
            pickle.dump(index, f)
        index = fif_index.load_index([path], cache_dir=d, num_workers=1)
        assert missing in index[path]

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


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("bads", [(), ("b",)])
@pytest.mark.parametrize("all_channels", [False, True])
def test_eeg_coordinates(indexed, bads, all_channels, tmp_path, monkeypatch):
    path = str(tmp_path / "x_raw.fif")
    _write_fif(path, 256 * 30, bads=bads)
    monkeypatch.setenv("AUDIOTOOLS_FIF_INDEX_DIR", str(tmp_path))
    loader = AudioLoader(sources=[path], EEG=True, eeg_all_channels=all_channels,
                         eeg_coordinates=not all_channels)
    if not indexed:
        loader.fif_index.clear()
    item = AudioDataset(loader, 256, duration=5.0)[7]
    with mne.io.read_raw_fif(path, preload=False, verbose=False) as raw:
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
        state = np.random.RandomState(7)
        start = int(state.uniform(0, 25) * 256)
        if not all_channels:
            picks = [int(state.choice(picks))]
        expected = raw.get_data(picks=picks, start=start, stop=start + 1280)
        assert item["signal"].audio_data.shape == (1, len(picks), 1280)
        assert torch.equal(item["signal"].audio_data[0], torch.from_numpy(expected).float())
        assert item["signal"].metadata["offset"] == start / 256
        assert item["channel_names"] == [raw.ch_names[i] for i in picks]
        xyz = np.array([raw.info["chs"][i]["loc"][:3] for i in picks], np.float32)
        np.testing.assert_array_equal(item["channel_xyz"], xyz)
    if indexed:
        loader.fif_index[path]["channel_xyz"][picks[0]] = np.nan
        with pytest.raises(ValueError, match="Missing EEG coordinates"):
            AudioDataset(loader, 256, duration=5.0)[7]
