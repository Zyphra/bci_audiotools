"""Fast random access into MNE ``.fif`` raw files.

``mne.io.read_raw_fif(path, preload=False)`` is lazy about the *data*, but it
still has to locate every data buffer, and MNE-written files carry no tag
directory, so it walks the file tag by tag. Python opens files with a buffer of
``st_blksize`` bytes, which is 4 MiB on Lustre, so that walk ends up reading
the whole file. Opening a file per training example therefore costs a full
file read per example.

Here each file is parsed once into a small index entry (where the data buffers
are, sample rate, EEG picks, calibrations). With that entry an excerpt is a
single ``pread`` of exactly the rows needed. The files themselves stay plain
MNE-readable ``.fif``; the index is only a cache and can always be rebuilt.
"""
import hashlib
import multiprocessing
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np

# FIFFT_* type codes of raw data buffers -> numpy dtype (FIF is big-endian).
_DTYPES = {2: ">i2", 16: ">i2", 3: ">i4", 4: ">f4", 5: ">f8"}
_TAG_HEADER = 16  # bytes: kind, type, size, next


def index_fif(path: str, buffering: int = 0):
    """Parse one FIF file and return a small picklable index entry.

    Parameters
    ----------
    path : str
        Path to a raw ``.fif`` file.
    buffering : int, optional
        Buffer size used while walking the file's tags, by default 0
        (unbuffered), which reads only the 16-byte tag headers.

    Raises
    ------
    ValueError
        If the file layout is not supported (split files, data skips,
        mixed buffer types). Such files must be read with MNE instead.
    """
    import mne
    from mne._fiff.constants import FIFF
    from mne._fiff.meas_info import read_meas_info
    from mne._fiff.open import fiff_open
    from mne._fiff.tree import dir_tree_find

    with open(path, "rb", buffering=buffering) as f, mne.utils.use_log_level("ERROR"):
        fid, tree, _ = fiff_open(f)
        info, meas = read_meas_info(fid, tree)
        node = dir_tree_find(meas, FIFF.FIFFB_RAW_DATA)
        if len(node) != 1:
            raise ValueError(f"Expected exactly one raw data block: {path}")
        ents = node[0]["directory"]
        if any(e.kind == FIFF.FIFF_DATA_SKIP for e in ents):
            raise ValueError(f"FIF files with data skips are not supported: {path}")
        for ref in dir_tree_find(tree, FIFF.FIFFB_REF):
            if any(e.kind == FIFF.FIFF_REF_FILE_NAME for e in ref["directory"]):
                raise ValueError(f"Split FIF files are not supported: {path}")
        ents = [e for e in ents if e.kind == FIFF.FIFF_DATA_BUFFER]

    types = {e.type for e in ents}
    if len(types) != 1 or next(iter(types)) not in _DTYPES:
        raise ValueError(f"Unsupported data buffer types {types}: {path}")
    dtype = _DTYPES[next(iter(types))]
    nchan = int(info["nchan"])
    row = nchan * np.dtype(dtype).itemsize
    pos = np.array([e.pos + _TAG_HEADER for e in ents], np.int64)
    nsamp = np.array([e.size // row for e in ents], np.int64)
    bounds = np.concatenate([[0], np.cumsum(nsamp)])

    st = os.stat(path)
    entry = {
        "sfreq": float(info["sfreq"]),
        "n_times": int(bounds[-1]),
        "nchan": nchan,
        "dtype": dtype,
        "eeg_picks": np.asarray(
            mne.pick_types(info, meg=False, eeg=True, exclude="bads"), np.int32
        ),
        "channel_names": list(info["ch_names"]),
        "channel_xyz": np.array([ch["loc"][:3] for ch in info["chs"]], np.float32),
        "cals": np.array([ch["range"] * ch["cal"] for ch in info["chs"]], np.float64),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }
    # MNE writes equal-sized buffers back to back; store that as two numbers
    # instead of one position per buffer.
    uniform = len(pos) == 1 or (
        np.all(nsamp[:-1] == nsamp[0])
        and nsamp[-1] <= nsamp[0]
        and np.all(np.diff(pos) == nsamp[0] * row + _TAG_HEADER)
    )
    if uniform:
        entry["pos0"] = int(pos[0])
        entry["buf_nsamp"] = int(nsamp[0])
    else:
        entry["pos"] = pos
        entry["bounds"] = bounds
    return entry


def read_excerpt(path: str, entry: dict, start: int, stop: int, picks: List[int]):
    """Read samples ``start:stop`` of channels ``picks`` using an index entry.

    Returns the same array as
    ``mne.io.read_raw_fif(path).get_data(picks=picks, start=start, stop=stop)``:
    float64 of shape ``(len(picks), stop - start)``.
    """
    nchan = entry["nchan"]
    dtype = np.dtype(entry["dtype"])
    row = nchan * dtype.itemsize
    if "pos0" in entry:
        nbuf = entry["buf_nsamp"]
        bufs = np.arange(start // nbuf, (stop - 1) // nbuf + 1, dtype=np.int64)
        firsts = bufs * nbuf
        lasts = np.minimum(firsts + nbuf, entry["n_times"])
        pos = entry["pos0"] + bufs * (nbuf * row + _TAG_HEADER)
    else:
        bounds = entry["bounds"]
        b0 = int(np.searchsorted(bounds, start, side="right") - 1)
        b1 = int(np.searchsorted(bounds, stop - 1, side="right") - 1)
        pos = entry["pos"][b0 : b1 + 1]
        firsts, lasts = bounds[b0 : b1 + 1], bounds[b0 + 1 : b1 + 2]

    # The buffers are contiguous on disk, so one read covers all rows we need.
    lo = int(pos[0] + (start - firsts[0]) * row)
    hi = int(pos[-1] + (stop - firsts[-1]) * row)
    fd = os.open(path, os.O_RDONLY)
    try:
        blob = os.pread(fd, hi - lo, lo)
    finally:
        os.close(fd)
    if len(blob) != hi - lo:
        raise IOError(f"Short read from {path}; is the FIF index stale?")

    out = np.empty((len(picks), stop - start), np.float64)
    for p, first, last in zip(pos, firsts, lasts):
        a, b = max(start, first), min(stop, last)
        buf = np.frombuffer(
            blob, dtype, count=int(b - a) * nchan, offset=int(p + (a - first) * row) - lo
        ).reshape(-1, nchan)
        out[:, a - start : b - start] = buf[:, picks].T
    out *= entry["cals"][picks][:, None]
    return out


def _index_or_none(path):
    try:
        return index_fif(path)
    except Exception as e:  # unsupported or unreadable: leave it to MNE
        return f"{type(e).__name__}: {e}"


def _stat(path):
    try:
        st = os.stat(path)
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None


def _cache_file(paths: List[str], cache_dir: str = None):
    if cache_dir is None:
        cache_dir = os.environ.get(
            "AUDIOTOOLS_FIF_INDEX_DIR", "~/.cache/audiotools/fif_index"
        )
    key = hashlib.sha1("\n".join(paths).encode()).hexdigest()
    return Path(cache_dir).expanduser() / f"{key}.pkl"


def load_index(paths: List[str], cache_dir: str = None, num_workers: int = None):
    """Return ``{path: entry}`` for all ``paths``, indexing files as needed.

    Entries are cached on disk (one pickle per list of paths, in
    ``$AUDIOTOOLS_FIF_INDEX_DIR`` or ``~/.cache/audiotools/fif_index``) and are
    rebuilt for files whose size or modification time changed. Files that
    cannot be indexed are left out; read those with MNE.
    """
    paths = [str(p) for p in paths]
    cache_file = _cache_file(paths, cache_dir)
    index = {}
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                index = pickle.load(f)
        except Exception:
            index = {}

    with ThreadPoolExecutor(32) as pool:
        stats = dict(zip(paths, pool.map(_stat, paths)))
    todo = [
        p
        for p in paths
        if stats[p] is not None
        and (p not in index or "channel_xyz" not in index[p]
             or "channel_names" not in index[p]
             or (index[p]["size"], index[p]["mtime_ns"]) != stats[p])
    ]
    if todo:
        if num_workers is None:
            num_workers = min(32, os.cpu_count() or 1, len(todo))
        print(f"Indexing {len(todo)} FIF files -> {cache_file}")
        if num_workers > 1:
            with ProcessPoolExecutor(
                num_workers, mp_context=multiprocessing.get_context("fork")
            ) as pool:
                entries = list(pool.map(_index_or_none, todo, chunksize=8))
        else:
            entries = [_index_or_none(p) for p in todo]
        failed = [(p, e) for p, e in zip(todo, entries) if isinstance(e, str)]
        for p, e in zip(todo, entries):
            index.pop(p, None)
            if not isinstance(e, str):
                index[p] = e
        if failed:
            warnings.warn(
                f"{len(failed)} FIF files could not be indexed and will be read "
                f"with MNE (slow). First: {failed[0][0]}: {failed[0][1]}"
            )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_file)
    return {p: index[p] for p in paths if p in index}
