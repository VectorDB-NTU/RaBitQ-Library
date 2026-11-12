"""Independent search parameters, GIL release, and fail-fast index access."""

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread
from time import monotonic, sleep

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

BUSY = "Index is busy: conflicting operation in progress"
CASES = [
    ("ivf", 1),
    ("ivf", 4),
    ("ivf", 32),
    ("ivf_hnsw", 4),
    ("ivf_flat_rabitq", 4),
    ("hnsw", 1),
    ("hnsw", 9),
    ("qg", 32),
    ("qg", 4),
    ("qg", 8),
]


def make_index(kind, bits=4, metric="l2", fill=True):
    rng = np.random.default_rng(713)
    data = rng.standard_normal((129, 65)).astype(np.float32)
    labels = (np.arange(len(data)) % 5).astype(np.uint32)
    centroids = np.stack([data[labels == i].mean(axis=0) for i in range(5)])
    if kind.startswith("ivf"):
        initializer = kind.removeprefix("ivf_") if "_" in kind else "flat"
        index = IvfIndex(65, 5, nbits=bits, metric=metric, initializer=initializer)
    elif kind == "hnsw":
        index = HnswIndex(
            65,
            num_clusters=5,
            M=8,
            ef_construction=64,
            nbits=bits,
            metric=metric,
            seed=42,
        )
    else:
        index = SymqgIndex(
            65, max_degree=32, ef_construction=64, metric=metric, nbits=bits
        )
    if fill:
        if kind == "qg":
            index.add(data, num_threads=1)
        else:
            index.set_centroids(centroids)
            index.add(data, cluster_ids=labels, num_threads=1)
    return index, data


def parameters(kind, wide):
    if kind.startswith("ivf"):
        return {"nprobe": 5 if wide else 1, "high_accuracy": wide}
    return {"ef": 96 if wide else 5}


@pytest.mark.parametrize("kind,bits", CASES)
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_concurrent_search_matches_serial_and_loaded(kind, bits, metric, tmp_path):
    index, _ = make_index(kind, bits, metric)
    # Exercise conversions and ownership of a temporary contiguous float32 copy.
    queries = np.random.default_rng(18).standard_normal((17, 130))[:, ::2]
    expected = [index.search(queries, 5, **parameters(kind, wide)) for wide in (0, 1)]
    path = str(tmp_path / "concurrent.index")
    index.save(path)
    for current in (index, type(index).load(path)):
        start = Barrier(4)

        def search(worker):
            wide = worker % 2
            start.wait(timeout=10)
            for count in (0, 1, 17, 17):
                actual = current.search(
                    queries[:count],
                    5,
                    num_threads=1 + worker // 2,
                    **parameters(kind, wide),
                )
                for result, reference in zip(actual, expected[wide], strict=True):
                    np.testing.assert_array_equal(result, reference[:count])

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(search, range(4)))


class PausedArray:
    """Hold an operation open deterministically during Python input conversion."""

    def __init__(self, array):
        self.array = array
        self.entered = Event()
        self.resume = Event()

    def __array__(self, dtype=None, copy=None):
        self.entered.set()
        if not self.resume.wait(timeout=15):
            raise RuntimeError("test did not release input conversion")
        return np.array(self.array, dtype=dtype, copy=True)


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
def test_active_search_allows_reads_and_rejects_mutations(kind, tmp_path):
    index, data = make_index(kind)
    queries = PausedArray(data[:3])
    summary = repr(index)
    options = parameters(kind, True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(index.search, queries, 5, **options)
        try:
            assert queries.entered.wait(timeout=10)
            assert index.dim == 65
            assert repr(index) == summary
            index.search(data[:1], 5, **options)
            index.save(str(tmp_path / "reading.index"))
            mutations = [lambda: index.add(data[:1])]
            if kind != "qg":
                mutations += [
                    lambda: index.remove([0]),
                    lambda: index.train(data),
                    lambda: index.set_centroids(data[:5]),
                ]
            if kind == "hnsw":
                mutations.append(lambda: index.resize(256))
            for mutate in mutations:
                with pytest.raises(RuntimeError, match=f"^{BUSY}$"):
                    mutate()
        finally:
            queries.resume.set()
        running.result(timeout=10)
    # Failed conflicts must not leave the access state busy.
    if kind == "hnsw":
        index.resize(256)
    elif kind == "qg":
        # add takes write access before it rejects a second graph.
        with pytest.raises(RuntimeError, match="does not support incremental addition"):
            index.add(data)
    else:
        index.add(data[:1])
    index.search(data[:1], 5, **options)


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
def test_active_mutation_rejects_reads_and_other_mutations(kind, tmp_path):
    # SymphonyQG adds all vectors at once, so its index starts empty.
    index, data = make_index(kind, fill=kind != "qg")
    vectors = PausedArray(data if kind == "qg" else data[:1])
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(index.add, vectors, num_threads=1)
        try:
            assert vectors.entered.wait(timeout=10)
            calls = [
                lambda: index.search(data[:1], 5, **parameters(kind, True)),
                lambda: index.save(str(tmp_path / "busy.index")),
                lambda: index.dim,
                lambda: len(index),
                lambda: repr(index),
                lambda: str(index),
                lambda: index.add(data[:1]),
            ]
            if kind != "qg":
                calls += [
                    lambda: index.remove([0]),
                    lambda: index.train(data),
                    lambda: index.set_centroids(data[:5]),
                ]
            if kind == "hnsw":
                calls.append(lambda: index.resize(256))
            for call in calls:
                with pytest.raises(RuntimeError, match=f"^{BUSY}$"):
                    call()
            other, other_data = make_index(kind)
            other.search(other_data[:1], 5, **parameters(kind, True))
        finally:
            vectors.resume.set()
        running.result(timeout=10)
    assert len(index) == (len(data) if kind == "qg" else len(data) + 1)
    index.search(data[:1], 5, **parameters(kind, True))


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
def test_search_releases_gil(kind):
    index, data = make_index(kind)
    queries = np.tile(data, (32, 1))
    ready, progress = Event(), Event()
    active = stopped = False
    options = parameters(kind, True)

    def python_worker():
        ready.set()
        while not active and not stopped:
            sleep(0.001)
        if active:
            progress.set()

    worker = Thread(target=python_worker)
    worker.start()
    previous = sys.getswitchinterval()
    try:
        assert ready.wait(timeout=10)
        # Suppress interpreter timeslices across the whole probe. Repeated native
        # calls allow a delayed worker to run; one short search is not a deadline.
        sys.setswitchinterval(100)
        deadline = monotonic() + 5
        # Arm with a Python assignment: signaling an Event here could itself
        # release the GIL on a contended lock and falsely satisfy the probe.
        active = True
        while not progress.is_set() and monotonic() < deadline:
            index.search(queries, 5, num_threads=1, **options)
        # Capture before restoring normal scheduling or joining the worker.
        released = progress.is_set()
    finally:
        stopped = True
        sys.setswitchinterval(previous)
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert released, "Python worker made no progress during native searches"


def test_native_exception_reacquires_gil_and_releases_access():
    index, data = make_index("qg")
    queries = data[:2].copy()
    queries[1, 0] = np.nan
    # The native search checks coordinates after releasing the GIL.
    with pytest.raises(ValueError, match="must contain only finite values"):
        index.search(queries, 5, ef=64)
    # add takes write access before it rejects a second graph.
    with pytest.raises(RuntimeError, match="does not support incremental addition"):
        index.add(data)
    ids, distances = index.search(data[:2], 5, ef=64)
    assert ids.shape == distances.shape == (2, 5)
