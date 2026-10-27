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
    ("qg", 0),
    ("qg", 4),
    ("qg", 8),
]


def make_index(kind, bits=4, metric="l2"):
    rng = np.random.default_rng(713)
    data = rng.standard_normal((129, 65)).astype(np.float32)
    labels = (np.arange(len(data)) % 5).astype(np.uint32)
    centroids = np.stack([data[labels == i].mean(axis=0) for i in range(5)])
    if kind.startswith("ivf"):
        initializer = kind.removeprefix("ivf_") if "_" in kind else "flat"
        index = IvfIndex(65, len(data), 5, bits, metric, initializer)
    elif kind == "hnsw":
        index = HnswIndex(65, len(data) + 8, 8, 64, bits, metric, 42)
    else:
        index = SymqgIndex(65, 32, metric, bits)

    def build(vectors=data):
        if kind == "qg":
            index.build(vectors, 64, 1)
        else:
            index.build(vectors, centroids, labels, 1)

    build()
    return index, data, build


def parameters(kind, wide):
    if kind.startswith("ivf"):
        return {"nprobe": 5 if wide else 1, "high_accuracy": wide}
    return {"ef": 96 if wide else 5}


@pytest.mark.parametrize("kind,bits", CASES)
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_concurrent_search_matches_serial_and_loaded(kind, bits, metric, tmp_path):
    index, _, _ = make_index(kind, bits, metric)
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
                method = (
                    current.search_batch
                    if kind == "qg" and worker >= 2
                    else current.search
                )
                actual = method(
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
    index, data, build = make_index(kind)
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
            mutations = [build]
            if kind != "qg":
                mutations += [lambda: index.add(data[:1]), lambda: index.remove([0])]
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
    else:
        build()
    index.search(data[:1], 5, **options)


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
def test_active_mutation_rejects_reads_and_other_mutations(kind, tmp_path):
    index, data, build = make_index(kind)
    vectors = PausedArray(data[:1] if kind == "hnsw" else data)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(index.add if kind == "hnsw" else build, vectors)
        try:
            assert vectors.entered.wait(timeout=10)
            calls = [
                lambda: index.search(data[:1], 5, **parameters(kind, True)),
                lambda: index.save(str(tmp_path / "busy.index")),
                lambda: index.dim,
                lambda: index.is_built,
                lambda: repr(index),
                lambda: str(index),
                build,
            ]
            if kind != "qg":
                calls += [lambda: index.add(data[:1]), lambda: index.remove([0])]
            if kind == "hnsw":
                calls.append(lambda: index.resize(256))
            for call in calls:
                with pytest.raises(RuntimeError, match=f"^{BUSY}$"):
                    call()
            other, other_data, _ = make_index(kind)
            other.search(other_data[:1], 5, **parameters(kind, True))
        finally:
            vectors.resume.set()
        running.result(timeout=10)
    assert index.is_built
    index.search(data[:1], 5, **parameters(kind, True))


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
def test_search_releases_gil(kind):
    index, data, _ = make_index(kind)
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
    index, data, build = make_index("qg")
    with pytest.raises(ValueError, match="SearchBuffer capacity is too large"):
        index.search(data[:2], 5, ef=int(np.iinfo(np.uintp).max))
    build()
    ids, distances = index.search(data[:2], 5, ef=64)
    assert ids.shape == distances.shape == (2, 5)
