"""Thread requests are safe and preserve search results across all indexes."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

MAX_THREADS = int(np.iinfo(np.uintp).max)


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("build_threads", [0, 1, MAX_THREADS])
def test_build_and_search_thread_counts(kind, metric, build_threads):
    rng = np.random.default_rng(431)
    data = rng.standard_normal((96, 64)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    if kind == "ivf":
        index = IvfIndex(64, len(data), 1, nbits=8, metric=metric)
        search_args = {"nprobe": 1}
    elif kind == "hnsw":
        index = HnswIndex(
            64, len(data), M=16, ef_construction=96, nbits=8, metric=metric
        )
        search_args = {"ef": 96}
    else:
        index = SymqgIndex(64, max_degree=32, metric=metric)
        search_args = {"ef": 96}
    if kind == "qg":
        index.build(data, ef_construction=96, num_threads=build_threads)
    else:
        index.build(
            data,
            data.mean(axis=0, keepdims=True),
            np.zeros(len(data), dtype=np.uint32),
            num_threads=build_threads,
        )

    # Compare searches on the same graph; parallel construction may change its topology.
    expected_ids, expected_distances = index.search(
        data, k=5, num_threads=1, **search_args
    )
    np.testing.assert_array_equal(expected_ids[:, 0], np.arange(len(data)))
    assert np.isfinite(expected_distances).all()
    for threads in (0, MAX_THREADS):
        ids, distances = index.search(data, k=5, num_threads=threads, **search_args)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(distances, expected_distances)
        # A small or empty query batch must remain valid with an oversized request.
        for count in (0, 1):
            ids, distances = index.search(
                data[:count], k=5, num_threads=threads, **search_args
            )
            np.testing.assert_array_equal(ids, expected_ids[:count])
            np.testing.assert_array_equal(distances, expected_distances[:count])
