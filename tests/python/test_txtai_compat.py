"""txtai's rabitqlib backend, call for call, must keep working in 0.8.

Mirrors txtai.ann.dense.rabitq.RabitQ at neuml/txtai 4d5e05f (index, append,
delete, search, count, save, load). txtai trains centroids with scikit-learn;
any centroids exercise the same calls, so these come from the data. The only
warnings allowed are the compatibility layer's DeprecationWarnings.
"""

import math
import warnings

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex

SENTINEL = 0xFFFFFFFF


@pytest.fixture
def embeddings():
    vectors = np.random.default_rng(3).normal(size=(400, 64)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def index_like_txtai(mode, vectors):
    rows = vectors.shape[0]
    clusters = max(1, min(round(4 * math.sqrt(rows)), rows))
    centroids = np.ascontiguousarray(vectors[:clusters], dtype=np.float32)
    clusterids = np.ascontiguousarray(
        np.argmax(vectors @ centroids.T, axis=1), dtype=np.uint32
    )
    dim = vectors.shape[1]
    if mode == "ivf":
        backend = IvfIndex(
            dim=dim, max_elements=rows, num_clusters=clusters, nbits=1, metric="ip"
        )
    else:
        backend = HnswIndex(
            dim=dim,
            max_elements=rows,
            M=16,
            ef_construction=200,
            nbits=1,
            metric="ip",
            random_seed=100,
        )
    backend.build(vectors, centroids, clusterids)
    return backend


def append_like_txtai(backend, vectors):
    backend.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return backend.num_stored_points  # txtai's next id offset


def delete_like_txtai(backend, ids):
    ids = [uid for uid in ids if 0 <= uid < backend.num_stored_points]
    if ids:
        backend.remove(np.array(ids, dtype=np.int64))


def search_like_txtai(backend, mode, queries, limit):
    count = backend.num_points
    if not count:
        return [[] for _ in queries]
    k = min(limit, count)
    if mode == "ivf":
        nprobe = max(1, round(backend.num_clusters / 16))
        ids, distances = backend.search(queries, k, nprobe)
    else:
        ids, distances = backend.search(queries, k, None or 0)
    return [
        [
            (int(uid), float(1.0 - dist))
            for uid, dist in zip(uids, dists)
            if uid != SENTINEL
        ]
        for uids, dists in zip(ids.tolist(), distances.tolist())
    ]


@pytest.mark.parametrize("mode", ["ivf", "hnsw"])
def test_txtai_calls(mode, embeddings, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = index_like_txtai(mode, embeddings[:300])
        results = search_like_txtai(backend, mode, embeddings[:5], 3)
        assert [hits[0][0] for hits in results] == list(range(5))

        if mode == "ivf":
            # append and delete exist for ivf only.
            assert append_like_txtai(backend, embeddings[300:]) == 400
            delete_like_txtai(backend, [0, 1, 399, 1000])
            assert backend.num_points == 397
            results = search_like_txtai(backend, mode, embeddings[300:305], 3)
            assert results[0][0][0] == 300

        path = str(tmp_path / f"{mode}.index")
        backend.save(path)
        loaded = IvfIndex.load(path) if mode == "ivf" else HnswIndex.load(path)
        assert search_like_txtai(loaded, mode, embeddings[:5], 3) == (
            search_like_txtai(backend, mode, embeddings[:5], 3)
        )
    categories = {w.category for w in caught}
    assert categories == {DeprecationWarning}, [str(w.message) for w in caught]
