"""Tests for IvfIndex: construction, search, properties, error handling, save/load."""

from pathlib import Path

import numpy as np
import pytest
from conftest import DIM, N_CLUSTERS, N_QUERIES, N_VECTORS, brute_force_knn, recall_at_k
from rabitqlib import IvfIndex

_TOPK = 10
_NPROBE_ALL = N_CLUSTERS  # probe every cluster → deterministic coverage


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built_ivf(base_data, clusters):
    idx = IvfIndex(DIM, N_VECTORS, N_CLUSTERS, nbits=4)
    centroids, cluster_ids = clusters
    idx.build(base_data, centroids, cluster_ids)
    return idx


# ── construction ──────────────────────────────────────────────────────────────


def test_is_built(built_ivf):
    assert built_ivf.is_built


def test_properties(built_ivf):
    assert built_ivf.dim == DIM
    assert built_ivf.num_clusters == N_CLUSTERS
    assert built_ivf.nbits == 4
    assert built_ivf.metric == "l2"
    assert built_ivf.max_elements == N_VECTORS


def test_fast_quantization_builds(base_data, clusters):
    idx = IvfIndex(DIM, N_VECTORS, N_CLUSTERS, nbits=4)
    centroids, cluster_ids = clusters
    idx.build(base_data, centroids, cluster_ids, fast_quantization=True)
    assert idx.is_built


# ── search output shape and dtype ─────────────────────────────────────────────


def test_search_output_shape(built_ivf, query_data):
    ids, dists = built_ivf.search(query_data, k=_TOPK, nprobe=_NPROBE_ALL)
    assert ids.shape == (N_QUERIES, _TOPK)
    assert dists.shape == (N_QUERIES, _TOPK)


def test_search_output_dtype(built_ivf, query_data):
    ids, dists = built_ivf.search(query_data, k=_TOPK, nprobe=_NPROBE_ALL)
    assert np.issubdtype(ids.dtype, np.integer)
    assert dists.dtype == np.float32


def test_search_ids_in_range(built_ivf, query_data):
    ids, _ = built_ivf.search(query_data, k=_TOPK, nprobe=_NPROBE_ALL)
    assert np.all(ids < N_VECTORS)


def test_search_distances_nonneg(built_ivf, query_data):
    _, dists = built_ivf.search(query_data, k=_TOPK, nprobe=_NPROBE_ALL)
    assert np.all(dists >= 0)


def test_single_query(built_ivf, query_data):
    ids, dists = built_ivf.search(query_data[:1], k=1, nprobe=_NPROBE_ALL)
    assert ids.shape == (1, 1)
    assert dists.shape == (1, 1)


def test_k_equals_one(built_ivf, query_data):
    ids, dists = built_ivf.search(query_data, k=1, nprobe=1)
    assert ids.shape == (N_QUERIES, 1)
    assert dists.shape == (N_QUERIES, 1)


def test_unfilled_results_use_sentinels():
    data = np.zeros((4, DIM), dtype=np.float32)
    data[1:] = 100.0
    centroids = np.stack((data[0], data[1]))
    cluster_ids = np.array([0, 1, 1, 1], dtype=np.uint32)
    idx = IvfIndex(DIM, 4, 2, nbits=4)
    idx.build(data, centroids, cluster_ids)

    ids, dists = idx.search(data[:1], k=3, nprobe=1)
    assert ids[0, 0] == 0
    np.testing.assert_array_equal(ids[0, 1:], np.iinfo(np.uint32).max)
    assert np.all(np.isinf(dists[0, 1:]))


# ── search correctness ────────────────────────────────────────────────────────


def test_self_retrieval(built_ivf, base_data):
    """Probing all clusters: each database vector must be its own nearest neighbor."""
    probes = base_data[:10]
    ids, _ = built_ivf.search(probes, k=1, nprobe=_NPROBE_ALL)
    for i in range(10):
        assert i in ids[i], f"Vector {i} not found in its own top-1 result"


def test_recall_vs_brute_force(built_ivf, base_data, query_data):
    """Approximate recall should exceed 0.5 when probing all clusters."""
    k = 5
    approx_ids, _ = built_ivf.search(query_data, k=k, nprobe=_NPROBE_ALL)
    exact_ids, _ = brute_force_knn(base_data, query_data, k)
    r = recall_at_k(approx_ids, exact_ids, k)
    assert r >= 0.5, f"Recall {r:.3f} too low"


# ── optional parameters ───────────────────────────────────────────────────────


def test_high_accuracy_false(built_ivf, query_data):
    ids, dists = built_ivf.search(
        query_data, k=_TOPK, nprobe=_NPROBE_ALL, high_accuracy=False
    )
    assert ids.shape == (N_QUERIES, _TOPK)


@pytest.mark.parametrize("nbits", [4, 8])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_full_quantization_centroid_distances_roundtrip(nbits, metric, tmp_path):
    # Exercise a tail batch and a padded dimension, including a zero residual.
    rng = np.random.default_rng(42)
    data = rng.standard_normal((33, 129)).astype(np.float32)
    data[0] = 0
    centroid = np.zeros((1, 129), dtype=np.float32)
    idx = IvfIndex(129, len(data), 1, nbits=nbits, metric=metric)
    idx.build(
        data,
        centroid,
        np.zeros(len(data), dtype=np.uint32),
        fast_quantization=False,
        num_threads=1,
    )

    ids, distances = idx.search(centroid, k=len(data), nprobe=1, num_threads=1)
    expected = (
        np.einsum("ij,ij->i", data, data)
        if metric == "l2"
        else np.ones(len(data), dtype=np.float32)
    )
    np.testing.assert_array_equal(np.sort(ids[0]), np.arange(len(data)))
    np.testing.assert_allclose(distances[0], expected[ids[0]], rtol=2e-5, atol=2e-5)

    path = str(tmp_path / "quantized.index")
    idx.save(path)
    loaded = IvfIndex.load(path)
    loaded_ids, loaded_distances = loaded.search(
        centroid, k=len(data), nprobe=1, num_threads=1
    )
    np.testing.assert_array_equal(loaded_ids, ids)
    np.testing.assert_array_equal(loaded_distances, distances)


def test_multithreaded_search_matches_single(built_ivf, query_data):
    ids1, dists1 = built_ivf.search(
        query_data, k=_TOPK, nprobe=_NPROBE_ALL, num_threads=1
    )
    ids2, dists2 = built_ivf.search(
        query_data, k=_TOPK, nprobe=_NPROBE_ALL, num_threads=2
    )
    np.testing.assert_array_equal(ids1, ids2)
    np.testing.assert_allclose(dists1, dists2, rtol=1e-5)


# ── error handling ────────────────────────────────────────────────────────────


def test_wrong_data_dim_raises(clusters):
    idx = IvfIndex(DIM, N_VECTORS, N_CLUSTERS, nbits=4)
    centroids, cluster_ids = clusters
    bad_data = np.zeros((N_VECTORS, DIM + 1), dtype=np.float32)
    with pytest.raises(Exception):
        idx.build(bad_data, centroids, cluster_ids)


def test_wrong_centroid_dim_raises(base_data, clusters):
    idx = IvfIndex(DIM, N_VECTORS, N_CLUSTERS, nbits=4)
    _, cluster_ids = clusters
    bad_centroids = np.zeros((N_CLUSTERS, DIM + 1), dtype=np.float32)
    with pytest.raises(Exception):
        idx.build(base_data, bad_centroids, cluster_ids)


def test_wrong_cluster_ids_length_raises(base_data, clusters):
    idx = IvfIndex(DIM, N_VECTORS, N_CLUSTERS, nbits=4)
    centroids, _ = clusters
    bad_ids = np.zeros(N_VECTORS + 5, dtype=np.uint32)
    with pytest.raises(Exception):
        idx.build(base_data, centroids, bad_ids)


def test_wrong_query_dim_raises(built_ivf):
    bad_queries = np.zeros((5, DIM + 1), dtype=np.float32)
    with pytest.raises(Exception):
        built_ivf.search(bad_queries, k=1, nprobe=1)


def test_search_before_build_raises(query_data):
    idx = IvfIndex(DIM, N_VECTORS, N_CLUSTERS, nbits=4)
    with pytest.raises(Exception):
        idx.search(query_data[:1], k=1, nprobe=1)


# ── save / load roundtrip ─────────────────────────────────────────────────────


def test_save_load_roundtrip(built_ivf, query_data, tmp_path):
    path = str(tmp_path / "ivf.index")
    built_ivf.save(path)

    loaded = IvfIndex.load(path)
    assert loaded.is_built
    assert loaded.dim == built_ivf.dim
    assert loaded.num_clusters == built_ivf.num_clusters
    assert loaded.nbits == built_ivf.nbits

    ids_orig, dists_orig = built_ivf.search(query_data, k=_TOPK, nprobe=_NPROBE_ALL)
    ids_load, dists_load = loaded.search(query_data, k=_TOPK, nprobe=_NPROBE_ALL)
    np.testing.assert_array_equal(ids_orig, ids_load)
    np.testing.assert_allclose(dists_orig, dists_load, rtol=1e-5)


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("high_accuracy", [False, True])
@pytest.mark.parametrize("fast_quantization", [False, True])
def test_raw_reranking(metric, high_accuracy, fast_quantization, tmp_path):
    rng = np.random.default_rng(71)
    data = rng.standard_normal((65, 65)).astype(np.float32)
    data[0] = 0  # zero residual, padded dimension, and tail batches
    queries = rng.standard_normal((3, 65)).astype(np.float32)
    idx = IvfIndex(65, len(data), 3, 32, metric)
    cluster_ids = np.arange(len(data), dtype=np.uint32) % 2  # empty third cluster
    idx.build(
        data,
        np.zeros((3, 65), dtype=np.float32),
        cluster_ids,
        fast_quantization=fast_quantization,
        num_threads=2,
    )
    expected = (
        np.sum((queries[:, None] - data) ** 2, axis=2)
        if metric == "l2"
        else 1 - queries @ data.T
    )
    original = data.copy()
    data[:] = 1000  # the index must own the raw data
    path = tmp_path / "raw.index"
    idx.save(str(path))
    loaded = IvfIndex.load(str(path))
    assert idx.nbits == loaded.nbits == 32
    assert loaded.dim == 65
    assert loaded.metric == metric
    for k in (5, len(data)):
        ids, distances = idx.search(queries, k, 3, high_accuracy, 2)
        loaded_ids, loaded_distances = loaded.search(queries, k, 3, high_accuracy, 2)
        np.testing.assert_array_equal(loaded_ids, ids)
        np.testing.assert_array_equal(loaded_distances, distances)
        np.testing.assert_allclose(
            distances, np.take_along_axis(expected, ids, axis=1), rtol=2e-5, atol=2e-5
        )
        assert np.all(np.diff(distances, axis=1) >= 0)
        if k == len(data):
            np.testing.assert_array_equal(ids, np.argsort(expected, axis=1))

    # The reranking region contains precisely original floats in cluster order,
    # with no extra-bit codes/factors or padded coordinates.
    payload = path.read_bytes()
    ids_bytes = len(data) * np.dtype(np.uint32).itemsize
    raw_bytes = original.nbytes
    stored = np.frombuffer(
        payload[-ids_bytes - raw_bytes : -ids_bytes], dtype=np.float32
    )
    order = np.argsort(cluster_ids, kind="stable")
    np.testing.assert_array_equal(stored.reshape(original.shape), original[order])
    one_bit = IvfIndex(65, len(data), 3, 1, metric)
    one_bit.build(original, np.zeros((3, 65), dtype=np.float32), cluster_ids)
    one_bit_path = tmp_path / "one-bit.index"
    one_bit.save(str(one_bit_path))
    assert len(payload) - one_bit_path.stat().st_size == raw_bytes + 12


def test_legacy_ivf_fixture(tmp_path):
    path = Path(__file__).parent / "fixtures" / "ivf_legacy_4bit.index"
    idx = IvfIndex.load(str(path))
    assert (idx.dim, idx.max_elements, idx.num_clusters, idx.nbits, idx.metric) == (
        64,
        3,
        1,
        4,
        "l2",
    )
    ids, distances = idx.search(np.zeros((1, 64), np.float32), 3, 1)
    np.testing.assert_array_equal(np.sort(ids), [[0, 1, 2]])
    np.testing.assert_allclose(distances, [[0, 14, 14]], atol=2e-5)
    saved = tmp_path / "legacy.index"
    idx.save(str(saved))
    assert saved.read_bytes() == path.read_bytes()


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "ex_bits",
        "dimension",
        "count",
        "clusters",
        "truncated_header",
        "truncated_payload",
    ],
)
def test_raw_index_rejects_invalid_files(tmp_path, damage):
    data = np.zeros((2, 65), dtype=np.float32)
    idx = IvfIndex(65, 2, 1, 32)
    idx.build(data, data[:1], np.zeros(2, dtype=np.uint32))
    path = tmp_path / "invalid.index"
    idx.save(str(path))
    payload = bytearray(path.read_bytes())
    if damage == "version":
        payload[8:12] = (2).to_bytes(4, "little")
    elif damage == "ex_bits":
        payload[36:44] = (1).to_bytes(8, "little")
    elif damage == "dimension":
        payload[20:28] = (2**64 - 1).to_bytes(8, "little")
    elif damage == "count":
        payload[12:20] = (2**64 - 1).to_bytes(8, "little")
    elif damage == "clusters":
        payload[28:36] = (2**64 - 1).to_bytes(8, "little")
    elif damage == "truncated_header":
        payload = payload[:9]
    else:
        payload = payload[:-1]
    path.write_bytes(payload)
    with pytest.raises(RuntimeError):
        IvfIndex.load(str(path))


@pytest.mark.parametrize("nbits", [0, 10, 31, 33])
def test_invalid_ivf_bits(nbits):
    with pytest.raises(ValueError, match="IVF bits must be in"):
        IvfIndex(3, 2, 1, nbits)


@pytest.mark.parametrize("nbits", [1, 2, 3, 4, 5, 6, 7, 8, 9, 32])
def test_automatic_high_accuracy(nbits, tmp_path):
    rng = np.random.default_rng(29)
    data = rng.standard_normal((65, 65)).astype(np.float32)
    queries = rng.standard_normal((3, 65)).astype(np.float32)
    index = IvfIndex(65, len(data), 1, nbits)
    index.build(
        data, np.zeros((1, 65), dtype=np.float32), np.zeros(len(data), dtype=np.uint32)
    )
    path = tmp_path / "auto.index"
    index.save(str(path))
    for idx in (index, IvfIndex.load(str(path))):
        expected = idx.search(queries, 7, 1, high_accuracy=4 <= nbits <= 9)
        for actual in (
            idx.search(queries, 7, 1),
            idx.search(queries, 7, 1, high_accuracy=None),
        ):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
