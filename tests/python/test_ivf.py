"""Tests for IvfIndex: construction, search, properties, error handling, save/load."""

import numpy as np
import pytest
from rabitqlib import IvfIndex

from conftest import DIM, N_CLUSTERS, N_VECTORS, N_QUERIES, brute_force_knn, recall_at_k

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


def test_multithreaded_search_matches_single(built_ivf, query_data):
    ids1, dists1 = built_ivf.search(query_data, k=_TOPK, nprobe=_NPROBE_ALL, num_threads=1)
    ids2, dists2 = built_ivf.search(query_data, k=_TOPK, nprobe=_NPROBE_ALL, num_threads=2)
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
