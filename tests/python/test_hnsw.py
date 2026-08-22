"""Tests for HnswIndex: construction, search, properties, error handling, save/load."""

import numpy as np
import pytest
from conftest import DIM, N_CLUSTERS, N_QUERIES, N_VECTORS, brute_force_knn, recall_at_k
from rabitqlib import HnswIndex

_TOPK = 10
_EF = 50  # generous ef for correctness tests on a small dataset


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built_hnsw(base_data, clusters):
    idx = HnswIndex(DIM, N_VECTORS, M=8, ef_construction=50, nbits=4)
    centroids, cluster_ids = clusters
    idx.build(base_data, centroids, cluster_ids)
    return idx


# ── construction ──────────────────────────────────────────────────────────────


def test_is_built(built_hnsw):
    assert built_hnsw.is_built


def test_properties(built_hnsw):
    assert built_hnsw.dim == DIM
    assert built_hnsw.nbits == 4
    assert built_hnsw.metric == "l2"
    assert built_hnsw.max_elements == N_VECTORS
    assert built_hnsw.num_clusters == N_CLUSTERS


def test_fast_quantization_builds(base_data, clusters):
    idx = HnswIndex(DIM, N_VECTORS, M=8, ef_construction=50, nbits=4)
    centroids, cluster_ids = clusters
    idx.build(base_data, centroids, cluster_ids, fast_quantization=True)
    assert idx.is_built


# ── search output shape and dtype ─────────────────────────────────────────────


def test_search_output_shape(built_hnsw, query_data):
    ids, dists = built_hnsw.search(query_data, k=_TOPK, ef=_EF)
    assert ids.shape == (N_QUERIES, _TOPK)
    assert dists.shape == (N_QUERIES, _TOPK)


def test_search_output_dtype(built_hnsw, query_data):
    ids, dists = built_hnsw.search(query_data, k=_TOPK, ef=_EF)
    assert np.issubdtype(ids.dtype, np.integer)
    assert dists.dtype == np.float32


def test_search_ids_in_range(built_hnsw, query_data):
    ids, _ = built_hnsw.search(query_data, k=_TOPK, ef=_EF)
    assert np.all(ids < N_VECTORS)


def test_search_distances_nonneg(built_hnsw, query_data):
    _, dists = built_hnsw.search(query_data, k=_TOPK, ef=_EF)
    assert np.all(dists >= 0)


def test_ef_default(built_hnsw, query_data):
    """ef=0 should use the internal default (max(k, 10))."""
    ids, dists = built_hnsw.search(query_data, k=_TOPK)
    assert ids.shape == (N_QUERIES, _TOPK)


def test_single_query(built_hnsw, query_data):
    ids, dists = built_hnsw.search(query_data[:1], k=1, ef=_EF)
    assert ids.shape == (1, 1)


# ── search correctness ────────────────────────────────────────────────────────


def test_self_retrieval(built_hnsw, base_data):
    """Each database vector must be its own nearest neighbor at high ef."""
    probes = base_data[:10]
    ids, _ = built_hnsw.search(probes, k=1, ef=200)
    for i in range(10):
        assert i in ids[i], f"Vector {i} not found in its own top-1 result"


def test_recall_vs_brute_force(built_hnsw, base_data, query_data):
    """Approximate recall should exceed 0.5 at ef=50 on a small dataset."""
    k = 5
    approx_ids, _ = built_hnsw.search(query_data, k=k, ef=_EF)
    exact_ids, _ = brute_force_knn(base_data, query_data, k)
    r = recall_at_k(approx_ids, exact_ids, k)
    assert r >= 0.5, f"Recall {r:.3f} too low"


# ── error handling ────────────────────────────────────────────────────────────


def test_wrong_data_dim_raises(clusters):
    idx = HnswIndex(DIM, N_VECTORS, M=8, ef_construction=50, nbits=4)
    centroids, cluster_ids = clusters
    bad_data = np.zeros((N_VECTORS, DIM + 1), dtype=np.float32)
    with pytest.raises(Exception):
        idx.build(bad_data, centroids, cluster_ids)


def test_wrong_query_dim_raises(built_hnsw):
    bad_queries = np.zeros((5, DIM + 1), dtype=np.float32)
    with pytest.raises(Exception):
        built_hnsw.search(bad_queries, k=1)


# ── save / load roundtrip ─────────────────────────────────────────────────────


def test_save_load_roundtrip(built_hnsw, query_data, tmp_path):
    path = str(tmp_path / "hnsw.index")
    built_hnsw.save(path)

    loaded = HnswIndex.load(path)
    assert loaded.is_built
    assert loaded.dim == built_hnsw.dim
    assert loaded.nbits == built_hnsw.nbits

    ids_orig, dists_orig = built_hnsw.search(query_data, k=_TOPK, ef=_EF)
    ids_load, dists_load = loaded.search(query_data, k=_TOPK, ef=_EF)
    np.testing.assert_array_equal(ids_orig, ids_load)
    np.testing.assert_allclose(dists_orig, dists_load, rtol=1e-5)
