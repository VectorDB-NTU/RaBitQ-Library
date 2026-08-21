"""Tests for SymqgIndex: construction, search, properties, error handling, save/load."""

import numpy as np
import pytest
from rabitqlib import SymqgIndex

from conftest import DIM, N_VECTORS, N_QUERIES, brute_force_knn, recall_at_k

_TOPK = 10
_EF = 50
_MAX_DEGREE = 32  # must be a multiple of fastscan::kBatchSize (32)
_EF_BUILD = 50


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def built_symqg(base_data):
    idx = SymqgIndex(DIM, max_degree=_MAX_DEGREE)
    idx.build(base_data, ef_construction=_EF_BUILD)
    return idx


# ── construction ──────────────────────────────────────────────────────────────

def test_is_built(built_symqg):
    assert built_symqg.is_built


def test_properties(built_symqg):
    assert built_symqg.dim == DIM
    assert built_symqg.max_degree == _MAX_DEGREE
    assert built_symqg.num_points == N_VECTORS
    assert built_symqg.metric == "l2"


# ── search output shape and dtype ─────────────────────────────────────────────

def test_search_output_shape(built_symqg, query_data):
    ids, dists = built_symqg.search(query_data, k=_TOPK, ef=_EF)
    assert ids.shape == (N_QUERIES, _TOPK)
    assert dists.shape == (N_QUERIES, _TOPK)


def test_search_output_dtype(built_symqg, query_data):
    ids, dists = built_symqg.search(query_data, k=_TOPK, ef=_EF)
    assert np.issubdtype(ids.dtype, np.integer)
    assert dists.dtype == np.float32


def test_search_ids_in_range(built_symqg, query_data):
    ids, _ = built_symqg.search(query_data, k=_TOPK, ef=_EF)
    assert np.all(ids < N_VECTORS)


def test_search_distances_nonneg(built_symqg, query_data):
    _, dists = built_symqg.search(query_data, k=_TOPK, ef=_EF)
    assert np.all(dists >= 0)


def test_single_query(built_symqg, query_data):
    ids, dists = built_symqg.search(query_data[:1], k=1, ef=_EF)
    assert ids.shape == (1, 1)


# ── search correctness ────────────────────────────────────────────────────────

def test_self_retrieval(built_symqg, base_data):
    """Each database vector must be its own nearest neighbor at high ef."""
    probes = base_data[:10]
    ids, _ = built_symqg.search(probes, k=1, ef=200)
    for i in range(10):
        assert i in ids[i], f"Vector {i} not found in its own top-1 result"


def test_recall_vs_brute_force(built_symqg, base_data, query_data):
    """Approximate recall should exceed 0.5 at ef=50 on a small dataset."""
    k = 5
    approx_ids, _ = built_symqg.search(query_data, k=k, ef=_EF)
    exact_ids, _ = brute_force_knn(base_data, query_data, k)
    r = recall_at_k(approx_ids, exact_ids, k)
    assert r >= 0.5, f"Recall {r:.3f} too low"


# ── error handling ────────────────────────────────────────────────────────────

def test_wrong_data_dim_raises():
    idx = SymqgIndex(DIM, max_degree=_MAX_DEGREE)
    bad_data = np.zeros((N_VECTORS, DIM + 1), dtype=np.float32)
    with pytest.raises(Exception):
        idx.build(bad_data, ef_construction=_EF_BUILD)


def test_wrong_query_dim_raises(built_symqg):
    bad_queries = np.zeros((5, DIM + 1), dtype=np.float32)
    with pytest.raises(Exception):
        built_symqg.search(bad_queries, k=1, ef=_EF)


def test_search_before_build_raises():
    idx = SymqgIndex(DIM, max_degree=_MAX_DEGREE)
    queries = np.zeros((5, DIM), dtype=np.float32)
    with pytest.raises(Exception):
        idx.search(queries, k=1, ef=_EF)


# ── save / load roundtrip ─────────────────────────────────────────────────────

def test_save_load_roundtrip(built_symqg, query_data, tmp_path):
    path = str(tmp_path / "symqg.index")
    built_symqg.save(path)

    loaded = SymqgIndex.load(path)
    assert loaded.is_built
    assert loaded.dim == built_symqg.dim
    assert loaded.max_degree == built_symqg.max_degree
    assert loaded.num_points == built_symqg.num_points

    ids_orig, dists_orig = built_symqg.search(query_data, k=_TOPK, ef=_EF)
    ids_load, dists_load = loaded.search(query_data, k=_TOPK, ef=_EF)
    np.testing.assert_array_equal(ids_orig, ids_load)
    np.testing.assert_allclose(dists_orig, dists_load, rtol=1e-5)
