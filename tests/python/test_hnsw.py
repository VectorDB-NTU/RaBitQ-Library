"""Tests for HnswIndex: construction, search, properties, error handling, save/load."""

from pathlib import Path

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


def test_parallel_build_preserves_self_retrieval(base_data, clusters):
    # Keep every neighbor so insertion-order-dependent pruning cannot isolate a
    # point. This checks parallel construction, not sparse-graph recall.
    idx = HnswIndex(DIM, N_VECTORS, M=N_VECTORS, ef_construction=N_VECTORS, nbits=4)
    centroids, cluster_ids = clusters
    idx.build(base_data, centroids, cluster_ids, num_threads=8)
    ids, distances = idx.search(base_data[:10], k=1, ef=N_VECTORS)
    np.testing.assert_array_equal(ids[:, 0], np.arange(10))
    assert np.isfinite(distances).all()


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


def test_out_of_range_cluster_id_raises(base_data, clusters):
    idx = HnswIndex(DIM, N_VECTORS, M=8, ef_construction=50, nbits=4)
    centroids, cluster_ids = clusters
    invalid_ids = cluster_ids.copy()
    invalid_ids[0] = len(centroids)
    with pytest.raises(ValueError, match="cluster_ids contains an out-of-range value"):
        idx.build(base_data, centroids, invalid_ids)


def test_wrong_query_dim_raises(built_hnsw):
    bad_queries = np.zeros((5, DIM + 1), dtype=np.float32)
    with pytest.raises(Exception):
        built_hnsw.search(bad_queries, k=1)


def test_search_before_build_raises(query_data):
    idx = HnswIndex(DIM, N_VECTORS, M=8, ef_construction=50, nbits=4)
    with pytest.raises(Exception):
        idx.search(query_data[:1], k=1)


# ── save / load roundtrip ─────────────────────────────────────────────────────


def test_save_rejects_unopenable_destination(built_hnsw, tmp_path):
    with pytest.raises(RuntimeError, match="HNSW: cannot open index file for writing"):
        built_hnsw.save(str(tmp_path / "missing" / "hnsw.index"))


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="/dev/full is unavailable")
def test_save_reports_write_or_close_failure(built_hnsw):
    with pytest.raises(RuntimeError, match="HNSW: failed to write index file"):
        built_hnsw.save("/dev/full")


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


def test_load_rejects_count_larger_than_capacity(built_hnsw, tmp_path):
    path = tmp_path / "hnsw-invalid-count.index"
    built_hnsw.save(str(path))
    with path.open("r+b") as index_file:
        index_file.write((1).to_bytes(np.dtype(np.uintp).itemsize, byteorder="little"))
    with pytest.raises(RuntimeError, match="HNSW"):
        HnswIndex.load(str(path))


# ── add and resize ────────────────────────────────────────────────────────────


@pytest.fixture
def partial_hnsw(base_data, clusters):
    """Built from the first 400 vectors, with room for the remaining 100."""
    idx = HnswIndex(DIM, N_VECTORS, M=8, ef_construction=50, nbits=4)
    centroids, cluster_ids = clusters
    idx.build(base_data[:400], centroids, cluster_ids[:400])
    return idx


def test_add_returns_dense_labels(partial_hnsw, base_data, clusters):
    _, cluster_ids = clusters
    assert partial_hnsw.num_points == 400
    ids = partial_hnsw.add(base_data[400:], cluster_ids=cluster_ids[400:])
    np.testing.assert_array_equal(ids, np.arange(400, N_VECTORS))
    assert partial_hnsw.num_points == N_VECTORS


def test_add_makes_points_searchable(partial_hnsw, base_data, clusters):
    # A sparse graph can isolate the odd point, as it can after build, so this
    # asks for near-perfect self-retrieval rather than exact.
    _, cluster_ids = clusters
    partial_hnsw.add(base_data[400:], cluster_ids=cluster_ids[400:])
    ids, distances = partial_hnsw.search(base_data[400:], k=1, ef=N_VECTORS)
    found = np.count_nonzero(ids[:, 0] == np.arange(400, N_VECTORS))
    assert found >= 0.95 * (N_VECTORS - 400)
    assert np.isfinite(distances).all()


def test_add_into_a_dense_graph_retrieves_every_point(base_data, clusters):
    # Keeping every neighbor removes the pruning that isolates points, so the
    # added points must all be found.
    centroids, cluster_ids = clusters
    idx = HnswIndex(DIM, N_VECTORS, M=N_VECTORS, ef_construction=N_VECTORS, nbits=4)
    idx.build(base_data[:400], centroids, cluster_ids[:400])
    idx.add(base_data[400:], cluster_ids=cluster_ids[400:])
    ids, _ = idx.search(base_data[400:], k=1, ef=N_VECTORS)
    np.testing.assert_array_equal(ids[:, 0], np.arange(400, N_VECTORS))


def test_add_without_cluster_ids_routes(partial_hnsw, base_data, clusters):
    _, cluster_ids = clusters
    ids = partial_hnsw.add(base_data[400:])
    np.testing.assert_array_equal(ids, np.arange(400, N_VECTORS))
    # Round-robin clusters, so routing should recover the original assignment for
    # most points; the centroids are close together and a few land elsewhere.
    found, _ = partial_hnsw.search(base_data[400:], k=1, ef=N_VECTORS)
    assert np.count_nonzero(found[:, 0] == np.arange(400, N_VECTORS)) >= 0.95 * 100


def test_add_preserves_earlier_points(partial_hnsw, base_data, clusters):
    _, cluster_ids = clusters
    before, _ = partial_hnsw.search(base_data[:10], k=1, ef=N_VECTORS)
    partial_hnsw.add(base_data[400:], cluster_ids=cluster_ids[400:])
    after, _ = partial_hnsw.search(base_data[:10], k=1, ef=N_VECTORS)
    np.testing.assert_array_equal(before[:, 0], after[:, 0])


def test_add_beyond_capacity_raises(partial_hnsw, base_data):
    with pytest.raises(ValueError, match="capacity"):
        partial_hnsw.add(base_data)
    assert partial_hnsw.num_points == 400


def test_add_rejects_bad_cluster_ids(partial_hnsw, base_data):
    rows = N_VECTORS - 400
    with pytest.raises(ValueError, match="cluster_ids"):
        partial_hnsw.add(base_data[400:], cluster_ids=np.full(rows, N_CLUSTERS))
    with pytest.raises(ValueError, match="cluster_ids"):
        partial_hnsw.add(base_data[400:], cluster_ids=-np.ones(rows, dtype=np.int64))
    with pytest.raises(ValueError, match="cluster_ids"):
        partial_hnsw.add(
            base_data[400:], cluster_ids=np.zeros(rows - 1, dtype=np.int64)
        )
    assert partial_hnsw.num_points == 400


def test_add_rejects_wrong_dimension(partial_hnsw):
    with pytest.raises(ValueError, match="dimension"):
        partial_hnsw.add(np.zeros((4, DIM + 1), dtype=np.float32))


def test_add_before_build_raises(base_data):
    idx = HnswIndex(DIM, N_VECTORS, M=8, ef_construction=50, nbits=4)
    with pytest.raises(RuntimeError, match="built or loaded"):
        idx.add(base_data[:4])


def test_resize_grows_capacity_and_allows_more(base_data, clusters):
    centroids, cluster_ids = clusters
    idx = HnswIndex(DIM, 400, M=8, ef_construction=50, nbits=4)
    idx.build(base_data[:400], centroids, cluster_ids[:400])
    assert idx.max_elements == 400

    with pytest.raises(ValueError, match="capacity"):
        idx.add(base_data[400:])

    idx.resize(N_VECTORS)
    assert idx.max_elements == N_VECTORS
    assert idx.num_points == 400

    idx.add(base_data[400:], cluster_ids=cluster_ids[400:])
    assert idx.num_points == N_VECTORS
    ids, _ = idx.search(base_data[:10], k=1, ef=N_VECTORS)
    np.testing.assert_array_equal(ids[:, 0], np.arange(10))


def test_resize_below_element_count_raises(partial_hnsw):
    with pytest.raises(ValueError, match="element count"):
        partial_hnsw.resize(10)
    assert partial_hnsw.max_elements == N_VECTORS


def test_added_points_survive_save_and_load(
    partial_hnsw, base_data, clusters, tmp_path
):
    _, cluster_ids = clusters
    partial_hnsw.add(base_data[400:], cluster_ids=cluster_ids[400:])
    before_ids, before_dists = partial_hnsw.search(base_data, k=_TOPK, ef=_EF)

    path = str(tmp_path / "hnsw-added.index")
    partial_hnsw.save(path)
    loaded = HnswIndex.load(path)

    assert loaded.num_points == N_VECTORS
    assert loaded.max_elements == N_VECTORS
    after_ids, after_dists = loaded.search(base_data, k=_TOPK, ef=_EF)
    np.testing.assert_array_equal(before_ids, after_ids)
    np.testing.assert_allclose(before_dists, after_dists, rtol=1e-5)
