"""Tests for HnswIndex: construction, search, properties, error handling, save/load."""

import re
from pathlib import Path

import format2_file
import numpy as np
import pytest
from conftest import DIM, N_CLUSTERS, N_QUERIES, N_VECTORS, brute_force_knn, recall_at_k
from rabitqlib import HnswIndex

_TOPK = 10
_EF = 50  # generous ef for correctness tests on a small dataset


# ── fixtures ──────────────────────────────────────────────────────────────────


def _built(
    data, centroids, cluster_ids, num_threads=1, fast_quantization=False, **options
):
    """An index over data with the given centroids; the first add links the points."""
    # Seeded, so that a failure of a graph-dependent assertion can be replayed.
    options.setdefault("seed", 100)
    index = HnswIndex(data.shape[1], num_clusters=len(centroids), **options)
    index.set_centroids(centroids)
    index.add(
        data,
        cluster_ids=cluster_ids,
        num_threads=num_threads,
        fast_quantization=fast_quantization,
    )
    return index


@pytest.fixture(scope="module")
def built_hnsw(base_data, clusters):
    centroids, cluster_ids = clusters
    return _built(base_data, centroids, cluster_ids, M=8, ef_construction=50, nbits=4)


# ── construction ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("m", [0, 1])
def test_rejects_too_small_m(m):
    with pytest.raises(ValueError, match="^HNSW M must be at least 2$"):
        HnswIndex(DIM, M=m)


def test_properties(built_hnsw):
    assert len(built_hnsw) == N_VECTORS
    assert built_hnsw.dim == DIM
    assert built_hnsw.nbits == 4
    assert built_hnsw.metric == "l2"
    assert built_hnsw.capacity == N_VECTORS
    assert built_hnsw.num_clusters == N_CLUSTERS


def test_seed_reproduces_serial_build_and_search(tmp_path):
    rng = np.random.default_rng(47)
    data = rng.standard_normal((65, 65), dtype=np.float32)
    queries = rng.standard_normal((5, 65), dtype=np.float32)
    centroids = data[:2].copy()
    clusters = (np.arange(len(data)) % 2).astype(np.uint32)
    snapshots = []
    results = []
    for run, seed in enumerate((42, 42, 43)):
        index = _built(
            data, centroids, clusters, M=8, ef_construction=32, nbits=4, seed=seed
        )
        path = tmp_path / f"seed-{run}.index"
        index.save(str(path))
        snapshots.append(path.read_bytes())
        results.append(index.search(queries, k=5, ef_search=65, num_threads=1))
    assert snapshots[0] == snapshots[1]
    assert snapshots[0] != snapshots[2]
    for actual, expected in zip(results[1], results[0], strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_parallel_build_preserves_self_retrieval(base_data, clusters):
    # M equals the point count, so the base layer holds 2M >= N - 1 links and never
    # prunes a reverse link: every point stays reachable under any thread
    # interleaving. This checks parallel construction, not sparse-graph recall.
    centroids, cluster_ids = clusters
    idx = _built(
        base_data,
        centroids,
        cluster_ids,
        num_threads=8,
        M=N_VECTORS,
        ef_construction=N_VECTORS,
        nbits=4,
    )
    ids, distances = idx.search(base_data, k=1, ef_search=N_VECTORS)
    np.testing.assert_array_equal(ids[:, 0], np.arange(N_VECTORS))
    assert np.isfinite(distances).all()


# ── search output shape and dtype ─────────────────────────────────────────────


@pytest.mark.parametrize("nq,k", [(N_QUERIES, _TOPK), (1, 1)])
def test_search_output_contract(built_hnsw, query_data, nq, k):
    ids, dists = built_hnsw.search(query_data[:nq], k=k, ef_search=_EF)
    assert ids.shape == dists.shape == (nq, k)
    assert ids.dtype == np.int64
    assert dists.dtype == np.float32
    assert np.all(ids < N_VECTORS)
    assert np.isfinite(dists).all()
    assert np.all(dists >= 0)


@pytest.mark.parametrize("k", [1, 10, 17])
def test_ef_default(built_hnsw, query_data, k):
    """None selects the internal default, max(k, 64); 0 is rejected."""
    expected_ids, expected_distances = built_hnsw.search(
        query_data, k=k, ef_search=max(k, 64)
    )
    for kwargs in ({}, {"ef_search": None}):
        ids, distances = built_hnsw.search(query_data, k=k, **kwargs)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(distances, expected_distances)
    with pytest.raises(
        ValueError, match="^ef_search must be at least 1; pass None for the default$"
    ):
        built_hnsw.search(query_data, k=k, ef_search=0)
    # A breadth below k is raised to k.
    at_k = built_hnsw.search(query_data, k=k, ef_search=k)
    below_k = built_hnsw.search(query_data, k=k, ef_search=1)
    np.testing.assert_array_equal(below_k[0], at_k[0])
    np.testing.assert_array_equal(below_k[1], at_k[1])


# ── search correctness ────────────────────────────────────────────────────────


@pytest.mark.parametrize("fast_quantization", [False, True])
def test_self_retrieval(built_hnsw, base_data, clusters, fast_quantization):
    """Each database vector must be its own nearest neighbor at high ef."""
    idx = built_hnsw
    if fast_quantization:
        centroids, cluster_ids = clusters
        idx = _built(
            base_data,
            centroids,
            cluster_ids,
            fast_quantization=True,
            M=8,
            ef_construction=50,
            nbits=4,
        )
    assert len(idx) == N_VECTORS
    probes = base_data[:10]
    ids, _ = idx.search(probes, k=1, ef_search=200)
    for i in range(10):
        assert i in ids[i], f"Vector {i} not found in its own top-1 result"


def test_recall_vs_brute_force(built_hnsw, base_data, query_data):
    """Approximate recall should exceed 0.5 at ef_search=50 on a small dataset."""
    k = 5
    approx_ids, _ = built_hnsw.search(query_data, k=k, ef_search=_EF)
    exact_ids, _ = brute_force_knn(base_data, query_data, k)
    r = recall_at_k(approx_ids, exact_ids, k)
    assert r >= 0.5, f"Recall {r:.3f} too low"


# ── error handling ────────────────────────────────────────────────────────────


def test_wrong_data_dim_raises(clusters):
    centroids, cluster_ids = clusters
    idx = HnswIndex(DIM, M=8, ef_construction=50, nbits=4, num_clusters=len(centroids))
    idx.set_centroids(centroids)
    bad_data = np.zeros((N_VECTORS, DIM + 1), dtype=np.float32)
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            f"data dimension does not match index dim; expected last dimension {DIM}, got shape {bad_data.shape}"
        )
        + "$",
    ):
        idx.add(bad_data, cluster_ids=cluster_ids, num_threads=1)


def test_out_of_range_cluster_id_raises(base_data, clusters):
    centroids, cluster_ids = clusters
    idx = HnswIndex(DIM, M=8, ef_construction=50, nbits=4, num_clusters=len(centroids))
    idx.set_centroids(centroids)
    invalid_ids = cluster_ids.copy()
    invalid_ids[0] = len(centroids)
    with pytest.raises(
        ValueError, match=re.escape("cluster_ids must be in [0, num_clusters)")
    ):
        idx.add(base_data, cluster_ids=invalid_ids, num_threads=1)
    assert len(idx) == 0


def test_wrong_query_dim_raises(built_hnsw):
    bad_queries = np.zeros((5, DIM + 1), dtype=np.float32)
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            f"query dimension does not match index dim; expected last dimension {DIM}, got shape {bad_queries.shape}"
        )
        + "$",
    ):
        built_hnsw.search(bad_queries, k=1)


def test_search_before_build_raises(query_data):
    idx = HnswIndex(DIM, M=8, ef_construction=50, nbits=4)
    with pytest.raises(
        RuntimeError, match="^HnswIndex must be trained or loaded before search$"
    ):
        idx.search(query_data[:1], k=1)


# ── save / load roundtrip ─────────────────────────────────────────────────────


def test_save_rejects_unopenable_destination(built_hnsw, tmp_path):
    with pytest.raises(RuntimeError, match="HNSW cannot open index file for writing"):
        built_hnsw.save(str(tmp_path / "missing" / "hnsw.index"))


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="/dev/full is unavailable")
def test_save_reports_write_or_close_failure(built_hnsw):
    with pytest.raises(RuntimeError, match="HNSW index file cannot be written"):
        built_hnsw.save("/dev/full")


def test_save_load_roundtrip(built_hnsw, query_data, tmp_path):
    path = str(tmp_path / "hnsw.index")
    built_hnsw.save(path)

    loaded = HnswIndex.load(path)
    assert len(loaded) == len(built_hnsw)
    assert loaded.dim == built_hnsw.dim
    assert loaded.nbits == built_hnsw.nbits

    ids_orig, dists_orig = built_hnsw.search(query_data, k=_TOPK, ef_search=_EF)
    ids_load, dists_load = loaded.search(query_data, k=_TOPK, ef_search=_EF)
    np.testing.assert_array_equal(ids_orig, ids_load)
    np.testing.assert_allclose(dists_orig, dists_load, rtol=1e-5)


def test_load_rejects_count_larger_than_capacity(built_hnsw, tmp_path):
    path = tmp_path / "hnsw-invalid-count.index"
    built_hnsw.save(str(path))
    # The capacity is the first field of the graph parameters, which are checksummed.
    payload = bytearray(path.read_bytes())
    offset, _ = format2_file.sections(payload)[format2_file.HNSW_PARAMS]
    payload[offset : offset + 8] = (1).to_bytes(8, "little")
    format2_file.reseal(payload)
    path.write_bytes(payload)
    with pytest.raises(
        RuntimeError, match="^HNSW index file has invalid graph parameters$"
    ):
        HnswIndex.load(str(path))


# ── add and reserve ───────────────────────────────────────────────────────────


@pytest.fixture
def partial_hnsw(base_data, clusters):
    """Built from the first 400 vectors, with room for the remaining 100."""
    centroids, cluster_ids = clusters
    idx = HnswIndex(
        DIM, M=8, ef_construction=50, nbits=4, num_clusters=len(centroids), seed=100
    )
    idx.set_centroids(centroids)
    idx.reserve(N_VECTORS)
    idx.add(base_data[:400], cluster_ids=cluster_ids[:400], num_threads=1)
    return idx


def test_add_returns_dense_labels_and_preserves_search(
    partial_hnsw, base_data, clusters
):
    _, cluster_ids = clusters
    assert partial_hnsw.num_points == 400
    before, _ = partial_hnsw.search(base_data[:10], k=1, ef_search=N_VECTORS)
    # One thread, as the build used, so the grown graph and `after` are reproducible.
    ids = partial_hnsw.add(
        base_data[400:], cluster_ids=cluster_ids[400:], num_threads=1
    )
    np.testing.assert_array_equal(ids, np.arange(400, N_VECTORS))
    assert partial_hnsw.num_points == N_VECTORS
    # A sparse graph can isolate the odd point, as it can after build, so this
    # asks for near-perfect self-retrieval rather than exact.
    ids, distances = partial_hnsw.search(base_data[400:], k=1, ef_search=N_VECTORS)
    found = np.count_nonzero(ids[:, 0] == np.arange(400, N_VECTORS))
    assert found >= 0.95 * (N_VECTORS - 400)
    assert np.isfinite(distances).all()
    after, _ = partial_hnsw.search(base_data[:10], k=1, ef_search=N_VECTORS)
    np.testing.assert_array_equal(before[:, 0], after[:, 0])


def test_add_into_a_dense_graph_retrieves_every_point(base_data, clusters):
    # M equals the point count, so the base layer holds 2M >= N - 1 links and never
    # prunes a reverse link: even a parallel add keeps every point reachable.
    centroids, cluster_ids = clusters
    idx = _built(
        base_data[:400],
        centroids,
        cluster_ids[:400],
        M=N_VECTORS,
        ef_construction=N_VECTORS,
        nbits=4,
    )
    idx.add(base_data[400:], cluster_ids=cluster_ids[400:])
    ids, _ = idx.search(base_data[400:], k=1, ef_search=N_VECTORS)
    np.testing.assert_array_equal(ids[:, 0], np.arange(400, N_VECTORS))


def test_add_without_cluster_ids_preserves_self_retrieval(partial_hnsw, base_data):
    ids = partial_hnsw.add(base_data[400:])
    np.testing.assert_array_equal(ids, np.arange(400, N_VECTORS))
    # Automatically routed points must remain searchable even though nearest-centroid
    # routing need not reproduce the round-robin assignments used during construction.
    found, _ = partial_hnsw.search(base_data[400:], k=1, ef_search=N_VECTORS)
    assert np.count_nonzero(found[:, 0] == np.arange(400, N_VECTORS)) >= 0.95 * 100


def test_add_beyond_capacity_grows_it(partial_hnsw, base_data):
    capacity = partial_hnsw.capacity
    partial_hnsw.add(base_data)
    assert partial_hnsw.num_points == 400 + N_VECTORS
    # Geometric growth: at least double, so repeated small adds stay cheap.
    assert partial_hnsw.capacity == max(400 + N_VECTORS, 2 * capacity)


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


def test_add_before_build_trains_first(base_data):
    idx = HnswIndex(DIM, M=8, ef_construction=50, nbits=4)
    np.testing.assert_array_equal(idx.add(base_data[:4]), np.arange(4))
    assert idx.is_trained
    assert idx.num_clusters == 4  # fewer rows than clusters: one centroid per row
    with pytest.raises(RuntimeError, match="trained or loaded"):
        HnswIndex(DIM, M=8, ef_construction=50, nbits=4).add(
            base_data[:4], cluster_ids=np.zeros(4, dtype=np.uint32)
        )


def test_reserve_grows_capacity_and_allows_more(base_data, clusters):
    centroids, cluster_ids = clusters
    idx = _built(
        base_data[:400], centroids, cluster_ids[:400], M=8, ef_construction=50, nbits=4
    )
    assert idx.capacity == 400  # the first add allocates exactly its rows

    idx.reserve(N_VECTORS)
    assert idx.capacity == N_VECTORS
    assert idx.num_points == 400

    idx.add(base_data[400:], cluster_ids=cluster_ids[400:], num_threads=1)
    assert idx.num_points == N_VECTORS
    ids, _ = idx.search(base_data[:10], k=1, ef_search=N_VECTORS)
    np.testing.assert_array_equal(ids[:, 0], np.arange(10))


def test_reserve_never_shrinks(partial_hnsw):
    # 450 lies between the 400 stored points and the capacity, where the
    # resize of 0.8 shrank the index.
    for smaller in (0, 10, 450, N_VECTORS):
        partial_hnsw.reserve(smaller)
        assert partial_hnsw.capacity == N_VECTORS
    with pytest.raises(ValueError, match="^num_points must be non-negative$"):
        partial_hnsw.reserve(-1)
    with pytest.raises(TypeError):
        partial_hnsw.reserve(1.5)
    assert partial_hnsw.capacity == N_VECTORS


@pytest.mark.parametrize("nbits", [1, 9])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_reserve_across_allocation_threshold_preserves_search(tmp_path, nbits, metric):
    rng = np.random.default_rng(73)
    data = rng.standard_normal((97, 65)).astype(np.float32)
    queries = rng.standard_normal((7, 65)).astype(np.float32)
    centroids = data.mean(axis=0, keepdims=True)
    index = _built(
        data,
        centroids,
        np.zeros(len(data), dtype=np.uint32),
        M=8,
        nbits=nbits,
        metric=metric,
    )
    expected_ids, expected_distances = index.search(queries, k=7, ef_search=97)

    # The level-0 allocation moves from below 2 MiB to above it.
    capacity = 32768
    index.reserve(capacity)
    path = str(tmp_path / "reserved.index")
    index.save(path)
    index = HnswIndex.load(path)
    assert index.capacity == capacity
    assert index.num_points == len(data)
    ids, distances = index.search(queries, k=7, ef_search=97)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(distances, expected_distances)


def test_added_points_survive_save_and_load(
    partial_hnsw, base_data, clusters, tmp_path
):
    _, cluster_ids = clusters
    partial_hnsw.add(base_data[400:], cluster_ids=cluster_ids[400:])
    before_ids, before_dists = partial_hnsw.search(base_data, k=_TOPK, ef_search=_EF)

    path = str(tmp_path / "hnsw-added.index")
    partial_hnsw.save(path)
    loaded = HnswIndex.load(path)

    assert loaded.num_points == N_VECTORS
    assert loaded.capacity == N_VECTORS
    after_ids, after_dists = loaded.search(base_data, k=_TOPK, ef_search=_EF)
    np.testing.assert_array_equal(before_ids, after_ids)
    np.testing.assert_allclose(before_dists, after_dists, rtol=1e-5)


# ── remove ────────────────────────────────────────────────────────────────────

_NO_ID = -1


def _removable(nbits=4, metric="l2", count=120, dim=64, seed=61):
    """A small built index, its points, and as many new points to add."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((count * 2, dim)).astype(np.float32)
    centroids = data[:count].mean(axis=0, keepdims=True)
    idx = _built(
        data[:count],
        centroids,
        np.zeros(count, dtype=np.uint32),
        M=8,
        ef_construction=50,
        nbits=nbits,
        metric=metric,
    )
    return idx, data[:count], data[count:]


def _everything(index, queries):
    """With k and ef at the point count, search visits every reachable point."""
    return index.search(
        queries, k=index.num_stored_points, ef_search=index.num_stored_points
    )


# C++ HnswRemoveTest covers intermediate widths; keep both storage endpoints here.
@pytest.mark.parametrize("nbits", [1, 9])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_remove_excludes_points_and_survives_reload(tmp_path, nbits, metric):
    idx, old, new = _removable(nbits=nbits, metric=metric)
    count = len(old)
    queries = new[:5]
    before_ids, before_dists = _everything(idx, queries)
    plain_path = tmp_path / "plain.index"
    idx.save(str(plain_path))

    removed = np.array([0, 5, 6, count - 1, 5])
    assert idx.remove(removed) == 4
    assert idx.remove(removed) == 0
    assert idx.remove([]) == 0
    assert idx.num_points == count - 4
    assert idx.num_stored_points == count

    def check(index):
        ids, dists = _everything(index, queries)
        alive = count - 4
        for q in range(len(queries)):
            keep = ~np.isin(before_ids[q], removed)
            np.testing.assert_array_equal(ids[q, :alive], before_ids[q][keep])
            np.testing.assert_array_equal(dists[q, :alive], before_dists[q][keep])
            assert np.all(ids[q, alive:] == _NO_ID)
            assert np.all(np.isinf(dists[q, alive:]))
        top, top_dists = index.search(queries, k=10, ef_search=10)
        assert not np.isin(top, removed).any()
        assert np.isfinite(top_dists).all()

    check(idx)
    path = tmp_path / "removed.index"
    idx.save(str(path))
    loaded = HnswIndex.load(str(path))
    check(loaded)
    assert loaded.remove(removed) == 0

    # New points link through removed ones, which stay hidden. Compare the two
    # loaded copies: save/load preserves the graph but not the random generator
    # state used for future insertion levels. The original can grow a different
    # graph, with a different set of reachable points.
    plain = HnswIndex.load(str(plain_path))
    for index in (idx, loaded, plain):
        index.reserve(count + len(new))
        # One thread: graphs grown in parallel differ, and the searches compare them.
        index.add(new, num_threads=1)
        assert index.num_stored_points == count + len(new)
        assert index.num_points == count + len(new) - (0 if index is plain else 4)

    # The original still supports add after remove, and its resulting graph must
    # round-trip exactly without any further insertions.
    original_ids, original_dists = _everything(idx, queries)
    assert not np.isin(original_ids, removed).any()
    grown_path = tmp_path / "grown.index"
    idx.save(str(grown_path))
    reloaded_ids, reloaded_dists = _everything(HnswIndex.load(str(grown_path)), queries)
    np.testing.assert_array_equal(reloaded_ids, original_ids)
    np.testing.assert_array_equal(reloaded_dists, original_dists)

    ids, dists = _everything(loaded, queries)
    plain_ids, plain_dists = _everything(plain, queries)
    for q in range(len(queries)):
        keep = ~np.isin(plain_ids[q], removed) & (plain_ids[q] != _NO_ID)
        found = np.count_nonzero(keep)
        assert found >= 0.95 * (count + len(new) - 4)
        np.testing.assert_array_equal(ids[q, :found], plain_ids[q][keep])
        np.testing.assert_array_equal(dists[q, :found], plain_dists[q][keep])
        assert np.all(ids[q, found:] == _NO_ID)
        assert np.all(np.isinf(dists[q, found:]))


def test_remove_rejects_invalid_ids_without_removing_anything():
    idx, old, new = _removable()
    count = len(old)
    before = _everything(idx, new[:3])
    # 2**32 + 1 would wrap to point 1 if it were cast to uint32 first.
    for bad in (-1, count, 2**32 + 1):
        with pytest.raises(ValueError, match="ids"):
            idx.remove([0, 1, bad])
    with pytest.raises(ValueError, match="1D"):
        idx.remove(np.zeros((2, 2), dtype=np.int64))
    after = _everything(idx, new[:3])
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert idx.remove([0, 1]) == 2

    unbuilt = HnswIndex(64, M=8, ef_construction=50, nbits=4)
    with pytest.raises(RuntimeError, match="trained or loaded"):
        unbuilt.remove([0])


def test_removing_every_point_returns_only_sentinels():
    idx, old, new = _removable()
    assert idx.remove(np.arange(len(old))) == len(old)
    ids, dists = idx.search(new[:2], k=5, ef_search=10)
    assert np.all(ids == _NO_ID)
    assert np.all(np.isinf(dists))


@pytest.mark.parametrize(
    "bad",
    [
        [2.7],
        [-0.5],
        # Integral floats are refused too: the dtype is the contract.
        np.array([1.0, 2.0]),
        np.array([True, False]),  # a boolean mask is not a list of ids
        ["3"],
        np.array([1, 2], dtype=object),
    ],
)
def test_remove_refuses_non_integer_ids(bad):
    idx, old, new = _removable()
    before = _everything(idx, new[:3])
    with pytest.raises(ValueError, match="integers"):
        idx.remove(bad)
    after = _everything(idx, new[:3])
    np.testing.assert_array_equal(before[0], after[0])


def test_remove_accepts_any_integer_dtype_and_empty_input():
    idx, _, _ = _removable()
    assert idx.remove([]) == 0
    assert idx.remove(np.array([], dtype=np.float64)) == 0  # no dtype to check
    assert idx.remove([1, 2]) == 2
    assert idx.remove(np.array([3, 4], dtype=np.uint8)) == 2
    assert idx.remove(np.array([5], dtype=np.int16)) == 1


def test_legacy_padding_round_trip_and_add(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "hnsw_legacy_padding65_4bit.index"
    index = HnswIndex.load(str(fixture))
    query = np.zeros((1, 65), dtype=np.float32)
    ids, distances = index.search(query, k=3, ef_search=8)
    np.testing.assert_array_equal(ids, [[0, 1, 2]])
    np.testing.assert_allclose(distances, [[0, 14, 14]], atol=1e-5)
    # Expected distances were recorded with the old 64-padding implementation.
    nonzero_query = ((np.arange(65) % 9 - 4) / 4).astype(np.float32)[None, :]
    nonzero_ids, nonzero_distances = index.search(nonzero_query, k=3, ef_search=8)
    np.testing.assert_array_equal(nonzero_ids, [[0, 2, 1]])
    np.testing.assert_allclose(
        nonzero_distances, [[27.81249809, 33.36483002, 50.26016998]], rtol=2e-6
    )
    path = tmp_path / "migrated.index"
    index.save(str(path))
    assert format2_file.padded_dim(path.read_bytes()) == 128
    loaded = HnswIndex.load(str(path))
    actual = loaded.search(query, k=3, ef_search=8)
    np.testing.assert_array_equal(actual[0], ids)
    np.testing.assert_array_equal(actual[1], distances)
    loaded.add(np.ones((1, 65), dtype=np.float32))
    assert loaded.num_points == 4


def test_huge_ef_is_clamped_to_the_stored_point_count():
    data = np.random.default_rng(5).normal(size=(80, 64)).astype(np.float32)
    index = _built(
        data, data.mean(axis=0, keepdims=True), np.zeros(80, dtype=np.uint32), nbits=4
    )
    exhaustive_ids, exhaustive_distances = index.search(data[:3], k=5, ef_search=80)
    ids, distances = index.search(data[:3], k=5, ef_search=2**40)
    np.testing.assert_array_equal(ids, exhaustive_ids)
    np.testing.assert_array_equal(distances, exhaustive_distances)
