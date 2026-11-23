"""Tests for SymqgIndex: construction, search, properties, error handling, save/load."""

import os
import re

import format2_file
import numpy as np
import pytest
from conftest import DIM, N_QUERIES, N_VECTORS, brute_force_knn, recall_at_k
from rabitqlib import SymqgIndex

_TOPK = 10
_EF = 50
_MAX_DEGREE = 32  # must be a multiple of fastscan::kBatchSize (32)
_EF_BUILD = 50


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built_symqg(base_data):
    # A seeded single-thread build is reproducible, so the self-retrieval test can
    # require every probe (#216).
    idx = SymqgIndex(DIM, max_degree=_MAX_DEGREE, ef_construction=_EF_BUILD, seed=42)
    idx.add(base_data, num_threads=1)
    return idx


# ── construction ──────────────────────────────────────────────────────────────


def test_properties(built_symqg):
    assert len(built_symqg) == N_VECTORS
    assert built_symqg.dim == DIM
    assert built_symqg.max_degree == _MAX_DEGREE
    assert built_symqg.num_points == N_VECTORS
    assert built_symqg.metric == "l2"
    assert built_symqg.nbits == 32


# ── search output shape and dtype ─────────────────────────────────────────────


@pytest.mark.parametrize("nq,k", [(N_QUERIES, _TOPK), (1, 1)])
def test_search_output_contract(built_symqg, query_data, nq, k):
    ids, dists = built_symqg.search(query_data[:nq], k=k, ef_search=_EF)
    assert ids.shape == dists.shape == (nq, k)
    assert ids.dtype == np.int64
    assert dists.dtype == np.float32
    assert np.all(ids < N_VECTORS)
    assert np.isfinite(dists).all()
    assert np.all(dists >= 0)


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_batch_search_matches_individual_queries(tmp_path, bits, metric):
    rng = np.random.default_rng(73)
    data = rng.standard_normal((97, 65)).astype(np.float32)
    queries = rng.standard_normal((19, 130)).astype(np.float32)[:, ::2]
    assert not queries.flags.c_contiguous
    queries[4] = queries[0]
    index = SymqgIndex(
        65, max_degree=32, metric=metric, nbits=bits or 32, ef_construction=48
    )
    index.add(data, num_threads=1)
    path = str(tmp_path / "batch.index")
    index.save(path)
    restored = SymqgIndex.load(path)

    # C++ owns the full ef/thread matrix. These scenarios retain serial, automatic
    # and parallel scheduling, changing windows, and strided inputs while sharing a
    # single build for each metric/storage pair.
    for k, ef, threads in [
        (7, 64, 1),
        (1, 1, 4),
        (7, 64, 4),
        (10, 16, 0),
        (10, 128, 2),
        (1, 64, 8),
    ]:
        individual = [
            index.search(query.copy()[None, :], k=k, ef_search=ef) for query in queries
        ]
        expected_ids = np.concatenate([result[0] for result in individual])
        expected_distances = np.concatenate([result[1] for result in individual])
        ids, distances = restored.search(
            queries, k=k, ef_search=ef, num_threads=threads
        )
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(
            distances.view(np.uint32), expected_distances.view(np.uint32)
        )
        empty_ids, empty_distances = restored.search(
            queries[:0], k=k, ef_search=ef, num_threads=threads
        )
        assert empty_ids.shape == empty_distances.shape == (0, k)


def test_search_validates_inputs(built_symqg, query_data):
    with pytest.raises(ValueError, match="queries must be a 2D NumPy array"):
        built_symqg.search(query_data[0], k=1, ef_search=_EF)
    with pytest.raises(ValueError, match="query dimension does not match index dim"):
        built_symqg.search(query_data[:, :-1], k=1, ef_search=_EF)
    with pytest.raises(ValueError, match=r"k must be between 1 and 2\*\*31"):
        built_symqg.search(query_data, k=0, ef_search=_EF)
    unbuilt = SymqgIndex(DIM, max_degree=_MAX_DEGREE)
    with pytest.raises(RuntimeError, match="add them or load an index before search"):
        unbuilt.search(query_data, k=1, ef_search=_EF)


# ── search correctness ────────────────────────────────────────────────────────


def test_self_retrieval(built_symqg, base_data):
    """Each database vector must be its own nearest neighbor at high ef."""
    probes = base_data[:10]
    ids, _ = built_symqg.search(probes, k=1, ef_search=200)
    for i in range(10):
        assert i in ids[i], f"Vector {i} not found in its own top-1 result"


def test_recall_vs_brute_force(built_symqg, base_data, query_data):
    """Approximate recall should exceed 0.5 at ef_search=50 on a small dataset."""
    k = 5
    approx_ids, _ = built_symqg.search(query_data, k=k, ef_search=_EF)
    exact_ids, _ = brute_force_knn(base_data, query_data, k)
    r = recall_at_k(approx_ids, exact_ids, k)
    assert r >= 0.5, f"Recall {r:.3f} too low"


# ── error handling ────────────────────────────────────────────────────────────


def test_wrong_data_dim_raises():
    idx = SymqgIndex(DIM, max_degree=_MAX_DEGREE, ef_construction=_EF_BUILD)
    bad_data = np.zeros((N_VECTORS, DIM + 1), dtype=np.float32)
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            f"data dimension does not match index dim; expected last dimension {DIM}, got shape {bad_data.shape}"
        )
        + "$",
    ):
        idx.add(bad_data, num_threads=1)


def test_wrong_query_dim_raises(built_symqg):
    bad_queries = np.zeros((5, DIM + 1), dtype=np.float32)
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            f"query dimension does not match index dim; expected last dimension {DIM}, got shape {bad_queries.shape}"
        )
        + "$",
    ):
        built_symqg.search(bad_queries, k=1, ef_search=_EF)


def test_save_before_build_raises(tmp_path):
    idx = SymqgIndex(DIM, max_degree=_MAX_DEGREE)
    with pytest.raises(
        RuntimeError,
        match="^SymqgIndex has no vectors; add them or load an index before save$",
    ):
        idx.save(str(tmp_path / "unbuilt.index"))


def test_search_before_build_raises():
    idx = SymqgIndex(DIM, max_degree=_MAX_DEGREE)
    queries = np.zeros((5, DIM), dtype=np.float32)
    with pytest.raises(
        RuntimeError,
        match="^SymqgIndex has no vectors; add them or load an index before search$",
    ):
        idx.search(queries, k=1, ef_search=_EF)


def test_search_raises_ef_to_k_and_defaults_it(built_symqg, query_data):
    # As in HnswIndex: no ef_search selects max(k, 64), and it is at least k.
    at_k = built_symqg.search(query_data, k=10, ef_search=10)
    for ef in (9, 1):
        raised = built_symqg.search(query_data, k=10, ef_search=ef)
        np.testing.assert_array_equal(raised[0], at_k[0])
    default = built_symqg.search(query_data, k=3, ef_search=64)
    np.testing.assert_array_equal(
        built_symqg.search(query_data, 3, None)[0], default[0]
    )
    np.testing.assert_array_equal(built_symqg.search(query_data, 3)[0], default[0])
    with pytest.raises(
        ValueError, match="^ef_search must be at least 1; pass None for the default$"
    ):
        built_symqg.search(query_data, 3, 0)


def test_invalid_degree_raises():
    with pytest.raises(
        ValueError, match="^max_degree must be a positive multiple of 32$"
    ):
        SymqgIndex(DIM, max_degree=16)


def test_save_propagates_write_failure(built_symqg):
    if not os.path.exists("/dev/full"):
        pytest.skip("requires a failing output device")
    with pytest.raises(RuntimeError):
        built_symqg.save("/dev/full")


@pytest.mark.parametrize("bits", [4, 8])
def test_qg_quant_build_search_and_roundtrip(bits, base_data, query_data, tmp_path):
    idx = SymqgIndex(DIM, max_degree=_MAX_DEGREE, nbits=bits, ef_construction=_EF_BUILD)
    idx.add(base_data, num_threads=1)
    assert idx.nbits == bits

    ids, dists = idx.search(query_data, k=5, ef_search=_EF)
    assert np.all(ids < N_VECTORS)
    assert np.all(np.isfinite(dists))

    path = str(tmp_path / f"symqg_quant{bits}.index")
    idx.save(path)
    loaded = SymqgIndex.load(path)
    assert loaded.nbits == bits
    loaded_ids, loaded_dists = loaded.search(query_data, k=5, ef_search=_EF)
    np.testing.assert_array_equal(loaded_ids, ids)
    np.testing.assert_allclose(loaded_dists, dists, rtol=1e-5)


# ── save / load roundtrip ─────────────────────────────────────────────────────


def test_save_load_roundtrip(built_symqg, query_data, tmp_path):
    path = str(tmp_path / "symqg.index")
    built_symqg.save(path)

    loaded = SymqgIndex.load(path)
    assert len(loaded) == len(built_symqg)
    assert loaded.dim == built_symqg.dim
    assert loaded.max_degree == built_symqg.max_degree
    assert loaded.num_points == built_symqg.num_points

    ids_orig, dists_orig = built_symqg.search(query_data, k=_TOPK, ef_search=_EF)
    ids_load, dists_load = loaded.search(query_data, k=_TOPK, ef_search=_EF)
    np.testing.assert_array_equal(ids_orig, ids_load)
    np.testing.assert_allclose(dists_orig, dists_load, rtol=1e-5)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_code_only_build_saves_the_specified_layout(bits, metric, tmp_path):
    rng = np.random.default_rng(851)
    data = rng.normal(size=(65, 65)).astype(np.float32)
    queries = rng.normal(size=(3, 65)).astype(np.float32)
    index = SymqgIndex(65, max_degree=32, metric=metric, nbits=bits, ef_construction=64)
    index.add(data, num_threads=1)
    data.fill(np.nan)
    path = tmp_path / "code_only.index"
    index.save(str(path))
    payload = path.read_bytes()
    assert payload[:8] == format2_file.MAGIC
    # Format-2 sections: the FHT/Kac rotator blob, the centroid, and per vertex its
    # single-code record, neighbor batches and neighbor IDs.
    padded = 96
    assert format2_file.padded_dim(payload) == padded
    row_bytes = padded * bits // 8 + 8 + padded * 4 + 256 + 128
    sizes = {kind: size for kind, (_, size) in format2_file.sections(payload).items()}
    assert sizes == {
        format2_file.ROTATOR: 64 + padded // 2,
        format2_file.QG_PARAMS: 32,
        format2_file.QG_CENTROID: padded * 4,
        format2_file.QG_RECORDS: 65 * row_bytes,
    }
    ids, distances = index.search(queries, 10, 64)
    assert np.isfinite(distances).all()
    assert all(len(set(row)) == 10 for row in ids)
    loaded = SymqgIndex.load(str(path))
    loaded_ids, loaded_distances = loaded.search(queries, 10, 64)
    np.testing.assert_array_equal(loaded_ids, ids)
    np.testing.assert_array_equal(loaded_distances, distances)


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_constant_vectors_preserve_returned_distance(metric, tmp_path):
    """Neighbor corrections must not shift the vertex distances returned to Python."""
    data = np.full((65, 64), 0.125, dtype=np.float32)
    queries = np.full((2, 64), 0.0625, dtype=np.float32)
    index = SymqgIndex(64, max_degree=32, metric=metric, ef_construction=64)
    index.add(data, num_threads=1)
    ids, distances = index.search(queries, k=5, ef_search=64)
    expected = 0.5 if metric == "ip" else 0.25
    np.testing.assert_allclose(distances, expected, rtol=0, atol=1e-6)
    assert np.all(ids < len(data))
    path = str(tmp_path / "constant.symqg")
    index.save(path)
    loaded_ids, loaded_distances = SymqgIndex.load(path).search(
        queries, k=5, ef_search=64
    )
    np.testing.assert_array_equal(loaded_ids, ids)
    np.testing.assert_allclose(loaded_distances, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("init", [None, "pipnn", "random"])
def test_initialization_and_roundtrip(tmp_path, bits, metric, init):
    rng = np.random.default_rng(17)
    data = rng.standard_normal((160, 65)).astype(np.float32)
    queries = rng.standard_normal((8, 65)).astype(np.float32)
    kwargs = {} if init is None else {"init": init}
    index = SymqgIndex(
        65,
        max_degree=32,
        metric=metric,
        nbits=bits or 32,
        ef_construction=128,
        **kwargs,
    )
    index.add(data, num_threads=2)
    assert len(index) == len(data)
    ids, distances = index.search(queries, k=10, ef_search=160, num_threads=2)
    assert np.all(ids < len(data))
    assert np.isfinite(distances).all()
    assert all(len(set(row)) == 10 for row in ids)
    exact = (
        brute_force_knn(data, queries, 10)[0]
        if metric == "l2"
        else np.argsort(-(queries @ data.T), axis=1)[:, :10]
    )
    assert recall_at_k(ids, exact, 10) >= (0.7 if bits else 0.95)
    if bits == 0:
        expected = (
            np.sum((queries[:, None, :] - data[ids]) ** 2, axis=2)
            if metric == "l2"
            else 1 - np.einsum("qd,qkd->qk", queries, data[ids])
        )
        np.testing.assert_allclose(distances, expected, rtol=1e-5, atol=1e-5)
    path = str(tmp_path / "native_pipnn.index")
    index.save(path)
    restored = SymqgIndex.load(path)
    restored_ids, restored_distances = restored.search(queries, k=10, ef_search=160)
    np.testing.assert_array_equal(ids, restored_ids)
    np.testing.assert_array_equal(distances, restored_distances)


def test_native_pipnn_rejects_invalid_build_arguments():
    with pytest.raises(ValueError, match="init must be 'random' or 'pipnn'"):
        SymqgIndex(DIM, max_degree=32, ef_construction=64, init="unknown")
    with pytest.raises(
        ValueError, match="ef_construction must be positive and fit uint32"
    ):
        SymqgIndex(DIM, max_degree=32, ef_construction=0, init="pipnn")


def test_default_initialization_is_pipnn():
    assert SymqgIndex(DIM).init == "pipnn"
