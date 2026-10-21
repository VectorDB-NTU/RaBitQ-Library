"""Contiguous arrays may still require an aligned copy at the native boundary."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, QGKMeans, RaBitQKMeans, SymqgIndex


def unaligned_copy(values, offset):
    result = np.ndarray(
        values.shape,
        dtype=values.dtype,
        buffer=bytearray(values.nbytes + offset),
        offset=offset,
    )
    result[:] = values
    result.flags.writeable = False
    assert result.flags.c_contiguous
    assert not result.flags.aligned
    return result


@pytest.mark.parametrize("model_type", [RaBitQKMeans, QGKMeans])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_unaligned_training_matches_aligned_input(model_type, offset):
    data = np.random.default_rng(42).standard_normal((97, 65), dtype=np.float32)
    unaligned = unaligned_copy(data, offset)
    expected = model_type(65, 33, niter=2, num_threads=1, seed=42)
    actual = model_type(65, 33, niter=2, num_threads=1, seed=42)
    assert actual.train(unaligned) == expected.train(data)
    for name in ("centroids", "assignments", "distances", "obj"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(unaligned, data)


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_indexes_accept_unaligned_build_and_query_arrays(kind, offset):
    rng = np.random.default_rng(43)
    data = rng.standard_normal((65, 65), dtype=np.float32)
    centroids = data[:2].copy()
    labels = (np.arange(len(data)) % 2).astype(np.uint32)
    queries = rng.standard_normal((3, 65), dtype=np.float32)
    unaligned_data = unaligned_copy(data, offset)
    unaligned_centroids = unaligned_copy(centroids, offset)
    unaligned_queries = unaligned_copy(queries, offset)
    if kind == "ivf":
        index = IvfIndex(65, len(data), 2, 4)
        parameters = {"nprobe": 2}
    elif kind == "hnsw":
        index = HnswIndex(65, len(data), M=8, ef_construction=32, nbits=4)
        parameters = {"ef": 65}
    else:
        index = SymqgIndex(65, 32)
        parameters = {"ef": 65}
    if kind == "qg":
        index.build(unaligned_data, ef_construction=32, num_threads=1)
    else:
        index.build(unaligned_data, unaligned_centroids, labels, num_threads=1)
    expected = index.search(queries, k=5, num_threads=1, **parameters)
    actual = index.search(unaligned_queries, k=5, num_threads=1, **parameters)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    np.testing.assert_array_equal(unaligned_data, data)
    np.testing.assert_array_equal(unaligned_centroids, centroids)
    np.testing.assert_array_equal(unaligned_queries, queries)
