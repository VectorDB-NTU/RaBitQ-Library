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
def test_indexes_accept_unaligned_build_and_query_arrays(kind, offset, tmp_path):
    rng = np.random.default_rng(43)
    data = rng.standard_normal((65, 65), dtype=np.float32)
    centroids = data[:2].copy()
    labels = (np.arange(len(data)) % 2).astype(np.uint32)
    queries = rng.standard_normal((3, 65), dtype=np.float32)
    unaligned_data = unaligned_copy(data, offset)
    unaligned_centroids = unaligned_copy(centroids, offset)
    unaligned_queries = unaligned_copy(queries, offset)

    def make_index():
        if kind == "ivf":
            return IvfIndex(65, len(data), 2, 4)
        if kind == "hnsw":
            return HnswIndex(
                65, len(data), M=8, ef_construction=32, nbits=4, random_seed=42
            )
        return SymqgIndex(65, 32)

    if kind == "ivf":
        parameters = {"nprobe": 2}
    else:
        parameters = {"ef": 65}

    if kind != "qg":
        reference = make_index()
        reference.build(data, centroids, labels, num_threads=1)
        reference_path = tmp_path / "aligned.index"
        reference.save(str(reference_path))
        expected_bytes = reference_path.read_bytes()
        expected = reference.search(queries, k=5, num_threads=1, **parameters)

    # Isolate each conversion so a damaged centroid cannot hide behind data damage.
    inputs = [(unaligned_data, centroids)]
    if kind != "qg":
        inputs += [(data, unaligned_centroids), (unaligned_data, unaligned_centroids)]
    for vectors, centers in inputs:
        # IVF samples a rotation at construction; rebuild a loaded reference to
        # reuse it. HNSW's explicit seed reproduces its rotation and serial graph.
        index = IvfIndex.load(str(reference_path)) if kind == "ivf" else make_index()
        if kind == "qg":
            index.build(vectors, ef_construction=32, num_threads=1)
            ids, distances = index.search(queries, k=len(data), **parameters)
            for row in ids:
                np.testing.assert_array_equal(np.sort(row), np.arange(len(data)))
            exact = np.sum(
                (queries.astype(np.float64)[:, None] - data.astype(np.float64)) ** 2,
                axis=2,
            )
            np.testing.assert_allclose(
                distances, np.take_along_axis(exact, ids, axis=1), rtol=2e-6, atol=1e-5
            )
            expected = index.search(queries, k=5, num_threads=1, **parameters)
        else:
            index.build(vectors, centers, labels, num_threads=1)
            actual_path = tmp_path / "unaligned.index"
            index.save(str(actual_path))
            assert actual_path.read_bytes() == expected_bytes
        for values in (queries, unaligned_queries):
            actual = index.search(values, k=5, num_threads=1, **parameters)
            for result, reference_result in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(result, reference_result)
    np.testing.assert_array_equal(unaligned_data, data)
    np.testing.assert_array_equal(unaligned_centroids, centroids)
    np.testing.assert_array_equal(unaligned_queries, queries)
