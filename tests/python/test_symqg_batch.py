"""Batch scheduling must preserve scalar search, including quantized indexes."""

import numpy as np
import pytest
from rabitqlib import SymqgIndex


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_parallel_batch_matches_scalar(bits, metric, tmp_path):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(257, 65)).astype(np.float32)
    queries = rng.normal(size=(19, 65)).astype(np.float32)
    idx = SymqgIndex(65, 32, metric, bits)
    idx.build(data, 64, 1)
    for ef in (16, 64, 128):
        for k in (1, 10):
            expected_ids, expected_distances = idx.search(queries, k, ef, 1)
            for threads in (0, 2, 4, 8):
                ids, distances = idx.search(queries, k, ef, threads)
                np.testing.assert_array_equal(ids, expected_ids)
                np.testing.assert_array_equal(
                    distances.view(np.uint32), expected_distances.view(np.uint32)
                )
    path = str(tmp_path / "batch.index")
    idx.save(path)
    loaded = SymqgIndex.load(path)
    expected = idx.search(queries, 10, 64, 1)
    actual = loaded.search(queries, 10, 64, 4)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_empty_and_noncontiguous_batches(base_data, query_data):
    built_symqg = SymqgIndex(base_data.shape[1], 32)
    built_symqg.build(base_data, 64, 1)
    for threads in (0, 1, 4):
        ids, distances = built_symqg.search(query_data[:0], 5, 64, threads)
        assert ids.shape == distances.shape == (0, 5)
        subset = query_data[::2]
        actual = built_symqg.search(subset, 5, 64, threads)
        expected = built_symqg.search(np.ascontiguousarray(subset), 5, 64, 1)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
