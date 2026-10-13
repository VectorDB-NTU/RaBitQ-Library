"""Removal accepts contiguous integer views without assuming pointer alignment."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("readonly", [False, True])
def test_remove_integer_views_preserve_input_and_validate_atomically(
    kind, offset, readonly
):
    count, dim = 8, 64
    data = np.random.default_rng(42).normal(size=(count, dim)).astype(np.float32)
    index = (
        IvfIndex(dim, count, 1, 4)
        if kind == "ivf"
        else HnswIndex(dim, count, M=2, ef_construction=10, nbits=4)
    )
    index.build(data, np.zeros((1, dim), np.float32), np.zeros(count, np.uint32))
    search_args = {"nprobe": 1} if kind == "ivf" else {"ef": count}
    before = index.search(data, k=1, **search_args)

    def make_ids(values):
        result = np.ndarray(
            len(values),
            dtype=np.int64,
            buffer=bytearray(8 * len(values) + offset),
            offset=offset,
        )
        result[:] = values
        result.flags.writeable = not readonly
        assert result.flags.c_contiguous
        assert result.flags.aligned == (offset == 0)
        return result

    bad = make_ids([0, count])
    with pytest.raises(ValueError, match=r"^ids must be in \[0, num_stored_points\)$"):
        index.remove(bad)
    assert index.num_points == count
    for actual, expected in zip(
        index.search(data, k=1, **search_args), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(bad, [0, count])

    ids = make_ids([0, 3, 0])
    assert index.remove(ids) == 2
    assert index.remove(ids) == 0
    assert index.num_points == count - 2
    assert index.num_stored_points == count
    assert not np.isin(index.search(data, k=1, **search_args)[0], [0, 3]).any()
    np.testing.assert_array_equal(ids, [0, 3, 0])
