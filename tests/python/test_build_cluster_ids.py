"""Build must validate original assignments before narrowing them to point IDs."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
@pytest.mark.parametrize(
    "bad",
    [2**32, -(2**32), 0.5, -1, 1, np.uint64(2**64 - 1), True, "0"],
)
def test_build_rejects_invalid_cluster_ids(kind, bad):
    data = np.ones((2, 64), dtype=np.float32)
    centroids = np.zeros((1, 64), dtype=np.float32)
    index = (
        IvfIndex(64, 2, 1, 4)
        if kind == "ivf"
        else HnswIndex(64, 2, M=2, ef_construction=10, nbits=4)
    )
    ids = np.full(2, bad)
    with pytest.raises(ValueError, match="cluster_ids"):
        index.build(data, centroids, ids)
    assert not index.is_built
    index.build(data, centroids, np.zeros(2, dtype=np.int64))
    assert index.is_built


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
@pytest.mark.parametrize("layout", ["native", "strided", "swapped", "unaligned"])
def test_build_uint32_cluster_ids_validate_and_preserve_input(kind, layout):
    data = np.ones((2, 64), dtype=np.float32)
    data[1] = 2
    centroids = data.copy()
    index = (
        IvfIndex(64, 2, 2, 4)
        if kind == "ivf"
        else HnswIndex(64, 2, M=2, ef_construction=10, nbits=4)
    )
    if layout == "strided":
        ids = np.zeros(4, dtype=np.uint32)[::2]
    elif layout == "swapped":
        ids = np.zeros(2, dtype=np.dtype(np.uint32).newbyteorder("S"))
    elif layout == "unaligned":
        ids = np.ndarray(2, dtype=np.uint32, buffer=bytearray(9), offset=1)
    else:
        ids = np.zeros(2, dtype=np.uint32)
    ids[1] = np.iinfo(np.uint32).max
    with pytest.raises(ValueError, match="cluster_ids"):
        index.build(data, centroids, ids, num_threads=1)
    assert not index.is_built
    ids[1] = 1
    ids.flags.writeable = False
    before = ids.copy()
    index.build(data, centroids, ids, num_threads=1)
    assert index.is_built
    np.testing.assert_array_equal(ids, before)
