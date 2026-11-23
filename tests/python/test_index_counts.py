"""All index wrappers distinguish stored records and live points; HNSW adds capacity."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


def check_counts(index, capacity, stored, active):
    """capacity is None for the indexes without one."""
    assert (index.num_stored_points, index.num_points) == (stored, active)
    assert 0 <= index.num_points <= index.num_stored_points
    assert getattr(index, "capacity", None) == capacity
    assert not hasattr(index, "max_elements")
    if capacity is not None:
        assert index.num_stored_points <= capacity


# Metric forwarding is covered by portable-index tests; counts are metric-independent.
@pytest.mark.parametrize(
    "kind,bits",
    [("ivf", 1), ("ivf", 4), ("ivf", 32), ("hnsw", 1), ("hnsw", 9)]
    + [("symqg", bits) for bits in (32, 4, 8)],
)
def test_count_contract_through_lifecycle(tmp_path, kind, bits):
    metric = "l2"
    count, dim = 40, 65
    data = np.random.default_rng(42).normal(size=(count + 4, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    centroid = np.zeros((1, dim), np.float32)
    assignments = np.zeros(count + 4, np.uint32)
    if kind == "ivf":
        index = IvfIndex(dim, 1, nbits=bits, metric=metric)
        capacity = None
    elif kind == "hnsw":
        index = HnswIndex(
            dim, num_clusters=1, M=8, ef_construction=40, nbits=bits, metric=metric
        )
        index.reserve(64)
        capacity = 64
    else:
        index = SymqgIndex(
            dim, max_degree=32, ef_construction=40, nbits=bits, metric=metric
        )
        capacity = None
    check_counts(index, capacity, 0, 0)
    if kind == "symqg":
        index.add(data[:count], num_threads=1)
    else:
        index.set_centroids(centroid)
        index.add(data[:count], cluster_ids=assignments[:count], num_threads=1)
    check_counts(index, capacity, count, count)

    path = tmp_path / "counts.index"
    if kind == "symqg":
        index.save(str(path))
        check_counts(SymqgIndex.load(str(path)), None, count, count)
        return

    assert index.remove([0, 1, 0]) == 2
    assert index.remove([0, 1]) == 0
    check_counts(index, capacity, count, count - 2)
    # A live-count bound would incorrectly reject this valid high ID.
    assert index.remove([count - 1]) == 1
    check_counts(index, capacity, count, count - 3)
    with pytest.raises(ValueError, match="ids"):
        index.remove([2, count])
    check_counts(index, capacity, count, count - 3)
    if kind == "hnsw":
        index.reserve(count - 3)  # never shrinks
        check_counts(index, capacity, count, count - 3)
        index.reserve(80)
        capacity = 80
        check_counts(index, capacity, count, count - 3)

    ids = index.add(data[count : count + 3], cluster_ids=assignments[:3])
    np.testing.assert_array_equal(ids, np.arange(count, count + 3))
    stored = count + 3
    check_counts(index, capacity, stored, count)
    index.save(str(path))
    index = type(index).load(str(path))
    check_counts(index, capacity, stored, count)
    assert index.remove([0, 1, count - 1]) == 0
    check_counts(index, capacity, stored, count)

    assert index.remove(np.arange(stored)) == count
    check_counts(index, capacity, stored, 0)
    index.save(str(path))
    index = type(index).load(str(path))
    check_counts(index, capacity, stored, 0)
    args = {"nprobe": 1} if kind == "ivf" else {"ef_search": stored}
    hits, _ = index.search(data[:1], k=1, **args)
    assert np.all(hits == -1)

    np.testing.assert_array_equal(
        index.add(data[-1:], cluster_ids=assignments[:1]), [stored]
    )
    stored += 1
    check_counts(index, capacity, stored, 1)
    index.save(str(path))
    check_counts(type(index).load(str(path)), capacity, stored, 1)
