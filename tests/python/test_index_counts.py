"""All index wrappers distinguish capacity, stored records and live points."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


def check_counts(index, capacity, stored, active):
    assert (index.max_elements, index.num_stored_points, index.num_points) == (
        capacity,
        stored,
        active,
    )
    assert 0 <= index.num_points <= index.num_stored_points <= index.max_elements


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize(
    "kind,bits",
    [("ivf", 1), ("ivf", 4), ("ivf", 32), ("hnsw", 1), ("hnsw", 9)]
    + [("symqg", bits) for bits in (0, 4, 8)],
)
def test_count_contract_through_lifecycle(tmp_path, kind, bits, metric):
    count, dim = 40, 65
    data = np.random.default_rng(42).normal(size=(count + 4, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    centroid = np.zeros((1, dim), np.float32)
    assignments = np.zeros(count + 4, np.uint32)
    if kind == "ivf":
        index = IvfIndex(dim, count, 1, bits, metric=metric)
        capacity = count
    elif kind == "hnsw":
        index = HnswIndex(dim, 64, M=8, ef_construction=40, nbits=bits, metric=metric)
        capacity = 64
    else:
        index = SymqgIndex(dim, max_degree=32, quantization_bits=bits, metric=metric)
        capacity = 0
    check_counts(index, capacity, 0, 0)
    if kind == "symqg":
        index.build(data[:count], ef_construction=40, num_threads=1)
        capacity = count
    else:
        index.build(data[:count], centroid, assignments[:count])
    check_counts(index, capacity, count, count)

    path = tmp_path / "counts.index"
    if kind == "symqg":
        index.save(str(path))
        check_counts(SymqgIndex.load(str(path)), count, count, count)
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
        with pytest.raises(ValueError):
            index.resize(count - 3)
        index.resize(80)
        capacity = 80
        check_counts(index, capacity, count, count - 3)

    ids = index.add(data[count : count + 3], assignments[:3])
    np.testing.assert_array_equal(ids, np.arange(count, count + 3))
    stored = count + 3
    if kind == "ivf":
        capacity = stored
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
    args = {"nprobe": 1} if kind == "ivf" else {"ef": stored}
    hits, _ = index.search(data[:1], k=1, **args)
    assert np.all(hits == np.iinfo(np.uint32).max)

    np.testing.assert_array_equal(index.add(data[-1:], assignments[:1]), [stored])
    stored += 1
    if kind == "ivf":
        capacity = stored
    check_counts(index, capacity, stored, 1)
    index.save(str(path))
    check_counts(type(index).load(str(path)), capacity, stored, 1)
