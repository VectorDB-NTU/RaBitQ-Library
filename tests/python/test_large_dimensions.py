"""FHT-based indexes support large and padded dimensions across persistence."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


@pytest.mark.parametrize("dim", [4096, 16384, 16385, 65536])
@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
def test_large_dimension_round_trip(tmp_path, dim, metric, kind):
    count = 40
    data = np.random.default_rng(42).standard_normal((count, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    if kind == "ivf":
        index = IvfIndex(dim, count, 1, nbits=4, metric=metric)
        search_args = {"nprobe": 1}
    elif kind == "hnsw":
        index = HnswIndex(
            dim, count, M=count, ef_construction=count, nbits=4, metric=metric
        )
        search_args = {"ef": count}
    else:
        index = SymqgIndex(dim, max_degree=32, quantization_bits=4, metric=metric)
        search_args = {"ef": count}

    if kind == "qg":
        index.build(data, ef_construction=count, num_threads=2)
    else:
        index.build(data, data.mean(axis=0, keepdims=True), np.zeros(count, np.uint32))
    queries = data[:3]
    ids, distances = index.search(queries, k=1, **search_args)
    np.testing.assert_array_equal(ids[:, 0], np.arange(3))
    assert np.isfinite(distances).all()

    path = tmp_path / "large.index"
    index.save(str(path))
    loaded = type(index).load(str(path))
    restored_ids, restored_distances = loaded.search(queries, k=1, **search_args)
    np.testing.assert_array_equal(restored_ids, ids)
    np.testing.assert_array_equal(restored_distances, distances)


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "qg"])
def test_rejects_dimensions_above_fht_limit(kind):
    dim = 65537
    with pytest.raises(ValueError, match="Unsupported dimension for FhtKacRotator"):
        if kind == "ivf":
            IvfIndex(dim, 40, 1, nbits=4)
        elif kind == "hnsw":
            HnswIndex(dim, 40, nbits=4)
        else:
            index = SymqgIndex(dim, max_degree=32, quantization_bits=4)
            index.build(np.zeros((40, dim), dtype=np.float32), ef_construction=40)
