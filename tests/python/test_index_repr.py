"""Concise metadata summaries across index lifecycle and count changes."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

SUMMARIES = {
    "ivf": (
        "IvfIndex(dim=64, metric='ip', nbits={bits}, num_points={live}, "
        "num_stored_points={stored}, max_elements={capacity}, "
        "num_clusters={clusters}, initializer='flat', is_built={built})"
    ),
    "hnsw": (
        "HnswIndex(dim=64, metric='ip', nbits={bits}, num_points={live}, "
        "num_stored_points={stored}, max_elements={capacity}, "
        "num_clusters={clusters}, is_built={built})"
    ),
    "symqg": (
        "SymqgIndex(dim=64, metric='ip', quantization_bits={bits}, "
        "num_points={live}, max_degree=32, is_built={built})"
    ),
}


@pytest.fixture
def vectors():
    return np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)


def unbuilt_index(kind, bits):
    if kind == "ivf":
        return IvfIndex(64, 80, 1, bits, metric="innerproduct")
    if kind == "hnsw":
        return HnswIndex(64, 96, nbits=bits, metric="innerproduct")
    return SymqgIndex(64, 32, metric="innerproduct", quantization_bits=bits)


def build_index(index, vectors):
    if isinstance(index, SymqgIndex):
        index.build(vectors, 40)
    else:
        index.build(
            vectors, vectors.mean(axis=0, keepdims=True), np.zeros(80, dtype=np.uint32)
        )


@pytest.mark.parametrize(
    "kind,bits", [("ivf", 4), ("ivf", 32), ("hnsw", 8), ("symqg", 0), ("symqg", 8)]
)
def test_repr_lifecycle(kind, bits, vectors, tmp_path):
    index = unbuilt_index(kind, bits)
    capacity = 96 if kind == "hnsw" else 80
    before = SUMMARIES[kind].format(
        bits=bits,
        live=0,
        stored=0,
        capacity=capacity,
        clusters=0 if kind == "hnsw" else 1,
        built=False,
    )
    assert repr(index) == str(index) == before

    build_index(index, vectors)
    after = SUMMARIES[kind].format(
        bits=bits, live=80, stored=80, capacity=capacity, clusters=1, built=True
    )
    reference = index.search(vectors[:2], 3, 32)
    assert repr(index) == str(index) == after
    assert repr(index) == after
    result = index.search(vectors[:2], 3, 32)
    np.testing.assert_array_equal(result[0], reference[0])
    np.testing.assert_array_equal(result[1], reference[1])

    path = tmp_path / "summary.index"
    index.save(path)
    restored = type(index).load(path)
    assert repr(restored) == str(restored) == after


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
def test_repr_tracks_live_stored_and_capacity(kind, vectors, tmp_path):
    index = unbuilt_index(kind, 4)
    assert isinstance(index, (IvfIndex, HnswIndex))
    build_index(index, vectors)
    assert index.remove([0]) == 1
    capacity = 96 if kind == "hnsw" else 80
    assert repr(index) == SUMMARIES[kind].format(
        bits=4, live=79, stored=80, capacity=capacity, clusters=1, built=True
    )

    index.add(vectors[:1])
    if isinstance(index, HnswIndex):
        index.resize(128)
        capacity = 128
    else:
        capacity = 81
    expected = SUMMARIES[kind].format(
        bits=4, live=80, stored=81, capacity=capacity, clusters=1, built=True
    )
    assert repr(index) == expected
    path = tmp_path / "updated.index"
    index.save(path)
    assert repr(type(index).load(path)) == expected
