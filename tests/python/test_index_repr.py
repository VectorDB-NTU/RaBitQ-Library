"""Concise metadata summaries across index lifecycle and count changes."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

SUMMARIES = {
    "ivf": (
        "IvfIndex(dim=64, metric='ip', nbits={bits}, num_points={live}, "
        "num_stored_points={stored}, max_elements={capacity}, "
        "num_clusters={clusters}, initializer='flat', is_trained={built})"
    ),
    "hnsw": (
        "HnswIndex(dim=64, metric='ip', nbits={bits}, num_points={live}, "
        "num_stored_points={stored}, max_elements={capacity}, "
        "num_clusters={clusters}, is_trained={built})"
    ),
    "symqg": (
        "SymqgIndex(dim=64, metric='ip', nbits={bits}, "
        "num_points={live}, max_degree=32)"
    ),
}


@pytest.fixture
def vectors():
    return np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)


def unbuilt_index(kind, bits):
    if kind == "ivf":
        return IvfIndex(64, 1, nbits=bits, metric="innerproduct")
    if kind == "hnsw":
        index = HnswIndex(64, nbits=bits, num_clusters=1, metric="innerproduct")
        # Keep capacity distinct from the 80 stored points.
        index.resize(96)
        return index
    return SymqgIndex(
        64, max_degree=32, ef_construction=40, metric="innerproduct", nbits=bits
    )


def build_index(index, vectors):
    if isinstance(index, SymqgIndex):
        index.add(vectors, num_threads=1)
    else:
        index.set_centroids(vectors.mean(axis=0, keepdims=True))
        index.add(vectors, cluster_ids=np.zeros(80, dtype=np.uint32), num_threads=1)


@pytest.mark.parametrize(
    "kind,bits", [("ivf", 4), ("ivf", 32), ("hnsw", 8), ("symqg", 32), ("symqg", 8)]
)
def test_repr_lifecycle(kind, bits, vectors, tmp_path):
    index = unbuilt_index(kind, bits)
    before = SUMMARIES[kind].format(
        bits=bits,
        live=0,
        stored=0,
        capacity=96 if kind == "hnsw" else 0,
        clusters=1,
        built=False,
    )
    assert repr(index) == str(index) == before

    build_index(index, vectors)
    capacity = 96 if kind == "hnsw" else 80
    after = SUMMARIES[kind].format(
        bits=bits,
        live=80,
        stored=80,
        capacity=capacity,
        clusters=1,
        built=True,
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
