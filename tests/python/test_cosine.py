import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

DIM = 65
COUNT = 300


def normalized(rows):
    """Normalize as the cosine metric specifies: the squared norm summed in float64
    in coordinate order (cumsum is sequential), then each coordinate times its
    reciprocal square root, rounded to float32. Zero rows stay zero."""
    wide = np.asarray(rows, dtype=np.float64)
    squared_norm = np.cumsum(wide * wide, axis=1)[:, -1]
    scale = np.zeros_like(squared_norm)
    nonzero = squared_norm > 0
    scale[nonzero] = 1.0 / np.sqrt(squared_norm[nonzero])
    return (wide * scale[:, None]).astype(np.float32)


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(17)
    rows = rng.standard_normal((COUNT + 5, DIM), dtype=np.float32)
    rows *= (1 + np.arange(len(rows)) % 7)[:, None].astype(np.float32)
    return rows[:COUNT], rows[COUNT:]


def make_index(kind, metric, nbits):
    if kind == "ivf":
        return IvfIndex(DIM, 4, nbits=nbits, metric=metric, seed=7)
    if kind == "hnsw":
        return HnswIndex(DIM, nbits=nbits, num_clusters=4, M=8, metric=metric, seed=7)
    return SymqgIndex(DIM, nbits=nbits, ef_construction=40, metric=metric, seed=7)


CASES = [
    ("ivf", 1),
    ("ivf", 5),
    ("ivf", 32),
    ("hnsw", 1),
    ("hnsw", 5),
    ("symqg", 4),
    ("symqg", 32),
]


@pytest.mark.parametrize("kind,nbits", CASES)
def test_cosine_matches_inner_product_over_normalized_vectors(
    kind, nbits, vectors, tmp_path
):
    data, queries = vectors
    cosine = make_index(kind, "cosine", nbits)
    ip = make_index(kind, "ip", nbits)
    # One thread, so that training and IVF HNSW routing are reproducible.
    cosine.add(data, num_threads=1)
    ip.add(normalized(data), num_threads=1)
    assert cosine.metric == "cosine"
    assert "metric='cosine'" in repr(cosine)

    expected = ip.search(normalized(queries), 10)
    path = tmp_path / f"{kind}.index"
    cosine.save(path)
    loaded = type(cosine).load(path)
    assert loaded.metric == "cosine"
    for index in (cosine, loaded):
        ids, distances = index.search(queries, 10, num_threads=2)
        np.testing.assert_array_equal(ids, expected[0])
        np.testing.assert_array_equal(distances, expected[1])


@pytest.mark.parametrize("kind,nbits", [("ivf", 32), ("symqg", 32)])
def test_cosine_distance_of_a_zero_vector_is_one(kind, nbits, vectors):
    # Raw vectors give exact distances.
    data = vectors[0].copy()
    data[0] = 0
    index = make_index(kind, "cosine", nbits)
    index.add(data, num_threads=1)
    _, distances = index.search(np.zeros((1, DIM), dtype=np.float32), 10)
    np.testing.assert_array_equal(distances, np.ones((1, 10), dtype=np.float32))
    breadth = {"nprobe": 4} if kind == "ivf" else {"ef": COUNT}
    ids, distances = index.search(data[1:2], COUNT, **breadth)
    assert distances[0, list(ids[0]).index(0)] == 1.0


def test_cosine_distances_are_one_minus_cosine(vectors):
    data, queries = vectors
    index = make_index("ivf", "cosine", 32)
    index.add(data, num_threads=1)
    ids, distances = index.search(queries, 10, nprobe=4)
    exact = 1.0 - normalized(queries) @ normalized(data).T
    np.testing.assert_allclose(
        distances, np.take_along_axis(exact, ids, axis=1), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
@pytest.mark.parametrize("setup", ["train", "set_centroids"])
def test_cosine_trained_or_given_centroids_match_inner_product(kind, setup, vectors):
    data, queries = vectors
    cosine = make_index(kind, "cosine", 5)
    ip = make_index(kind, "ip", 5)
    if setup == "train":
        cosine.train(data[:200], num_threads=1)
        ip.train(normalized(data[:200]), num_threads=1)
    else:
        # Centroids are used as given.
        centroids = normalized(data[:4])
        cosine.set_centroids(centroids)
        ip.set_centroids(centroids)
    cosine.add(data, num_threads=1)
    ip.add(normalized(data), num_threads=1)
    ids, distances = cosine.search(queries, 10)
    expected_ids, expected_distances = ip.search(normalized(queries), 10)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(distances, expected_distances)


@pytest.mark.parametrize("kind,label", [("ivf", "IVF"), ("hnsw", "HNSW")])
def test_cosine_training_checks_the_original_data(kind, label, vectors):
    data = vectors[0].copy()
    data[3, 5] = np.nan
    index = make_index(kind, "cosine", 5)
    with pytest.raises(
        ValueError, match=rf"^{label} training data must contain only finite values$"
    ):
        index.train(data)
    assert not index.is_trained
