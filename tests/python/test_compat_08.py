"""The 0.7 call forms the 0.8 compatibility layer keeps; removed in 0.9."""

import inspect
import warnings

import numpy as np
import pytest
from conftest import DIM, N_VECTORS
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


def single_deprecation(call):
    """Run call and return its result and its one DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    assert [w.category for w in caught] == [DeprecationWarning], [
        str(w.message) for w in caught
    ]
    # Attributed to the caller, so users see their own line.
    assert caught[0].filename == __file__
    assert "python_migration" in str(caught[0].message)
    return result, str(caught[0].message)


def quietly(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return call()


@pytest.mark.parametrize(
    "make",
    [
        lambda: IvfIndex(DIM, N_VECTORS, 4, 4),
        lambda: IvfIndex(DIM, N_VECTORS, 4, nbits=4),
        lambda: IvfIndex(DIM, N_VECTORS, num_clusters=4, nbits=4),
        lambda: IvfIndex(
            dim=DIM, max_elements=N_VECTORS, num_clusters=4, nbits=4, metric="ip"
        ),
    ],
)
def test_ivf_point_count_forms(make, base_data, clusters):
    index, message = single_deprecation(make)
    assert message.startswith(
        "IvfIndex(dim, max_elements, num_clusters, nbits, ...) is deprecated"
    )
    assert index.max_elements == N_VECTORS
    assert index.num_clusters == 4
    assert index.seed is None
    assert not index.is_trained


@pytest.mark.parametrize(
    "make",
    [
        lambda: HnswIndex(DIM, N_VECTORS),
        lambda: HnswIndex(DIM, N_VECTORS, 8, 50, 4, "ip", 5),
        lambda: HnswIndex(dim=DIM, max_elements=N_VECTORS, M=8, nbits=4, random_seed=5),
    ],
)
def test_hnsw_max_elements_forms(make):
    index, message = single_deprecation(make)
    assert message.startswith("HnswIndex(dim, max_elements, ...) is deprecated")
    assert index.max_elements == N_VECTORS


def test_old_forms_keep_old_defaults():
    hnsw = quietly(lambda: HnswIndex(DIM, N_VECTORS))
    assert hnsw.nbits == 8
    assert hnsw.seed == 100
    assert hnsw.num_clusters == 16
    build = inspect.signature(IvfIndex.build).parameters
    assert build["num_threads"].default == 1
    assert inspect.signature(HnswIndex.build).parameters["num_threads"].default == 1


def test_random_seed_keyword():
    index, message = single_deprecation(lambda: HnswIndex(DIM, nbits=4, random_seed=7))
    assert message.startswith("HnswIndex(random_seed=...) is deprecated; use seed=")
    assert index.seed == 7
    with pytest.raises(
        TypeError, match="^HnswIndex\\(\\) got both seed and its old name random_seed$"
    ):
        HnswIndex(DIM, nbits=4, seed=1, random_seed=2)


@pytest.mark.parametrize(
    "make,message",
    [
        # Mixing forms: the old form has no seed, and the new one no max_elements.
        (
            lambda: IvfIndex(DIM, N_VECTORS, 4, nbits=4, seed=1),
            "IvfIndex() got an unexpected keyword argument 'seed' (read as the 0.7 form",
        ),
        (
            lambda: IvfIndex(DIM, 4, nbits=4, max_elements=N_VECTORS),
            "IvfIndex() multiple values for argument 'max_elements' (read as the 0.7 form",
        ),
        (
            lambda: HnswIndex(DIM, max_elements=10, num_clusters=4),
            "HnswIndex() got an unexpected keyword argument 'num_clusters' (read as the 0.7 form",
        ),
        (
            lambda: HnswIndex(DIM, 10, seed=4),
            "HnswIndex() got an unexpected keyword argument 'seed' (read as the 0.7 form",
        ),
    ],
)
def test_mixed_forms_raise_instead_of_guessing(make, message):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(TypeError) as error:
            make()
    assert str(error.value).startswith(message)


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
def test_build_and_is_built(kind, base_data, clusters):
    centroids, cluster_ids = clusters
    if kind == "ivf":
        index = quietly(lambda: IvfIndex(DIM, N_VECTORS, len(centroids), 4))
    else:
        index = quietly(lambda: HnswIndex(DIM, N_VECTORS, nbits=4))
    built, _ = single_deprecation(lambda: index.is_built)
    assert built is False
    _, message = single_deprecation(
        lambda: index.build(base_data, centroids, cluster_ids)
    )
    assert message.startswith(f"{type(index).__name__}.build() is deprecated")
    assert index.is_trained
    assert len(index) == N_VECTORS
    built, message = single_deprecation(lambda: index.is_built)
    assert built is True
    assert message.startswith(f"{type(index).__name__}.is_built is deprecated")


def test_build_on_new_form_indexes(base_data, clusters, tmp_path):
    centroids, cluster_ids = clusters
    ivf = IvfIndex(DIM, len(centroids), nbits=4, seed=3)
    single_deprecation(lambda: ivf.build(base_data, centroids, cluster_ids))
    reference = IvfIndex(DIM, len(centroids), nbits=4, seed=3)
    reference.set_centroids(centroids)
    reference.add(base_data, cluster_ids=cluster_ids, num_threads=1)
    ivf.save(tmp_path / "a.index")
    reference.save(tmp_path / "b.index")
    assert (tmp_path / "a.index").read_bytes() == (tmp_path / "b.index").read_bytes()

    # HNSW build takes any number of centroids, as in 0.7.
    hnsw = HnswIndex(DIM, nbits=4)
    assert hnsw.num_clusters == 16
    single_deprecation(lambda: hnsw.build(base_data, centroids, cluster_ids))
    assert hnsw.num_clusters == len(centroids)
    assert hnsw.max_elements == N_VECTORS


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
def test_positional_add_options(kind, base_data):
    index = (
        IvfIndex(DIM, 2, nbits=4)
        if kind == "ivf"
        else HnswIndex(DIM, nbits=4, num_clusters=2)
    )
    index.set_centroids(base_data[:2])
    labels = np.zeros(10, dtype=np.uint32)
    ids, message = single_deprecation(lambda: index.add(base_data[:10], labels, 1))
    assert message.startswith(
        f"passing cluster_ids, num_threads to {type(index).__name__}.add() by position is deprecated"
    )
    np.testing.assert_array_equal(ids, np.arange(10))
    with pytest.raises(
        TypeError, match="got multiple values for argument 'cluster_ids'"
    ):
        quietly(lambda: index.add(base_data[:10], labels, cluster_ids=labels))
    with pytest.raises(
        TypeError, match=r"takes at most 4 positional arguments \(5 given\)"
    ):
        index.add(base_data[:10], labels, 1, False, 3)
    assert len(index) == 10


def test_failed_build_leaves_a_new_form_index_unchanged(base_data, clusters):
    centroids, cluster_ids = clusters
    bad = base_data.copy()
    bad[7, 3] = np.nan
    ivf = IvfIndex(DIM, len(centroids), nbits=4)
    hnsw = HnswIndex(DIM, nbits=4)
    for index in (ivf, hnsw):
        with pytest.raises(ValueError, match="must contain only finite values"):
            quietly(lambda: index.build(bad, centroids, cluster_ids))
        assert not index.is_trained
        assert len(index) == 0
    assert hnsw.max_elements == 1
    bad_centroids = centroids.copy()
    bad_centroids[0, 0] = np.inf
    with pytest.raises(ValueError, match="must contain only finite values"):
        quietly(lambda: hnsw.build(base_data, bad_centroids, cluster_ids))
    assert hnsw.max_elements == 1


def test_hnsw_build_on_a_trained_index_replaces_centroids(
    base_data, clusters, tmp_path
):
    centroids, cluster_ids = clusters
    trained = HnswIndex(DIM, nbits=4, num_clusters=4)
    trained.train(base_data)
    single_deprecation(lambda: trained.build(base_data, centroids, cluster_ids))
    assert trained.num_clusters == len(centroids)
    # The same index the 0.7 form builds: both link with the exact vectors.
    built = quietly(lambda: HnswIndex(DIM, N_VECTORS, nbits=4))
    quietly(lambda: built.build(base_data, centroids, cluster_ids))
    trained.save(tmp_path / "a.index")
    built.save(tmp_path / "b.index")
    assert (tmp_path / "a.index").read_bytes() == (tmp_path / "b.index").read_bytes()


@pytest.mark.parametrize(
    "make,bits",
    [
        (lambda: SymqgIndex(DIM, 32), 0),
        (lambda: SymqgIndex(DIM, 32, "ip", 4), 4),
        (lambda: SymqgIndex(DIM, 32, quantization_bits=8), 8),
    ],
)
def test_symqg_positional_form(make, bits):
    index, message = single_deprecation(make)
    assert message.startswith(
        "SymqgIndex(dim, max_degree, metric, quantization_bits) is deprecated"
    )
    assert index.nbits == (32 if bits == 0 else bits)
    assert index.max_degree == 32
    assert (index.ef_construction, index.init) == (200, "pipnn")


@pytest.mark.parametrize("bits,nbits", [(0, 32), (4, 4), (8, 8)])
def test_symqg_quantization_bits_keyword(bits, nbits, base_data, tmp_path):
    index, message = single_deprecation(
        lambda: SymqgIndex(dim=DIM, max_degree=32, quantization_bits=bits, seed=4)
    )
    assert message.startswith("SymqgIndex(quantization_bits=...) is deprecated")
    reference = SymqgIndex(DIM, nbits=nbits, seed=4)
    for graph in (index, reference):
        graph.add(base_data, num_threads=1)
    index.save(tmp_path / "a.index")
    reference.save(tmp_path / "b.index")
    assert (tmp_path / "a.index").read_bytes() == (tmp_path / "b.index").read_bytes()
    value, message = single_deprecation(lambda: index.quantization_bits)
    assert value == bits
    assert message.startswith("SymqgIndex.quantization_bits is deprecated")


@pytest.mark.parametrize(
    "make,error,message",
    [
        (
            lambda: SymqgIndex(DIM, nbits=4, quantization_bits=4),
            TypeError,
            "^SymqgIndex\\(\\) got both nbits and its old form quantization_bits$",
        ),
        (
            lambda: SymqgIndex(DIM, quantization_bits=2),
            ValueError,
            "^quantization_bits must be 0, 4, or 8$",
        ),
        (
            lambda: SymqgIndex(DIM, 32, nbits=4),
            TypeError,
            "got an unexpected keyword argument 'nbits' \\(read as the 0.7 form",
        ),
        (
            lambda: SymqgIndex(DIM, 32, "l2", 3),
            ValueError,
            "^quantization_bits must be 0, 4, or 8$",
        ),
    ],
)
def test_symqg_mixed_or_invalid_old_forms(make, error, message):
    with pytest.raises(error, match=message):
        quietly(make)


def test_symqg_build_search_batch_and_is_built(base_data, query_data):
    index = quietly(lambda: SymqgIndex(DIM, 32))
    built, _ = single_deprecation(lambda: index.is_built)
    assert built is False
    _, message = single_deprecation(lambda: index.build(base_data, 50))
    assert message.startswith("SymqgIndex.build() is deprecated")
    assert len(index) == N_VECTORS
    built, _ = single_deprecation(lambda: index.is_built)
    assert built is True
    batch, message = single_deprecation(lambda: index.search_batch(query_data, 5, 40))
    assert message.startswith("SymqgIndex.search_batch() is deprecated; use search()")
    expected = index.search(query_data, 5, 40)
    np.testing.assert_array_equal(batch[0], expected[0])
    np.testing.assert_array_equal(batch[1], expected[1])
    # As in 0.7, build replaces the graph, even after add or another build.
    quietly(lambda: index.build(base_data[:100], 50, 1, "random"))
    assert len(index) == 100
    added = SymqgIndex(DIM)
    added.add(base_data[:80])
    quietly(lambda: added.build(base_data, 50))
    assert len(added) == N_VECTORS


def test_symqg_build_draws_a_new_seed_unless_one_was_given(base_data):
    unseeded = quietly(lambda: SymqgIndex(DIM, 32))
    seeds = set()
    for init in ("pipnn", "random"):
        quietly(lambda init=init: unseeded.build(base_data, 50, 1, init))
        seeds.add(unseeded.seed)
        # The settings of the current graph.
        assert (unseeded.ef_construction, unseeded.init) == (50, init)
    assert len(seeds) == 2
    seeded = SymqgIndex(DIM, seed=3)
    for _ in range(2):
        quietly(lambda: seeded.build(base_data, 50))
        assert seeded.seed == 3
