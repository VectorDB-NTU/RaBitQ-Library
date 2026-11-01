"""The 0.8 train / set_centroids / add / search workflow of IvfIndex and HnswIndex."""

import inspect
import warnings

import numpy as np
import pytest
from conftest import DIM, N_VECTORS, brute_force_knn, recall_at_k
from rabitqlib import HnswIndex, IvfIndex, RaBitQKMeans

KINDS = ["ivf", "hnsw"]


def new_index(kind, metric="l2", **kwargs):
    if kind == "ivf":
        kwargs.setdefault("seed", 7)
        return IvfIndex(
            DIM, kwargs.pop("num_clusters", 4), nbits=4, metric=metric, **kwargs
        )
    return HnswIndex(
        DIM,
        nbits=4,
        num_clusters=kwargs.pop("num_clusters", 4),
        metric=metric,
        **kwargs,
    )


def search(index, queries, k):
    if isinstance(index, IvfIndex):
        return index.search(queries, k, index.num_clusters)
    return index.search(queries, k, 200)


def saved_bytes(index, path):
    index.save(path)
    return path.read_bytes()


@pytest.fixture(autouse=True)
def no_deprecations():
    # The 0.8 API must never go through the compatibility layer.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        yield


def test_constructor_options_are_keyword_only():
    ivf = inspect.signature(IvfIndex.__init__).parameters
    hnsw = inspect.signature(HnswIndex.__init__).parameters
    keyword = inspect.Parameter.KEYWORD_ONLY
    assert [p for p in ivf if ivf[p].kind is not keyword] == [
        "self",
        "dim",
        "num_clusters",
    ]
    assert [p for p in hnsw if hnsw[p].kind is not keyword] == ["self", "dim"]
    assert ivf["nbits"].default is inspect.Parameter.empty
    assert hnsw["nbits"].default is inspect.Parameter.empty
    assert ivf["seed"].default is None
    assert hnsw["seed"].default == 100
    assert hnsw["num_clusters"].default == 16
    for cls in (IvfIndex, HnswIndex):
        add = inspect.signature(cls.add).parameters
        assert [p for p in add if add[p].kind is not keyword] == ["self", "data"]
        assert add["num_threads"].default == 0
        # train is native; pybind11 documents its signature on the first line.
        assert ", x: object, *, num_threads:" in cls.train.__doc__.splitlines()[0]
        assert cls.train.__doc__.splitlines()[0].endswith("= None) -> None")


@pytest.mark.parametrize(
    "make,message",
    [
        # Python 3.11 omits "keyword-only" from inspect's message; 3.12+ includes it.
        (
            lambda: IvfIndex(DIM, 4),
            r"^IvfIndex\(\) missing a required (keyword-only )?argument: 'nbits'$",
        ),
        (
            lambda: HnswIndex(DIM),
            r"^HnswIndex\(\) missing a required (keyword-only )?argument: 'nbits'$",
        ),
        (
            lambda: IvfIndex(DIM, 4, nbits=4, max_degree=8),
            r"^IvfIndex\(\) got an unexpected keyword argument 'max_degree'$",
        ),
    ],
)
def test_constructor_argument_errors(make, message):
    with pytest.raises(TypeError, match=message):
        make()


def test_positional_nbits_is_read_as_the_old_form_and_explained():
    with pytest.raises(
        TypeError, match=r"missing a required argument: 'nbits' \(read as the 0\.7 form"
    ):
        IvfIndex(DIM, 4, 4)


@pytest.mark.parametrize("cls", [IvfIndex, HnswIndex])
@pytest.mark.parametrize(
    "seed,error,message",
    [
        (-1, ValueError, "seed must be an integer in"),
        (True, TypeError, "seed must be an integer in"),
        (1.5, TypeError, "seed must be an integer in"),
    ],
)
def test_seed_validation(cls, seed, error, message):
    with pytest.raises(error, match=message):
        cls(DIM, 4, nbits=4, seed=seed) if cls is IvfIndex else cls(
            DIM, nbits=4, seed=seed
        )


def test_ivf_seed_range_and_random_default():
    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        IvfIndex(DIM, 4, nbits=4, seed=2**32)
    drawn = [IvfIndex(DIM, 4, nbits=4, seed=None).seed for _ in range(4)]
    assert all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in drawn)
    assert len(set(drawn)) > 1
    assert IvfIndex(DIM, 4, nbits=4, seed=np.uint32(9)).seed == 9


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_cp_defaults_follow_metric_and_seed(metric):
    ivf = IvfIndex(DIM, 4, nbits=4, metric=metric, seed=11)
    hnsw = HnswIndex(DIM, nbits=4, metric=metric, seed=2**40 + 5)
    for index, seed in ((ivf, 11), (hnsw, 5)):
        assert index.cp.niter == 25
        assert index.cp.max_points_per_centroid == 256
        assert index.cp.spherical is (metric == "ip")
        assert index.cp.seed == seed
        assert index.cp.num_threads == 0
    assert hnsw.seed == 2**40 + 5
    # cp is the index's own object: changes persist, and assignment copies.
    ivf.cp.niter = 3
    assert ivf.cp.niter == 3
    other = RaBitQKMeans(DIM, 2, niter=7).cp
    ivf.cp = other
    other.niter = 8
    assert ivf.cp.niter == 7


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_train_add_search(kind, metric, base_data, query_data):
    index = new_index(kind, metric)
    assert not index.is_trained and len(index) == 0
    index.train(base_data)
    assert index.is_trained and len(index) == 0
    ids = index.add(base_data)
    np.testing.assert_array_equal(ids, np.arange(N_VECTORS, dtype=np.uint32))
    assert len(index) == index.num_points == N_VECTORS
    if metric == "l2":
        exact, _ = brute_force_knn(base_data, query_data, 10)
        found, _ = search(index, query_data, 10)
        assert recall_at_k(found, exact, 10) > 0.8


@pytest.mark.parametrize("kind", KINDS)
def test_train_uses_cp_as_rabitqkmeans_does(kind, base_data, tmp_path):
    # set_centroids with the centroids RaBitQKMeans trains from cp must give the
    # same index as train, so train really clusters with cp.
    trained = new_index(kind)
    trained.cp.niter = 3
    trained.cp.num_threads = 1
    trained.train(base_data)
    trained.add(base_data, num_threads=1)

    model = RaBitQKMeans(
        DIM,
        4,
        niter=3,
        max_points_per_centroid=256,
        seed=trained.cp.seed,
        num_threads=1,
    )
    model.train(base_data)
    given = new_index(kind)
    given.set_centroids(model.centroids)
    given.add(base_data, num_threads=1)
    assert saved_bytes(trained, tmp_path / "a.index") == saved_bytes(
        given, tmp_path / "b.index"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_train_num_threads_overrides_cp_only_for_the_call(kind, base_data):
    index = new_index(kind)
    index.cp.num_threads = 3
    index.train(base_data, num_threads=1)
    assert index.cp.num_threads == 3


@pytest.mark.parametrize("kind", KINDS)
def test_retraining_is_allowed_only_while_empty(kind, base_data, tmp_path):
    index = new_index(kind)
    index.train(base_data)
    index.set_centroids(base_data[:4])
    index.train(base_data)
    index.add(base_data[:50])
    name = type(index).__name__
    message = f"^{name} cannot be trained once it stores points$"
    before = search(index, base_data[:5], 3)
    with pytest.raises(RuntimeError, match=message):
        index.train(base_data)
    with pytest.raises(RuntimeError, match=message):
        index.set_centroids(base_data[:4])
    after = search(index, base_data[:5], 3)
    np.testing.assert_array_equal(before[0], after[0])

    index.save(tmp_path / "trained.index")
    loaded = type(index).load(tmp_path / "trained.index")
    assert loaded.is_trained
    assert loaded.seed is None
    with pytest.raises(RuntimeError, match=message):
        loaded.set_centroids(base_data[:4])


@pytest.mark.parametrize("kind", KINDS)
def test_set_centroids_requires_num_clusters_by_dim(kind, base_data):
    index = new_index(kind)
    for bad in (base_data[:3], base_data[:4, :32], base_data[0]):
        with pytest.raises(ValueError) as error:
            index.set_centroids(bad)
        if bad.ndim == 2:
            assert str(error.value).startswith(
                "centroids must have shape (num_clusters, dim); expected (4, 64), got shape"
            )
    assert not index.is_trained


@pytest.mark.parametrize("kind", KINDS)
def test_train_rejects_too_few_rows_and_bad_values(kind, base_data):
    index = new_index(kind)
    with pytest.raises(
        ValueError,
        match=r"^x must have at least num_clusters rows; expected at least 4, got shape \(3, 64\)$",
    ):
        index.train(base_data[:3])
    bad = base_data.copy()
    bad[3, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        index.train(bad)
    assert not index.is_trained


@pytest.mark.parametrize("kind", KINDS)
def test_operations_before_training_raise_and_change_nothing(kind, base_data, tmp_path):
    index = new_index(kind)
    name = type(index).__name__
    operations = {
        "add": lambda: index.add(base_data),
        "search": lambda: search(index, base_data[:1], 1),
        "remove": lambda: index.remove([0]),
        "save": lambda: index.save(tmp_path / "untrained.index"),
    }
    if kind == "ivf":
        operations["reserve"] = lambda: index.reserve(0, 10)
    for operation, call in operations.items():
        with pytest.raises(
            RuntimeError, match=f"^{name} must be trained or loaded before {operation}$"
        ):
            call()
    assert not (tmp_path / "untrained.index").exists()
    assert not index.is_trained and len(index) == 0
    index.train(base_data)
    index.add(base_data)
    assert len(index) == N_VECTORS


@pytest.mark.parametrize("kind", KINDS)
def test_trained_index_without_points_cannot_be_saved(kind, base_data, tmp_path):
    index = new_index(kind)
    index.train(base_data)
    path = tmp_path / "empty.index"
    with pytest.raises(
        RuntimeError,
        match=f"^{type(index).__name__} has no points to save; add points first$",
    ):
        index.save(path)
    assert not path.exists()
    index.add(base_data)
    index.save(path)
    assert type(index).load(path).num_points == N_VECTORS


@pytest.mark.parametrize("kind", KINDS)
def test_search_k_is_bounded_by_stored_points(kind, base_data):
    # Not by HNSW capacity, which grows geometrically.
    index = new_index(kind)
    index.train(base_data)
    bound = "num_stored_points" if kind == "hnsw" else "max_elements"
    with pytest.raises(ValueError, match=f"^k must be between 1 and {bound}$"):
        search(index, base_data[:1], 1)
    index.add(base_data[:10])
    index.add(base_data[10:11])
    assert search(index, base_data[:1], 11)[0].shape == (1, 11)
    with pytest.raises(ValueError, match=f"^k must be between 1 and {bound}$"):
        search(index, base_data[:1], 12)


@pytest.mark.parametrize("kind", KINDS)
def test_len_counts_live_points(kind, base_data):
    index = new_index(kind)
    index.train(base_data)
    index.add(base_data)
    assert index.remove([0, 1, 1]) == 2
    assert len(index) == index.num_points == N_VECTORS - 2
    assert index.num_stored_points == N_VECTORS


def test_same_seed_reproduces_an_ivf_index(base_data, tmp_path):
    first = IvfIndex(DIM, 4, nbits=4)
    second = IvfIndex(DIM, 4, nbits=4, seed=first.seed)
    for index in (first, second):
        index.cp.num_threads = 1
        index.train(base_data)
        index.add(base_data, num_threads=1)
    assert saved_bytes(first, tmp_path / "a.index") == saved_bytes(
        second, tmp_path / "b.index"
    )


def test_ivf_point_count_index_cannot_be_trained(base_data):
    with pytest.warns(DeprecationWarning):
        index = IvfIndex(DIM, N_VECTORS, 4, 4)
    message = r"^IvfIndex created with max_elements is filled by build; create IvfIndex\(dim, num_clusters, nbits=\.\.\.\) to train or set centroids$"
    with pytest.raises(RuntimeError, match=message):
        index.train(base_data)
    with pytest.raises(RuntimeError, match=message):
        index.set_centroids(base_data[:4])
    assert index.max_elements == N_VECTORS


def test_hnsw_capacity_grows_and_resize_preallocates(base_data):
    index = HnswIndex(DIM, nbits=4, num_clusters=4)
    assert index.max_elements == 1
    index.train(base_data)
    index.add(base_data[:100])
    assert index.max_elements == 100
    index.add(base_data[100:110])
    assert index.max_elements == 200  # doubled
    preallocated = HnswIndex(DIM, nbits=4, num_clusters=4)
    preallocated.resize(1000)
    preallocated.train(base_data)
    preallocated.add(base_data)
    assert preallocated.max_elements == 1000


def test_hnsw_search_ef_none_means_the_default(base_data):
    index = HnswIndex(DIM, nbits=4, num_clusters=4)
    index.train(base_data)
    index.add(base_data, num_threads=1)
    expected = index.search(base_data[:10], 5, 0)
    for found in (
        index.search(base_data[:10], 5, None),
        index.search(base_data[:10], 5),
    ):
        np.testing.assert_array_equal(found[0], expected[0])
        np.testing.assert_array_equal(found[1], expected[1])


def test_hnsw_failed_add_keeps_the_capacity(base_data):
    index = HnswIndex(DIM, nbits=4, num_clusters=4)
    index.train(base_data)
    index.add(base_data[:100])
    bad = base_data.copy()
    bad[150, 0] = np.nan
    with pytest.raises(ValueError, match="must contain only finite values"):
        index.add(bad)
    assert index.max_elements == 100
    assert len(index) == 100


def test_hnsw_num_clusters_is_validated():
    with pytest.raises(ValueError, match=r"^num_clusters must be in \[1, "):
        HnswIndex(DIM, nbits=4, num_clusters=0)


def test_hnsw_first_add_matches_deprecated_build(base_data, clusters, tmp_path):
    # The first add into an empty index links like the 0.7 build, so the two
    # paths give the same bytes when single-threaded.
    centroids, cluster_ids = clusters
    added = HnswIndex(
        DIM, nbits=4, num_clusters=len(centroids), M=8, ef_construction=50
    )
    added.set_centroids(centroids)
    added.add(base_data, cluster_ids=cluster_ids, num_threads=1)
    with pytest.warns(DeprecationWarning):
        built = HnswIndex(DIM, N_VECTORS, M=8, ef_construction=50, nbits=4)
        built.build(base_data, centroids, cluster_ids)
    assert saved_bytes(added, tmp_path / "a.index") == saved_bytes(
        built, tmp_path / "b.index"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_add_without_cluster_ids_routes_to_trained_centroids(kind, base_data):
    index = new_index(kind)
    index.train(base_data)
    index.add(base_data[:250])
    index.add(base_data[250:])
    found, _ = search(index, base_data[:20], 1)
    assert np.mean(found[:, 0] == np.arange(20)) >= 0.9
