"""The call forms that 0.9 removed fail plainly; none is read as something else.

Covers the 0.7 forms (issue #213), the 0.8 index spellings (#232) and the 0.8
k-means spellings (#233).
"""

import pickle

import numpy as np
import pytest
import rabitqlib
from conftest import DIM, N_VECTORS
from rabitqlib import (
    HnswIndex,
    IvfIndex,
    QGKMeans,
    QGKMeansParameters,
    RaBitQKMeans,
    RaBitQKMeansParameters,
    SymqgIndex,
)


@pytest.fixture(scope="module")
def ivf(base_data):
    index = IvfIndex(DIM, 8, nbits=4, routing="flat", seed=3)
    index.add(base_data, num_threads=1)
    return index


@pytest.fixture(scope="module")
def hnsw(base_data):
    index = HnswIndex(DIM, nbits=4, num_clusters=4, seed=3)
    index.add(base_data, num_threads=1)
    return index


@pytest.fixture(scope="module")
def symqg(base_data):
    index = SymqgIndex(DIM, ef_construction=60, seed=3)
    index.add(base_data, num_threads=1)
    return index


@pytest.mark.parametrize(
    "make",
    [
        # IvfIndex(dim, max_elements, num_clusters, nbits, metric, initializer)
        lambda: IvfIndex(DIM, 1000, 64),
        lambda: IvfIndex(DIM, N_VECTORS, 4, 4),
        lambda: IvfIndex(DIM, N_VECTORS, 4, 4, "l2", "flat"),
        lambda: IvfIndex(dim=DIM, max_elements=N_VECTORS, num_clusters=4, nbits=4),
        lambda: IvfIndex(DIM, N_VECTORS, num_clusters=4, nbits=4),
        lambda: IvfIndex(DIM, 4, initializer="flat"),
        # HnswIndex(dim, max_elements, M, ef_construction, nbits, metric, random_seed)
        lambda: HnswIndex(DIM, N_VECTORS),
        lambda: HnswIndex(DIM, N_VECTORS, 16, 200, 4, "l2", 100),
        lambda: HnswIndex(DIM, max_elements=N_VECTORS),
        lambda: HnswIndex(DIM, random_seed=5),
        # SymqgIndex(dim, max_degree, metric, quantization_bits)
        lambda: SymqgIndex(DIM, 32),
        lambda: SymqgIndex(DIM, 32, "l2", 4),
        lambda: SymqgIndex(DIM, quantization_bits=0),
    ],
)
def test_removed_constructor_forms_raise_type_error(make):
    with pytest.raises(TypeError, match="incompatible constructor arguments") as error:
        make()
    # Only the public constructor is offered as the alternative.
    assert "\n    1. " in str(error.value)
    assert "\n    2. " not in str(error.value)


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
def test_add_options_are_keyword_only(kind, request, base_data):
    index = request.getfixturevalue(kind)
    cluster_ids = np.zeros(4, dtype=np.uint32)
    stored = index.num_stored_points
    for call in (
        lambda: index.add(base_data[:4], cluster_ids),
        lambda: index.add(base_data[:4], cluster_ids, 1),
        lambda: index.add(base_data[:4], cluster_ids, 1, False),
    ):
        with pytest.raises(TypeError, match="incompatible function arguments"):
            call()
    assert index.num_stored_points == stored


def test_ivf_search_keeps_only_nprobe_positional(ivf, query_data):
    expected = ivf.search(query_data, 3, 4, high_accuracy=False, num_threads=1)
    # Four probes find other neighbors than the default of one, so a dropped
    # argument would fail the comparison below.
    assert not np.array_equal(
        expected[0], ivf.search(query_data, 3, high_accuracy=False)[0]
    )
    for call in (
        lambda: ivf.search(query_data, 3, 4, False),
        lambda: ivf.search(query_data, 3, 4, False, 1),
    ):
        with pytest.raises(TypeError, match="incompatible function arguments"):
            call()
    actual = ivf.search(query_data, 3, nprobe=4, high_accuracy=False)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("kind", ["hnsw", "symqg"])
def test_graph_search_takes_ef_search_and_keyword_threads(kind, request, query_data):
    index = request.getfixturevalue(kind)
    # A breadth of k finds other neighbors than the default of max(k, 64), so a
    # dropped argument would fail the comparison below.
    expected = index.search(query_data, 3, ef_search=3)
    assert not np.array_equal(expected[0], index.search(query_data, 3)[0])
    for call in (
        lambda: index.search(query_data, 3, ef=3),
        lambda: index.search(query_data, 3, 3, 1),
        lambda: index.search(query_data, 3, 3, ef=3),
    ):
        with pytest.raises(TypeError, match="incompatible function arguments"):
            call()
    actual = index.search(query_data, 3, 3, num_threads=1)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("zero", [0, np.int64(0), False, np.float32(0)])
def test_zero_breadth_is_rejected(zero, ivf, hnsw, symqg, query_data):
    message = "^nprobe must be at least 1; pass None for the default$"
    for call in (
        lambda: ivf.search(query_data, 3, zero),
        lambda: ivf.search(query_data, 3, nprobe=zero),
    ):
        with pytest.raises(ValueError, match=message):
            call()
    message = "^ef_search must be at least 1; pass None for the default$"
    for index in (hnsw, symqg):
        for call in (
            lambda index=index: index.search(query_data, 3, zero),
            lambda index=index: index.search(query_data, 3, ef_search=zero),
        ):
            with pytest.raises(ValueError, match=message):
                call()


def test_none_and_zero_threads_are_still_accepted(ivf, hnsw, symqg, query_data):
    for index in (ivf, hnsw, symqg):
        expected = index.search(query_data, 3)
        for options in ({"num_threads": None}, {"num_threads": 0}):
            actual = index.search(query_data, 3, None, **options)
            np.testing.assert_array_equal(actual[0], expected[0])


REMOVED_MEMBERS = {
    "ivf": [
        "build",
        "_build",
        "is_built",
        "max_elements",
        "capacity",
        "initializer",
        "reserve",
    ],
    "hnsw": ["build", "_build", "is_built", "max_elements", "resize"],
    "symqg": [
        "build",
        "_build",
        "search_batch",
        "quantization_bits",
        "_quantization_bits",
        "is_built",
        "max_elements",
        "capacity",
    ],
}


@pytest.mark.parametrize(
    "kind,name",
    [(kind, name) for kind, names in REMOVED_MEMBERS.items() for name in names],
)
def test_removed_index_members_raise_attribute_error(kind, name, request):
    index = request.getfixturevalue(kind)
    assert not hasattr(type(index), name)
    with pytest.raises(AttributeError, match=f"has no attribute '{name}'"):
        getattr(index, name)
    # An assignment must not create the attribute either.
    with pytest.raises(AttributeError):
        setattr(index, name, 1)


def test_renamed_index_members(ivf, hnsw):
    assert ivf.routing == "flat"
    assert hnsw.capacity >= hnsw.num_stored_points == N_VECTORS
    for index, name in ((ivf, "routing"), (hnsw, "capacity")):
        with pytest.raises(AttributeError):
            setattr(index, name, getattr(index, name))
    assert f"capacity={hnsw.capacity}" in repr(hnsw)
    assert "routing='flat'" in repr(ivf)
    for index in (ivf, hnsw):
        assert "max_elements" not in repr(index)
        assert "initializer" not in repr(index)
    with pytest.raises(ValueError, match="^routing must be 'auto', 'flat'"):
        IvfIndex(DIM, 8, routing="unknown")
    untrained = IvfIndex(DIM, 4)
    with pytest.raises(
        RuntimeError,
        match="^IvfIndex must be trained or loaded before reserve_cluster$",
    ):
        untrained.reserve_cluster(0, 1)


@pytest.mark.parametrize("name", ["quantization_bits", "graph_degree", "ef_build"])
def test_removed_kmeans_parameters(name):
    with pytest.raises(
        AttributeError,
        match=f"^QGKMeans\\(\\) got an unexpected keyword argument '{name}'$",
    ):
        QGKMeans(DIM, 80, **{name: 32})
    cp = QGKMeansParameters()
    with pytest.raises(AttributeError):
        getattr(cp, name)
    with pytest.raises(AttributeError):
        setattr(cp, name, 32)


def test_qgkmeans_parameters_use_the_index_names(base_data):
    cp = QGKMeansParameters()
    assert (cp.nbits, cp.max_degree, cp.ef_construction) == (32, 32, 240)
    model = QGKMeans(DIM, 80, nbits=4, max_degree=64, ef_construction=100)
    assert (model.cp.nbits, model.cp.max_degree, model.cp.ef_construction) == (
        4,
        64,
        100,
    )
    for name in ("nbits", "max_degree", "ef_construction"):
        with pytest.raises(AttributeError, match="unexpected keyword argument"):
            RaBitQKMeans(DIM, 4, **{name: 32})
    # 0 was raw storage in 0.8; it is not read as 32.
    with pytest.raises(ValueError, match="^QGKMeans nbits must be 4, 8, or 32$"):
        QGKMeans(DIM, 80, niter=1, nbits=0).train(base_data)
    with pytest.raises(
        ValueError, match="^QGKMeans ef_construction and ef_search must be positive$"
    ):
        QGKMeans(DIM, 80, niter=1, ef_construction=0).train(base_data)


@pytest.mark.parametrize("cls", [QGKMeans, RaBitQKMeans])
def test_final_assignment_is_a_string(cls, base_data):
    assert cls(DIM, 80).cp.final_assignment == "approximate"
    for mode in ("exact", "approximate"):
        model = cls(DIM, 80, niter=2, num_threads=1, seed=1, final_assignment=mode)
        assert model.cp.final_assignment == mode
        model.train(base_data)
    cp = cls(DIM, 80).cp
    for invalid in ("Exact", "EXACT", "", "symphonyqg"):
        with pytest.raises(
            ValueError, match="^final_assignment must be 'approximate' or 'exact'$"
        ):
            cls(DIM, 80, final_assignment=invalid)
        with pytest.raises(
            ValueError, match="^final_assignment must be 'approximate' or 'exact'$"
        ):
            cp.final_assignment = invalid
    # The 0.8 enum members were integers underneath.
    with pytest.raises(
        TypeError, match="could not convert parameter 'final_assignment' from int"
    ):
        cls(DIM, 80, final_assignment=1)
    with pytest.raises(TypeError):
        cp.final_assignment = 1
    assert cp.final_assignment == "approximate"


def test_index_training_parameters_take_the_string(base_data):
    for index in (IvfIndex(DIM, 8, seed=3), HnswIndex(DIM, num_clusters=8, seed=3)):
        assert isinstance(index.cp, RaBitQKMeansParameters)
        index.cp.final_assignment = "exact"
        assert index.cp.final_assignment == "exact"
        with pytest.raises(ValueError, match="^final_assignment must be"):
            index.cp.final_assignment = "Exact"
        with pytest.raises(TypeError):
            index.cp.final_assignment = 1
        assert index.cp.final_assignment == "exact"  # unchanged by the failures
        index.train(base_data, num_threads=1)
        assert index.is_trained


@pytest.mark.parametrize(
    "name",
    ["FinalAssignmentMode", "QGKMeansIterationStats", "RaBitQKMeansIterationStats"],
)
def test_names_that_left_the_package(name):
    assert name not in rabitqlib.__all__
    assert not hasattr(rabitqlib, name)
    with pytest.raises(ImportError):
        exec(f"from rabitqlib import {name}")


def test_kmeans_iteration_stats_is_one_class(base_data):
    stats = rabitqlib.KMeansIterationStats
    assert stats.__name__ == stats.__qualname__ == "KMeansIterationStats"
    for cls in (QGKMeans, RaBitQKMeans):
        model = cls(DIM, 80, niter=2, num_threads=1, seed=1)
        model.train(base_data)
        assert len(model.iteration_stats) == 2
        assert all(type(item) is stats for item in model.iteration_stats)
    # Pickle and pydoc find the class by this name in its module.
    assert stats.__module__ == "rabitqlib._rabitqlib"
    assert pickle.loads(pickle.dumps(stats)) is stats
