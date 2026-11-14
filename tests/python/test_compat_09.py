"""The 0.8 spellings that 0.9 renames: both work in 0.8.5, and the old one warns."""

import _thread
import inspect
import pickle
import sys
import time
import warnings

import numpy as np
import pytest
import rabitqlib
from conftest import DIM
from rabitqlib import HnswIndex, IvfIndex, QGKMeans, RaBitQKMeans, SymqgIndex


def single_deprecation(call, at=None):
    """Run call and return its result and its one DeprecationWarning.

    The warning must point at the code that used the old form: the lines of call,
    or of the helper function at when call reaches the old form through it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    assert [w.category for w in caught] == [DeprecationWarning], [
        str(w.message) for w in caught
    ]
    # Attributed to the caller, so users see their own line.
    assert caught[0].filename == __file__
    lines = [
        line for _, _, line in (at or call).__code__.co_lines() if line is not None
    ]
    assert min(lines) <= caught[0].lineno <= max(lines), (caught[0].lineno, lines)
    assert "python_migration" in str(caught[0].message)
    return result, str(caught[0].message)


def silently(call):
    """Run call, failing on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return call()


def assert_same(left, right):
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])


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


def test_routing_replaces_initializer(ivf):
    assert silently(lambda: ivf.routing) == "flat"
    value, message = single_deprecation(lambda: ivf.initializer)
    assert value == "flat"
    assert message.startswith("IvfIndex.initializer is deprecated; use routing")
    old, message = single_deprecation(lambda: IvfIndex(DIM, 8, initializer="hnsw"))
    assert message.startswith("IvfIndex(initializer=...) is deprecated; use routing=")
    assert silently(lambda: old.routing) == "hnsw"
    assert silently(lambda: IvfIndex(DIM, 8).routing) == "flat"  # 'auto' resolves
    with pytest.raises(
        TypeError,
        match="^IvfIndex\\(\\) got both routing and its old name initializer$",
    ):
        IvfIndex(DIM, 8, routing="flat", initializer="flat")
    with pytest.raises(ValueError, match="^routing must be 'auto', 'flat'"):
        IvfIndex(DIM, 8, routing="unknown")
    assert "routing" in inspect.signature(IvfIndex).parameters
    assert "initializer" not in inspect.signature(IvfIndex).parameters


@pytest.mark.parametrize("kind", ["ivf", "symqg"])
def test_max_elements_is_the_stored_count(kind, request):
    index = request.getfixturevalue(kind)
    value, message = single_deprecation(lambda: index.max_elements)
    assert value == index.num_stored_points
    assert message.startswith(
        f"{type(index).__name__}.max_elements is deprecated; use num_stored_points"
    )
    assert not hasattr(type(index), "capacity")


def test_hnsw_capacity_replaces_max_elements(hnsw):
    capacity = silently(lambda: hnsw.capacity)
    value, message = single_deprecation(lambda: hnsw.max_elements)
    assert value == capacity >= hnsw.num_stored_points
    assert message.startswith("HnswIndex.max_elements is deprecated; use capacity")


def test_hnsw_reserve_only_grows(base_data):
    index = HnswIndex(DIM, nbits=4, num_clusters=4, seed=3)
    silently(lambda: index.reserve(500))
    assert index.capacity == 500
    index.add(base_data, num_threads=1)
    for smaller in (0, len(base_data), 499, np.int64(500)):
        silently(lambda smaller=smaller: index.reserve(smaller))
        assert index.capacity == 500
    silently(lambda: index.reserve(600))
    assert index.capacity == 600
    with pytest.raises(ValueError, match="^num_points must be non-negative$"):
        index.reserve(-1)
    with pytest.raises(TypeError):
        index.reserve(1.5)
    # resize still shrinks, as in 0.8, and warns.
    _, message = single_deprecation(lambda: index.resize(len(base_data)))
    assert message.startswith("HnswIndex.resize() is deprecated; use reserve()")
    assert index.capacity == len(base_data)


def test_reserve_cluster_replaces_ivf_reserve(base_data):
    index = IvfIndex(DIM, 4, nbits=4, routing="flat", seed=3)
    index.train(base_data, num_threads=1)
    silently(lambda: index.reserve_cluster(0, 500))
    silently(lambda: index.reserve_cluster(cluster_id=1, num_points=np.int64(500)))
    _, message = single_deprecation(lambda: index.reserve(2, 500))
    assert message.startswith("IvfIndex.reserve() is deprecated; use reserve_cluster()")
    index.add(base_data, num_threads=1)
    assert len(index) == len(base_data)
    # Both names reach the extension: it rejects the same arguments for each.
    for reserve in (index.reserve_cluster, index.reserve):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="^cluster_id must be in"):
                reserve(4, 1)
            with pytest.raises(ValueError, match="^num_points must be non-negative$"):
                reserve(0, -1)
    untrained = IvfIndex(DIM, 4)
    with pytest.raises(
        RuntimeError,
        match="^IvfIndex must be trained or loaded before reserve_cluster$",
    ):
        untrained.reserve_cluster(0, 1)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(
            RuntimeError, match="^IvfIndex must be trained or loaded before reserve$"
        ),
    ):
        untrained.reserve(0, 1)


@pytest.mark.parametrize("kind", ["hnsw", "symqg"])
def test_ef_search_replaces_ef(kind, request, query_data):
    index = request.getfixturevalue(kind)
    name = type(index).__name__
    queries = query_data
    # A breadth of k finds other neighbors than the default of max(k, 64), so a
    # wrapper that dropped the argument would fail the comparisons below.
    expected = silently(lambda: index.search(queries, 3, ef_search=3))
    assert not np.array_equal(expected[0], index.search(queries, 3)[0])
    assert_same(silently(lambda: index.search(queries, 3, 3)), expected)
    assert_same(
        silently(lambda: index.search(queries, k=3, ef_search=3, num_threads=1)),
        expected,
    )
    actual, message = single_deprecation(lambda: index.search(queries, 3, ef=3))
    assert message.startswith(f"{name}.search(ef=...) is deprecated; use ef_search=")
    assert_same(actual, expected)
    actual, message = single_deprecation(lambda: index.search(queries, 3, 3, 1))
    assert message.startswith(
        f"passing num_threads to {name}.search() by position is deprecated"
    )
    assert_same(actual, expected)
    # Every spelling reaches the extension, which rejects a negative value.
    for call in (
        lambda: index.search(queries, 3, -1),
        lambda: index.search(queries, 3, ef_search=-1),
        lambda: index.search(queries, 3, ef=-1),
        lambda: index.search(queries, 3, num_threads=-1),
        lambda: index.search(queries, 3, 3, -1),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(TypeError):
                call()
    parameters = inspect.signature(type(index).search).parameters
    assert list(parameters) == ["self", "queries", "k", "ef_search", "num_threads"]
    assert parameters["num_threads"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "symqg"])
def test_malformed_search_calls_raise_before_any_warning(kind, request, query_data):
    index = request.getfixturevalue(kind)
    name = type(index).__name__
    queries = query_data
    breadth = "nprobe" if kind == "ivf" else "ef_search"
    calls = [
        (lambda: index.search(queries, 3, 4, **{breadth: 4}), "multiple values"),
        (lambda: index.search(queries, 3, 4, 1, 2, 3), "positional arguments"),
    ]
    if kind == "ivf":
        calls.append(
            (
                lambda: index.search(queries, 3, 4, False, high_accuracy=False),
                "got multiple values for argument 'high_accuracy'",
            )
        )
        calls.append(
            (
                lambda: index.search(queries, 3, 4, False, 1, num_threads=1),
                "got multiple values for argument 'num_threads'",
            )
        )
    else:
        calls.append(
            (
                lambda: index.search(queries, 3, 4, 1, num_threads=1),
                "got multiple values for argument 'num_threads'",
            )
        )
        calls.append(
            (
                lambda: index.search(queries, 3, ef=4, ef_search=4),
                "got both ef_search and its old name ef",
            )
        )
        calls.append(
            (
                lambda: index.search(queries, 3, 4, ef=4),
                "got both ef_search and its old name ef",
            )
        )
    for call, message in calls:
        # silently turns a warning into an error, so a TypeError here means the
        # call was rejected first.
        with pytest.raises(TypeError, match=f"^{name}\\.search\\(\\) .*{message}"):
            silently(call)


@pytest.mark.parametrize("kind", ["hnsw", "symqg"])
@pytest.mark.parametrize("zero", [0, np.int64(0), False, np.float32(0)])
def test_zero_breadth_selects_the_default_and_warns(kind, zero, request, base_data):
    index = request.getfixturevalue(kind)
    queries = base_data[:5]
    expected = silently(lambda: index.search(queries, 3))
    assert_same(silently(lambda: index.search(queries, 3, None)), expected)
    for call in (
        lambda: index.search(queries, 3, zero),
        lambda: index.search(queries, 3, ef_search=zero),
    ):
        actual, message = single_deprecation(call)
        assert message.startswith(
            f"{type(index).__name__}.search(ef_search=0) is deprecated; pass None"
        )
        assert_same(actual, expected)


def test_ivf_search_options_become_keyword_only(ivf, base_data):
    queries = base_data[:5]
    expected = silently(
        lambda: ivf.search(queries, 3, 4, high_accuracy=False, num_threads=1)
    )
    assert_same(
        silently(lambda: ivf.search(queries, 3, nprobe=4, high_accuracy=False)),
        expected,
    )
    actual, message = single_deprecation(lambda: ivf.search(queries, 3, 4, False))
    assert message.startswith(
        "passing high_accuracy to IvfIndex.search() by position is deprecated"
    )
    assert_same(actual, expected)
    actual, message = single_deprecation(lambda: ivf.search(queries, 3, 4, False, 1))
    assert message.startswith(
        "passing high_accuracy, num_threads to IvfIndex.search() by position"
    )
    assert_same(actual, expected)
    parameters = inspect.signature(IvfIndex.search).parameters
    assert [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ] == ["high_accuracy", "num_threads"]


@pytest.mark.parametrize("zero", [0, np.int64(0), False, np.float32(0)])
def test_zero_nprobe_selects_the_default_and_warns(ivf, zero, base_data):
    queries = base_data[:5]
    expected = silently(lambda: ivf.search(queries, 3))
    for call in (
        lambda: ivf.search(queries, 3, zero),
        lambda: ivf.search(queries, 3, nprobe=zero),
    ):
        actual, message = single_deprecation(call)
        assert message.startswith("IvfIndex.search(nprobe=0) is deprecated; pass None")
        assert_same(actual, expected)


def test_hnsw_seed_above_32_bits_warns():
    for seed in (0, 100, 2**32 - 1, np.uint32(7)):
        assert silently(lambda seed=seed: HnswIndex(DIM, seed=seed)).seed == seed
    index, message = single_deprecation(lambda: HnswIndex(DIM, seed=2**40 + 5))
    assert message.startswith("HnswIndex seeds above 2**32 - 1 are deprecated")
    assert index.seed == 2**40 + 5
    assert index.cp.seed == 5  # the low 32 bits, as before


def test_deprecated_search_batch_warns_once(symqg, query_data):
    expected = silently(lambda: symqg.search(query_data, 3, 3))
    assert not np.array_equal(expected[0], symqg.search(query_data, 3)[0])
    actual, message = single_deprecation(
        lambda: symqg.search_batch(query_data, 3, 3, 1)
    )
    assert message.startswith("SymqgIndex.search_batch() is deprecated; use search()")
    assert_same(actual, expected)


def test_unchanged_forms_stay_silent(ivf, hnsw, symqg, query_data):
    # 0.9 changes the HnswIndex default seed without a warning in this release.
    assert silently(lambda: HnswIndex(DIM)).seed == 100
    # num_threads=0 keeps meaning every available CPU.
    for index in (ivf, hnsw, symqg):
        silently(lambda index=index: index.search(query_data, 3, num_threads=0))


def test_old_spelling_keeps_its_error_message():
    with (
        pytest.warns(DeprecationWarning, match="use routing="),
        pytest.raises(ValueError, match="^initializer must be 'auto', 'flat'"),
    ):
        IvfIndex(DIM, 8, initializer="unknown")


def test_new_members_are_named_after_their_class(ivf, hnsw):
    assert HnswIndex.reserve.__qualname__ == "HnswIndex.reserve"
    assert IvfIndex.reserve_cluster.__qualname__ == "IvfIndex.reserve_cluster"
    for cls in (IvfIndex, HnswIndex, SymqgIndex):
        assert cls.search.__qualname__ == f"{cls.__name__}.search"
    with pytest.raises(
        TypeError, match="^HnswIndex\\.reserve\\(\\) missing 1 required"
    ):
        hnsw.reserve()
    with pytest.raises(AttributeError, match="'routing' of 'IvfIndex'"):
        ivf.routing = "flat"
    with pytest.raises(AttributeError, match="'capacity' of 'HnswIndex'"):
        hnsw.capacity = 1


def test_new_members_are_documented():
    for member in (
        IvfIndex.reserve_cluster,
        HnswIndex.reserve,
        IvfIndex.search,
        HnswIndex.search,
        SymqgIndex.search,
    ):
        assert member.__doc__.strip()
        assert "(self:" not in member.__doc__  # no stale native signature line
    for cls in (HnswIndex, SymqgIndex):
        assert "ef_search : int, optional" in cls.search.__doc__
        assert "\nef : int" not in cls.search.__doc__
    assert "routing : {'auto'" in IvfIndex.__doc__
    assert "initializer :" not in IvfIndex.__doc__
    assert "reserve can preallocate it" in HnswIndex.__doc__
    assert "reserve_cluster can preallocate room" in IvfIndex.add.__doc__
    assert "centroid initializer" not in IvfIndex.add.__doc__


# --- k-means: the names of issue #233 -----------------------------------------


def qg_kmeans(**parameters):
    return QGKMeans(DIM, 80, niter=2, num_threads=1, seed=1, **parameters)


def test_qgkmeans_takes_the_index_parameter_names(base_data):
    new = silently(lambda: qg_kmeans(nbits=4, max_degree=64, ef_construction=100))
    assert (new.cp.nbits, new.cp.max_degree, new.cp.ef_construction) == (4, 64, 100)
    for keyword, value, hint, current in (
        ("quantization_bits", 4, "use nbits=, with 32 instead of 0", "nbits"),
        ("graph_degree", 64, "use max_degree=", "max_degree"),
        ("ef_build", 100, "use ef_construction=", "ef_construction"),
    ):
        model, message = single_deprecation(
            lambda keyword=keyword, value=value: qg_kmeans(**{keyword: value}),
            at=qg_kmeans,
        )
        assert message.startswith(f"QGKMeans({keyword}=...) is deprecated; {hint}")
        # The old keyword reaches the same setting, which differs from its default.
        assert (
            getattr(model.cp, current)
            == value
            != getattr(QGKMeans(DIM, 80).cp, current)
        )
        with pytest.raises(
            TypeError,
            match=f"^QGKMeans\\(\\) got both {current} and its old name {keyword}$",
        ):
            silently(
                lambda keyword=keyword, value=value, current=current: qg_kmeans(
                    **{keyword: value, current: value}
                )
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = qg_kmeans(quantization_bits=4, graph_degree=64, ef_build=100)
    new.train(base_data)
    old.train(base_data)
    np.testing.assert_array_equal(new.centroids, old.centroids)
    np.testing.assert_array_equal(new.assignments, old.assignments)


def test_qgkmeans_nbits_uses_32_for_raw_vectors():
    assert silently(lambda: QGKMeans(DIM, 80).cp.nbits) == 32
    for nbits, stored in ((32, 0), (4, 4), (8, 8), (np.int64(32), 0)):
        model = silently(lambda nbits=nbits: qg_kmeans(nbits=nbits))
        assert model.cp.nbits == nbits
        with pytest.warns(DeprecationWarning, match="use nbits"):
            assert model.cp.quantization_bits == stored
    raw, _ = single_deprecation(lambda: qg_kmeans(quantization_bits=0), at=qg_kmeans)
    assert raw.cp.nbits == 32
    for invalid in (0, 16, 33, True):
        with pytest.raises(ValueError, match="^nbits must be 4, 8, or 32$"):
            qg_kmeans(nbits=invalid)
        with pytest.raises(ValueError, match="^nbits must be 4, 8, or 32$"):
            QGKMeans(DIM, 80).cp.nbits = invalid
    with pytest.raises(TypeError, match="could not convert parameter 'nbits'"):
        qg_kmeans(nbits="4")


def test_qgkmeans_parameter_attributes_are_renamed():
    cp = QGKMeans(DIM, 80).cp
    for old, new, value in (
        ("graph_degree", "max_degree", 64),
        ("ef_build", "ef_construction", 100),
        ("quantization_bits", "nbits", 8),
    ):
        silently(lambda new=new, value=value: setattr(cp, new, value))
        assert silently(lambda new=new: getattr(cp, new)) == value
        read, message = single_deprecation(lambda old=old: getattr(cp, old))
        assert read == value
        assert message.startswith(f"QGKMeansParameters.{old} is deprecated; use {new}")
        _, message = single_deprecation(
            lambda old=old: setattr(cp, old, 4 if new == "nbits" else 32)
        )
        assert message.startswith(f"QGKMeansParameters.{old} is deprecated; use {new}")
        assert getattr(cp, new) == (4 if new == "nbits" else 32)
    for name in ("nbits", "max_degree", "ef_construction", "graph_degree"):
        with pytest.raises(AttributeError, match="unexpected keyword argument"):
            RaBitQKMeans(DIM, 4, **{name: 32})


@pytest.mark.parametrize("cls", [QGKMeans, RaBitQKMeans])
def test_final_assignment_takes_a_string(cls, base_data):
    def trained(final_assignment):
        model = cls(
            DIM, 80, niter=2, num_threads=1, seed=1, final_assignment=final_assignment
        )
        model.train(base_data)
        return model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        modes = rabitqlib.FinalAssignmentMode
    for text, member in (("exact", modes.Exact), ("approximate", modes.Approximate)):
        by_text = silently(lambda text=text: trained(text))
        # A member is still accepted, silently: its name warned where it was read.
        by_member = silently(lambda member=member: trained(member))
        assert by_text.cp.final_assignment == member  # reads back as the member
        np.testing.assert_array_equal(by_text.assignments, by_member.assignments)
        np.testing.assert_array_equal(by_text.distances, by_member.distances)
        cp = cls(DIM, 80).cp
        silently(lambda cp=cp, text=text: setattr(cp, "final_assignment", text))
        assert cp.final_assignment == member
    for invalid in ("Exact", "EXACT", "", "symphonyqg"):
        with pytest.raises(
            ValueError, match="^final_assignment must be 'approximate' or 'exact'$"
        ):
            cls(DIM, 80, final_assignment=invalid)
    with pytest.raises(
        TypeError, match="could not convert parameter 'final_assignment'"
    ):
        cls(DIM, 80, final_assignment=1)


def test_index_training_parameters_take_the_string(base_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        modes = rabitqlib.FinalAssignmentMode
    for index in (IvfIndex(DIM, 8, seed=3), HnswIndex(DIM, num_clusters=8, seed=3)):
        for value, member in (
            ("exact", modes.Exact),
            ("approximate", modes.Approximate),
            (modes.Exact, modes.Exact),
            (modes.SymphonyQG, modes.Approximate),
        ):
            silently(
                lambda index=index, value=value: setattr(
                    index.cp, "final_assignment", value
                )
            )
            assert index.cp.final_assignment == member
        with pytest.raises(
            ValueError, match="^final_assignment must be 'approximate' or 'exact'$"
        ):
            index.cp.final_assignment = "Exact"
        with pytest.raises(
            TypeError,
            match="^final_assignment must be 'approximate' or 'exact', not int$",
        ):
            index.cp.final_assignment = 1
        assert index.cp.final_assignment == modes.Approximate  # unchanged by failures
        silently(lambda index=index: index.train(base_data, num_threads=1))
        assert index.is_trained


def _import_final_assignment_mode():
    from rabitqlib import FinalAssignmentMode

    return FinalAssignmentMode


def _import_qg_stats():
    from rabitqlib import QGKMeansIterationStats

    return QGKMeansIterationStats


def _import_rabitq_stats():
    from rabitqlib import RaBitQKMeansIterationStats

    return RaBitQKMeansIterationStats


@pytest.mark.parametrize(
    "name,importer,message",
    [
        (
            "FinalAssignmentMode",
            _import_final_assignment_mode,
            "FinalAssignmentMode is deprecated; pass 'approximate'",
        ),
        (
            "QGKMeansIterationStats",
            _import_qg_stats,
            "QGKMeansIterationStats is deprecated; use KMeansIterationStats",
        ),
        (
            "RaBitQKMeansIterationStats",
            _import_rabitq_stats,
            "RaBitQKMeansIterationStats is deprecated; use KMeansIterationStats",
        ),
    ],
)
def test_names_that_leave_the_package_warn_once(name, importer, message):
    value, text = single_deprecation(lambda: getattr(rabitqlib, name))
    assert text.startswith(message)
    imported, text = single_deprecation(importer, at=importer)
    assert text.startswith(message)
    assert imported is value
    assert name in rabitqlib.__all__
    # hasattr reads the name too.
    found, text = single_deprecation(lambda: hasattr(rabitqlib, name))
    assert found and text.startswith(message)


def test_old_names_can_be_read_without_a_python_caller():
    # A thread whose entry point is a builtin has no Python frame for the warning
    # to point at; the read must still succeed. list.extend and map are builtins,
    # so getattr runs in the new thread with no Python code calling it.
    read = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected = rabitqlib.FinalAssignmentMode
        _thread.start_new_thread(
            read.extend, (map(getattr, [rabitqlib], ["FinalAssignmentMode"]),)
        )
        # A failed read leaves the list empty, so this waits out the deadline.
        deadline = time.monotonic() + 5
        while not read and time.monotonic() < deadline:
            time.sleep(0.01)
    assert read == [expected]


def test_kmeans_iteration_stats_is_one_class(base_data):
    stats = silently(lambda: rabitqlib.KMeansIterationStats)
    assert stats.__name__ == stats.__qualname__ == "KMeansIterationStats"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert rabitqlib.QGKMeansIterationStats is stats
        assert rabitqlib.RaBitQKMeansIterationStats is stats
    for cls in (QGKMeans, RaBitQKMeans):
        model = cls(DIM, 80, niter=2, num_threads=1, seed=1)
        model.train(base_data)
        assert all(type(item) is stats for item in model.iteration_stats)
    with pytest.raises(
        AttributeError, match="^module 'rabitqlib' has no attribute 'missing'$"
    ):
        rabitqlib.missing


def test_star_import_still_provides_the_old_names():
    namespace = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec("from rabitqlib import *", namespace)
    assert set(rabitqlib.__all__) <= set(namespace)
    assert [w.category for w in caught] == [DeprecationWarning] * 3


def test_old_attribute_keeps_the_08_encoding():
    cp = QGKMeans(DIM, 80).cp
    cp.nbits = 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cp.quantization_bits = 0  # raw vectors in the 0.8 encoding
        assert cp.quantization_bits == 0
    assert cp.nbits == 32


def test_subclass_warnings_name_the_subclass():
    class Clustering(QGKMeans):
        pass

    def make():
        return Clustering(DIM, 80, graph_degree=64)

    model, message = single_deprecation(make, at=make)
    assert message.startswith(
        "Clustering(graph_degree=...) is deprecated; use max_degree="
    )
    assert model.cp.max_degree == 64


@pytest.mark.parametrize("cls", [QGKMeans, RaBitQKMeans])
@pytest.mark.parametrize("name", ["unknown", "mro", "__dict__"])
def test_unknown_kmeans_keyword_message_is_unchanged(cls, name):
    with pytest.raises(
        AttributeError,
        match=f"^{cls.__name__}\\(\\) got an unexpected keyword argument '{name}'$",
    ):
        cls(DIM, 80, **{name: 1})


def test_train_error_names_the_keyword_the_caller_used(base_data):
    old_message = "^QGKMeans ef_build and ef_search must be positive$"
    with pytest.raises(
        ValueError, match="^QGKMeans ef_construction and ef_search must be positive$"
    ):
        QGKMeans(DIM, 80, niter=1, ef_construction=0).train(base_data)
    # Every call that 0.8 accepts keeps the message 0.8 gave it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = QGKMeans(DIM, 80, niter=1, ef_build=0)
    with pytest.raises(ValueError, match=old_message):
        old.train(base_data)
    with pytest.raises(ValueError, match=old_message):
        QGKMeans(DIM, 80, niter=1, ef_search=0).train(base_data)

    class WithoutInit(QGKMeans):
        def __init__(self):
            self.d, self.k, self._impl = DIM, 80, None
            self.cp = rabitqlib.QGKMeansParameters()
            self.cp.ef_search = 0

    with pytest.raises(ValueError, match=old_message):
        WithoutInit().train(base_data)
    # Other errors pass through as the extension raised them.
    with pytest.raises(
        ValueError, match="^QGKMeans graph_build_iterations must be positive$"
    ) as raised:
        QGKMeans(DIM, 80, niter=1, ef_construction=100, graph_build_iterations=0).train(
            base_data
        )
    assert raised.value.__context__ is None  # not re-raised as a new object
    # The 0.8 name with a value only nbits would accept is still rejected at train.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wrong = QGKMeans(DIM, 80, niter=1, quantization_bits=32)
    with pytest.raises(ValueError, match="quantization_bits must be 0, 4, or 8"):
        wrong.train(base_data)


def test_statistics_class_pickles_by_reference():
    stats = rabitqlib.KMeansIterationStats
    assert pickle.loads(pickle.dumps(stats)) is stats
    assert stats.__module__ == "rabitqlib._rabitqlib"
    assert getattr(sys.modules[stats.__module__], "KMeansIterationStats") is stats
