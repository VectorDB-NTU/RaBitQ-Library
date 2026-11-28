"""Tests for the QGEMeans Python API."""

import numpy as np
import pytest
from conftest import DIM
from rabitqlib import QGEMeans, QGEMeansParameters

K = 33


def qgemeans(**kwargs):
    parameters = {
        "niter": 3,
        "num_threads": 1,
        "ef_construction": K,
        "ef_search": K,
    }
    parameters.update(kwargs)
    return QGEMeans(DIM, K, **parameters)


def nearest(x, centroids):
    distances = ((x[:, None, :].astype(np.float64) - centroids[None]) ** 2).sum(-1)
    return distances.argmin(1), distances.min(1)


def test_parameters_use_index_names_and_defaults():
    cp = QGEMeansParameters()
    assert cp.niter == 8
    assert cp.batch_size == 0
    assert not cp.verbose
    assert cp.seed == 42
    assert cp.num_threads == 0
    assert cp.final_assignment == "approximate"
    assert cp.max_degree == 32
    assert cp.ef_construction == 240
    assert cp.ef_search == 16
    assert cp.graph_build_iterations == 1
    for old in ("graph_degree", "ef_build", "epochs", "quantization_bits", "nbits"):
        assert not hasattr(cp, old)


def test_constructor_sets_keyword_parameters():
    model = qgemeans(batch_size=64, seed=7, verbose=True, final_assignment="exact")
    assert (model.d, model.k) == (DIM, K)
    assert model.cp.batch_size == 64
    assert model.cp.seed == 7
    assert model.cp.verbose
    assert model.cp.final_assignment == "exact"
    assert model.centroids is None
    assert model.obj is None
    assert model.iteration_stats is None


@pytest.mark.parametrize("name", ["epochs", "graph_degree", "ef_build", "nbits"])
def test_constructor_rejects_parameters_of_other_methods(name):
    with pytest.raises(
        AttributeError,
        match=f"QGEMeans\\(\\) got an unexpected keyword argument '{name}'",
    ):
        QGEMeans(DIM, K, **{name: 1})


def test_final_assignment_rejects_unknown_strings():
    with pytest.raises(
        ValueError, match="final_assignment must be 'approximate' or 'exact'"
    ):
        qgemeans(final_assignment="symphonyqg")


def test_train_populates_outputs_per_iteration(base_data):
    model = qgemeans()
    objective = model.train(base_data)

    assert model.centroids.shape == (K, DIM)
    assert model.centroids.dtype == np.float32
    assert model.assignments.shape == (base_data.shape[0],)
    assert model.assignments.dtype == np.uint32
    assert model.distances.shape == (base_data.shape[0],)
    assert model.obj.shape == (model.cp.niter,)
    stats = model.iteration_stats
    assert [s.iteration for s in stats] == [1, 2, 3]
    assert objective == model.obj[-1] == stats[-1].obj
    assert all(s.shift >= 0 for s in stats)
    # Cost-based revival moves centroids that cost little to remove.
    assert sum(s.nsplit for s in stats) > 0
    assert model.final_obj == pytest.approx(
        float(model.distances.sum(dtype=np.float64))
    )


def test_distances_are_exact_for_returned_assignments(base_data):
    model = qgemeans()
    model.train(base_data)
    selected = model.centroids[model.assignments]
    expected = ((base_data.astype(np.float64) - selected) ** 2).sum(1)
    np.testing.assert_allclose(model.distances, expected, rtol=1e-4, atol=1e-4)


def test_exact_final_assignment_returns_the_nearest_centroid(base_data):
    model = qgemeans(final_assignment="exact")
    model.train(base_data)
    ids, distances = nearest(base_data, model.centroids)
    np.testing.assert_array_equal(model.assignments, ids)
    np.testing.assert_allclose(model.distances, distances, rtol=1e-4, atol=1e-4)


def test_final_assignment_does_not_change_trained_centroids(base_data):
    approximate = qgemeans()
    approximate.train(base_data)
    exact = qgemeans(final_assignment="exact")
    exact.train(base_data)
    np.testing.assert_array_equal(approximate.centroids, exact.centroids)
    np.testing.assert_array_equal(approximate.obj, exact.obj)


@pytest.mark.parametrize("mode", ["approximate", "exact"])
def test_assign_reproduces_training_assignments(base_data, mode):
    model = qgemeans(final_assignment=mode)
    model.train(base_data)
    ids, distances = model.assign(base_data)
    np.testing.assert_array_equal(ids, model.assignments)
    np.testing.assert_array_equal(distances, model.distances)
    empty_ids, empty_distances = model.assign(np.empty((0, DIM), dtype=np.float32))
    assert empty_ids.shape == empty_distances.shape == (0,)


def test_assign_before_training_raises(base_data):
    with pytest.raises(RuntimeError, match="QGEMeans has not been trained"):
        qgemeans().assign(base_data)


def test_training_is_deterministic_across_thread_counts(base_data):
    single = qgemeans(num_threads=1)
    single.train(base_data)
    multi = qgemeans(num_threads=4)
    multi.train(base_data)
    np.testing.assert_array_equal(single.centroids, multi.centroids)
    np.testing.assert_array_equal(single.assignments, multi.assignments)
    np.testing.assert_array_equal(single.obj, multi.obj)


def test_parameters_can_be_changed_through_cp(base_data):
    model = qgemeans()
    model.cp.niter = 1
    model.cp.batch_size = 50
    model.train(base_data)
    assert model.obj.shape == (1,)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"niter": 0}, "QGEMeans d, k, and niter must be positive"),
        ({"max_degree": 48}, "QGEMeans max_degree must be a positive multiple of 32"),
        (
            {"ef_construction": 0},
            "QGEMeans ef_construction and ef_search must be positive",
        ),
        ({"ef_search": 1}, "QGEMeans ef_search must be at least 2"),
        (
            {"graph_build_iterations": 0},
            "QGEMeans graph_build_iterations must be positive",
        ),
    ],
)
def test_invalid_parameters_raise_with_python_names(base_data, parameters, message):
    model = qgemeans(**parameters)
    with pytest.raises(ValueError, match=f"^{message}$"):
        model.train(base_data)


def test_rejects_too_few_clusters_and_dimensions(base_data):
    with pytest.raises(
        ValueError, match="QGEMeans requires more centroids than max_degree"
    ):
        QGEMeans(DIM, 32).train(base_data)
    with pytest.raises(
        ValueError,
        match=r"QGEMeans currently requires dimensions in the range \[64, 65536\]",
    ):
        QGEMeans(32, K).train(base_data[:, :32])
    with pytest.raises(ValueError, match="QGEMeans requires n >= k"):
        qgemeans().train(base_data[: K - 1])


def test_failed_refit_keeps_previous_results(base_data):
    model = qgemeans()
    model.train(base_data)
    centroids = model.centroids.copy()
    bad = base_data.copy()
    bad[3, 5] = np.nan
    with pytest.raises(ValueError, match="QGEMeans x must contain only finite values"):
        model.train(bad)
    np.testing.assert_array_equal(model.centroids, centroids)


def test_train_rejects_wrong_dimension(base_data):
    with pytest.raises(ValueError, match="x dimension does not match d"):
        qgemeans().train(base_data[:, :-1])
