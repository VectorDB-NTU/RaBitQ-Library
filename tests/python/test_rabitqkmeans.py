"""Tests for flat RaBitQ k-means, independently of the cluster count."""

import numpy as np
import pytest
from rabitqlib import (
    FinalAssignmentMode,
    QGKMeans,
    QGKMeansParameters,
    RaBitQKMeans,
    RaBitQKMeansIterationStats,
    RaBitQKMeansParameters,
)


def test_flat_parameters_and_outputs():
    assert RaBitQKMeans is not QGKMeans
    assert RaBitQKMeansParameters is not QGKMeansParameters
    parameters = RaBitQKMeansParameters()
    assert parameters.niter == 25
    assert parameters.min_points_per_centroid == 39
    assert parameters.early_stop_threshold == 0.0
    assert parameters.final_assignment == FinalAssignmentMode.Approximate
    model = RaBitQKMeans(65, 33, niter=2, num_threads=1, seed=123)
    assert model.cp.seed == 123
    assert model.centroids is None
    assert model.obj is None
    model.cp.niter = 1
    data = np.random.default_rng(751).standard_normal((97, 65), dtype=np.float32)
    objective = model.train(data)
    assert model.centroids.shape == (33, 65)
    assert model.centroids.dtype == np.float32
    assert model.assignments.shape == (len(data),)
    assert model.assignments.dtype == np.uint32
    assert model.distances.shape == (len(data),)
    assert model.obj.shape == (1,)
    assert objective == model.obj[-1]
    assert model.final_obj == model.distances.sum(dtype=np.float64)
    assert isinstance(model.iteration_stats[0], RaBitQKMeansIterationStats)


@pytest.mark.parametrize(
    "parameter",
    [
        "graph_degree",
        "ef_build",
        "ef_search",
        "graph_build_iterations",
        "quantization_bits",
    ],
)
def test_flat_rejects_graph_only_parameters(parameter):
    with pytest.raises(AttributeError, match=parameter):
        RaBitQKMeans(65, 16, **{parameter: 0})


@pytest.mark.parametrize(
    "clusters,message", [(0, "must be positive"), (98, "requires n >= k")]
)
def test_flat_rejects_invalid_cluster_counts(clusters, message):
    model = RaBitQKMeans(65, clusters, niter=1, num_threads=1)
    with pytest.raises(ValueError, match=message):
        model.train(np.zeros((97, 65), dtype=np.float32))


def test_flat_native_centroids_before_training_raise():
    from rabitqlib._rabitqlib import _RaBitQKMeans

    model = _RaBitQKMeans(65, 16, RaBitQKMeansParameters())
    with pytest.raises(RuntimeError, match="^RaBitQKMeans has not been trained$"):
        _ = model.centroids


@pytest.mark.parametrize("clusters", [1, 2, 16, 31, 32, 33, 65])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_flat_train_refit_and_return_exact_selected_distances(
    clusters, spherical, exact
):
    rng = np.random.default_rng(573)
    x = rng.standard_normal((97, 65), dtype=np.float32)
    mode = FinalAssignmentMode.Exact if exact else FinalAssignmentMode.Approximate
    parameters = dict(
        niter=3,
        num_threads=2,
        spherical=spherical,
        min_points_per_centroid=1,
        final_assignment=mode,
    )
    model = RaBitQKMeans(65, clusters, **parameters)
    for count, seed in ((97, 42), (65, 77)):
        x += rng.standard_normal(x.shape, dtype=np.float32) * np.float32(0.125)
        fit = x[:count]
        original = x.copy()
        model.cp.seed = seed
        model.train(fit)
        fresh = RaBitQKMeans(65, clusters, seed=seed, **parameters)
        fresh.train(fit)
        np.testing.assert_array_equal(x, original)
        np.testing.assert_array_equal(model.centroids, fresh.centroids)
        np.testing.assert_array_equal(model.assignments, fresh.assignments)
        np.testing.assert_array_equal(model.obj, fresh.obj)
        centers = model.centroids.astype(np.float64)
        data = fit.astype(np.float64)
        if spherical:
            np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-6)
            distances = 1.0 - data @ centers.T
        else:
            delta = data[:, None, :] - centers[None, :, :]
            distances = np.einsum("nkd,nkd->nk", delta, delta)
        selected = distances[np.arange(len(fit)), model.assignments]
        np.testing.assert_allclose(model.distances, selected, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(
            model.final_obj, selected.sum(), rtol=1e-6, atol=1e-5
        )
        if exact:
            np.testing.assert_array_equal(
                model.assignments, np.argmin(distances, axis=1)
            )


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("clusters", [16, 65])
def test_flat_duplicates_and_zero(clusters, spherical):
    for value in (0.0, 1.0):
        x = np.zeros((97, 64), dtype=np.float32)
        x[:, 0] = value
        model = RaBitQKMeans(
            64,
            clusters,
            niter=3,
            num_threads=2,
            spherical=spherical,
            min_points_per_centroid=1,
            final_assignment=FinalAssignmentMode.Approximate,
        )
        model.train(x)
        np.testing.assert_array_equal(model.centroids, x[:clusters])
        np.testing.assert_array_equal(model.assignments, 0)
        expected = 1.0 - value if spherical else 0.0
        np.testing.assert_array_equal(model.distances, expected)


@pytest.mark.parametrize("dimension", [63, 65537])
def test_flat_keeps_dimension_limits(dimension):
    model = RaBitQKMeans(dimension, 2, niter=1, num_threads=1)
    with pytest.raises(ValueError, match="dimension"):
        model.train(np.zeros((2, dimension), dtype=np.float32))


def test_flat_refit_crosses_previous_cluster_threshold():
    data = np.random.default_rng(514).standard_normal((129, 65), dtype=np.float32)
    model = RaBitQKMeans(65, 32, niter=2, num_threads=1)
    for clusters in (32, 33, 16):
        model.k = clusters
        model.train(data)
        fresh = RaBitQKMeans(65, clusters, niter=2, num_threads=1)
        fresh.train(data)
        np.testing.assert_array_equal(model.centroids, fresh.centroids)
        np.testing.assert_array_equal(model.assignments, fresh.assignments)
        np.testing.assert_array_equal(model.distances, fresh.distances)
        np.testing.assert_array_equal(model.obj, fresh.obj)
