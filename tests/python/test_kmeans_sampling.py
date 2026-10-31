"""Training-set sampling and assignment of new vectors for both k-means classes."""

import numpy as np
import pytest
from rabitqlib import (
    FinalAssignmentMode,
    IvfIndex,
    QGKMeans,
    QGKMeansParameters,
    RaBitQKMeans,
    RaBitQKMeansParameters,
)

DIM = 64
# QGKMeans requires more centroids than its default graph degree of 32.
CLUSTERS = {QGKMeans: 33, RaBitQKMeans: 4}
MODES = [FinalAssignmentMode.Approximate, FinalAssignmentMode.Exact]


def make_model(model_type, **kwargs):
    parameters = {"niter": 3, "num_threads": 1, "min_points_per_centroid": 1}
    parameters.update(kwargs)
    return model_type(DIM, CLUSTERS[model_type], **parameters)


def clustered(count, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((5, DIM)).astype(np.float32) * 4
    noise = rng.standard_normal((count, DIM), dtype=np.float32)
    return centers[np.arange(count) % 5] + noise


def selected_distances(x, centroids, assignments):
    residual = x.astype(np.float64) - centroids[assignments].astype(np.float64)
    return np.einsum("ij,ij->i", residual, residual)


@pytest.mark.parametrize(
    "parameters_type", [QGKMeansParameters, RaBitQKMeansParameters]
)
def test_max_points_per_centroid_disables_sampling_by_default(parameters_type):
    parameters = parameters_type()
    assert parameters.max_points_per_centroid == 0
    parameters.max_points_per_centroid = 256
    assert parameters.max_points_per_centroid == 256


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_max_points_per_centroid_keyword_and_cp(model_type):
    assert make_model(model_type).cp.max_points_per_centroid == 0
    model = make_model(model_type, max_points_per_centroid=np.int64(7))
    assert model.cp.max_points_per_centroid == 7
    model.cp.max_points_per_centroid = 0
    assert model.cp.max_points_per_centroid == 0


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
@pytest.mark.parametrize("value", [-1, 1.5, "many"])
def test_max_points_per_centroid_rejects_invalid_values(model_type, value):
    with pytest.raises(TypeError) as error:
        model_type(DIM, CLUSTERS[model_type], max_points_per_centroid=value)
    assert str(error.value) == (
        f"{model_type.__name__}() could not convert parameter "
        f"'max_points_per_centroid' from {type(value).__name__}"
    )
    model = make_model(model_type)
    with pytest.raises(TypeError):
        model.cp.max_points_per_centroid = value
    assert model.cp.max_points_per_centroid == 0


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_assign_before_training_raises(model_type):
    model = make_model(model_type)
    with pytest.raises(
        RuntimeError, match=f"^{model_type.__name__} has not been trained$"
    ):
        model.assign(np.zeros((1, DIM), dtype=np.float32))
    native = model._implementation_type(DIM, CLUSTERS[model_type], model.cp)
    with pytest.raises(
        RuntimeError, match=f"^{model_type.__name__} has not been trained$"
    ):
        native.assign(np.zeros((1, DIM), dtype=np.float32))


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_assign_validates_input_like_train(model_type):
    count = 2 * CLUSTERS[model_type]
    valid = clustered(count, 3)
    nonfinite = valid.copy()
    nonfinite[-1, -1] = np.nan
    out_of_range = valid.copy()
    out_of_range[0, 0] = 1e8
    invalid_inputs = [
        valid[:, :-1],
        valid[0],
        valid[None],
        nonfinite,
        out_of_range,
        "not numeric",
    ]
    model = make_model(model_type)
    model.train(valid)
    for invalid in invalid_inputs:
        with pytest.raises((TypeError, ValueError)) as from_train:
            make_model(model_type).train(invalid)
        with pytest.raises(type(from_train.value)) as from_assign:
            model.assign(invalid)
        assert str(from_assign.value) == str(from_train.value)


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
@pytest.mark.parametrize("mode", MODES)
def test_assign_returns_documented_types_for_new_vectors(model_type, mode):
    model = make_model(model_type, final_assignment=mode)
    model.train(clustered(3 * CLUSTERS[model_type], 5))
    queries = clustered(45, 7)
    # A float64, Fortran-ordered input exercises the converting copy.
    result = model.assign(np.asfortranarray(queries, dtype=np.float64))
    assert isinstance(result, tuple)
    assignments, distances = result
    assert assignments.shape == distances.shape == (45,)
    assert assignments.dtype == np.uint32
    assert distances.dtype == np.float32
    assert np.all(assignments < model.k)
    np.testing.assert_allclose(
        distances,
        selected_distances(queries, model.centroids, assignments),
        rtol=1e-5,
        atol=1e-5,
    )
    if mode == FinalAssignmentMode.Exact:
        all_distances = np.stack(
            [
                selected_distances(queries, model.centroids, np.full(45, cluster))
                for cluster in range(model.k)
            ],
            axis=1,
        )
        np.testing.assert_array_equal(assignments, np.argmin(all_distances, axis=1))
    empty_assignments, empty_distances = model.assign(np.empty((0, DIM), np.float32))
    assert empty_assignments.shape == empty_distances.shape == (0,)
    assert empty_assignments.dtype == np.uint32
    assert empty_distances.dtype == np.float32


# assign() uses no labels or search hints from training. It reproduces train's
# final labels in Exact mode, for RaBitQKMeans (same training center) and for
# any sampled fit, whose final pass is assign() itself. Unsampled approximate
# QGKMeans training reuses graph search hints.
@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("threads", [1, 4])
def test_assign_on_training_data_matches_final_assignment(
    model_type, mode, sampled, threads
):
    clusters = CLUSTERS[model_type]
    x = clustered(2 * clusters + int(sampled), 11)
    model = make_model(
        model_type,
        final_assignment=mode,
        max_points_per_centroid=2,
        num_threads=threads,
    )
    model.train(x)
    assignments, distances = model.assign(x)
    reproducible = (
        sampled or mode == FinalAssignmentMode.Exact or model_type is RaBitQKMeans
    )
    if reproducible:
        np.testing.assert_array_equal(assignments, model.assignments)
        np.testing.assert_array_equal(distances, model.distances)
    np.testing.assert_allclose(
        distances,
        selected_distances(x, model.centroids, assignments),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
@pytest.mark.parametrize("mode", MODES)
def test_assign_is_independent_of_the_batch(model_type, mode):
    model = make_model(model_type, final_assignment=mode, num_threads=4)
    model.train(clustered(6 * CLUSTERS[model_type], 29))
    queries = clustered(70, 31)
    assignments, distances = model.assign(queries)
    for size in (1, 33):
        parts = [model.assign(queries[i : i + size]) for i in range(0, 70, size)]
        np.testing.assert_array_equal(
            np.concatenate([p[0] for p in parts]), assignments
        )
        np.testing.assert_array_equal(np.concatenate([p[1] for p in parts]), distances)


# C++ tests check the verbose sampling message, which is not flushed per line.
@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_sampling_at_the_limit_matches_disabled_sampling(model_type):
    limit = 2 * CLUSTERS[model_type]
    x = clustered(limit, 13)
    at_limit = make_model(model_type, max_points_per_centroid=2)
    at_limit.train(x)
    disabled = make_model(model_type, max_points_per_centroid=0)
    disabled.train(x)
    for name in ("centroids", "assignments", "distances", "obj"):
        np.testing.assert_array_equal(getattr(at_limit, name), getattr(disabled, name))
    assert at_limit.final_obj == disabled.final_obj


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
@pytest.mark.parametrize("mode", MODES)
def test_sampled_fit_covers_all_points_and_is_seeded(model_type, mode):
    clusters = CLUSTERS[model_type]
    x = clustered(6 * clusters + 1, 17)
    model = make_model(model_type, final_assignment=mode, max_points_per_centroid=2)
    model.train(x)
    assert model.assignments.shape == model.distances.shape == (len(x),)
    assert model.final_obj == model.distances.sum(dtype=np.float64)
    np.testing.assert_allclose(
        model.distances,
        selected_distances(x, model.centroids, model.assignments),
        rtol=1e-5,
        atol=1e-5,
    )
    same_seed = make_model(model_type, final_assignment=mode, max_points_per_centroid=2)
    same_seed.train(x)
    for name in ("centroids", "assignments", "distances", "obj"):
        np.testing.assert_array_equal(getattr(model, name), getattr(same_seed, name))
    other_seed = make_model(
        model_type, final_assignment=mode, max_points_per_centroid=2, seed=43
    )
    other_seed.train(x)
    assert not np.array_equal(other_seed.centroids, model.centroids)


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_failed_refit_keeps_previous_model_for_assign(model_type):
    clusters = CLUSTERS[model_type]
    x = clustered(4 * clusters, 19)
    model = make_model(model_type, max_points_per_centroid=1)
    model.train(x)
    expected = model.assign(x)
    invalid = x.copy()
    invalid[-1, 0] = np.inf
    with pytest.raises(ValueError, match="x must contain only finite values$"):
        model.train(invalid)
    for actual, previous in zip(model.assign(x), expected):
        np.testing.assert_array_equal(actual, previous)


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_sampled_results_build_an_ivf_index(model_type):
    clusters = CLUSTERS[model_type]
    x = clustered(4 * clusters, 23)
    model = make_model(model_type, max_points_per_centroid=1)
    model.train(x)
    index = IvfIndex(DIM, len(x), clusters, 4)
    index.build(x, model.centroids, model.assignments)
    ids, _ = index.search(x[:3], 5, nprobe=clusters)
    for query, neighbors in enumerate(ids):
        assert query in neighbors
