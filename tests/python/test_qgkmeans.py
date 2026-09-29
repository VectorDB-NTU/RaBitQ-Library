"""Tests for the FAISS-style QGKMeans Python API."""

import numpy as np
import pytest
from conftest import DIM
from rabitqlib import FinalAssignmentMode, QGKMeans, QGKMeansParameters, RaBitQKMeans

K = 33


def qgkmeans(**kwargs):
    parameters = {
        "niter": 2,
        "num_threads": 1,
        "min_points_per_centroid": 1,
        "ef_build": K,
        "ef_search": K,
        "graph_build_iterations": 2,
    }
    parameters.update(kwargs)
    return QGKMeans(DIM, K, **parameters)


def test_parameters_use_faiss_names_and_defaults():
    cp = QGKMeansParameters()
    assert cp.niter == 25
    assert cp.graph_build_iterations == 1
    assert cp.min_points_per_centroid == 39
    assert cp.early_stop_threshold == 0.0
    assert cp.quantization_bits == 0


def test_constructor_accepts_faiss_style_keyword_parameters():
    clustering = qgkmeans(verbose=True, seed=123)
    assert clustering.d == DIM
    assert clustering.k == K
    assert clustering.cp.niter == 2
    assert clustering.cp.verbose
    assert clustering.cp.seed == 123
    assert clustering.centroids is None
    assert clustering.obj is None


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_unknown_parameter_raises(model_type):
    with pytest.raises(AttributeError):
        model_type(DIM, K, unknown_parameter=True)


def test_train_populates_faiss_style_outputs(base_data):
    clustering = qgkmeans()
    final_obj = clustering.train(base_data)

    assert clustering.centroids.shape == (K, DIM)
    assert clustering.centroids.dtype == np.float32
    assert clustering.assignments.shape == (base_data.shape[0],)
    assert clustering.distances.shape == (base_data.shape[0],)
    assert clustering.obj.shape == (clustering.cp.niter,)
    assert len(clustering.iteration_stats) == clustering.cp.niter
    assert final_obj == clustering.obj[-1]
    assert clustering.iteration_stats[-1].obj == clustering.obj[-1]


def test_parameters_can_be_changed_through_cp(base_data):
    clustering = qgkmeans()
    clustering.cp.niter = 1
    clustering.train(base_data)

    assert clustering.obj.shape == (1,)
    assert len(clustering.iteration_stats) == 1


@pytest.mark.parametrize("bits", [0, 8])
def test_exact_final_assignment_has_no_nearest_centroid_violations(base_data, bits):
    clustering = qgkmeans(
        final_assignment=FinalAssignmentMode.Exact, quantization_bits=bits
    )
    clustering.train(base_data)

    differences = (
        base_data[:, np.newaxis, :] - clustering.centroids[np.newaxis, :, :]
    ).astype(np.float64)
    exact_distances = np.einsum("nkd,nkd->nk", differences, differences)
    exact_assignments = np.argmin(exact_distances, axis=1)
    selected_distances = exact_distances[
        np.arange(base_data.shape[0]), exact_assignments
    ]

    np.testing.assert_array_equal(clustering.assignments, exact_assignments)
    np.testing.assert_allclose(clustering.distances, selected_distances, rtol=1e-6)


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("threads", [1, 4])
def test_exact_final_assignment_handles_matrix_tile_tails(spherical, threads):
    x = np.random.default_rng(341).standard_normal((321, 65)).astype(np.float32)
    if spherical:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    clustering = QGKMeans(
        65,
        257,
        niter=1,
        num_threads=threads,
        min_points_per_centroid=1,
        ef_build=40,
        ef_search=40,
        graph_build_iterations=1,
        spherical=spherical,
        final_assignment=FinalAssignmentMode.Exact,
    )
    clustering.train(x)

    original = x.astype(np.float64)
    centroids = clustering.centroids.astype(np.float64)
    if spherical:
        exact_distances = 1.0 - original @ centroids.T
    else:
        differences = original[:, None, :] - centroids[None, :, :]
        exact_distances = np.einsum("nkd,nkd->nk", differences, differences)
    exact_assignments = np.argmin(exact_distances, axis=1)
    selected = exact_distances[np.arange(x.shape[0]), exact_assignments]
    np.testing.assert_array_equal(clustering.assignments, exact_assignments)
    np.testing.assert_allclose(clustering.distances, selected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        clustering.final_obj, clustering.distances.sum(dtype=np.float64), rtol=1e-14
    )


def test_exact_final_assignment_preserves_shifted_l2_accuracy():
    x = np.random.default_rng(85).uniform(-0.125, 0.125, size=(129, 65))
    x = (x + 100000.0).astype(np.float32)
    clustering = QGKMeans(
        65,
        K,
        niter=1,
        num_threads=4,
        min_points_per_centroid=1,
        final_assignment=FinalAssignmentMode.Exact,
    )
    clustering.train(x)

    differences = (
        x.astype(np.float64)[:, None, :]
        - clustering.centroids.astype(np.float64)[None, :, :]
    )
    exact_distances = np.einsum("nkd,nkd->nk", differences, differences)
    exact_assignments = np.argmin(exact_distances, axis=1)
    selected = exact_distances[np.arange(x.shape[0]), exact_assignments]
    np.testing.assert_array_equal(clustering.assignments, exact_assignments)
    np.testing.assert_array_equal(clustering.distances, selected.astype(np.float32))


@pytest.mark.parametrize("spherical", [False, True])
def test_exact_final_assignment_uses_first_duplicate_centroid(spherical):
    x = np.zeros((65, 65), dtype=np.float32)
    x[:, 0] = 1.0
    clustering = QGKMeans(
        65,
        K,
        niter=1,
        num_threads=4,
        min_points_per_centroid=1,
        spherical=spherical,
        final_assignment=FinalAssignmentMode.Exact,
    )
    clustering.train(x)

    np.testing.assert_array_equal(clustering.centroids, x[:K])
    np.testing.assert_array_equal(clustering.assignments, 0)
    np.testing.assert_array_equal(clustering.distances, 0.0)
    assert clustering.final_obj == 0.0


def test_spherical_mode_returns_normalized_centroids(base_data):
    normalized = base_data / np.linalg.norm(base_data, axis=1, keepdims=True)
    clustering = qgkmeans(
        spherical=True,
        final_assignment=FinalAssignmentMode.Exact,
    )
    clustering.train(normalized)

    np.testing.assert_allclose(
        np.linalg.norm(clustering.centroids, axis=1),
        np.ones(K),
        rtol=1e-5,
        atol=1e-6,
    )
    cosine_distances = 1.0 - normalized @ clustering.centroids.T
    exact_assignments = np.argmin(cosine_distances, axis=1)
    selected_distances = cosine_distances[
        np.arange(normalized.shape[0]), exact_assignments
    ]
    np.testing.assert_array_equal(clustering.assignments, exact_assignments)
    np.testing.assert_allclose(clustering.distances, selected_distances, atol=1e-6)


def test_train_rejects_wrong_dimension():
    clustering = qgkmeans()
    with pytest.raises(Exception, match="dimension"):
        clustering.train(np.zeros((K, DIM + 1), dtype=np.float32))


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("spherical", [False, True])
def test_graph_returns_exact_selected_distances(bits, spherical):
    x = np.random.default_rng(42).standard_normal((100, 65)).astype(np.float32)
    if spherical:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    clustering = QGKMeans(
        65,
        K,
        quantization_bits=bits,
        spherical=spherical,
        niter=2,
        num_threads=2,
        min_points_per_centroid=1,
        ef_build=K,
        ef_search=K,
        graph_build_iterations=2,
    )
    clustering.train(x)
    assert len(clustering.iteration_stats) == 2
    assert np.isfinite(clustering.obj).all()
    assert np.isfinite(clustering.centroids).all()
    selected = clustering.centroids[clustering.assignments].astype(np.float64)
    original = x.astype(np.float64)
    if spherical:
        expected = 1.0 - np.einsum("ij,ij->i", original, selected)
        np.testing.assert_allclose(
            np.linalg.norm(clustering.centroids, axis=1), 1, atol=1e-6
        )
    else:
        delta = original - selected
        expected = np.einsum("ij,ij->i", delta, delta)
    np.testing.assert_allclose(clustering.distances, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bits", [1, 7, 9, 32])
def test_train_rejects_unsupported_quantization_bits(base_data, bits):
    clustering = qgkmeans(quantization_bits=bits)
    with pytest.raises(
        ValueError, match="^QGKMeans quantization_bits must be 0, 4, or 8$"
    ):
        clustering.train(base_data)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_nonfinite_training_input_preserves_previous_fit(
    base_data, invalid, model_type
):
    clustering = model_type(DIM, K, niter=2, num_threads=1)
    clustering.train(base_data)
    expected = clustering.centroids
    invalid_data = base_data.copy()
    invalid_data[-1, -1] = invalid
    with pytest.raises(
        ValueError, match=f"^{model_type.__name__} x must contain only finite values$"
    ):
        clustering.train(invalid_data)
    np.testing.assert_array_equal(clustering.centroids, expected)


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_failed_first_fit_keeps_results_unavailable(model_type):
    clustering = model_type(DIM, K, niter=2, num_threads=1)
    with pytest.raises(ValueError, match="^x dimension does not match d$"):
        clustering.train(np.zeros((K, DIM + 1), dtype=np.float32))
    assert clustering.centroids is None
    assert clustering.obj is None


@pytest.mark.parametrize("model_type", [QGKMeans, RaBitQKMeans])
def test_failed_refit_keeps_previous_results(base_data, model_type):
    clustering = model_type(DIM, K, niter=2, num_threads=1)
    clustering.train(base_data)
    expected_centroids = clustering.centroids
    expected_assignments = clustering.assignments
    with pytest.raises(ValueError, match="^x dimension does not match d$"):
        clustering.train(np.zeros((K, DIM + 1), dtype=np.float32))
    np.testing.assert_array_equal(clustering.centroids, expected_centroids)
    np.testing.assert_array_equal(clustering.assignments, expected_assignments)


def test_zero_input_converges_without_exhausting_iterations():
    clustering = qgkmeans(niter=10)
    clustering.train(np.zeros((100, DIM), dtype=np.float32))
    assert len(clustering.iteration_stats) < 10
    np.testing.assert_array_equal(clustering.distances, 0)


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_final_objective_measures_returned_assignments(base_data, bits):
    clustering = qgkmeans(niter=1, quantization_bits=bits)
    assert clustering.final_obj is None
    returned = clustering.train(base_data)
    assert returned == clustering.obj[-1]
    assert clustering.final_obj == clustering.distances.sum(dtype=np.float64)
    assert clustering.final_obj < returned


def test_native_centroids_before_training_raise():
    from rabitqlib._rabitqlib import _QGKMeans

    clustering = _QGKMeans(DIM, K, QGKMeansParameters())
    with pytest.raises(RuntimeError, match="^QGKMeans has not been trained$"):
        _ = clustering.centroids


@pytest.mark.parametrize("spherical", [False, True])
def test_empty_cluster_recovery_preserves_constant_data(spherical):
    x = np.zeros((100, DIM), dtype=np.float32)
    x[:, 0] = 1
    clustering = qgkmeans(niter=10, spherical=spherical)
    clustering.train(x)
    assert len(clustering.iteration_stats) < 10
    assert clustering.final_obj == 0
    np.testing.assert_array_equal(clustering.centroids, x[:K])


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("threads", [1, 4])
def test_single_empty_cluster_preserves_duplicate_centroids(bits, threads):
    # n == k includes every input row in initialization. Exactly one duplicate
    # centroid is empty, and its only eligible donor contains the identical pair.
    x = np.eye(DIM, dtype=np.float32)[:K].copy()
    x[1] = x[0]
    clustering = qgkmeans(
        niter=1,
        num_threads=threads,
        quantization_bits=bits,
        final_assignment=FinalAssignmentMode.Exact,
    )
    clustering.train(x)

    assert clustering.iteration_stats[0].nsplit == 1
    expected = sorted(tuple(row) for row in x)
    actual = sorted(tuple(row) for row in clustering.centroids)
    np.testing.assert_array_equal(actual, expected)
    assert clustering.final_obj == 0


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("spherical", [False, True])
def test_final_objective_does_not_regress_after_last_update(bits, spherical):
    x = np.random.default_rng(739).standard_normal((2048, 65)).astype(np.float32)
    if spherical:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    clustering = QGKMeans(
        65,
        256,
        niter=4,
        num_threads=2,
        min_points_per_centroid=1,
        quantization_bits=bits,
        spherical=spherical,
    )
    clustering.train(x)
    assert clustering.final_obj <= clustering.obj[-1] * (1 + 1e-6)
    assert clustering.final_obj == clustering.distances.sum(dtype=np.float64)


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("spherical", [False, True])
def test_pipnn_default_rebuilds_and_returns_exact_selected_distances(bits, spherical):
    x = np.random.default_rng(73).standard_normal((100, 65)).astype(np.float32)
    if spherical:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    model = QGKMeans(
        65,
        K,
        niter=3,
        num_threads=2,
        min_points_per_centroid=1,
        quantization_bits=bits,
        spherical=spherical,
    )
    for shift in (0.0, 0.02):
        data = x + np.float32(shift)
        if spherical:
            data /= np.linalg.norm(data, axis=1, keepdims=True)
        model.train(data)
        chosen = model.centroids[model.assignments].astype(np.float64)
        original = data.astype(np.float64)
        if spherical:
            expected = 1 - np.einsum("ij,ij->i", original, chosen)
        else:
            delta = original - chosen
            expected = np.einsum("ij,ij->i", delta, delta)
        np.testing.assert_allclose(model.distances, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(model.final_obj, expected.sum(), rtol=1e-6)


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_finite_out_of_range_input_preserves_previous_fit(base_data, bits):
    model = qgkmeans(quantization_bits=bits)
    model.train(base_data)
    expected = model.centroids
    with pytest.raises(
        ValueError, match="^QGKMeans x exceeds the safe float32 coordinate range$"
    ):
        model.train(np.full_like(base_data, 1e19))
    np.testing.assert_array_equal(model.centroids, expected)


@pytest.mark.parametrize("verbose", [False, True])
def test_training_size_warning_is_opt_in(capfd, verbose):
    model = qgkmeans(niter=1, min_points_per_centroid=39, verbose=verbose)
    model.train(np.zeros((K, DIM), dtype=np.float32))
    captured = capfd.readouterr()
    assert ("please provide at least" in captured.err) == verbose


@pytest.mark.parametrize("dim", [4096, 4097, 65535, 65536])
@pytest.mark.parametrize("bits", [0, 4, 8])
def test_large_dimensions(dim, bits):
    x = np.zeros((33, dim), dtype=np.float32)
    x[np.arange(33), np.arange(33)] = 1
    x[:, -1] = np.arange(33) / 64
    model = QGKMeans(
        dim,
        33,
        niter=1,
        num_threads=2,
        quantization_bits=bits,
    )
    model.train(x)
    assert model.centroids.shape == (33, dim)
    assert model.final_obj == 0
    np.testing.assert_array_equal(x, model.centroids[model.assignments])


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("threads", [1, 4])
def test_grouped_training_preserves_centroid_means_with_dimension_tail(bits, threads):
    x = np.repeat(np.eye(K, 65, dtype=np.float32) * 16, 3, axis=0)
    x[:, -1] = np.tile(np.array([-0.125, 0, 0.125], dtype=np.float32), K)
    model = QGKMeans(
        65,
        K,
        niter=8,
        num_threads=threads,
        min_points_per_centroid=1,
        quantization_bits=bits,
        ef_build=K,
        ef_search=K,
    )
    model.train(x)
    labels = model.assignments
    counts = np.bincount(labels, minlength=K)
    assert np.all(counts > 0)
    expected = np.array(
        [x[labels == cluster].mean(axis=0, dtype=np.float64) for cluster in range(K)],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(model.centroids, expected)


@pytest.mark.parametrize("bits", [0, 4, 8])
@pytest.mark.parametrize("spherical", [False, True])
def test_refit_recomputes_sums_for_mutated_storage_seed_and_count(bits, spherical):
    rng = np.random.default_rng(823)
    storage = np.zeros((257, 1024), dtype=np.float32)
    storage[:, :8] = rng.standard_normal((257, 8)).astype(np.float32)
    original_pointer = storage.ctypes.data

    def make_model(seed):
        return QGKMeans(
            1024,
            65,
            niter=3,
            num_threads=1,
            seed=seed,
            min_points_per_centroid=1,
            ef_build=128,
            ef_search=1,
            quantization_bits=bits,
            spherical=spherical,
        )

    model = make_model(42)
    model.train(storage[:193])
    for count, seed in [(193, 42), (193, 77), (257, 77), (129, 77)]:
        # Keep both the backing storage and view's starting pointer unchanged.
        storage *= np.float32(0.375)
        storage[:, :8] += rng.standard_normal((257, 8)).astype(np.float32)
        assert storage.ctypes.data == original_pointer
        data = storage[:count]
        assert data.ctypes.data == original_pointer
        before = data.copy()
        model.cp.seed = seed
        model.train(data)
        fresh = make_model(seed)
        fresh.train(data)
        np.testing.assert_array_equal(
            model.centroids.view(np.uint32), fresh.centroids.view(np.uint32)
        )
        np.testing.assert_array_equal(model.assignments, fresh.assignments)
        np.testing.assert_array_equal(
            model.distances.view(np.uint32), fresh.distances.view(np.uint32)
        )
        np.testing.assert_array_equal(
            model.obj.view(np.uint64), fresh.obj.view(np.uint64)
        )
        assert model.final_obj == fresh.final_obj
        np.testing.assert_array_equal(data, before)


def test_approximate_final_assignment_keeps_graph_mode_alias():
    assert FinalAssignmentMode.Approximate == FinalAssignmentMode.SymphonyQG
    assert QGKMeansParameters().final_assignment == FinalAssignmentMode.Approximate


@pytest.mark.parametrize(
    "clusters,degree",
    [(1, 32), (2, 32), (16, 32), (31, 32), (32, 32), (33, 64), (64, 64)],
)
@pytest.mark.parametrize(
    "mode", [FinalAssignmentMode.Approximate, FinalAssignmentMode.Exact]
)
def test_graph_rejects_clusters_at_or_below_graph_degree(clusters, degree, mode):
    model = QGKMeans(
        64, clusters, niter=1, num_threads=1, graph_degree=degree, final_assignment=mode
    )
    with pytest.raises(
        ValueError, match="^QGKMeans requires more centroids than the graph degree$"
    ):
        model.train(np.zeros((65, 64), dtype=np.float32))
