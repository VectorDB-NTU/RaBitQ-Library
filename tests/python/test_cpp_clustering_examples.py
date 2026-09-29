"""File-based C++ clustering examples interoperate with C++ index builders."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

BIN = Path(__file__).resolve().parents[2] / "bin"


def run_cpp(name, *args):
    executable = BIN / (name + (".exe" if os.name == "nt" else ""))
    if not executable.is_file():
        message = f"Build the C++ example target {name} first"
        if os.environ.get("RABITQ_REQUIRE_CPP_EXAMPLES") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return subprocess.run(
        [str(executable), *map(str, args)],
        env=dict(os.environ, OMP_THREAD_LIMIT="2", OMP_NUM_THREADS="2"),
        capture_output=True,
        text=True,
        timeout=60,
    )


def read_vecs(path, rows, cols, dtype):
    words = np.fromfile(path, dtype=np.uint32).reshape(rows, cols + 1)
    np.testing.assert_array_equal(words[:, 0], cols)
    return words[:, 1:].copy().view(dtype)


@pytest.mark.parametrize("method,k", [("qgkmeans", 33), ("rabitqkmeans", 16)])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_cpp_clustering_file_outputs(tmp_path, method, k, metric):
    data = np.random.default_rng(42).standard_normal((512, 65)).astype(np.float32)
    if metric == "ip":
        data /= np.linalg.norm(data, axis=1, keepdims=True)
    words = np.empty((len(data), data.shape[1] + 1), dtype=np.uint32)
    words[:, 0] = data.shape[1]
    words[:, 1:] = data.view(np.uint32)
    source = tmp_path / "data.fvecs"
    words.tofile(source)
    centroids_path = tmp_path / "centroids.fvecs"
    labels_path = tmp_path / "labels.ivecs"
    result = run_cpp(
        method,
        source,
        k,
        centroids_path,
        labels_path,
        *([metric] if metric == "ip" else []),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    centroids = read_vecs(centroids_path, k, data.shape[1], np.float32)
    labels = read_vecs(labels_path, len(data), 1, np.uint32).ravel()
    assert np.isfinite(centroids).all()
    original, centers = data.astype(np.float64), centroids.astype(np.float64)
    if metric == "ip":
        np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-6)
        distances = 1 - original @ centers.T
    else:
        distances = np.sum((original[:, None, :] - centers) ** 2, axis=2)
    np.testing.assert_array_equal(labels, distances.argmin(axis=1))
    np.testing.assert_array_equal(np.fromfile(source, dtype=np.uint32), words.ravel())

    index_path = tmp_path / "index.bin"
    if method == "qgkmeans":
        result = run_cpp(
            "ivf_rabitq_indexing",
            source,
            centroids_path,
            labels_path,
            3,
            index_path,
            metric,
        )
    else:
        result = run_cpp(
            "hnsw_rabitq_indexing",
            source,
            centroids_path,
            labels_path,
            16,
            32,
            5,
            index_path,
            metric,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert index_path.stat().st_size > 0


@pytest.mark.parametrize("method", ["qgkmeans", "rabitqkmeans"])
def test_cpp_clustering_synthetic_demo(method):
    result = run_cpp(method)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "final assignment count=" in result.stdout


@pytest.mark.parametrize("method", ["qgkmeans", "rabitqkmeans"])
@pytest.mark.parametrize("count", ["0", "-1", "16x", "184467440737095516160"])
def test_cpp_clustering_rejects_invalid_count(tmp_path, method, count):
    result = run_cpp(method, "unused.fvecs", count, tmp_path / "c", tmp_path / "ids")
    assert result.returncode == 1
    assert "num_clusters must be a positive integer" in result.stderr
    assert not (tmp_path / "c").exists()
    assert not (tmp_path / "ids").exists()


@pytest.mark.parametrize("method", ["qgkmeans", "rabitqkmeans"])
def test_cpp_clustering_usage_and_errors(tmp_path, method):
    result = run_cpp(method, "unused.fvecs")
    assert result.returncode == 1
    assert "Usage:" in result.stderr
    args = [tmp_path / "missing.fvecs", 33, tmp_path / "c", tmp_path / "ids"]
    result = run_cpp(method, *args, "cosine")
    assert result.returncode == 1
    assert "metric must be l2 or ip" in result.stderr
    result = run_cpp(method, *args)
    assert result.returncode == 1
    assert "File does not exist:" in result.stderr


@pytest.mark.parametrize("required", [False, True])
def test_missing_cpp_example_fails_when_required(tmp_path, monkeypatch, required):
    monkeypatch.setitem(run_cpp.__globals__, "BIN", tmp_path)
    monkeypatch.setenv("RABITQ_REQUIRE_CPP_EXAMPLES", "1" if required else "0")
    expected = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(
        expected, match=r"Build the C\+\+ example target qgkmeans first"
    ):
        run_cpp("qgkmeans")
