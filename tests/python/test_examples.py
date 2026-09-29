"""Clustering, indexing and querying examples run without importing Faiss."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

EXAMPLES = Path(__file__).resolve().parents[2] / "sample" / "python"

# Fail on an accidental import, even when a runtime combination happens to coexist.
BOOTSTRAP = """
import runpy
import sys
from pathlib import Path

blocked, script, *args = sys.argv[1:]
class BlockImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == blocked:
            raise AssertionError(f'{script} must not import {blocked}')
sys.meta_path.insert(0, BlockImport())
sys.path.insert(0, str(Path(script).parent))
sys.argv = [script, *args]
runpy.run_path(script, run_name='__main__')
"""


def run_example(name, *args, blocked="faiss"):
    env = os.environ.copy()
    # Reproduce Windows redirected output even on hosts using UTF-8 by default.
    env["PYTHONIOENCODING"] = "cp1252:strict"
    env["OMP_NUM_THREADS"] = "2"
    for key in (
        "DYLD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "KMP_DUPLICATE_LIB_OK",
    ):
        env.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            BOOTSTRAP,
            blocked,
            str(EXAMPLES / name),
            *map(str, args),
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="cp1252",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def write_vecs(path, values):
    words = np.empty((len(values), values.shape[1] + 1), dtype=np.int32)
    words[:, 0] = values.shape[1]
    words[:, 1:] = values.view(np.int32)
    words.tofile(path)


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "symqg"])
@pytest.mark.parametrize("elapsed", [0.0, 0.5])
def test_query_timing_handles_unchanged_clock(kind, elapsed, monkeypatch, capsys):
    name = "symqg_querying" if kind == "symqg" else f"{kind}_rabitq_querying"
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Wall-clock timestamps can be identical for a short search on Windows.
    monkeypatch.setattr(module, "time", lambda: 100.0, raising=False)
    ticks = iter([100.0, 100.0 + elapsed])
    monkeypatch.setattr(module, "perf_counter", lambda: next(ticks), raising=False)
    monkeypatch.setattr(module, "NPROBES" if kind == "ivf" else "EFS", [1])
    ids = np.array([[0], [1]], dtype=np.uint32)
    index = SimpleNamespace(
        dim=2, num_clusters=2, search=lambda *a, **kw: (ids, np.zeros((2, 1)))
    )
    index_name = {"ivf": "IvfIndex", "hnsw": "HnswIndex", "symqg": "SymqgIndex"}[kind]
    monkeypatch.setattr(module, index_name, SimpleNamespace(load=lambda path: index))
    monkeypatch.setattr(module, "read_fvecs", lambda path: np.zeros((2, 2)))
    monkeypatch.setattr(module, "read_ivecs", lambda path: ids)
    module.main(
        SimpleNamespace(
            index_file="index",
            query_file="queries",
            gt_file="truth",
            topk=1,
            test_rounds=1,
            num_threads=1,
            metric="l2",
            use_hacc=None,
        )
    )
    row = capsys.readouterr().out.splitlines()[-1].split()
    qps, recall = map(float, row[1:])
    if elapsed == 0:
        assert np.isnan(qps)
    else:
        assert qps == pytest.approx(4.0)
    assert recall == 1.0


@pytest.fixture
def example_data(tmp_path):
    data = np.random.default_rng(641).standard_normal((128, 65)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    path = tmp_path / "base.fvecs"
    write_vecs(path, data)
    return path, data


def check_saved_index_and_query(tmp_path, kind, metric, data):
    index_path = tmp_path / "test.index"
    index_type = {"ivf": IvfIndex, "hnsw": HnswIndex, "symqg": SymqgIndex}[kind]
    index = index_type.load(str(index_path))
    queries = data[:4].copy()
    search_args = {"nprobe": 2} if kind == "ivf" else {"ef": len(data)}
    ids, distances = index.search(queries, k=5, num_threads=2, **search_args)
    assert index.metric == metric
    np.testing.assert_array_equal(ids[:, 0], np.arange(len(queries)))
    assert np.isfinite(distances).all()
    restored_path = tmp_path / "restored.index"
    index.save(str(restored_path))
    actual = index_type.load(str(restored_path)).search(
        queries, k=5, num_threads=2, **search_args
    )
    np.testing.assert_array_equal(actual[0], ids)
    np.testing.assert_array_equal(actual[1], distances)

    query_path = tmp_path / "query.fvecs"
    truth_path = tmp_path / "truth.ivecs"
    exact = (
        np.sum((queries[:, None] - data) ** 2, axis=2)
        if metric == "l2"
        else 1 - queries @ data.T
    )
    write_vecs(query_path, queries)
    write_vecs(truth_path, np.argsort(exact, axis=1)[:, :5].astype(np.int32))
    name = "symqg_querying.py" if kind == "symqg" else f"{kind}_rabitq_querying.py"
    extra = ["--metric", metric] if kind == "symqg" else []
    output = run_example(
        name,
        index_path,
        query_path,
        truth_path,
        "--topk",
        5,
        "--test-rounds",
        1,
        "--num-threads",
        2,
        *extra,
    )
    assert "Recall" in output


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
@pytest.mark.parametrize("method,count", [("rabitq", 2), ("qg", 33)])
def test_separate_clustering_and_indexing(
    tmp_path, example_data, metric, kind, method, count
):
    data_path, data = example_data
    clusters = tmp_path / "clusters.npz"
    run_example(
        "kmeans_clustering.py",
        data_path,
        clusters,
        "--num-clusters",
        count,
        "--metric",
        metric,
        "--num-threads",
        2,
        *(["--method", method] if method == "rabitq" else []),
    )
    with np.load(clusters, allow_pickle=False) as saved:
        assert saved["centroids"].shape == (count, data.shape[1])
        assert saved["centroids"].dtype == np.float32
        assert saved["cluster_ids"].shape == (len(data),)
        assert saved["cluster_ids"].dtype == np.uint32
        assert (saved["cluster_ids"] < count).all()
        assert saved["metric"].item() == metric
        check_exact_cluster_labels(
            data, saved["centroids"], saved["cluster_ids"], metric
        )
    run_example(
        f"{kind}_rabitq_indexing.py",
        data_path,
        tmp_path / "test.index",
        "--clusters",
        clusters,
        "--metric",
        metric,
        "--total-bits",
        5,
        "--num-threads",
        2,
    )
    check_saved_index_and_query(tmp_path, kind, metric, data)


def check_exact_cluster_labels(data, centroids, labels, metric):
    original = data.astype(np.float64)
    centers = centroids.astype(np.float64)
    if metric == "ip":
        np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-6)
        distances = 1.0 - original @ centers.T
    else:
        delta = original[:, None, :] - centers[None, :, :]
        distances = np.einsum("nkd,nkd->nk", delta, delta)
    np.testing.assert_array_equal(labels, np.argmin(distances, axis=1))


@pytest.mark.parametrize("argument", [None, "l2", "ip", "innerproduct"])
def test_legacy_clustering_formats_without_faiss(tmp_path, example_data, argument):
    data_path, data = example_data
    centroids_path = tmp_path / "centroids.fvecs"
    labels_path = tmp_path / "labels.ivecs"
    run_example(
        EXAMPLES.parents[1] / "python" / "ivf.py",
        data_path,
        16,
        centroids_path,
        labels_path,
        *([] if argument is None else [argument]),
        "--method",
        "rabitq",
    )
    centroid_words = np.fromfile(centroids_path, dtype=np.int32).reshape(16, -1)
    np.testing.assert_array_equal(centroid_words[:, 0], data.shape[1])
    centroids = centroid_words[:, 1:].copy().view(np.float32)
    label_words = np.fromfile(labels_path, dtype=np.int32).reshape(len(data), 2)
    np.testing.assert_array_equal(label_words[:, 0], 1)
    labels = label_words[:, 1]
    assert np.isfinite(centroids).all()
    assert ((0 <= labels) & (labels < 16)).all()
    metric = "ip" if argument in ("ip", "innerproduct") else "l2"
    check_exact_cluster_labels(data, centroids, labels, metric)


@pytest.mark.parametrize(
    "clusters,dimension,message",
    [(0, 64, "num_clusters"), (129, 64, "num_clusters"), (2, 63, "dimension")],
)
def test_clustering_rejects_invalid_sizes(clusters, dimension, message, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(
        "kmeans_clustering", EXAMPLES / "kmeans_clustering.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = np.zeros((128, dimension), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        module.cluster_data(data, clusters, "l2", 1, method="rabitq")


def test_clustering_default_requires_graph_compatible_cluster_count(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(
        "kmeans_clustering", EXAMPLES / "kmeans_clustering.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="more centroids than the graph degree"):
        module.cluster_data(np.zeros((65, 64), dtype=np.float32), 2, "l2", 1)


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_symqg_examples_do_not_import_faiss(tmp_path, example_data, metric):
    data_path, data = example_data
    run_example(
        "symqg_indexing.py",
        data_path,
        tmp_path / "test.index",
        "--metric",
        metric,
        "--num-threads",
        2,
        "--ef-construction",
        len(data),
    )
    check_saved_index_and_query(tmp_path, "symqg", metric, data)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("metric", "ip", "clustering metric does not match index metric"),
        ("metric", ["l2"], "clustering metric does not match index metric"),
        ("centroids", np.zeros((2, 3), np.float32), "centroids must be float32"),
        ("centroids", np.zeros((2, 4), np.float64), "centroids must be float32"),
        (
            "centroids",
            np.full((2, 4), np.nan, np.float32),
            "centroids must contain only finite",
        ),
        ("cluster_ids", np.zeros(2, np.uint32), "one ID per data point"),
        ("cluster_ids", np.zeros(3, np.float32), "cluster_ids must be uint32"),
        ("cluster_ids", np.array([0, 1, 2], np.uint32), "out-of-range cluster ID"),
        ("metric", None, "must contain centroids, cluster_ids and metric"),
    ],
)
def test_reject_mismatched_clustering_file(tmp_path, field, value, message):
    spec = importlib.util.spec_from_file_location(
        "example_utils", EXAMPLES / "utils.py"
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    saved = {
        "centroids": np.zeros((2, 4), np.float32),
        "cluster_ids": np.array([0, 1, 0], np.uint32),
        "metric": "l2",
    }
    if value is None:
        del saved[field]
    else:
        saved[field] = value
    path = tmp_path / "clusters.npz"
    np.savez(path, **saved)
    with pytest.raises(ValueError, match=message):
        utils.load_clusters(path, (3, 4), "l2")
