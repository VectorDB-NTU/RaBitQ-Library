"""Example stages keep Faiss and RaBitQ in different Python processes."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

EXAMPLES = Path(__file__).resolve().parents[2] / "sample" / "python"
HAS_FAISS = importlib.util.find_spec("faiss") is not None

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
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def write_vecs(path, values):
    words = np.empty((len(values), values.shape[1] + 1), dtype=np.int32)
    words[:, 0] = values.shape[1]
    words[:, 1:] = values.view(np.int32)
    words.tofile(path)


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


@pytest.mark.skipif(
    not HAS_FAISS, reason="Faiss is required only for clustering examples"
)
@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
def test_separate_clustering_and_indexing(tmp_path, example_data, metric, kind):
    data_path, data = example_data
    clusters = tmp_path / "clusters.npz"
    run_example(
        "faiss_clustering.py",
        data_path,
        clusters,
        "--num-clusters",
        2,
        "--metric",
        metric,
        "--num-threads",
        2,
        blocked="rabitqlib",
    )
    with np.load(clusters, allow_pickle=False) as saved:
        assert saved["centroids"].shape == (2, data.shape[1])
        assert saved["centroids"].dtype == np.float32
        assert saved["cluster_ids"].shape == (len(data),)
        assert saved["cluster_ids"].dtype == np.uint32
        assert (saved["cluster_ids"] < 2).all()
        assert saved["metric"].item() == metric
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
