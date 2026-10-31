"""Benchmark protocol checks without external datasets or baseline packages."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Wheel tests may invoke pytest outside the checkout.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
build_indexes = importlib.import_module("benchmarks.build_indexes")
query_index = importlib.import_module("benchmarks.query_index")
reproduce = importlib.import_module("benchmarks.reproduce")


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    queries = np.arange(12, dtype="float32").reshape(3, 4)
    truth = np.array([[0], [1], [2]], dtype="int32")
    record = {
        "config": {
            "base": build_indexes.DATASETS["gist1m"][0],
            "socket": 0,
            "method": "rabitq-ivf",
            "bits": 5,
            "k": 1,
            "base_count": 8,
            "query_count": 3,
            "index_dir": str(tmp_path),
        },
        "search": {"cpu_placement": {"allowed_cpus": [0]}},
    }
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps(record))
    output = tmp_path / "sweep.json"
    monkeypatch.setattr(
        query_index.os, "sched_getaffinity", lambda _: {0}, raising=False
    )
    monkeypatch.setattr(query_index, "query_set", lambda *_: (queries, truth))
    monkeypatch.setattr(query_index, "load_index", lambda *_: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query_index.py",
            str(initial),
            "--dataset",
            "gist1m",
            "--output",
            str(output),
        ],
    )
    return initial, output


def test_recall_uses_timed_single_query_results(sweep, monkeypatch):
    _, output = sweep
    calls = []

    def search(index, config, queries, threads):
        # Batch and single-query paths deliberately return different neighbors.
        calls.append((len(queries), threads))
        ids = (
            (queries[:, :1] / 4).astype("int32")
            if len(queries) == 1
            else np.full((len(queries), 1), 7)
        )
        return ids, np.zeros_like(ids, dtype="float32")

    monkeypatch.setattr(query_index, "search", search)
    query_index.main()
    record = json.loads(output.read_text())
    assert set(calls) == {(1, 1)}
    assert record["recall_mode"] == "one_query_per_call"
    assert len(record["query_points"]) == 10
    assert all(point["recall"] == 1 for point in record["query_points"])
    assert all(point["one_query_service_s"] > 0 for point in record["query_points"])


def test_sweep_rejects_wrong_query_count_before_search(sweep, monkeypatch):
    initial, output = sweep
    record = json.loads(initial.read_text())
    record["config"]["query_count"] = 4
    initial.write_text(json.dumps(record))
    monkeypatch.setattr(
        query_index, "load_index", lambda *_: pytest.fail("loaded index")
    )
    with pytest.raises(SystemExit) as error:
        query_index.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "key,value",
    [("bits", 9), ("base_count", 980000), ("threads", 16), ("metric", "ip")],
)
def test_ann_record_rejects_wrong_settings(key, value):
    base, queries, truth, metric = build_indexes.DATASETS["gist1m"]
    cfg = dict(
        method="rabitq-ivf",
        metric=metric,
        base=str(ROOT / base),
        queries=str(ROOT / queries),
        groundtruth=str(ROOT / truth),
        base_count=1000000,
        query_count=1000,
        base_limit=None,
        query_limit=None,
        threads=48,
        socket=0,
        k=10,
        nlist=4096,
        opq=False,
        bits=5,
    )
    record = {"config": cfg, "data": {"dimension": 960}}
    build_indexes.validate_record(record, "gist1m", "rabitq-ivf-1p4")
    cfg[key] = value
    with pytest.raises(ValueError, match="unexpected ANN configuration"):
        build_indexes.validate_record(record, "gist1m", "rabitq-ivf-1p4")


@pytest.mark.parametrize("stage", ["all", "build", "query", "export"])
def test_partial_reproduction_selects_only_requested_work(monkeypatch, capsys, stage):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reproduce.py",
            "--datasets",
            "dbpedia",
            "--methods",
            "faiss-pqfs",
            "--stage",
            stage,
            "--dry-run",
        ],
    )
    reproduce.main()
    commands = capsys.readouterr().out
    assert "gist1m" not in commands
    assert "run_kmeans.py" not in commands
    assert ("download_datasets.py" in commands) == (stage != "export")
    assert ("build_indexes.py" in commands) == (stage in ("all", "build"))
    assert ("run_sweeps.py" in commands) == (stage in ("all", "query"))
    assert ("plot_results.py" in commands) == (stage in ("all", "export"))


def test_recall_rejects_missing_query_results():
    from benchmarks.run import recall_at_k

    with pytest.raises(ValueError, match="shapes differ"):
        recall_at_k(np.array([[0]]), np.array([[0], [1]]))


def test_kmeans_export_rejects_wrong_metric_before_writing(tmp_path):
    from benchmarks.run_kmeans import publish

    docs = tmp_path / "docs"
    docs.mkdir()
    page = docs / "kmeans.md"
    page.write_text("original content")
    results = tmp_path / "results/gist1m"
    results.mkdir(parents=True)
    record = {
        "config": {
            "dataset": "gist1m",
            "method": "faiss",
            "metric": "ip",
            "seed": 42,
            "socket": 0,
            "k": 4096,
            "threads": 48,
            "niter": 25,
            "nq": 1000,
            "n": 1000000,
            "d": 960,
        },
        "quality": {"nprobe": 40},
    }
    (results / "faiss.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unexpected benchmark configuration"):
        publish(tmp_path / "results", docs, datasets=["gist1m"], methods=["faiss"])
    assert page.read_text() == "original content"


def _write_sweep(directory, name, cpu):
    (directory / "gist1m").mkdir(parents=True, exist_ok=True)
    (directory / "sweeps" / "gist1m").mkdir(parents=True, exist_ok=True)
    record = {
        "config": {"query_count": 10},
        "build": {"cpu_placement": {"allowed_cpus": [0]}},
        "hardware": {"cpu": cpu},
    }
    (directory / "gist1m" / f"{name}.json").write_text(json.dumps(record))
    sweep = {
        "status": "single_query_set",
        "dataset": "gist1m",
        "query_count": 10,
        "query_points": [
            {"setting": {"nprobe": 1}, "recall": 0.9, "one_query_service_s": 0.01}
        ],
        "thread_environment": dict.fromkeys(
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), "1"
        ),
        "cpu_affinity": [0],
    }
    (directory / "sweeps" / "gist1m" / f"{name}.json").write_text(json.dumps(sweep))


@pytest.mark.parametrize("cpus", [("CPU A", "CPU A"), ("CPU A", "CPU B")])
def test_plot_subtitle_uses_the_recorded_cpu(tmp_path, monkeypatch, cpus):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    pytest.importorskip("matplotlib")
    plot_results = importlib.import_module("benchmarks.plot_results")
    monkeypatch.setattr(plot_results, "validate_record", lambda *_: None)
    monkeypatch.setattr(plot_results, "settings", lambda _: [{"nprobe": 1}])
    methods = ("rabitq-ivf-1p4", "faiss-rabitqfs-1p4")
    for name, cpu in zip(methods, cpus):
        _write_sweep(tmp_path / "in", name, cpu)
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_results.py",
            "--input-dir",
            str(tmp_path / "in"),
            "--output-dir",
            str(output),
            "--datasets",
            "gist1m",
            "--methods",
            *methods,
        ],
    )
    if cpus[0] != cpus[1]:
        with pytest.raises(ValueError, match="selected records span CPUs"):
            plot_results.main()
        assert not (output / "ivf-gist1m-qps-recall.svg").exists()
        return
    plot_results.main()
    assert "CPU: CPU A" in (output / "ivf-gist1m-qps-recall.svg").read_text()
