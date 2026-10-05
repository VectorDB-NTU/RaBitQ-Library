"""Sweep a retained index with one search thread on one fixed query set."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

if __package__:
    from .build_indexes import DATASETS
    from .run import (
        load_index,
        load_vectors,
        read_matrix,
        recall_at_k,
        search,
        validate_results,
    )
else:
    from build_indexes import DATASETS
    from run import (
        load_index,
        load_vectors,
        read_matrix,
        recall_at_k,
        search,
        validate_results,
    )


def query_set(dataset, k):
    _, query_path, truth_path, _ = DATASETS[dataset]
    queries, _ = load_vectors(query_path)
    truth, _ = read_matrix(truth_path)
    if len(queries) != len(truth) or truth.shape[1] < k:
        raise ValueError("query and ground-truth shapes differ")
    return queries, truth[:, :k]


def settings(cfg):
    method = cfg["method"]
    k = cfg["k"]
    if method in ("rabitq-ivf", "faiss-ivf-rabitq-fs", "faiss-ivf-pqfs"):
        probes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        factors = (
            (1, 2, 5, 10, 20)
            if cfg["bits"] == 1 or method == "faiss-ivf-pqfs"
            else (1,)
        )
        if method == "rabitq-ivf":
            factors = (1,)
        return [
            {"nprobe": probe, "k_factor": factor}
            for probe in probes
            for factor in factors
        ]
    return [{"ef_search": ef} for ef in (16, 32, 64, 128, 256, 512, 1024) if ef >= k]


def configure(index, cfg, setting):
    selected = cfg | setting
    method = cfg["method"]
    if method.startswith("faiss-ivf-"):
        import faiss

        faiss.ParameterSpace().set_index_parameter(index, "nprobe", selected["nprobe"])
        if hasattr(index, "k_factor"):
            index.k_factor = selected["k_factor"]
    elif method.startswith("faiss-hnsw-"):
        index.hnsw.efSearch = selected["ef_search"]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_record", type=Path)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    initial = json.loads(args.initial_record.read_text())
    cfg = initial["config"]
    if cfg["socket"] != 0 or os.sched_getaffinity(0) != set(
        initial["search"]["cpu_placement"]["allowed_cpus"]
    ):
        parser.error("run the sweep with numactl --cpunodebind=0 --membind=0")
    if Path(cfg["base"]).resolve() != Path(DATASETS[args.dataset][0]).resolve():
        parser.error(
            "index uses a different dataset; rebuild for the current query protocol"
        )
    k = cfg["k"]
    queries, truth = query_set(args.dataset, k)
    if len(queries) != cfg["query_count"]:
        parser.error("query count differs from the index build record")
    index = load_index(cfg, Path(cfg["index_dir"]) / "index.bin")
    points = []
    for setting in settings(cfg):
        selected = configure(index, cfg, setting)
        for query in queries[:20]:
            search(index, selected, query.reshape(1, -1), 1)
        results = []
        started = time.perf_counter()
        for query in queries:
            results.append(search(index, selected, query.reshape(1, -1), 1))
        service_s = time.perf_counter() - started
        ids = np.concatenate([result[0] for result in results])
        distances = np.concatenate([result[1] for result in results])
        validate_results(ids, distances, cfg["base_count"], k)
        recall = recall_at_k(ids, truth)
        points.append(
            {
                "setting": setting,
                "recall": recall,
                "one_query_service_s": service_s,
            }
        )
        print(f"SWEEP {setting} recall={recall:.4f}", flush=True)

    result = {
        "status": "single_query_set",
        "dataset": args.dataset,
        "initial_record": str(args.initial_record),
        "query_count": len(queries),
        "query_points": points,
        "recall_mode": "one_query_per_call",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_THREADING_LAYER",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
