"""Compare flat float32 k-means with a 25-iteration limit and normal early stopping."""

import argparse
import csv
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from .build_indexes import DATASET_SHAPES
    from .build_indexes import DATASETS as ANN_DATASETS
    from .run import (
        cpu_placement,
        exact_neighbors,
        load_vectors,
        read_matrix,
    )
else:
    from build_indexes import DATASET_SHAPES
    from build_indexes import DATASETS as ANN_DATASETS
    from run import (
        cpu_placement,
        exact_neighbors,
        load_vectors,
        read_matrix,
    )

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "qgkmeans": "QGKMeans",
    "faiss": "Faiss K-means",
    "superkmeans": "SuperKMeans",
}
DATASETS = {
    # Base path, query path, base count, dimensions, clusters.
    name: (*paths[:2], *DATASET_SHAPES[name], 4096)
    for name, paths in ANN_DATASETS.items()
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_base(cfg):
    return np.memmap(
        cfg["base_file"], mode="r", dtype="<f4", shape=(cfg["n"], cfg["d"])
    )


def train_python(method, x, k, threads, seed):
    if method == "qgkmeans":
        from rabitqlib import FinalAssignmentMode, QGKMeans

        start = time.perf_counter()
        model = QGKMeans(
            x.shape[1],
            k,
            niter=25,
            num_threads=threads,
            seed=seed,
            spherical=False,
            quantization_bits=0,
            graph_degree=32,
            ef_build=240,
            ef_search=16,
            graph_build_iterations=1,
            early_stop_threshold=0.0,
            final_assignment=FinalAssignmentMode.Exact,
        )
        model.train(x)
        total = time.perf_counter() - start
        return (
            model.centroids,
            model.assignments,
            {
                "total_s": total,
                "training_s": None,
                "assignment_s": None,
                "actual_iterations": len(model.iteration_stats),
                "timing_boundary": "train includes exact final assignment",
            },
        )
    if method == "superkmeans":
        from superkmeans import SuperKMeans

        start = time.perf_counter()
        model = SuperKMeans(
            k,
            x.shape[1],
            hierarchical=False,
            quantizer="f32",
            iters=25,
            sampling_fraction=1.0,
            n_threads=threads,
            seed=seed,
            use_blas_only=False,
            early_termination=True,
            tol=1e-4,
            sample_queries=False,
            angular=False,
        )
        centroids = model.train(x)
        trained = time.perf_counter()
        labels = model.assign(x, centroids)
        assigned = time.perf_counter()
        return (
            centroids,
            labels,
            {
                "total_s": assigned - start,
                "training_s": trained - start,
                "assignment_s": assigned - trained,
                "actual_iterations": len(model.iteration_stats),
                "timing_boundary": "train plus exhaustive final assignment",
            },
        )
    if method != "faiss":
        raise ValueError(f"unknown method: {method}")
    import faiss

    faiss.omp_set_num_threads(threads)
    start = time.perf_counter()
    model = faiss.Kmeans(
        x.shape[1],
        k,
        niter=25,
        nredo=1,
        seed=seed,
        spherical=False,
        max_points_per_centroid=len(x),
        early_stop_threshold=0.0,
        gpu=False,
    )
    model.train(x)
    trained = time.perf_counter()
    # The Kmeans index contains the final centroids and performs exhaustive L2 search.
    labels = model.index.search(x, 1)[1][:, 0]
    assigned = time.perf_counter()
    return (
        model.centroids,
        labels,
        {
            "total_s": assigned - start,
            "training_s": trained - start,
            "assignment_s": assigned - trained,
            "actual_iterations": len(model.iteration_stats),
            "timing_boundary": "train plus exhaustive final assignment",
        },
    )


def quality_metrics(base, centroids, labels, queries, truth, threads):
    """Final WCSS and SuperKMeans-style cluster coverage of exact top-10 neighbors."""
    import faiss

    k, d = centroids.shape
    nprobe = max(1, int(0.01 * k))
    if labels.shape != (len(base),) or np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("invalid final assignments")
    if not np.isfinite(centroids).all():
        raise ValueError("nonfinite centroids")
    if (
        truth.shape != (len(queries), 10)
        or np.any(truth < 0)
        or np.any(truth >= len(base))
    ):
        raise ValueError("ground truth must contain ten valid neighbors per query")
    wcss = 0.0
    for start in range(0, len(base), 8192):
        stop = start + 8192
        residual = base[start:stop].astype(np.float64) - centroids[
            labels[start:stop]
        ].astype(np.float64)
        wcss += float(np.einsum("ij,ij->", residual, residual))
    # Independent direct-float64 audit, outside timing, including evenly spaced vectors.
    max_gap = 0.0
    c64 = centroids.astype(np.float64)
    for i in np.linspace(0, len(base) - 1, min(128, len(base)), dtype=int):
        residual = c64 - base[i].astype(np.float64)
        distances = np.einsum("ij,ij->i", residual, residual)
        best = float(distances.min())
        gap = float(distances[labels[i]]) - best
        max_gap = max(max_gap, gap)
        if gap > 1e-5 * max(1.0, best):
            raise ValueError(f"final assignment audit failed at vector {i}: gap={gap}")
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    hits = 0
    for start in range(0, len(queries), 256):
        distances, ids = index.search(queries[start : start + 256], k)
        # Sort all centroid distances to make equal-distance cutoff ties deterministic.
        order = np.lexsort((ids, distances), axis=1)[:, :nprobe]
        probes = np.take_along_axis(ids, order, axis=1)
        neighbor_labels = labels[truth[start : start + len(probes)]]
        hits += int(
            (neighbor_labels[:, :, None] == probes[:, None, :]).any(axis=2).sum()
        )
    return {
        "wcss": wcss,
        "recall_at_10": hits / (10 * len(queries)),
        "nprobe": nprobe,
        "assignment_audit_max_l2_gap": max_gap,
    }


def worker(stage, config_path):
    cfg = json.loads(config_path.read_text())
    cpu_placement(cfg["socket"])
    x = load_base(cfg)
    folder = Path(cfg["work_dir"])
    if stage == "_prepare":
        queries = np.load(cfg["query_file"])
        truth = exact_neighbors(x, queries, 10, "l2", cfg["threads"])
        np.save(cfg["truth_file"], truth)
    elif stage == "_train":
        centroids, labels, timing = train_python(
            cfg["method"], x, cfg["k"], cfg["threads"], cfg["seed"]
        )
        centroids.astype("<f4").tofile(folder / "centroids.f32")
        labels.astype("<u4").tofile(folder / "assignments.u32")
        write_json(folder / "timing.json", timing)
    elif stage == "_evaluate":
        centroids = np.fromfile(folder / "centroids.f32", dtype="<f4").reshape(
            cfg["k"], cfg["d"]
        )
        labels = np.fromfile(folder / "assignments.u32", dtype="<u4")
        quality = quality_metrics(
            x,
            centroids,
            labels,
            np.load(cfg["query_file"]),
            np.load(cfg["truth_file"]),
            cfg["threads"],
        )
        write_json(folder / "quality.json", quality)
    else:
        raise ValueError(f"unknown stage: {stage}")


def publish(output, docs=ROOT / "docs/docs/benchmarks", datasets=None, methods=None):
    datasets = list(DATASETS) if datasets is None else datasets
    methods = list(METHODS) if methods is None else methods
    path = docs / "kmeans.md"
    text = path.read_text()
    csv_path = docs / "figures/kmeans.csv"
    rows = {}
    if csv_path.exists():
        with csv_path.open(newline="") as previous:
            rows = {
                (row["dataset"], row["method"]): row for row in csv.DictReader(previous)
            }
    for dataset in datasets:
        for method in methods:
            result = output / dataset / f"{method}.json"
            record = json.loads(result.read_text())
            cfg = record["config"]
            n, d, clusters = DATASETS[dataset][2:]
            if (
                cfg["dataset"] != dataset
                or cfg["method"] != method
                or cfg["metric"] != "l2"
                or cfg["seed"] != 42
                or cfg["socket"] != 0
                or cfg["k"] != clusters
                or cfg["threads"] != 48
                or cfg["niter"] != 25
                or cfg["nq"] != 1000
                or cfg["n"] != n
                or cfg["d"] != d
                or record["quality"]["nprobe"] != int(0.01 * clusters)
            ):
                raise ValueError(f"unexpected benchmark configuration: {result}")
            timing, quality = record["timing"], record["quality"]
            if (
                not 1 <= timing["actual_iterations"] <= 25
                or not math.isfinite(timing["total_s"])
                or timing["total_s"] <= 0
                or not math.isfinite(quality["wcss"])
                or quality["wcss"] < 0
                or not 0 <= quality["recall_at_10"] <= 1
            ):
                raise ValueError(f"invalid k-means measurements: {result}")
            rows[dataset, method] = {
                "dataset": dataset,
                "method": method,
                "clusters": clusters,
                "threads": 48,
                "iteration_limit": 25,
                **timing,
                **quality,
            }
    for dataset in datasets:
        nprobe = int(0.01 * DATASETS[dataset][4])
        table = [
            f"| Method | Total time (s) | WCSS | Recall@10 ({nprobe} probes) | Iterations |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for method, label in METHODS.items():
            row = rows.get((dataset, method))
            if row is None:
                table.append(f"| {label} | Pending | — | — | — |")
            else:
                table.append(
                    f"| {label} | {float(row['total_s']):.2f} | {float(row['wcss']):.9g} | {float(row['recall_at_10']):.6f} | {row['actual_iterations']} |"
                )
        text = replace_block(text, f"kmeans:{dataset}", "\n".join(table))
    path.write_text(text)
    fields = [
        "dataset",
        "method",
        "clusters",
        "threads",
        "iteration_limit",
        "actual_iterations",
        "total_s",
        "training_s",
        "assignment_s",
        "wcss",
        "recall_at_10",
        "nprobe",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for dataset in DATASETS:
        for method in METHODS:
            if (dataset, method) in rows:
                writer.writerow(
                    {**rows[dataset, method], "clusters": DATASETS[dataset][4]}
                )
    csv_path.write_text(stream.getvalue())


def replace_block(text, name, body):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"missing or repeated result marker: {name}")
    before, rest = text.split(start)
    _, after = rest.split(end)
    return before + start + "\n" + body + "\n" + end + after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="export saved results without running experiments",
    )
    parser.add_argument(
        "--no-publish", action="store_true", help="save results without updating docs"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    args = parser.parse_args()
    output = (
        args.output_dir
        or Path(
            "results/benchmarks-smoke/kmeans"
            if args.smoke
            else "results/benchmarks/kmeans"
        )
    ).resolve()
    if args.export_only:
        if args.smoke:
            parser.error("smoke results cannot be published")
        publish(output, datasets=args.datasets, methods=args.methods)
        return
    output.mkdir(parents=True, exist_ok=True)
    placement = cpu_placement(None if args.smoke else 0)
    if not args.smoke:
        cores = {
            Path(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read_text().strip()
            for c in placement["allowed_cpus"]
        }
        if len(cores) != 24 or len(placement["allowed_cpus"]) != 48:
            raise RuntimeError(
                "benchmark requires 24 physical cores and 48 logical CPUs"
            )
    threads = 2 if args.smoke else 48
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "MKL_THREADING_LAYER": "GNU",
        # Let MKL resolve the same GNU OpenMP library as the Python environment.
        "LD_LIBRARY_PATH": os.pathsep.join(
            p
            for p in (str(Path(sys.prefix) / "lib"), os.environ.get("LD_LIBRARY_PATH"))
            if p
        ),
    }
    metadata = {
        "rabitqlib_version": importlib.metadata.version("rabitqlib"),
        "faiss_version": importlib.metadata.version("faiss-cpu"),
        "superkmeans_version": importlib.metadata.version("superkmeans"),
        "cpu_placement": placement,
        "platform": platform.platform(),
    }
    failures = []
    for dataset in ("smoke",) if args.smoke else args.datasets:
        pending = [
            m for m in args.methods if not (output / dataset / f"{m}.json").exists()
        ]
        if not pending:
            continue
        print(f"PREPARE {dataset}", flush=True)
        with tempfile.TemporaryDirectory(prefix="rabitq-kmeans-") as tmp:
            staging = Path(tmp)
            if args.smoke:
                rng = np.random.default_rng(42)
                base = rng.standard_normal((2048, 192)).astype("<f4")
                queries = rng.standard_normal((20, 192)).astype("<f4")
            else:
                base, _ = load_vectors(DATASETS[dataset][0])
                queries, _ = load_vectors(DATASETS[dataset][1])
                n, d, _ = DATASETS[dataset][2:]
                if base.shape != (n, d) or queries.shape != (1000, d):
                    raise ValueError(f"unexpected dataset shape: {dataset}")
            cfg = {
                "dataset": dataset,
                "n": len(base),
                "d": base.shape[1],
                "nq": len(queries),
                "k": 288 if args.smoke else DATASETS[dataset][4],
                "niter": 25,
                "seed": 42,
                "threads": threads,
                "socket": None if args.smoke else 0,
                "metric": "l2",
                "base_file": str(staging / "base.f32"),
                "query_file": str(staging / "queries.npy"),
                "truth_file": str(staging / "truth.npy"),
                "work_dir": str(staging),
            }
            base.tofile(cfg["base_file"])
            np.save(cfg["query_file"], queries)
            del base, queries
            write_json(staging / "config.json", cfg)
            if dataset == "gist1m":
                truth, _ = read_matrix("data/gist/gist_groundtruth.ivecs")
                np.save(cfg["truth_file"], truth[:, :10].astype(np.int64))
            else:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "_prepare",
                        str(staging / "config.json"),
                    ],
                    env=env,
                    check=True,
                )
            for method in pending:
                work = staging / method
                work.mkdir()
                cfg.update(method=method, work_dir=str(work))
                write_json(work / "config.json", cfg)
                result = output / dataset / f"{method}.json"
                result.parent.mkdir(parents=True, exist_ok=True)
                log = output / "logs" / dataset / f"{method}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"START {dataset}/{method}: k={cfg['k']}, niter=25, threads={threads}",
                    flush=True,
                )
                started = datetime.now(timezone.utc).isoformat()
                try:
                    with log.open("w") as stream:
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "_train",
                            str(work / "config.json"),
                        ]
                        subprocess.run(
                            command, env=env, stdout=stream, stderr=stream, check=True
                        )
                        timing = json.loads((work / "timing.json").read_text())
                        if not 1 <= timing["actual_iterations"] <= 25:
                            raise ValueError("unexpected training iteration count")
                        subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "_evaluate",
                                str(work / "config.json"),
                            ],
                            env=env,
                            stdout=stream,
                            stderr=stream,
                            check=True,
                        )
                    quality = json.loads((work / "quality.json").read_text())
                    # Retain compact outputs for later checks; the staged input is temporary.
                    for name in ("centroids.f32", "assignments.u32"):
                        shutil.copyfile(work / name, result.with_suffix("." + name))
                    write_json(
                        result,
                        {
                            "config": cfg.copy(),
                            "metadata": metadata,
                            "started_at_utc": started,
                            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "timing": timing,
                            "quality": quality,
                        },
                    )
                    print(
                        f"DONE {dataset}/{method}: {timing['total_s']:.2f}s, WCSS={quality['wcss']:.9g}, recall={quality['recall_at_10']:.6f}, iterations={timing['actual_iterations']}",
                        flush=True,
                    )
                except (subprocess.CalledProcessError, ValueError) as error:
                    failures.append(f"{dataset}/{method}: {error}; see {log}")
                    print(f"FAIL {failures[-1]}", flush=True)
    write_json(output / "failures.json", failures)
    if failures:
        raise SystemExit(f"{len(failures)} failed measurements")
    if not args.smoke and not args.no_publish:
        publish(output, datasets=args.datasets, methods=args.methods)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "--config-file",
                "docs/mkdocs.yml",
            ],
            cwd=ROOT,
            check=True,
        )
    print("All requested k-means measurements completed.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1].startswith("_"):
        worker(sys.argv[1], Path(sys.argv[2]))
    else:
        main()
