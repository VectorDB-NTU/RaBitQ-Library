"""Build and check one CPU ANN configuration in isolated build/search processes."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

METHODS = (
    "rabitq-ivf",
    "rabitq-hnsw",
    "rabitq-symqg",
    "faiss-ivf-rabitq-fs",
    "faiss-ivf-pqfs",
    "faiss-hnsw-flat",
    "faiss-hnsw-sq4",
    "faiss-hnsw-sq8",
)


def import_search_library(method):
    if method.startswith("faiss-"):
        importlib.import_module("faiss")
    elif method.startswith("rabitq-"):
        importlib.import_module("rabitqlib")
    else:
        raise ValueError(f"unsupported benchmark method: {method}")


def read_matrix(path, limit=None):
    """Read a row-major fvecs/ivecs matrix, checking every selected row header."""
    path = Path(path)
    first = np.fromfile(path, dtype="<i4", count=1)
    if len(first) != 1 or first[0] <= 0:
        raise ValueError(f"invalid vector dimension in {path}")
    dim = int(first[0])
    row_bytes = (dim + 1) * 4
    size = path.stat().st_size
    if size % row_bytes:
        raise ValueError(f"incomplete vector row in {path}")
    rows = size // row_bytes
    count = rows if limit is None else min(rows, limit)
    if count == 0:
        raise ValueError(f"no vectors in {path}")
    matrix = np.memmap(path, dtype="<i4", mode="r", shape=(rows, dim + 1))
    if not np.all(matrix[:count, 0] == dim):
        raise ValueError(f"inconsistent vector dimensions in {path}")
    return np.ascontiguousarray(matrix[:count, 1:]).copy(), rows


def load_vectors(path, limit=None):
    values, rows = read_matrix(path, limit)
    return values.view("<f4"), rows


def exact_neighbors(base, queries, k, metric, threads):
    import faiss

    faiss.omp_set_num_threads(threads)
    index = (
        faiss.IndexFlatL2(base.shape[1])
        if metric == "l2"
        else faiss.IndexFlatIP(base.shape[1])
    )
    index.add(base)
    return index.search(queries, k)[1]


def build_index(cfg, base, path):
    method = cfg["method"]
    n, dim = base.shape
    started = time.perf_counter()
    training_s = 0.0
    if method.startswith("faiss-"):
        import faiss

        faiss.omp_set_num_threads(cfg["threads"])
        if method == "faiss-ivf-rabitq-fs":
            spec = f"HR{dim},IVF{cfg['nlist']},RaBitQfs{cfg['bits']}"
            if cfg["bits"] == 1:
                spec += ",RFlat"
        elif method == "faiss-ivf-pqfs":
            pq = f"PQ{dim // 2}x4fs"
            spec = f"IVF{cfg['nlist']},{pq},RFlat"
            if cfg["opq"]:
                spec = f"OPQ{dim // 2},{spec}"
        else:
            suffix = {
                "faiss-hnsw-flat": "",
                "faiss-hnsw-sq4": "_SQ4",
                "faiss-hnsw-sq8": "_SQ8",
            }[method]
            spec = f"HNSW{cfg['m']}{suffix}"
        metric = (
            faiss.METRIC_L2 if cfg["metric"] == "l2" else faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.index_factory(dim, spec, metric)
        if method.startswith("faiss-hnsw-"):
            index.hnsw.efConstruction = cfg["ef_construction"]
        stage_start = time.perf_counter()
        index.train(base)
        training_s = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        index.add(base)
        indexing_s = time.perf_counter() - stage_start
        if hasattr(index, "k_factor"):
            index.k_factor = cfg["k_factor"]
        details = {"faiss_factory": spec}
    elif method in ("rabitq-ivf", "rabitq-hnsw"):
        from rabitqlib import FinalAssignmentMode, IvfIndex, RaBitQKMeans

        cluster_count = cfg["nlist"] if method == "rabitq-ivf" else cfg["hnsw_clusters"]
        kmeans = RaBitQKMeans(
            dim,
            cluster_count,
            niter=10,
            num_threads=cfg["threads"],
            spherical=cfg["metric"] == "ip",
            seed=42,
            min_points_per_centroid=1,
            final_assignment=FinalAssignmentMode.Exact,
        )
        stage_start = time.perf_counter()
        kmeans.train(base)
        training_s = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        if method == "rabitq-ivf":
            index = IvfIndex(dim, n, cluster_count, cfg["bits"], cfg["metric"])
        else:
            from rabitqlib import HnswIndex

            index = HnswIndex(
                dim,
                n,
                cfg["m"],
                cfg["ef_construction"],
                cfg["bits"],
                cfg["metric"],
                random_seed=42,
            )
        index.build(
            base,
            kmeans.centroids,
            kmeans.assignments,
            num_threads=cfg["threads"],
        )
        indexing_s = time.perf_counter() - stage_start
        details = {
            "clustering": "RaBitQKMeans",
            "clustering_seed": 42,
            "cluster_count": cluster_count,
            "final_assignment": "exact",
        }
    elif method == "rabitq-symqg":
        from rabitqlib import SymqgIndex

        index = SymqgIndex(dim, cfg["degree"], cfg["metric"], cfg["quantization_bits"])
        stage_start = time.perf_counter()
        index.build(
            base,
            ef_construction=cfg["ef_construction"],
            num_threads=cfg["threads"],
            init="pipnn",
        )
        indexing_s = time.perf_counter() - stage_start
        details = {"initialization": "pipnn", "refinement_iterations": 1}
    else:
        raise ValueError(f"unsupported benchmark method: {method}")
    save_start = time.perf_counter()
    if method.startswith("faiss-"):
        faiss.write_index(index, str(path))
    else:
        index.save(str(path))
    finished = time.perf_counter()
    return {
        "training_s": training_s,
        "indexing_s": indexing_s,
        "build_s": save_start - started,
        "save_s": finished - save_start,
        "build_and_save_s": finished - started,
        **details,
    }


def load_index(cfg, path):
    method = cfg["method"]
    if method.startswith("faiss-"):
        import faiss

        index = faiss.read_index(str(path))
        if method.startswith("faiss-ivf-"):
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", cfg["nprobe"])
        else:
            index.hnsw.efSearch = cfg["ef_search"]
        if hasattr(index, "k_factor"):
            index.k_factor = cfg["k_factor"]
        return index
    if method == "rabitq-ivf":
        from rabitqlib import IvfIndex

        return IvfIndex.load(str(path))
    if method == "rabitq-hnsw":
        from rabitqlib import HnswIndex

        return HnswIndex.load(str(path))
    if method == "rabitq-symqg":
        from rabitqlib import SymqgIndex

        return SymqgIndex.load(str(path))
    raise ValueError(f"unsupported benchmark method: {method}")


def search(index, cfg, queries, threads):
    method = cfg["method"]
    k = cfg["k"]
    if method.startswith("faiss-"):
        import faiss

        faiss.omp_set_num_threads(threads)
        if method in ("faiss-ivf-rabitq-fs", "faiss-ivf-pqfs") and len(queries) > 100:
            batches = [
                index.search(queries[start : start + 100], k)
                for start in range(0, len(queries), 100)
            ]
            distances = np.concatenate([batch[0] for batch in batches])
            ids = np.concatenate([batch[1] for batch in batches])
        else:
            distances, ids = index.search(queries, k)
        return ids, distances
    if method == "rabitq-ivf":
        return index.search(queries, k=k, nprobe=cfg["nprobe"], num_threads=threads)
    if method in ("rabitq-hnsw", "rabitq-symqg"):
        return index.search(queries, k=k, ef=cfg["ef_search"], num_threads=threads)
    raise ValueError(f"unsupported benchmark method: {method}")


def validate_results(ids, distances, count, k):
    if ids.ndim != 2 or ids.shape[1] != k or distances.shape != ids.shape:
        raise ValueError("index returned an incorrect result shape")
    if np.any(ids < 0) or np.any(ids >= count):
        raise ValueError("index returned an invalid point ID")
    if any(len(set(row)) != k for row in ids.tolist()):
        raise ValueError("index returned duplicate point IDs")
    if not np.all(np.isfinite(distances)):
        raise ValueError("index returned a nonfinite distance")


def recall_at_k(ids, groundtruth):
    if (
        ids.ndim != 2
        or groundtruth.ndim != 2
        or len(ids) != len(groundtruth)
        or groundtruth.shape[1] < ids.shape[1]
    ):
        raise ValueError("result and ground-truth shapes differ")
    k = ids.shape[1]
    return float(
        np.mean(
            [len(set(row) & set(truth[:k])) / k for row, truth in zip(ids, groundtruth)]
        )
    )


def cpu_placement(socket):
    if socket is None:
        return None
    allowed = sorted(os.sched_getaffinity(0))
    sockets = sorted(
        {
            int(
                Path(
                    f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
                ).read_text()
            )
            for cpu in allowed
        }
    )
    if sockets != [socket]:
        raise RuntimeError(
            f"expected CPU socket {socket}, but process affinity spans sockets {sockets}"
        )
    return {"socket": socket, "allowed_cpus": allowed}


def worker(stage, directory):
    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text())
    placement = cpu_placement(cfg["socket"])
    if placement is not None and cfg["threads"] > len(placement["allowed_cpus"]):
        raise RuntimeError("thread count exceeds CPUs available on the selected socket")
    index_directory = (
        Path(cfg["index_dir"]) if cfg["index_dir"] else directory / "index"
    )
    path = index_directory / "index.bin"
    if stage == "_build":
        base = np.load(directory / "base.npy", mmap_mode="r")
        import_search_library(cfg["method"])
        details = build_index(cfg, base, path)
        details["cpu_placement"] = placement
        (directory / "build.json").write_text(json.dumps(details))
        return
    queries = np.load(directory / "queries.npy")
    groundtruth = np.load(directory / "groundtruth.npy")
    import_search_library(cfg["method"])
    load_start = time.perf_counter()
    index = load_index(cfg, path)
    load_s = time.perf_counter() - load_start
    search(index, cfg, queries[: min(len(queries), 20)], 1)
    ids, distances = search(index, cfg, queries, 1)
    validate_results(ids, distances, cfg["base_count"], cfg["k"])
    recall = recall_at_k(ids, groundtruth)
    (directory / "search.json").write_text(
        json.dumps(
            {
                "recall_at_k": recall,
                "load_s": load_s,
                "cpu_placement": placement,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base vectors in fvecs format")
    parser.add_argument(
        "--queries", required=True, help="query vectors in fvecs format"
    )
    parser.add_argument(
        "--groundtruth", help="ivecs file; only valid with the full base"
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--metric", choices=("l2", "ip"), default="l2")
    parser.add_argument("--base-limit", type=int)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--quantization-bits", type=int, default=0)
    parser.add_argument("--hnsw-clusters", type=int, default=16)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--k-factor", type=int, default=1)
    parser.add_argument("--opq", action="store_true")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument(
        "--socket",
        type=int,
        help="bind build and search CPUs and memory to this Linux NUMA socket",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="persist the built index here for later search-parameter sweeps",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name in (
        "nlist",
        "nprobe",
        "bits",
        "hnsw_clusters",
        "degree",
        "k_factor",
        "m",
        "ef_construction",
        "ef_search",
        "threads",
        "k",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.socket is not None:
        if args.socket < 0:
            parser.error("--socket must be nonnegative")
        if shutil.which("numactl") is None:
            parser.error("--socket requires numactl")
    for name in ("base_limit", "query_limit"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.method
        in (
            "rabitq-hnsw",
            "rabitq-symqg",
            "faiss-hnsw-flat",
            "faiss-hnsw-sq4",
            "faiss-hnsw-sq8",
        )
        and args.ef_search < args.k
    ):
        parser.error("--ef-search must be at least k for graph indexes")
    if args.nprobe > args.nlist:
        parser.error("--nprobe must be at most --nlist")
    if args.method == "rabitq-ivf" and args.bits not in (*range(1, 10), 32):
        parser.error("RaBitQ-Library IVF bits must be 1 through 9 or 32")
    if args.method == "rabitq-hnsw" and args.bits not in range(1, 10):
        parser.error("HNSW-RaBitQ bits must be 1 through 9")
    if args.method == "rabitq-symqg" and args.quantization_bits not in (0, 4, 8):
        parser.error("SymphonyQG quantization bits must be 0, 4, or 8")
    if args.method == "faiss-ivf-rabitq-fs" and args.bits not in range(1, 10):
        parser.error("Faiss RaBitQ FastScan bits must be 1 through 9")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.index_dir is not None and args.index_dir.exists():
        parser.error(f"index directory already exists: {args.index_dir}")
    base, total_base = load_vectors(args.base, args.base_limit)
    queries, _ = load_vectors(args.queries, args.query_limit)
    if base.shape[1] != queries.shape[1]:
        parser.error("base/query dimensions differ")
    if args.k > len(base):
        parser.error("k exceeds the base vector count")
    if args.nlist >= len(base):
        parser.error("nlist must be smaller than the base vector count")
    if args.method == "rabitq-ivf" and args.metric == "ip":
        norms = np.linalg.norm(base, axis=1)
        if not np.allclose(norms, 1, atol=1e-4):
            parser.error(
                "RaBitQKMeans spherical training requires normalized base vectors"
            )
    if args.groundtruth:
        if len(base) != total_base:
            parser.error("supplied ground truth is invalid for a truncated base")
        gt, _ = read_matrix(args.groundtruth, len(queries))
        if (
            gt.shape[0] != len(queries)
            or gt.shape[1] < args.k
            or np.any(gt[:, : args.k] < 0)
            or np.any(gt[:, : args.k] >= len(base))
        ):
            parser.error("ground truth shape or IDs are incompatible")
        gt = gt[:, : args.k].copy()
        gt_source = str(args.groundtruth)
    else:
        gt = exact_neighbors(base, queries, args.k, args.metric, args.threads)
        gt_source = "Faiss IndexFlat on selected base and queries"
    cfg = vars(args).copy()
    cfg["output"] = str(args.output)
    cfg["index_dir"] = str(args.index_dir.resolve()) if args.index_dir else None
    cfg["base_count"] = len(base)
    cfg["query_count"] = len(queries)
    with tempfile.TemporaryDirectory(prefix="rabitq-benchmark-") as tmp:
        directory = Path(tmp)
        index_directory = (
            Path(cfg["index_dir"]) if cfg["index_dir"] else directory / "index"
        )
        index_directory.mkdir(parents=True)
        np.save(directory / "base.npy", base)
        np.save(directory / "queries.npy", queries)
        np.save(directory / "groundtruth.npy", gt)
        (directory / "config.json").write_text(json.dumps(cfg))
        for stage in ("_build", "_search"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                stage,
                str(directory),
            ]
            if args.socket is not None:
                command = [
                    "numactl",
                    f"--cpunodebind={args.socket}",
                    f"--membind={args.socket}",
                    *command,
                ]
            subprocess.run(
                command,
                check=True,
            )
        build = json.loads((directory / "build.json").read_text())
        measured = json.loads((directory / "search.json").read_text())
        artifact_bytes = sum(
            p.stat().st_size for p in index_directory.rglob("*") if p.is_file()
        )
    packages = {
        name: importlib.metadata.version(name) for name in ("rabitqlib", "faiss-cpu")
    }
    cpuinfo = Path("/proc/cpuinfo")
    model_lines = (
        [
            line
            for line in cpuinfo.read_text().splitlines()
            if line.startswith("model name")
        ]
        if cpuinfo.exists()
        else []
    )
    cpu_model = (
        model_lines[0].split(":", 1)[1].strip() if model_lines else platform.processor()
    )
    record = {
        "schema_version": 3,
        "status": "built",
        "packages": packages,
        "hardware": {
            "machine": platform.machine(),
            "cpu": cpu_model,
            "os": platform.platform(),
        },
        "config": cfg,
        "data": {
            "groundtruth_source": gt_source,
            "dimension": base.shape[1],
            "total_base_rows": total_base,
        },
        "build": build,
        "search": measured,
        "serialized_index_bytes": artifact_bytes,
        "reproduction_command": shlex.join(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(record, out, indent=2)
        out.write("\n")
    print(
        f"{args.method}: recall@{args.k}={measured['recall_at_k']:.4f}; {args.output}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("_build", "_search"):
        worker(sys.argv[1], sys.argv[2])
    else:
        main()
