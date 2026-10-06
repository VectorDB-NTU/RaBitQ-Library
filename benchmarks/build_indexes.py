"""Build ANN indexes and verify search results before parameter sweeps."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run.py"

DATASETS = {
    "gist1m": (
        "data/gist/gist_base.fvecs",
        "data/gist/gist_query.fvecs",
        "data/gist/gist_groundtruth.ivecs",
        "l2",
    ),
    "dbpedia": (
        "data/dbpedia/base.fvecs",
        "data/dbpedia/queries.fvecs",
        "data/dbpedia/queries.ivecs",
        "ip",
    ),
}
DATASET_SHAPES = {"gist1m": (1000000, 960), "dbpedia": (999000, 1536)}

CASES = (
    ("rabitq-ivf-raw", "rabitq-ivf", ("--bits", "32")),
    ("rabitq-ivf-1p4", "rabitq-ivf", ("--bits", "5")),
    ("rabitq-ivf-1p8", "rabitq-ivf", ("--bits", "9")),
    ("faiss-rabitqfs-raw", "faiss-ivf-rabitq-fs", ("--bits", "1")),
    ("faiss-rabitqfs-1p4", "faiss-ivf-rabitq-fs", ("--bits", "5")),
    ("faiss-rabitqfs-1p8", "faiss-ivf-rabitq-fs", ("--bits", "9")),
    ("faiss-pqfs", "faiss-ivf-pqfs", ()),
    ("faiss-opq-pqfs", "faiss-ivf-pqfs", ("--opq",)),
    ("rabitq-symqg-raw", "rabitq-symqg", ("--quantization-bits", "0")),
    ("rabitq-symqg-4", "rabitq-symqg", ("--quantization-bits", "4")),
    ("rabitq-symqg-8", "rabitq-symqg", ("--quantization-bits", "8")),
    ("rabitq-hnsw-1p4", "rabitq-hnsw", ("--bits", "5")),
    ("rabitq-hnsw-1p8", "rabitq-hnsw", ("--bits", "9")),
    ("faiss-hnsw-flat", "faiss-hnsw-flat", ()),
    ("faiss-hnsw-sq4", "faiss-hnsw-sq4", ()),
    ("faiss-hnsw-sq8", "faiss-hnsw-sq8", ()),
)

LABELS = (
    "IVF RaBitQ raw",
    "IVF RaBitQ 1+4",
    "IVF RaBitQ 1+8",
    "Faiss IVF RaBitQfs raw + RFlat",
    "Faiss IVF RaBitQfs 1+4",
    "Faiss IVF RaBitQfs 1+8",
    "Faiss IVF PQfs + RFlat",
    "Faiss OPQ-IVF PQfs + RFlat",
    "SymphonyQG raw",
    "SymphonyQG 4-bit",
    "SymphonyQG 8-bit",
    "HNSW RaBitQ 1+4",
    "HNSW RaBitQ 1+8",
    "Faiss HNSW Flat",
    "Faiss HNSW SQ4",
    "Faiss HNSW SQ8",
)


def validate_record(record, dataset, name):
    """Reject results that do not match the published ANN configuration."""
    method, extra = next((m, e) for n, m, e in CASES if n == name)
    base, queries, truth, metric = DATASETS[dataset]
    n, dim = DATASET_SHAPES[dataset]
    expected = {
        "method": method,
        "metric": metric,
        "base_count": n,
        "query_count": 1000,
        "base_limit": None,
        "query_limit": None,
        "threads": 48,
        "socket": 0,
        "k": 10,
    }
    if "ivf" in method:
        expected.update(nlist=4096, opq="--opq" in extra)
    if "hnsw" in method:
        expected.update(m=16, ef_construction=200)
        if method == "rabitq-hnsw":
            expected["hnsw_clusters"] = 16
    if method == "rabitq-symqg":
        expected.update(degree=32, ef_construction=200)
    for option in ("--bits", "--quantization-bits"):
        if option in extra:
            expected[option[2:].replace("-", "_")] = int(extra[extra.index(option) + 1])
    cfg = record["config"]
    if (
        any(cfg.get(key) != value for key, value in expected.items())
        or record["data"]["dimension"] != dim
        or any(
            Path(cfg[key]).resolve() != (ROOT / path).resolve()
            for key, path in (
                ("base", base),
                ("queries", queries),
                ("groundtruth", truth),
            )
        )
    ):
        raise ValueError(f"unexpected ANN configuration: {dataset}/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/benchmarks/ann")
    )
    parser.add_argument("--methods", nargs="+", choices=[case[0] for case in CASES])
    parser.add_argument("--threads", type=int, default=48)
    args = parser.parse_args()

    failed = []
    for dataset in args.datasets:
        base, queries, truth, metric = DATASETS[dataset]
        for name, method, extra in CASES:
            if args.methods is not None and name not in args.methods:
                continue
            output = args.output_dir / dataset / f"{name}.json"
            index_dir = args.output_dir / "indexes" / dataset / name
            log = args.output_dir / "logs" / dataset / f"{name}.log"
            if output.exists():
                validate_record(json.loads(output.read_text()), dataset, name)
                print(f"SKIP {dataset} {name}: result exists", flush=True)
                continue
            if index_dir.exists():
                failed.append(f"{dataset}/{name}: incomplete index directory exists")
                print(f"SKIP {failed[-1]}", flush=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(RUNNER),
                "--base",
                base,
                "--queries",
                queries,
                "--groundtruth",
                truth,
                "--metric",
                metric,
                "--method",
                method,
                "--nlist",
                "4096",
                "--nprobe",
                "64",
                "--m",
                "16",
                "--degree",
                "32",
                "--hnsw-clusters",
                "16",
                "--ef-construction",
                "200",
                "--ef-search",
                "128",
                "--k-factor",
                "1",
                "--threads",
                str(args.threads),
                "--socket",
                "0",
                "--k",
                "10",
                "--index-dir",
                str(index_dir),
                "--output",
                str(output),
                *extra,
            ]
            print(f"START {dataset} {name}", flush=True)
            with log.open("w") as stream:
                stream.write(" ".join(command) + "\n")
                stream.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=stream)
            if result.returncode:
                failed.append(f"{dataset}/{name}: exit {result.returncode}")
                print(f"FAIL {failed[-1]} (see {log})", flush=True)
            else:
                print(f"DONE {dataset} {name}", flush=True)
    if failed:
        report = args.output_dir / "build_failures.json"
        report.write_text(json.dumps(failed, indent=2) + "\n")
        raise SystemExit(f"{len(failed)} index builds failed; see {report}")


if __name__ == "__main__":
    main()
