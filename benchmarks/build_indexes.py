"""Build ANN indexes and verify search results before parameter sweeps."""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Case:
    name: str
    label: str
    method: str
    family: str
    parameters: dict

    def cli_args(self):
        args = []
        for key, value in self.parameters.items():
            args.append("--" + key.replace("_", "-"))
            if value is not True:
                args.append(str(value))
        return args


CASES = (
    Case("rabitq-ivf-raw", "IVF-RaBitQ raw", "rabitq-ivf", "ivf", {"bits": 32}),
    Case("rabitq-ivf-1p4", "IVF-RaBitQ 1+4", "rabitq-ivf", "ivf", {"bits": 5}),
    Case("rabitq-ivf-1p8", "IVF-RaBitQ 1+8", "rabitq-ivf", "ivf", {"bits": 9}),
    Case(
        "faiss-rabitqfs-raw",
        "Faiss IVF RaBitQfs raw + RFlat",
        "faiss-ivf-rabitq-fs",
        "ivf",
        {"bits": 1},
    ),
    Case(
        "faiss-rabitqfs-1p4",
        "Faiss IVF RaBitQfs 1+4",
        "faiss-ivf-rabitq-fs",
        "ivf",
        {"bits": 5},
    ),
    Case(
        "faiss-rabitqfs-1p8",
        "Faiss IVF RaBitQfs 1+8",
        "faiss-ivf-rabitq-fs",
        "ivf",
        {"bits": 9},
    ),
    Case("faiss-pqfs", "Faiss IVF PQfs + RFlat", "faiss-ivf-pqfs", "ivf", {}),
    Case(
        "faiss-opq-pqfs",
        "Faiss OPQ-IVF PQfs + RFlat",
        "faiss-ivf-pqfs",
        "ivf",
        {"opq": True},
    ),
    Case(
        "rabitq-symqg-raw",
        "SymphonyQG raw",
        "rabitq-symqg",
        "graphs",
        {"quantization_bits": 0},
    ),
    Case(
        "rabitq-symqg-4",
        "SymphonyQG 4-bit",
        "rabitq-symqg",
        "graphs",
        {"quantization_bits": 4},
    ),
    Case(
        "rabitq-symqg-8",
        "SymphonyQG 8-bit",
        "rabitq-symqg",
        "graphs",
        {"quantization_bits": 8},
    ),
    Case("rabitq-hnsw-1p4", "HNSW-RaBitQ 1+4", "rabitq-hnsw", "graphs", {"bits": 5}),
    Case("rabitq-hnsw-1p8", "HNSW-RaBitQ 1+8", "rabitq-hnsw", "graphs", {"bits": 9}),
    Case("faiss-hnsw-flat", "Faiss HNSW Flat", "faiss-hnsw-flat", "graphs", {}),
    Case("faiss-hnsw-sq4", "Faiss HNSW SQ4", "faiss-hnsw-sq4", "graphs", {}),
    Case("faiss-hnsw-sq8", "Faiss HNSW SQ8", "faiss-hnsw-sq8", "graphs", {}),
)


def validate_record(record, dataset, name):
    """Reject results that do not match the published ANN configuration."""
    case = next(case for case in CASES if case.name == name)
    method = case.method
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
    if case.family == "ivf":
        expected.update(nlist=4096, opq=False)
    if "hnsw" in method:
        expected.update(m=16, ef_construction=200)
        if method == "rabitq-hnsw":
            expected["hnsw_clusters"] = 16
    if method == "rabitq-symqg":
        expected.update(degree=32, ef_construction=200)
    expected.update(case.parameters)
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
    parser.add_argument("--methods", nargs="+", choices=[case.name for case in CASES])
    parser.add_argument("--threads", type=int, default=48)
    args = parser.parse_args()

    failed = []
    for dataset in args.datasets:
        base, queries, truth, metric = DATASETS[dataset]
        for case in CASES:
            name, method = case.name, case.method
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
                *case.cli_args(),
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
