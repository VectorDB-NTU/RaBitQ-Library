"""Sweep search parameters on one fixed query set per dataset."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from build_indexes import CASES, DATASETS, ROOT

SWEEPER = ROOT / "benchmarks" / "query_index.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/benchmarks/ann")
    )
    parser.add_argument("--methods", nargs="+", choices=[case[0] for case in CASES])
    args = parser.parse_args()
    # One native search thread, including BLAS transforms.
    env = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    failed = []
    for dataset in args.datasets:
        for name, _, _ in CASES:
            if args.methods is not None and name not in args.methods:
                continue
            initial = args.output_dir / dataset / f"{name}.json"
            output = args.output_dir / "sweeps" / dataset / f"{name}.json"
            log = args.output_dir / "logs" / dataset / f"{name}-sweep.log"
            if output.exists():
                print(f"SKIP {dataset} {name}: sweep exists", flush=True)
                continue
            if not initial.exists():
                failed.append(f"{dataset}/{name}: initial record missing")
                print(f"SKIP {failed[-1]}", flush=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "numactl",
                "--cpunodebind=0",
                "--membind=0",
                sys.executable,
                str(SWEEPER),
                str(initial),
                "--dataset",
                dataset,
                "--output",
                str(output),
            ]
            print(f"START SWEEP {dataset} {name}", flush=True)
            with log.open("w") as stream:
                stream.write(" ".join(command) + "\n")
                stream.flush()
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=stream, stderr=stream
                )
            if result.returncode:
                failed.append(f"{dataset}/{name}: exit {result.returncode}")
                print(f"FAIL {failed[-1]} (see {log})", flush=True)
            else:
                print(f"DONE SWEEP {dataset} {name}", flush=True)
    report = args.output_dir / "sweep_failures.json"
    report.write_text(json.dumps(failed, indent=2) + "\n")
    if failed:
        raise SystemExit(f"{len(failed)} sweeps failed; see {report}")


if __name__ == "__main__":
    main()
