"""Reproduce all benchmarks or selected datasets, families, methods, and stages."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

if __package__:
    from .build_indexes import CASES, DATASETS, ROOT
    from .run_kmeans import METHODS as KMEANS_METHODS
else:
    from build_indexes import CASES, DATASETS, ROOT
    from run_kmeans import METHODS as KMEANS_METHODS


def main():
    families = {
        name: "ivf" if "ivf" in method else "graphs" for name, method, _ in CASES
    }
    families.update(dict.fromkeys(KMEANS_METHODS, "kmeans"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="default: GIST1M and DBpedia for all families",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("ivf", "graphs", "kmeans"),
        default=["ivf", "graphs", "kmeans"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=families,
        help="configuration names; omit to run all methods in the selected families",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "build", "query", "export"),
        default="all",
        help="build includes k-means fitting; query reuses saved ANN indexes",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running or writing results",
    )
    args = parser.parse_args()
    selected = [
        name
        for name, family in families.items()
        if family in args.benchmarks and (args.methods is None or name in args.methods)
    ]
    if args.methods is not None and set(args.methods) - set(selected):
        parser.error("--methods includes configurations outside --benchmarks")
    ann = [name for name in selected if families[name] != "kmeans"]
    kmeans = [name for name in selected if families[name] == "kmeans"]
    if args.stage == "query" and not ann:
        parser.error("query stage requires an IVF or graph method")
    if args.stage == "query":
        kmeans = []
    ann_datasets = args.datasets if ann else []
    kmeans_datasets = args.datasets if kmeans else []
    output = args.output_dir.resolve()
    env = dict(
        os.environ,
        MKL_THREADING_LAYER="GNU",
        OMP_NUM_THREADS="48",
        MKL_NUM_THREADS="48",
        OPENBLAS_NUM_THREADS="48",
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(Path(sys.prefix) / "lib"), env.get("LD_LIBRARY_PATH")) if p
    )
    placement = ["numactl", "--cpunodebind=0", "--membind=0"]
    commands = []
    if args.stage != "export":
        commands.append(
            [
                *placement,
                sys.executable,
                str(ROOT / "benchmarks/download_datasets.py"),
                "--datasets",
                *args.datasets,
            ]
        )
    ann_selection = ["--datasets", *ann_datasets, "--methods", *ann]
    kmeans_selection = ["--datasets", *kmeans_datasets, "--methods", *kmeans]
    if args.stage in ("all", "build"):
        if ann:
            commands.append(
                [
                    *placement,
                    sys.executable,
                    str(ROOT / "benchmarks/build_indexes.py"),
                    "--output-dir",
                    str(output / "ann"),
                    "--threads",
                    "48",
                    *ann_selection,
                ]
            )
        if kmeans:
            commands.append(
                [
                    *placement,
                    sys.executable,
                    str(ROOT / "benchmarks/run_kmeans.py"),
                    "--output-dir",
                    str(output / "kmeans"),
                    "--no-publish",
                    *kmeans_selection,
                ]
            )
    if args.stage in ("all", "query") and ann:
        commands.append(
            [
                *placement,
                sys.executable,
                str(ROOT / "benchmarks/run_sweeps.py"),
                "--output-dir",
                str(output / "ann"),
                *ann_selection,
            ]
        )
    if args.stage in ("all", "export"):
        if ann:
            commands.append(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/update_indexing.py"),
                    "--indexing-dir",
                    str(output / "ann"),
                    *ann_selection,
                ]
            )
            commands.append(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/plot_results.py"),
                    "--input-dir",
                    str(output / "ann"),
                    *ann_selection,
                ]
            )
        if kmeans:
            commands.append(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/run_kmeans.py"),
                    "--output-dir",
                    str(output / "kmeans"),
                    "--export-only",
                    *kmeans_selection,
                ]
            )
        commands.append(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "--config-file",
                str(ROOT / "docs/mkdocs.yml"),
            ]
        )
    if not args.dry_run and args.stage == "export":
        required = [
            output / folder / dataset / f"{name}.json"
            for name in ann + kmeans
            for dataset in (kmeans_datasets if name in kmeans else ann_datasets)
            for folder in (("kmeans",) if name in kmeans else ("ann", "ann/sweeps"))
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            parser.error("selected results are missing: " + ", ".join(missing))
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
