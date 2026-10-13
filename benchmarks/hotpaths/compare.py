"""Alternate two commands, retain raw evidence, and compare output fingerprints."""

import argparse
import csv
import io
import json
import os
import platform
import shlex
import statistics
import subprocess
import time
from pathlib import Path


def records(text):
    rows = list(
        csv.DictReader(
            io.StringIO(
                "\n".join(
                    line for line in text.splitlines() if not line.startswith("#")
                )
            )
        )
    )
    if (
        not rows
        or not {"operation", "repetition", "operations", "hash", "ns_per_operation"}
        <= rows[0].keys()
    ):
        raise ValueError("Command did not produce hot-path benchmark CSV")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before", required=True, help="command, parsed without a shell"
    )
    parser.add_argument(
        "--after", required=True, help="command, parsed without a shell"
    )
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, required=True, help="new evidence directory"
    )
    parser.add_argument(
        "--notes", required=True, help="revisions, compiler/flags and NUMA policy"
    )
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    cpuinfo = Path("/proc/cpuinfo")
    report = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "notes": args.notes,
        "commands": {
            "before": shlex.split(args.before),
            "after": shlex.split(args.after),
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
        },
        "runs": [],
        "status": "running",
    }
    if cpuinfo.exists():
        (args.output / "cpuinfo.txt").write_text(cpuinfo.read_text())
    samples = {"before": {}, "after": {}}

    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for pair in range(args.pairs):
            fingerprints = {}
            for version in (
                ("before", "after") if pair % 2 == 0 else ("after", "before")
            ):
                prefix = args.output / f"pair-{pair}-{version}"
                start = time.time()
                # Stream into files so failures/interruptions retain partial output.
                with (
                    prefix.with_suffix(".csv").open("w") as stdout,
                    prefix.with_suffix(".stderr").open("w") as stderr,
                ):
                    result = subprocess.run(
                        report["commands"][version],
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
                report["runs"].append(
                    {
                        "pair": pair,
                        "version": version,
                        "start": start,
                        "returncode": result.returncode,
                    }
                )
                save()
                if result.returncode:
                    raise RuntimeError(f"{version} failed; inspect {prefix}.stderr")
                rows = records(prefix.with_suffix(".csv").read_text())
                fingerprints[version] = [
                    (r["operation"], r["repetition"], r["operations"], r["hash"])
                    for r in rows
                ]
                for row in rows:
                    samples[version].setdefault(row["operation"], []).append(
                        float(row["ns_per_operation"])
                    )
            if fingerprints["before"] != fingerprints["after"]:
                raise RuntimeError(
                    f"Output fingerprints differ in pair {pair}; review numerical results before timing"
                )
        report["medians_ns"] = {
            v: {op: statistics.median(xs) for op, xs in ops.items()}
            for v, ops in samples.items()
        }
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()
    print(json.dumps(report["medians_ns"], indent=2))


if __name__ == "__main__":
    main()
