"""Plot QPS-recall curves and export measurements from single-query-set sweeps."""

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __package__:
    from .build_indexes import CASES, DATASETS, LABELS, validate_record
    from .query_index import settings
else:
    from build_indexes import CASES, DATASETS, LABELS, validate_record
    from query_index import settings


def frontier(points):
    """Keep measured points not dominated in both recall and throughput."""
    result = []
    best_qps = -math.inf
    for recall, qps in sorted(set(points), reverse=True):
        if qps > best_qps:
            result.append((recall, qps))
            best_qps = qps
    return list(reversed(result))


def observations(record):
    if record.get("status") != "single_query_set":
        raise ValueError("expected a single_query_set sweep record")
    for point in record["query_points"]:
        yield (
            point["setting"],
            point["recall"],
            record["query_count"] / point["one_query_service_s"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/docs/benchmarks/figures")
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[case[0] for case in CASES],
        default=[case[0] for case in CASES],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measurements = {}
    csv_path = args.output_dir / "qps-recall-points.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as previous:
            for row in csv.DictReader(previous):
                measurements.setdefault((row["dataset"], row["method"]), []).append(
                    (
                        json.loads(row["setting"]),
                        float(row["recall_at_10"]),
                        float(row["qps"]),
                    )
                )
    for dataset, name in product(args.datasets, args.methods):
        initial = json.loads((args.input_dir / dataset / f"{name}.json").read_text())
        validate_record(initial, dataset, name)
        record = json.loads(
            (args.input_dir / "sweeps" / dataset / f"{name}.json").read_text()
        )
        if (
            record["dataset"] != dataset
            or record["query_count"] != initial["config"]["query_count"]
            or [p["setting"] for p in record["query_points"]]
            != settings(initial["config"])
            or any(
                record["thread_environment"].get(key) != "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            )
            or record["cpu_affinity"]
            != initial["build"]["cpu_placement"]["allowed_cpus"]
        ):
            raise ValueError(f"sweep does not match its index build: {dataset}/{name}")
        measurements[dataset, name] = list(observations(record))
    colors = list(plt.get_cmap("tab10").colors)
    markers = ("o", "s", "^", "D", "v", "P")
    rows = []
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})

    for dataset, family in product(args.datasets, ("ivf", "graphs")):
        if not any(
            name in args.methods and ("ivf" if "ivf" in method else "graphs") == family
            for name, method, _ in CASES
        ):
            continue
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.subplots_adjust(top=0.85, bottom=0.32)
        title = {"gist1m": "GIST1M", "dbpedia": "DBpedia"}[dataset]
        family_title = "IVF indexes" if family == "ivf" else "Graph indexes"
        fig.suptitle(
            f"{title} · {family_title}  |  Recall@10 vs QPS", fontsize=20, y=0.975
        )
        fig.text(
            0.5,
            0.92,
            "One Intel Xeon Gold 6418H CPU · one query per call",
            ha="center",
            color="#444444",
        )
        handles = []
        recalls = []
        for ordinal, ((name, method, _), label) in enumerate(
            (case, label)
            for case, label in zip(CASES, LABELS, strict=True)
            if ("ivf" if "ivf" in case[1] else "graphs") == family
        ):
            if (dataset, name) not in measurements:
                continue
            points = []
            for setting, recall, qps in measurements[dataset, name]:
                if not (0 <= recall <= 1 and math.isfinite(qps) and qps > 0):
                    raise ValueError(f"invalid measurement: {dataset}/{name}")
                points.append((recall, qps))
            recalls.extend(recall for recall, _ in points)
            color = colors[ordinal]
            marker = markers[ordinal % len(markers)]
            style = "-" if name.startswith("rabitq-") else "--"
            handles.append(
                Line2D([], [], color=color, marker=marker, linestyle=style, label=label)
            )
            if points:
                x, y = zip(*points)
                ax.scatter(x, y, color=color, marker=marker, s=22, alpha=0.45)
                x, y = zip(*frontier(points))
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.4,
                    linestyle=style,
                )
        ax.set_title(
            "Query set · full sweep · 1 search thread · one timing pass", fontsize=11
        )
        ax.set_xlabel("Recall@10 (measured)")
        ax.set_ylabel("Queries / second (log scale)")
        recall_min = max(0, math.floor((min(recalls) - 0.005) * 20) / 20)
        ax.set_xlim(recall_min, 1.005)
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.09),
            ncol=2,
            frameon=False,
            fontsize=9,
        )
        fig.text(
            0.5,
            0.046,
            "Faint markers: all measured settings. Lines: per-method Pareto frontier; no fitted or extrapolated points.",
            ha="center",
            fontsize=10,
            color="#444444",
        )
        path = args.output_dir / f"{family}-{dataset}-qps-recall.svg"
        fig.savefig(path)
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
        plt.close(fig)
        print(f"Wrote {path}")
    for dataset in DATASETS:
        for name, _, _ in CASES:
            for setting, recall, qps in measurements.get((dataset, name), []):
                rows.append(
                    (
                        dataset,
                        name,
                        json.dumps(setting, sort_keys=True),
                        recall,
                        qps,
                    )
                )
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "dataset",
                "method",
                "setting",
                "recall_at_10",
                "qps",
            )
        )
        writer.writerows(rows)
    print(f"Exported {len(rows)} measurements")


if __name__ == "__main__":
    main()
