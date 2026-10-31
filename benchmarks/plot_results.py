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
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

if __package__:
    from .build_indexes import CASES, DATASETS, validate_record
    from .query_index import settings
else:
    from build_indexes import CASES, DATASETS, validate_record
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


# Each panel compares methods with similar per-dimension storage. Colors follow
# the index type within a figure (validated categorical slots; at most three per
# figure); line style and marker separate methods that share a color.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
PANELS = {
    "ivf": (
        (
            "Codes + float32 reranking",
            ("rabitq-ivf-raw", "faiss-rabitqfs-raw", "faiss-pqfs", "faiss-opq-pqfs"),
        ),
        ("1 + 4 bits", ("rabitq-ivf-1p4", "faiss-rabitqfs-1p4")),
        ("1 + 8 bits", ("rabitq-ivf-1p8", "faiss-rabitqfs-1p8")),
    ),
    "graphs": (
        ("Float32 vectors", ("rabitq-symqg-raw", "faiss-hnsw-flat")),
        ("4-5 bits", ("rabitq-symqg-4", "rabitq-hnsw-1p4", "faiss-hnsw-sq4")),
        ("8-9 bits", ("rabitq-symqg-8", "rabitq-hnsw-1p8", "faiss-hnsw-sq8")),
    ),
}
STYLES = {
    "rabitq-ivf": (BLUE, "o", "-"),
    "faiss-ivf-rabitq-fs": (ORANGE, "s", "--"),
    "faiss-pqfs": (AQUA, "^", "--"),
    "faiss-opq-pqfs": (AQUA, "D", ":"),
    "rabitq-symqg": (BLUE, "o", "-"),
    "rabitq-hnsw": (AQUA, "^", "-"),
    "faiss-hnsw-flat": (ORANGE, "s", "--"),
    "faiss-hnsw-sq4": (ORANGE, "s", "--"),
    "faiss-hnsw-sq8": (ORANGE, "s", "--"),
}
for _family, _panels in PANELS.items():
    if {name for _, names in _panels for name in names} != {
        case.name for case in CASES if case.family == _family
    }:
        raise RuntimeError(f"plot panels do not cover the {_family} cases")


def case_style(case):
    return STYLES.get(case.name) or STYLES[case.method]


def qps_label(value, _position):
    return f"{value / 1000:g}K" if value >= 1000 else f"{value:g}"


def plot_family(measurements, dataset, family, cpu):
    """Draw one dataset's QPS-recall panels for an index family."""
    cases = {case.name: case for case in CASES}
    panels = PANELS[family]
    series = {}
    for _, names in panels:
        for name in names:
            points = measurements.get((dataset, name))
            if not points:
                continue
            for _, recall, qps in points:
                if not (0 <= recall <= 1 and math.isfinite(qps) and qps > 0):
                    raise ValueError(f"invalid measurement: {dataset}/{name}")
            series[name] = [(recall, qps) for _, recall, qps in points]
    # Start at 0.5 recall, or lower so every method's best recall sits at least
    # 0.1 inside the axis; settings below the start are clipped.
    lowest_best = min(max(recall for recall, _ in points) for points in series.values())
    recall_min = min(0.5, math.floor((lowest_best - 0.1) * 10) / 10)

    fig, axes = plt.subplots(1, len(panels), figsize=(9, 4.7), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.81, bottom=0.31, wspace=0.16)
    title = {"gist1m": "GIST1M", "dbpedia": "DBpedia"}[dataset]
    family_title = "IVF indexes" if family == "ivf" else "Graph indexes"
    fig.text(
        0.012, 0.962, f"{title} · {family_title}", fontsize=14, weight="bold", color=INK
    )
    fig.text(
        0.012,
        0.91,
        f"Recall@10 vs queries per second · 1 search thread, 1 query per call · CPU: {cpu}",
        fontsize=9.5,
        color=SECONDARY,
    )
    for ax, (panel_title, names) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        handles = []
        for name in names:
            if name not in series:
                continue
            case = cases[name]
            color, marker, style = case_style(case)
            points = series[name]
            x, y = zip(*points)
            ax.scatter(x, y, color=color, marker=marker, s=12, alpha=0.3, linewidths=0)
            x, y = zip(*frontier(points))
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                linewidth=1.5,
                linestyle=style,
                solid_capstyle="round",
                dash_capstyle="round",
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    marker=marker,
                    linestyle=style,
                    label=case.label,
                )
            )
        ax.set_title(panel_title, fontsize=10.5, color=INK, loc="left", pad=6)
        ax.set_xlim(recall_min, 1.005)
        ax.set_xticks(
            [
                round(recall_min + 0.1 * i, 1)
                for i in range(round((1 - recall_min) * 10) + 1)
            ]
        )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(qps_label))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, which="major", color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=9, length=0)
        ax.set_xlabel("Recall@10", fontsize=9.5, color=SECONDARY)
        # Below the panel, so the legend never covers measured points.
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0, -0.2),
            borderaxespad=0,
            frameon=False,
            fontsize=8,
            handlelength=2.2,
            markerscale=0.8,
            labelcolor=INK,
        )
    axes[0].set_ylabel("Queries / second (log)", fontsize=9.5, color=SECONDARY)
    return fig


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
        choices=[case.name for case in CASES],
        default=[case.name for case in CASES],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measurements = {}
    cpus = set()
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
        cpus.add(initial["hardware"]["cpu"])
        measurements[dataset, name] = list(observations(record))
    if len(cpus) != 1:
        raise ValueError(f"selected records span CPUs {sorted(cpus)}")
    (cpu,) = cpus
    rows = []
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
        }
    )

    for dataset, family in product(args.datasets, ("ivf", "graphs")):
        if not any(
            case.name in args.methods and case.family == family for case in CASES
        ):
            continue
        fig = plot_family(measurements, dataset, family, cpu)
        path = args.output_dir / f"{family}-{dataset}-qps-recall.svg"
        fig.savefig(path, facecolor=SURFACE)
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
        plt.close(fig)
        print(f"Wrote {path}")
    for dataset in DATASETS:
        for case in CASES:
            name = case.name
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
