"""Regenerate homepage bars and tables from the committed benchmark CSVs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/docs/benchmarks/figures"
# These are not matched-memory configurations.
METHODS = (
    ("rabitq-ivf-1p4", "IVF-RaBitQ · 5-bit", "#635bff"),
    ("rabitq-hnsw-1p4", "HNSW-RaBitQ · 5-bit", "#149eaa"),
    ("rabitq-symqg-4", "SymphonyQG · 4-bit", "#e29838"),
)


def main():
    with (FIGURES / "qps-recall-points.csv").open() as stream:
        queries = list(csv.DictReader(stream))
    with (FIGURES / "indexing.csv").open() as stream:
        builds = list(csv.DictReader(stream))
    plt.rcParams.update(
        {"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "homepage"}
    )
    sections = []
    detail_sections = []
    readme_sections = []
    for dataset, title, description in (
        ("gist1m", "GIST1M", "1,000,000 vectors · 960 dimensions · L2"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        fig.subplots_adjust(left=0.065, right=0.98, top=0.72, bottom=0.22, wspace=0.36)
        fig.suptitle(
            title,
            x=0.065,
            y=0.98,
            ha="left",
            fontsize=20,
            fontweight="bold",
            color="#0f172a",
        )
        fig.text(
            0.065,
            0.88,
            "Fastest measured QPS at Recall@10 ≥ 90% · RaBitQ-Library 0.7.0",
            color="#475569",
            fontsize=12,
        )
        values = [[], []]
        table = [
            "| Configuration | Measured Recall@10 | Selected search setting |",
            "| --- | ---: | --- |",
        ]
        for method, label, _ in METHODS:
            rows = [
                r for r in queries if r["dataset"] == dataset and r["method"] == method
            ]
            assert rows, (dataset, method)
            eligible = [r for r in rows if float(r["recall_at_10"]) >= 0.90]
            assert eligible, (dataset, method, "90% recall not reached")
            selected = max(eligible, key=lambda r: float(r["qps"]))
            values[0].append(float(selected["qps"]))
            matches = [
                r for r in builds if r["dataset"] == dataset and r["method"] == method
            ]
            assert len(matches) == 1, (dataset, method)
            build = matches[0]
            gib = int(build["serialized_index_bytes"]) / 2**30
            values[1].append(gib)
            setting = json.loads(selected["setting"])
            parameter = (
                f"nprobe={setting['nprobe']}"
                if "nprobe" in setting
                else f"ef={setting['ef_search']}"
            )
            table.append(
                f"| {label} | {float(selected['recall_at_10']):.2%} | `{parameter}` |"
            )
        labels = ["IVF-RaBitQ\n5-bit", "HNSW-RaBitQ\n5-bit", "SymphonyQG\n4-bit"]
        colors = [m[2] for m in METHODS]
        for ax, numbers, heading, direction, precision in zip(
            axes,
            values,
            ("Queries / second", "Index size · GiB"),
            ("Higher is better", "Lower is better"),
            (0, 2),
        ):
            bars = ax.bar(range(3), numbers, width=0.55, color=colors, zorder=3)
            ax.bar_label(
                bars,
                labels=[f"{v:,.{precision}f}" for v in numbers],
                padding=7,
                fontsize=12,
                fontweight="bold",
                color="#0f172a",
            )
            ax.set_ylim(0, max(numbers) * 1.25)
            ax.set_xticks(range(3), labels, fontsize=11)
            ax.set_title(
                heading + "\n" + direction,
                loc="left",
                fontsize=12,
                pad=16,
                color="#334155",
            )
            ax.grid(axis="y", color="#e5e7eb", zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#cbd5e1")
            ax.tick_params(axis="both", length=0, colors="#475569")
        fig.text(
            0.065,
            0.035,
            "1 search thread · 1 query/call · Index size is serialized storage, not RAM",
            fontsize=10,
            color="#64748b",
        )
        filename = f"homepage-{dataset}.svg"
        path = FIGURES / filename
        fig.savefig(path, metadata={"Date": None})
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
        plt.close(fig)
        body = [
            description,
            "",
            f"![{title}: QPS at Recall@10 ≥ 90%, serialized index size](benchmarks/figures/{filename})",
        ]
        detail_sections.append(f"### {title}\n\n" + "\n".join(table))
        sections.append(f"**{title}**\n\n" + "\n".join(body))
        readme_body = sections[-1].replace(
            "](benchmarks/figures/", "](docs/docs/benchmarks/figures/"
        )
        readme_sections.append(readme_body)
    details = ROOT / "docs/docs/benchmarks/bench.md"
    start, end = "<!-- homepage-selection:start -->", "<!-- homepage-selection:end -->"
    before, rest = details.read_text().split(start)
    _, after = rest.split(end)
    details.write_text(
        before + start + "\n\n" + "\n\n".join(detail_sections) + "\n\n" + end + after
    )
    readme = ROOT / "README.md"
    start, end = "<!-- homepage-results:start -->", "<!-- homepage-results:end -->"
    before, rest = readme.read_text().split(start)
    _, after = rest.split(end)
    readme.write_text(
        before + start + "\n\n" + "\n\n".join(readme_sections) + "\n\n" + end + after
    )
    homepage = ROOT / "docs/docs/index.md"
    text = homepage.read_text()
    start, end = "<!-- homepage-results:start -->", "<!-- homepage-results:end -->"
    before, rest = text.split(start)
    _, after = rest.split(end)
    homepage.write_text(
        before + start + "\n\n" + "\n\n".join(sections) + "\n" + end + after
    )


if __name__ == "__main__":
    main()
