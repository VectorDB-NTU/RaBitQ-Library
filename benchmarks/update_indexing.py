"""Export indexing tables and CSV from completed 48-thread builds."""

import argparse
import csv
import io
import json
import math
from pathlib import Path

if __package__:
    from .build_indexes import CASES, DATASETS, LABELS, validate_record
else:
    from build_indexes import CASES, DATASETS, LABELS, validate_record

FIELDS = (
    "dataset",
    "family",
    "method",
    "training_s",
    "indexing_s",
    "build_s",
    "save_s",
    "serialized_index_bytes",
    "construction_threads",
)


def replace_block(text, name, body):
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected one generated block: {name}")
    before, rest = text.split(start)
    _, after = rest.split(end)
    return before + start + "\n" + body + "\n" + end + after


def update(indexing_dir, docs_dir, datasets=None, methods=None):
    datasets = list(DATASETS) if datasets is None else datasets
    cases = [case for case in CASES if methods is None or case[0] in methods]
    rows = {}
    csv_path = docs_dir / "figures/indexing.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as previous:
            rows = {
                (row["dataset"], row["method"]): tuple(row[key] for key in FIELDS)
                for row in csv.DictReader(previous)
            }
    for dataset in datasets:
        for name, method, _ in cases:
            path = indexing_dir / dataset / f"{name}.json"
            record = json.loads(path.read_text())
            validate_record(record, dataset, name)
            build = record["build"]
            values = [build[key] for key in FIELDS[3:7]]
            values.append(record["serialized_index_bytes"])
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(f"invalid indexing measurements: {path}")
            family = "ivf" if "ivf" in method else "graphs"
            rows[dataset, name] = (dataset, family, name, *values, 48)

    pages = {}
    for family in ("ivf", "graphs"):
        if not any(
            ("ivf" if "ivf" in method else "graphs") == family for _, method, _ in cases
        ):
            continue
        path = docs_dir / f"{family}.md"
        text = path.read_text()
        for dataset in datasets:
            table = [
                "| Method | Train s | Construct s | Total s | Save s | Index GiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for (name, method, _), label in zip(CASES, LABELS, strict=True):
                if ("ivf" if "ivf" in method else "graphs") != family:
                    continue
                row = rows.get((dataset, name))
                if row is None:
                    cells = ["Pending", *["—"] * 4]
                else:
                    values = [
                        *(float(value) for value in row[3:7]),
                        float(row[7]) / 2**30,
                    ]
                    cells = [f"{value:.2f}" for value in values]
                table.append("| " + label + " | " + " | ".join(cells) + " |")
            text = replace_block(text, f"indexing:{dataset}", "\n".join(table))
        pages[path] = text

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(
        rows[key] for key in ((d, c[0]) for d in DATASETS for c in CASES) if key in rows
    )
    pages[csv_path] = stream.getvalue()
    for path, text in pages.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
    print(f"Updated {len(cases) * len(datasets)} indexing results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexing-dir", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/docs/benchmarks"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=[case[0] for case in CASES])
    args = parser.parse_args()
    update(args.indexing_dir, args.docs_dir, args.datasets, args.methods)


if __name__ == "__main__":
    main()
