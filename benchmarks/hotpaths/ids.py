"""Time the production ID-conversion helper in a benchmark-only extension."""

import argparse
import hashlib
import importlib
import sys
import time
from pathlib import Path

import numpy as np


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--rows", type=positive, default=1_000_000)
    parser.add_argument("--iterations", type=positive, default=5)
    parser.add_argument("--repetitions", type=positive, default=21)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module_dir.resolve()))
    module = importlib.import_module("rabitq_ids_benchmark")
    extension = Path(module.__file__).resolve()
    if extension.parent != args.module_dir.resolve():
        raise RuntimeError("Imported benchmark module from an unexpected directory")
    print(
        f"# extension={extension} sha256={hashlib.sha256(extension.read_bytes()).hexdigest()}"
    )
    print(f"# numpy={np.__version__} rows={args.rows} threads=1 clusters=64")
    print("operation,repetition,operations,seconds,ns_per_operation,hash")
    for dtype in (np.uint32, np.int64):
        ids = np.arange(args.rows, dtype=dtype) % 64
        expected = int(ids[-1])
        if module.convert(ids, 64) != expected:
            raise RuntimeError("Incorrect converted ID")
        for rep in range(args.repetitions):
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                last = module.convert(ids, 64)
            elapsed = (time.perf_counter_ns() - start) / 1e9
            if last != expected:
                raise RuntimeError("Incorrect converted ID")
            print(
                f"ids_{np.dtype(dtype)},{rep},{args.iterations},{elapsed:.12g},"
                f"{elapsed * 1e9 / args.iterations:.12g},{last}"
            )


if __name__ == "__main__":
    main()
