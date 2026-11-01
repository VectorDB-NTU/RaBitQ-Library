import argparse
from time import time

from rabitqlib import SymqgIndex

from utils import read_fvecs

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
MAX_DEGREE = 32  # degree bound for SymphonyQG
EF_CONSTRUCTION = 200  # ef for indexing
METRIC = "l2"  # "l2" or "ip"
NBITS = 32  # 32 for raw SymphonyQG, or 4/8 for quantized SymphonyQG
NUM_THREADS = 16  # number of threads for build
INIT = "pipnn"  # "pipnn" or "random" initialization of SymphonyQG
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load data
    data = read_fvecs(args.data_file)
    print(f"Data shape: {data.shape}")

    # 2. Build the SymphonyQG graph; there is nothing to train.
    n, dim = data.shape
    print(
        f"\nBuilding SymphonyQG index: n={n}, dim={dim}, MaxDegree={args.max_degree}, "
        f"ef={args.ef_construction}, metric={args.metric}, "
        f"nbits={args.nbits}, init={args.init}"
    )

    idx = SymqgIndex(
        dim,
        nbits=args.nbits,
        max_degree=args.max_degree,
        ef_construction=args.ef_construction,
        init=args.init,
        metric=args.metric,
    )

    t0 = time()
    # add builds the whole graph at once; it cannot be extended later.
    idx.add(data, num_threads=args.num_threads)
    print(f"Indexing time: {time() - t0:.2f}s")

    idx.save(args.index_file)
    print(f"Index saved: {args.index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SymphonyQG Manager")
    parser.add_argument("data_file", type=str, help="Path to the data file")
    parser.add_argument("index_file", type=str, help="Path to save the index")
    parser.add_argument(
        "--max-degree",
        dest="max_degree",
        type=int,
        metavar="INT",
        default=MAX_DEGREE,
        help="Degree bound for SymphonyQG",
    )
    parser.add_argument(
        "--ef-construction",
        dest="ef_construction",
        type=int,
        metavar="INT",
        default=EF_CONSTRUCTION,
        help="EF parameter for index construction",
    )
    parser.add_argument(
        "--metric",
        dest="metric",
        type=str,
        default=METRIC,
        choices=["l2", "ip"],
        help="Distance metric (l2 or ip)",
    )
    parser.add_argument(
        "--nbits",
        dest="nbits",
        type=int,
        choices=[4, 8, 32],
        default=NBITS,
        help="Storage bits: 32 for raw SymphonyQG, or 4/8 for quantized SymphonyQG",
    )
    parser.add_argument(
        "--init",
        choices=["pipnn", "random"],
        default=INIT,
        help="SymphonyQG initialization (default: pipnn)",
    )
    parser.add_argument(
        "--num-threads",
        dest="num_threads",
        type=int,
        metavar="INT",
        default=NUM_THREADS,
        help="Number of threads for building the index",
    )

    args = parser.parse_args()
    main(args)
