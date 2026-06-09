import argparse
from time import time
from rabitqlib import SymqgIndex
from utils import read_fvecs

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
MAX_DEGREE      = 32            # degree bound for SymphonyQG
EF_CONSTRUCTION = 200           # ef for indexing
METRIC          = "l2"          # "l2" or "ip"
NUM_THREADS     = 1             # number of threads for build
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load data
    data = read_fvecs(args.data_file)
    print(f"Data shape: {data.shape}")

    # 3. Build SymphonyQG index
    n, dim = data.shape
    print(f"\nBuilding SymphonyQG index: n={n}, dim={dim}, MaxDegree={args.max_degree}, "
          f"ef={args.ef_construction}, metric={args.metric}")

    idx = SymqgIndex(dim=dim, max_degree=args.max_degree, metric=args.metric)
    
    t0 = time()
    idx.build(data, ef_construction=args.ef_construction, num_threads=args.num_threads)
    print(f"Indexing time: {time() - t0:.2f}s")

    idx.save(args.index_file)
    print(f"Index saved → {args.index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaBitQ SymphonyQG Manager")
    parser.add_argument("data_file", type=str, help="Path to the data file")
    parser.add_argument("index_file", type=str, help="Path to save the index")
    parser.add_argument("--max-degree", dest="max_degree", type=int, metavar="INT", default=MAX_DEGREE, help="Degree bound for SymphonyQG")
    parser.add_argument("--ef-construction", dest="ef_construction", type=int, metavar="INT", default=EF_CONSTRUCTION, help="EF parameter for index construction")
    parser.add_argument("--metric", dest="metric", type=str, default=METRIC, choices=["l2", "ip"], help="Distance metric (l2 or ip)")
    parser.add_argument("--num-threads", dest="num_threads", type=int, metavar="INT", default=NUM_THREADS, help="Number of threads for building the index")

    args = parser.parse_args()
    main(args)
