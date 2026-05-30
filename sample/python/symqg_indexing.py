from time import time
from rabitqlib import SymqgIndex
from utils import read_fvecs

# ──────────────────────────────────────────────
# Hardcoded configuration (mirrors C++ args)
# ──────────────────────────────────────────────
DATA_FILE       = "siftsmall/base.fvecs"  # path for data file
INDEX_FILE      = "symqg.index"  # path for saving index
MAX_DEGREE      = 32            # degree bound for SymphonyQG
EF_CONSTRUCTION = 200           # ef for indexing
METRIC          = "l2"          # "l2" or "ip"
NUM_THREADS     = 1             # number of threads for build
# ──────────────────────────────────────────────


def main() -> None:
    # 1. Load data
    data = read_fvecs(DATA_FILE)
    print(f"Data shape: {data.shape}")

    # 3. Build SymphonyQG index
    n, dim = data.shape
    print(f"\nBuilding SymphonyQG index: n={n}, dim={dim}, MaxDegree={MAX_DEGREE}, "
          f"ef={EF_CONSTRUCTION}, metric={METRIC}")

    idx = SymqgIndex(dim=dim, max_degree=MAX_DEGREE, metric="l2")
    
    t0 = time()
    idx.build(data, ef_construction=EF_CONSTRUCTION, num_threads=NUM_THREADS)
    print(f"Indexing time: {time() - t0:.2f}s")

    idx.save(INDEX_FILE)
    print(f"Index saved → {INDEX_FILE}")


if __name__ == "__main__":
    main()
