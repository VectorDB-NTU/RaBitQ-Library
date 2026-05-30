from time import time
from rabitqlib import HnswIndex
from utils import read_fvecs, cluster_data

# ──────────────────────────────────────────────
# Hardcoded configuration (mirrors C++ args)
# ──────────────────────────────────────────────
DATA_FILE       = "siftsmall/base.fvecs"  # path for data file
NUM_CLUSTERS    = 256           # number of clusters (K for IVF)
INDEX_FILE      = "hnsw.index"  # path for saving index
M               = 16            # degree bound for HNSW
EF_CONSTRUCTION = 200           # ef for indexing
TOTAL_BITS      = 8             # total number of bits for quantization
METRIC          = "l2"          # "l2" or "ip"
FASTER_QUANT    = False         # use faster quantization
NUM_THREADS     = 4             # number of threads for build
# ──────────────────────────────────────────────


def main() -> None:
    # 1. Load data
    data = read_fvecs(DATA_FILE)
    print(f"Data shape: {data.shape}")

    # 2. Cluster with FAISS
    centroids, cluster_ids = cluster_data(data, NUM_CLUSTERS, METRIC)
    print(f"Centroids: {centroids.shape}, cluster_ids: {cluster_ids.shape}")

    # 3. Build HNSW index
    n, dim = data.shape
    print(f"\nBuilding HNSW index: n={n}, dim={dim}, M={M}, "
          f"ef={EF_CONSTRUCTION}, bits={TOTAL_BITS}, metric={METRIC}")

    idx = HnswIndex(
        dim=dim,
        max_elements=n,
        M=M,
        ef_construction=EF_CONSTRUCTION,
        nbits=TOTAL_BITS,
        metric=METRIC,
    )

    t0 = time()
    idx.build(
        data,
        centroids,
        cluster_ids,
        num_threads=NUM_THREADS,
        fast_quantization=FASTER_QUANT,
    )
    print(f"Indexing time: {time() - t0:.2f}s")

    idx.save(INDEX_FILE)
    print(f"Index saved → {INDEX_FILE}")


if __name__ == "__main__":
    main()