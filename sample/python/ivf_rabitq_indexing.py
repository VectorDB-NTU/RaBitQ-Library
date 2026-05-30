from time import time
from rabitqlib import IvfIndex
from utils import read_fvecs, cluster_data

# ──────────────────────────────────────────────
# Hardcoded configuration (mirrors C++ args)
# ──────────────────────────────────────────────
DATA_FILE    = "siftsmall/base.fvecs"  # path for data file
NUM_CLUSTERS = 256            # arg2: number of clusters (K for IVF)
INDEX_FILE   = "ivf.index"   # arg5: path for saving index
TOTAL_BITS   = 8             # arg4: total number of bits for quantization
METRIC       = "l2"          # arg6: "l2" or "ip"
FASTER_QUANT = True         # arg7: use faster quantization
# ──────────────────────────────────────────────


def main() -> None:
    # 1. Load data
    data = read_fvecs(DATA_FILE)
    n, dim = data.shape
    print(f"Data loaded")
    print(f"\tN: {n}")
    print(f"\tDIM: {dim}")

    # 2. Cluster with FAISS
    centroids, cluster_ids = cluster_data(data, NUM_CLUSTERS, METRIC)
    print(f"Centroids: {centroids.shape}, cluster_ids: {cluster_ids.shape}")

    # 3. Build IVF index
    print(f"\nBuilding IVF index: bits={TOTAL_BITS}, metric={METRIC}, "
          f"faster_quant={FASTER_QUANT}")

    idx = IvfIndex(
        dim=dim,
        max_elements=n,
        num_clusters=NUM_CLUSTERS,
        nbits=TOTAL_BITS,
        metric=METRIC,
    )

    t0 = time()
    idx.build(data, centroids, cluster_ids, fast_quantization=FASTER_QUANT)
    elapsed_min = (time() - t0) / 60

    print("IVF constructed")
    idx.save(INDEX_FILE)
    print(f"Indexing time: {elapsed_min:.4f} min")
    print(f"Index saved → {INDEX_FILE}")


if __name__ == "__main__":
    main()