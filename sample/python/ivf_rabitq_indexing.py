import argparse
from time import time
from rabitqlib import IvfIndex
from utils import read_fvecs, cluster_data

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
NUM_CLUSTERS = 256          # number of clusters (K for IVF)
TOTAL_BITS   = 8            # total number of bits for quantization
METRIC       = "l2"         # "l2" or "ip"
FASTER_QUANT = True         # use faster quantization
NUM_THREADS  = 1            # number of threads for building the index
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load data
    data = read_fvecs(args.data_file)
    n, dim = data.shape
    print(f"Data loaded")
    print(f"\tN: {n}")
    print(f"\tDIM: {dim}")

    # 2. Cluster with FAISS
    centroids, cluster_ids = cluster_data(data, args.num_clusters, args.metric)
    print(f"Centroids: {centroids.shape}, cluster_ids: {cluster_ids.shape}")

    # 3. Build IVF index
    print(f"\nBuilding IVF index: bits={args.total_bits}, metric={args.metric}, "
          f"faster_quant={args.faster_quant}")

    idx = IvfIndex(
        dim=dim,
        max_elements=n,
        num_clusters=args.num_clusters,
        nbits=args.total_bits,
        metric=args.metric,
    )

    t0 = time()
    idx.build(data, centroids, cluster_ids, fast_quantization=args.faster_quant)
    elapsed_min = (time() - t0) / 60

    print("IVF constructed")
    idx.save(args.index_file)
    print(f"Indexing time: {elapsed_min:.4f} min")
    print(f"Index saved → {args.index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaBitQ IVF Manager")

    parser.add_argument("data_file", type=str, help="Path to the data file")
    parser.add_argument("index_file", type=str, help="Path to save the index")
    parser.add_argument("--num-clusters", dest="num_clusters", type=int, metavar="INT", default=NUM_CLUSTERS, help="Number of clusters (K for IVF)")
    parser.add_argument("--total-bits", dest="total_bits", type=int, metavar="INT", default=TOTAL_BITS, help="Total number of bits for quantization")
    parser.add_argument("--metric", dest="metric", type=str, default=METRIC, choices=["l2", "ip"], help="Distance metric (l2 or ip)")
    parser.add_argument("--faster-quant", dest="faster_quant", action="store_true", help="Use faster quantization method")
    parser.add_argument("--num-threads", dest="num_threads", type=int, metavar="INT", default=NUM_THREADS, help="Number of threads for building the index")
    args = parser.parse_args()
    main(args)