import argparse
from time import time

from rabitqlib import IvfIndex

from utils import load_clusters, read_fvecs

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
TOTAL_BITS = 8  # total number of bits for quantization
METRIC = "l2"  # "l2" or "ip"
FASTER_QUANT = True  # use faster quantization
NUM_THREADS = 16  # number of threads for building the index
NUM_CLUSTERS = 4096  # centroids to train when no clustering file is given
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load data
    data = read_fvecs(args.data_file)
    n, dim = data.shape
    print("Data loaded")
    print(f"\tN: {n}")
    print(f"\tDIM: {dim}")

    # 2. Train centroids in the index, or reuse a kmeans_clustering.py file.
    print(
        f"\nBuilding IVF index: bits={args.total_bits}, metric={args.metric}, "
        f"num_threads={args.num_threads}, faster_quant={args.faster_quant}"
    )
    t0 = time()
    if args.clusters is None:
        idx = IvfIndex(
            dim, args.num_clusters, nbits=args.total_bits, metric=args.metric
        )
        idx.train(data, num_threads=args.num_threads)
        cluster_ids = None
        print(f"Trained {args.num_clusters} centroids")
    else:
        centroids, cluster_ids = load_clusters(args.clusters, data.shape, args.metric)
        print(f"Centroids: {centroids.shape}, cluster_ids: {cluster_ids.shape}")
        idx = IvfIndex(dim, len(centroids), nbits=args.total_bits, metric=args.metric)
        idx.set_centroids(centroids)

    # 3. Add the vectors
    idx.add(
        data,
        cluster_ids=cluster_ids,
        num_threads=args.num_threads,
        fast_quantization=args.faster_quant,
    )
    elapsed_min = (time() - t0) / 60

    print("IVF constructed")
    idx.save(args.index_file)
    print(f"Indexing time: {elapsed_min:.4f} min")
    print(f"Index saved: {args.index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IVF RaBitQ Manager")

    parser.add_argument("data_file", type=str, help="Path to the data file")
    parser.add_argument("index_file", type=str, help="Path to save the index")
    parser.add_argument(
        "--clusters",
        default=None,
        help=(
            "Clustering file produced by kmeans_clustering.py for this data and "
            "metric; without it, the index trains its own centroids"
        ),
    )
    parser.add_argument(
        "--num-clusters",
        dest="num_clusters",
        type=int,
        metavar="INT",
        default=NUM_CLUSTERS,
        help="Number of centroids to train when --clusters is not given",
    )
    parser.add_argument(
        "--total-bits",
        dest="total_bits",
        type=int,
        metavar="INT",
        default=TOTAL_BITS,
        help="Quantization bits (1-9), or 32 to store raw vectors for reranking",
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
        "--faster-quant",
        dest="faster_quant",
        action="store_true",
        help="Use faster quantization method",
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
