"""Run Faiss in its own process and save clusters for the RaBitQ examples."""

import argparse
import os
from time import time

import faiss
import numpy as np

from utils import read_fvecs


def cluster_data(
    data: np.ndarray, num_clusters: int, metric: str, num_threads: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= num_clusters <= len(data):
        raise ValueError("num_clusters must be between 1 and the number of data points")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    if metric not in ("l2", "ip"):
        raise ValueError("metric must be l2 or ip")
    hardware_threads = os.cpu_count() or 1
    threads = (
        hardware_threads if num_threads == 0 else min(num_threads, hardware_threads)
    )
    faiss.omp_set_num_threads(threads)
    print(f"Clustering metric: {metric.upper()}, threads: {threads}")
    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(data.shape[1], f"IVF{num_clusters},Flat", faiss_metric)
    index.verbose = True
    start = time()
    index.train(data)
    print(f"IVF training time: {time() - start:.2f}s")
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    _, labels = index.quantizer.search(data, 1)
    return centroids, labels.ravel().astype(np.uint32)


def main(args) -> None:
    data = read_fvecs(args.data_file)
    centroids, cluster_ids = cluster_data(
        data, args.num_clusters, args.metric, args.num_threads
    )
    # Open the exact requested path; np.savez otherwise appends .npz automatically.
    with open(args.clusters_file, "wb") as output:
        np.savez(
            output, centroids=centroids, cluster_ids=cluster_ids, metric=args.metric
        )
    print(f"Clusters saved → {args.clusters_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", help="Input vectors in fvecs format")
    parser.add_argument(
        "clusters_file", help="Output clustering file (NumPy npz format)"
    )
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Threads for clustering (0: hardware count; larger requests are capped)",
    )
    main(parser.parse_args())
