"""Train RaBitQKMeans or QGKMeans and save clusters for the index examples."""

import argparse
from time import time

import numpy as np
from rabitqlib import FinalAssignmentMode, QGKMeans, RaBitQKMeans
from rabitqlib._rabitqlib import _available_cpu_count

from utils import read_fvecs


def cluster_data(
    data: np.ndarray,
    num_clusters: int,
    metric: str,
    num_threads: int,
    method: str = "qg",
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= num_clusters <= len(data):
        raise ValueError("num_clusters must be between 1 and the number of data points")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    if metric not in ("l2", "ip"):
        raise ValueError("metric must be l2 or ip")
    if method not in ("rabitq", "qg"):
        raise ValueError("method must be rabitq or qg")
    clustering_type = RaBitQKMeans if method == "rabitq" else QGKMeans
    available_threads = _available_cpu_count()
    threads = (
        available_threads if num_threads == 0 else min(num_threads, available_threads)
    )
    print(f"Clustering metric: {metric.upper()}, threads: {threads}")
    kmeans = clustering_type(
        data.shape[1],
        num_clusters,
        spherical=metric == "ip",
        num_threads=threads,
        final_assignment=FinalAssignmentMode.Exact,
        verbose=True,
    )
    start = time()
    kmeans.train(data)
    print(f"{clustering_type.__name__} training time: {time() - start:.2f}s")
    return kmeans.centroids, kmeans.assignments


def main(args) -> None:
    data = read_fvecs(args.data_file)
    centroids, cluster_ids = cluster_data(
        data, args.num_clusters, args.metric, args.num_threads, args.method
    )
    # Open the exact requested path; np.savez otherwise appends .npz automatically.
    with open(args.clusters_file, "wb") as output:
        np.savez(
            output, centroids=centroids, cluster_ids=cluster_ids, metric=args.metric
        )
    print(f"Clusters saved: {args.clusters_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", help="Input vectors in fvecs format")
    parser.add_argument(
        "clusters_file", help="Output clustering file (NumPy npz format)"
    )
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument(
        "--method",
        choices=["rabitq", "qg"],
        default="qg",
        help="rabitq for few centroids; qg for many (requires k > graph_degree)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Threads for clustering (0: available CPUs; larger requests are capped)",
    )
    main(parser.parse_args())
