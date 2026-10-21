import argparse
from time import time

from rabitqlib import FinalAssignmentMode, QGKMeans, RaBitQKMeans
from utils.io import read_fvecs, write_fvecs, write_ivecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster vectors and save fvecs/ivecs outputs."
    )
    parser.add_argument("data_path")
    parser.add_argument("num_clusters", type=int)
    parser.add_argument("centroids_path")
    parser.add_argument("cluster_id_path")
    parser.add_argument(
        "metric",
        nargs="?",
        default="l2",
        type=str.lower,
        choices=["l2", "ip", "innerproduct"],
    )
    parser.add_argument(
        "--method",
        choices=["rabitq", "qg"],
        default="qg",
        help="rabitq for few centroids; qg for many",
    )
    args = parser.parse_args()
    data_path = args.data_path
    K = args.num_clusters
    centroids_path = args.centroids_path
    cluster_id_path = args.cluster_id_path
    metric = "l2" if args.metric == "l2" else "ip"
    clustering_type = RaBitQKMeans if args.method == "rabitq" else QGKMeans
    print(f"Using {metric.upper()} metric with {clustering_type.__name__}")

    X = read_fvecs(data_path)

    dim = X.shape[1]

    t1 = time()

    # cluster data vectors
    kmeans = clustering_type(
        dim,
        K,
        spherical=metric == "ip",
        final_assignment=FinalAssignmentMode.Exact,
        verbose=True,
    )
    kmeans.train(X)

    t2 = time()
    print(f"Time for training ivf {t2 - t1} secs")

    centroids = kmeans.centroids
    cluster_id = kmeans.assignments.reshape(-1, 1)

    write_ivecs(cluster_id_path, cluster_id)
    write_fvecs(centroids_path, centroids)
