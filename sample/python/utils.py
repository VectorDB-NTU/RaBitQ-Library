import numpy as np

# ──────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────


def read_ivecs(filename: str) -> np.ndarray:
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} read, dim={d}")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename: str) -> np.ndarray:
    return read_ivecs(filename).view("float32")


# ──────────────────────────────────────────────
# Benchmarking utilities
# ──────────────────────────────────────────────


def compute_recall(ids: np.ndarray, gt: np.ndarray, topk: int) -> float:
    """Compute recall@topk: fraction of gt top-k found in returned top-k."""
    nq = ids.shape[0]
    total_correct = 0
    for i in range(nq):
        gt_set = set(gt[i, :topk].tolist())
        for j in range(topk):
            if ids[i, j] in gt_set:
                total_correct += 1
    return total_correct / (nq * topk)


# ──────────────────────────────────────────────
# Saved clustering results (no Faiss dependency)
# ──────────────────────────────────────────────


def load_clusters(
    filename: str, data_shape: tuple[int, int], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load results from faiss_clustering.py for the same data in the same order."""
    with np.load(filename, allow_pickle=False) as saved:
        if not {"centroids", "cluster_ids", "metric"}.issubset(saved.files):
            raise ValueError(
                "clustering file must contain centroids, cluster_ids and metric"
            )
        centroids = saved["centroids"]
        cluster_ids = saved["cluster_ids"]
        saved_metric = saved["metric"]
    if saved_metric.shape != () or saved_metric.item() != metric:
        raise ValueError("clustering metric does not match index metric")
    num_points, dim = data_shape
    if (
        centroids.dtype != np.float32
        or centroids.ndim != 2
        or centroids.shape[1] != dim
        or not 1 <= len(centroids) <= num_points
    ):
        raise ValueError(
            "centroids must be float32 with shape (num_clusters, data_dim)"
        )
    if not np.isfinite(centroids).all():
        raise ValueError("centroids must contain only finite values")
    if cluster_ids.dtype != np.uint32 or cluster_ids.shape != (num_points,):
        raise ValueError("cluster_ids must be uint32 with one ID per data point")
    if (cluster_ids >= len(centroids)).any():
        raise ValueError("cluster_ids contain an out-of-range cluster ID")
    return centroids, cluster_ids
