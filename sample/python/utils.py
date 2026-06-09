import numpy as np
import faiss
from time import time


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
# Clustering
# ──────────────────────────────────────────────

def cluster_data(
    X: np.ndarray,
    K: int,
    metric_str: str = "l2",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster X into K clusters using FAISS IVF.

    Returns
    -------
    centroids   : np.ndarray, shape (K, dim), float32
    cluster_ids : np.ndarray, shape (n,),     uint32
    """
    dim = X.shape[1]

    if metric_str == "ip":
        metric = faiss.METRIC_INNER_PRODUCT
        print("Clustering metric: InnerProduct")
    else:
        metric = faiss.METRIC_L2
        print("Clustering metric: L2")

    index = faiss.index_factory(dim, f"IVF{K},Flat", metric)
    index.verbose = True

    t0 = time()
    index.train(X)
    print(f"IVF training time: {time() - t0:.2f}s")

    centroids = index.quantizer.reconstruct_n(0, index.nlist)       # (K, dim) float32
    _, cluster_ids_2d = index.quantizer.search(X, 1)               # (n, 1)   int64
    cluster_ids = cluster_ids_2d.flatten().astype(np.uint32)        # (n,)     uint32

    return centroids, cluster_ids