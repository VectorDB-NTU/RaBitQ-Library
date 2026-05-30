import numpy as np
from time import time
from rabitqlib import IvfIndex
from utils import read_fvecs, read_ivecs, compute_recall

# ──────────────────────────────────────────────
# Hardcoded configuration (mirrors C++ args)
# ──────────────────────────────────────────────
INDEX_FILE  = "ivf.index"    # arg1: path for index
QUERY_FILE  = "siftsmall/query.fvecs"  # arg2: path for query file
GT_FILE     = "siftsmall/groundtruth.ivecs"     # arg3: path for groundtruth file
USE_HACC    = True           # arg4: use high accuracy fastscan
TOPK        = 100            # top-k results
TEST_ROUNDS = 3              # number of test rounds

NPROBES = [5, 10, 20, 40, 80, 120, 200]
# ──────────────────────────────────────────────


def main() -> None:
    # 1. Load queries and ground truth
    queries = read_fvecs(QUERY_FILE)
    gt      = read_ivecs(GT_FILE)
    nq      = queries.shape[0]
    print(f"Queries: {queries.shape}, GT: {gt.shape}")
    print(f"TopK: {TOPK}, use_hacc: {USE_HACC}")

    # 2. Load index
    idx = IvfIndex.load(INDEX_FILE)
    print(f"Index loaded — dim={idx.dim}, clusters={idx.num_clusters}")

    all_qps    = np.zeros((TEST_ROUNDS, len(NPROBES)))
    all_recall = np.zeros((TEST_ROUNDS, len(NPROBES)))

    # 3. Benchmark
    for r in range(TEST_ROUNDS):
        for l, nprobe in enumerate(NPROBES):
            if nprobe > idx.num_clusters:
                print(f"nprobe {nprobe} is larger than number of clusters, "
                      f"will use nprobe = num_clusters ({idx.num_clusters}).")

            t0 = time()
            ids, _ = idx.search(queries, k=TOPK, nprobe=nprobe, high_accuracy=USE_HACC)
            elapsed = time() - t0

            all_qps[r, l]    = nq / elapsed
            all_recall[r, l] = compute_recall(ids, gt, TOPK)

    avg_qps    = all_qps.mean(axis=0)
    avg_recall = all_recall.mean(axis=0)

    # 4. Print results table
    print(f"\n{'nprobe':<10}{'QPS':<14}{'Recall'}")
    print("-" * 35)
    for i, nprobe in enumerate(NPROBES):
        print(f"{nprobe:<10}{avg_qps[i]:<14.1f}{avg_recall[i]:.4f}")


if __name__ == "__main__":
    main()