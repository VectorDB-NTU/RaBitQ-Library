import numpy as np
from time import time
from rabitqlib import SymqgIndex
from utils import read_fvecs, read_ivecs, compute_recall

# ──────────────────────────────────────────────
# Hardcoded configuration (mirrors C++ args)
# ──────────────────────────────────────────────
INDEX_FILE  = "symqg.index"    # arg1: path for index
QUERY_FILE  = "siftsmall/query.fvecs"   # arg2: path for query file
GT_FILE     = "siftsmall/groundtruth.ivecs"      # arg3: path for groundtruth file
METRIC      = "l2"            # arg4: "l2" or "ip"
TOPK        = 10              # arg5: top-k results
NUM_THREADS = 1               # arg6: number of threads

EFS         = [10, 20, 40, 80, 120, 200, 400, 600, 800, 1000, 1500, 2000]
TEST_ROUNDS = 3
# ──────────────────────────────────────────────





def main() -> None:
    # 1. Load queries and ground truth
    queries = read_fvecs(QUERY_FILE)
    gt      = read_ivecs(GT_FILE)
    nq      = queries.shape[0]
    print(f"Queries: {queries.shape}, GT: {gt.shape}")

    # 2. Load index
    idx = SymqgIndex.load(INDEX_FILE)
    print(f"Index loaded — dim={idx.dim}, metric={METRIC}")
    print(f"TopK: {TOPK}")

    print("\nsearch start >.....\n")

    all_qps    = np.zeros((TEST_ROUNDS, len(EFS)))
    all_recall = np.zeros((TEST_ROUNDS, len(EFS)))

    for i_probe, ef in enumerate(EFS):
        for r in range(TEST_ROUNDS):
            t0 = time()
            ids, _ = idx.search(queries, k=TOPK, ef=ef, num_threads=NUM_THREADS)
            elapsed = time() - t0  # seconds

            qps    = nq / elapsed
            recall = compute_recall(ids, gt, TOPK)

            all_qps[r, i_probe]    = qps
            all_recall[r, i_probe] = recall

    avg_qps    = all_qps.mean(axis=0)
    avg_recall = all_recall.mean(axis=0)

    # 3. Print results table
    print(f"{'EF':<8}{'QPS':<14}{'Recall':<12}{'Per-round QPS'}")
    print("-" * 60)
    for i, ef in enumerate(EFS):
        per_round = "\t".join(f"{all_qps[r, i]:.1f}" for r in range(TEST_ROUNDS))
        print(f"{ef:<8}{avg_qps[i]:<14.1f}{avg_recall[i]:<12.4f}{per_round}")


if __name__ == "__main__":
    main()
