import argparse
import numpy as np
from time import time
from rabitqlib import SymqgIndex
from utils import read_fvecs, read_ivecs, compute_recall

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
METRIC      = "l2"            # "l2" or "ip"
TOPK        = 10              # top-k results
NUM_THREADS = 1               # number of threads
EFS         = [10, 20, 40, 80, 120, 200, 400, 600, 800, 1000, 1500, 2000]
TEST_ROUNDS = 3
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load queries and ground truth
    queries = read_fvecs(args.query_file)
    gt      = read_ivecs(args.gt_file)
    nq      = queries.shape[0]
    print(f"Queries: {queries.shape}, GT: {gt.shape}")

    # 2. Load index
    idx = SymqgIndex.load(args.index_file)
    print(f"Index loaded — dim={idx.dim}, metric={args.metric}")
    print(f"TopK: {args.topk}")

    print("\nsearch start >.....\n")

    all_qps    = np.zeros((args.test_rounds, len(EFS)))
    all_recall = np.zeros((args.test_rounds, len(EFS)))

    for i_probe, ef in enumerate(EFS):
        for r in range(args.test_rounds):
            t0 = time()
            ids, _ = idx.search(queries, k=args.topk, ef=ef, num_threads=args.num_threads)
            elapsed = time() - t0  # seconds

            qps    = nq / elapsed
            recall = compute_recall(ids, gt, args.topk)

            all_qps[r, i_probe]    = qps
            all_recall[r, i_probe] = recall

    avg_qps    = all_qps.mean(axis=0)
    avg_recall = all_recall.mean(axis=0)

    # 3. Print results table
    print(f"{'EF':<8}{'QPS':<14}{'Recall'}")
    print("-" * 35)
    for i, ef in enumerate(EFS):
        print(f"{ef:<8}{avg_qps[i]:<14.1f}{avg_recall[i]:<12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaBitQ SymphonyQG Querying Manager")
    parser.add_argument("index_file", type=str, help="Path to the SymphonyQG index file")
    parser.add_argument("query_file", type=str, help="Path to the query file")
    parser.add_argument("gt_file", type=str, help="Path to the ground truth file")
    parser.add_argument("--metric", dest="metric", type=str, default=METRIC, choices=["l2", "ip"], help="Distance metric (l2 or ip)")
    parser.add_argument("--topk", dest="topk", type=int, metavar="INT", default=TOPK, help="Number of top-k results to retrieve")
    parser.add_argument("--num-threads", dest="num_threads", type=int, metavar="INT", default=NUM_THREADS, help="Number of threads for searching")
    parser.add_argument("--test-rounds", dest="test_rounds", type=int, metavar="INT", default=TEST_ROUNDS, help="Number of test rounds to average results")
    args = parser.parse_args()
    main(args)
