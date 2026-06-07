import argparse
import numpy as np
from time import time
from rabitqlib import IvfIndex
from utils import read_fvecs, read_ivecs, compute_recall

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
USE_HACC    = True           # use high accuracy fastscan
TOPK        = 100            # top-k results
NUMBER_THREADS = 4           # number of threads for search
TEST_ROUNDS = 3              # number of test rounds
NPROBES = [5, 10, 20, 40, 80, 120, 200]
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load queries and ground truth
    queries = read_fvecs(args.query_file)
    gt      = read_ivecs(args.gt_file)
    nq      = queries.shape[0]
    print(f"Queries: {queries.shape}, GT: {gt.shape}")
    print(f"TopK: {args.topk}, use_hacc: {args.use_hacc}")

    # 2. Load index
    idx = IvfIndex.load(args.index_file)
    print(f"Index loaded — dim={idx.dim}, clusters={idx.num_clusters}")

    all_qps    = np.zeros((args.test_rounds, len(NPROBES)))
    all_recall = np.zeros((args.test_rounds, len(NPROBES)))

    # 3. Benchmark
    for r in range(args.test_rounds):
        for l, nprobe in enumerate(NPROBES):
            if nprobe > idx.num_clusters:
                print(f"nprobe {nprobe} is larger than number of clusters, "
                      f"will use nprobe = num_clusters ({idx.num_clusters}).")

            t0 = time()
            ids, _ = idx.search(queries, k=args.topk, nprobe=nprobe, high_accuracy=args.use_hacc, num_threads=args.num_threads)
            elapsed = time() - t0

            all_qps[r, l]    = nq / elapsed
            all_recall[r, l] = compute_recall(ids, gt, args.topk)

    avg_qps    = all_qps.mean(axis=0)
    avg_recall = all_recall.mean(axis=0)

    # 4. Print results table
    print(f"\n{'nprobe':<10}{'QPS':<14}{'Recall'}")
    print("-" * 35)
    for i, nprobe in enumerate(NPROBES):
        print(f"{nprobe:<10}{avg_qps[i]:<14.1f}{avg_recall[i]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaBitQ IVF Querying")

    parser.add_argument("index_file", type=str, help="Path to the IVF index file")
    parser.add_argument("query_file", type=str, help="Path to the query file")
    parser.add_argument("gt_file", type=str, help="Path to the ground truth file")
    parser.add_argument("--topk", dest="topk", type=int, metavar="INT", default=TOPK, help="Number of top results to retrieve")
    parser.add_argument("--use-hacc", dest="use_hacc", action="store_true", help="Use high accuracy fastscan method")
    parser.add_argument("--num-threads", dest="num_threads", type=int, metavar="INT", default=NUMBER_THREADS, help="Number of threads for search")
    parser.add_argument("--test-rounds", dest="test_rounds", type=int, metavar="INT", default=TEST_ROUNDS, help="Number of test rounds for averaging")
    args = parser.parse_args()

    main(args)