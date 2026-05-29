import numpy as np
from rabitqlib import *

def main():
    # Load index and run some queries
    idx = HnswIndex.load("hnsw.index", metric="l2")
    queries = np.random.random((10, idx.dim)).astype(np.float32)
    ids, dists = idx.search(queries, k=5, ef=50, num_threads=2)
    print("ids:\n", ids)
    print("dists:\n", dists)

if __name__ == '__main__':
    main()
