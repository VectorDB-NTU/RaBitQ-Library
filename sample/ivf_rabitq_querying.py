import numpy as np
from rabitqlib import *

def main():
    idx = IvfIndex.load("ivf.index")
    queries = np.random.random((10, idx.dim)).astype(np.float32)
    ids, dists = idx.search(queries, k=10, nprobe=4, high_accuracy=True, num_threads=2)
    print("ids:\n", ids)
    print("dists:\n", dists)

if __name__ == '__main__':
    main()
