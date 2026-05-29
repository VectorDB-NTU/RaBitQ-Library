import numpy as np
from rabitqlib import *

def main():
    idx = SymqgIndex.load("symqg.index")
    queries = np.random.random((10, idx.dim)).astype(np.float32)
    ids, dists = idx.search(queries, k=10, ef=100, num_threads=2)
    print("ids:\n", ids)
    print("dists:\n", dists)

if __name__ == '__main__':
    main()
