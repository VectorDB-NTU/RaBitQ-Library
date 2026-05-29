import numpy as np
from rabitqlib import *

def main():
    dim = 64
    n = 2000
    max_degree = 32

    data = np.random.random((n, dim)).astype(np.float32)

    idx = SymqgIndex(dim=dim, max_degree=max_degree, metric="l2")
    idx.build(data, ef_construction=200, num_threads=4)
    idx.save("symqg.index")
    print("Saved symqg.index")

if __name__ == '__main__':
    main()
