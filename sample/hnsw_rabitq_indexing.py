import numpy as np
from rabitqlib import *

def main():
    dim = 64
    n = 2000
    num_clusters = 16

    # Random data and simple clustering
    data = np.random.random((n, dim)).astype(np.float32)
    centroids = data[:num_clusters].copy()
    cluster_ids = np.random.randint(0, num_clusters, size=(n,), dtype=np.uint32)

    idx = HnswIndex(dim=dim, max_elements=n, M=16, ef_construction=200, nbits=8, metric="l2")
    idx.build(data, centroids, cluster_ids, num_threads=4, fast_quantization=False)
    idx.save("hnsw.index")
    print("Saved hnsw.index")

if __name__ == '__main__':
    main()
