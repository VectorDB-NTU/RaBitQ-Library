import numpy as np
from rabitqlib import *

def main():
    dim = 64
    n = 2000
    num_clusters = 32

    data = np.random.random((n, dim)).astype(np.float32)
    centroids = data[:num_clusters].copy()
    cluster_ids = np.random.randint(0, num_clusters, size=(n,), dtype=np.uint32)

    idx = IvfIndex(dim=dim, max_elements=n, num_clusters=num_clusters, nbits=8, metric="l2")
    idx.build(data, centroids, cluster_ids, fast_quantization=False)
    idx.save("ivf.index")
    print("Saved ivf.index")

if __name__ == '__main__':
    main()
