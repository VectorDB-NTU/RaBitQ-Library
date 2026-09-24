# Python examples

Run these commands from the repository root in your project Python environment.
Install `rabitqlib` and NumPy; the clustering step additionally needs `faiss-cpu`.

## IVF: cluster, then index

Faiss clustering and RaBitQ indexing run in separate Python processes. This keeps
their OpenMP runtimes separate, including when both packages come from pip wheels.
The intermediate `.npz` contains float32 `centroids`, uint32 `cluster_ids`, and the
`metric` (`l2` or `ip`). It can be reused for different RaBitQ bit widths.

```sh
python sample/python/faiss_clustering.py --num-clusters 4096 \
  data/gist/gist_base.fvecs data/gist/clusters_4096_l2.npz

python sample/python/ivf_rabitq_indexing.py --total-bits 5 \
  --clusters data/gist/clusters_4096_l2.npz \
  data/gist/gist_base.fvecs data/gist/ivf_4096_5.index

python sample/python/ivf_rabitq_querying.py \
  data/gist/ivf_4096_5.index data/gist/gist_query.fvecs data/gist/gist_groundtruth.ivecs
```

The indexing command now takes `--clusters` instead of `--num-clusters`; its cluster
count comes from the saved centroids. Always use the same base vectors in the same
row order for both stages. The loader checks the metric, dimensions, point count,
array types, and cluster ID range; it cannot detect reordered or replaced vectors.

For inner product, pass `--metric ip` to **both** clustering and indexing, and use
ground truth for that metric when evaluating recall. `--num-threads 0` selects the
hardware thread count; larger requests are capped. Each stage has its own setting.

## HNSW

HNSW uses the same saved-cluster format. Choose its cluster count in the Faiss stage:

```sh
python sample/python/faiss_clustering.py --num-clusters 16 \
  data/gist/gist_base.fvecs data/gist/clusters_16_l2.npz

python sample/python/hnsw_rabitq_indexing.py --total-bits 5 --degree 16 \
  --ef-construction 200 --clusters data/gist/clusters_16_l2.npz \
  data/gist/gist_base.fvecs data/gist/hnsw_5.index
```

HNSW querying is unchanged. SymphonyQG does not need Faiss clustering; run
`symqg_indexing.py` and `symqg_querying.py` directly. All querying scripts and the
shared file/recall utilities are independent of Faiss.

Run the stages as separate commands, not by importing both libraries into one
Python process or notebook kernel. This separation fixes the example workflow;
it does not change native-runtime coexistence for applications importing both.
