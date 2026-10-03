# Python examples

Run these commands from the repository root in your project Python environment
with `rabitqlib>=0.5.2` and NumPy installed. Clustering and indexing both use RaBitQ
Library; FAISS is needed only for `compare_with_faiss.py`. For unreleased changes,
[install from a checkout](../../CONTRIBUTING.md#python-changes).

## IVF: cluster, then index

`kmeans_clustering.py` uses `--method qg` (the default) for QGKMeans graph
assignment or `--method rabitq` for RaBitQKMeans flat assignment. Both make exact
final assignments. The saved `.npz` contains float32 `centroids`, uint32
`cluster_ids`, and `metric` (`l2` or `ip`), and can be reused across index
configurations. Choose the method explicitly; the script does not switch based
on the cluster count.

```sh
python sample/python/kmeans_clustering.py --method qg --num-clusters 4096 \
  data/gist/gist_base.fvecs data/gist/clusters_4096_l2.npz

python sample/python/ivf_rabitq_indexing.py --total-bits 5 \
  --clusters data/gist/clusters_4096_l2.npz \
  data/gist/gist_base.fvecs data/gist/ivf_4096_5.index

python sample/python/ivf_rabitq_querying.py \
  data/gist/ivf_4096_5.index data/gist/gist_query.fvecs data/gist/gist_groundtruth.ivecs
```

Use the same base vectors in the same row order for clustering and indexing.
The loader checks the metric, dimensions, count, types, and ID range, but cannot
detect replaced or reordered vectors. See [QGKMeans](../../docs/docs/clustering.md#qgkmeans)
and [RaBitQKMeans](../../docs/docs/clustering.md#rabitqkmeans) for input limits and assignment
behavior.

For inner product, pass `--metric ip` to both stages and provide normalized
training vectors. Use queries and ground truth prepared for the same metric.
`--num-threads 0` uses the hardware thread count; each stage accepts its own limit.

## HNSW

Use [RaBitQKMeans](../../docs/docs/clustering.md#rabitqkmeans) for this small-cluster
example, with the same clustering file format:

```sh
python sample/python/kmeans_clustering.py --method rabitq --num-clusters 16 \
  data/gist/gist_base.fvecs data/gist/clusters_16_l2.npz

python sample/python/hnsw_rabitq_indexing.py --total-bits 5 --degree 16 \
  --ef-construction 200 --clusters data/gist/clusters_16_l2.npz \
  data/gist/gist_base.fvecs data/gist/hnsw_5.index
```

Query with `hnsw_rabitq_querying.py`. SymphonyQG builds directly from vectors;
use `symqg_indexing.py` and `symqg_querying.py` without a clustering stage.
