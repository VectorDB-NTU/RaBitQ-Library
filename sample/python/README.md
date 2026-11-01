# Python examples

Run these commands from the repository root in your project Python environment
with `rabitqlib>=0.8.1` and NumPy installed. When working from source,
[install the library from the same checkout](../../.github/CONTRIBUTING.md#python-changes)
so the examples and native extension use matching APIs. Clustering and indexing
both use RaBitQ Library; FAISS is needed only for `compare_with_faiss.py`.

## IVF-RaBitQ

The indexing script trains the centroids itself unless you pass a clustering
file:

```sh
python sample/python/ivf_rabitq_indexing.py --total-bits 5 --num-clusters 4096 \
  data/gist/gist_base.fvecs data/gist/ivf_4096_5.index

python sample/python/ivf_rabitq_querying.py \
  data/gist/ivf_4096_5.index data/gist/gist_query.fvecs data/gist/gist_groundtruth.ivecs
```

To reuse one clustering across index configurations, cluster once and pass the
file with `--clusters`. `kmeans_clustering.py` uses `--method qg` (the default) for QGKMeans graph
assignment or `--method rabitq` for RaBitQKMeans flat assignment. Both make exact
final assignments. The saved `.npz` contains float32 `centroids`, uint32
`cluster_ids`, and `metric` (`l2` or `ip`). Choose the method explicitly; the script does not switch based
on the cluster count.

```sh
python sample/python/kmeans_clustering.py --method qg --num-clusters 4096 \
  data/gist/gist_base.fvecs data/gist/clusters_4096_l2.npz

python sample/python/ivf_rabitq_indexing.py --total-bits 5 \
  --clusters data/gist/clusters_4096_l2.npz \
  data/gist/gist_base.fvecs data/gist/ivf_4096_5.index
```

Use the same base vectors in the same row order for clustering and indexing.
The loader checks the metric, dimensions, count, types, and ID range, but cannot
detect replaced or reordered vectors. See [QGKMeans](../../docs/docs/clustering.md#qgkmeans)
and [RaBitQKMeans](../../docs/docs/clustering.md#rabitqkmeans) for input limits and assignment
behavior.

For inner product, pass `--metric ip` to every stage and provide normalized
vectors. Use queries and ground truth prepared for the same metric.
`--num-threads 0` uses the library's available logical CPU count, capped by
`OMP_NUM_THREADS`; each stage accepts its own limit. See
[threading](../../docs/docs/index/usage.md#threading-and-file-paths) for affinity
and OpenMP placement behavior.

## HNSW-RaBitQ

The indexing script trains 16 centroids by default (`--num-clusters`):

```sh
python sample/python/hnsw_rabitq_indexing.py --total-bits 5 --degree 16 \
  --ef-construction 200 data/gist/gist_base.fvecs data/gist/hnsw_5.index
```

It also accepts `--clusters` with a file from `kmeans_clustering.py`; use
`--method rabitq` there for a small cluster count. With more than one thread
(`--num-threads`), the graph links vectors in a nondeterministic order; pass
`--num-threads 1` for a reproducible index.

Query with `hnsw_rabitq_querying.py`. SymphonyQG builds directly from vectors;
use `symqg_indexing.py` and `symqg_querying.py` without a clustering stage.

## Separate centroid and assignment files

The helper formerly at `python/ivf.py` now lives at
`sample/python/clustering_cli/ivf.py`. It writes centroids as `.fvecs` and
assignments as `.ivecs` for workflows using separate files:

```sh
python sample/python/clustering_cli/ivf.py data/gist/gist_base.fvecs 4096 \
  data/gist/centroids.fvecs data/gist/assignments.ivecs --method qg
```

Use `--method rabitq` for flat assignment or the optional positional `ip` metric
with normalized input. For reusable Python index inputs in `.npz`, use
`kmeans_clustering.py` above.
