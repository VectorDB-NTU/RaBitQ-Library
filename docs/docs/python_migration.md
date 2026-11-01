# Migrating Python code from 0.7 to 0.8

In 0.8, `IvfIndex`, `HnswIndex` and `SymqgIndex` follow a `train` → `add` →
`search` workflow, as Faiss indexes do. IVF-RaBitQ and HNSW-RaBitQ learn their
own centroids, so you no longer run k-means and pass the centroids and
assignments to `build`; SymphonyQG has nothing to train and builds its graph in
`add`. Code written for 0.7
still runs in 0.8. Each old form raises a `DeprecationWarning` at your call site
and will be removed in **0.9**.

Unless an item below says otherwise, a 0.7 call that raises a deprecation
warning keeps its 0.7 behavior. Calls that are valid in both versions, such as
`add(x)` and `search`, follow the 0.8 rules.

## IVF-RaBitQ and HNSW-RaBitQ

### Old and new calls

| 0.7 | 0.8 |
| --- | --- |
| `IvfIndex(dim, max_elements, num_clusters, nbits, metric, initializer)` | `IvfIndex(dim, num_clusters, nbits=..., metric=..., initializer=..., seed=None)`; storage grows on `add` |
| `IvfIndex(dim=..., max_elements=..., num_clusters=..., nbits=...)` | the same, without `max_elements` |
| `HnswIndex(dim, max_elements, M=16, ef_construction=200, nbits=8, metric="l2", random_seed=100)` | `HnswIndex(dim, nbits=..., num_clusters=16, M=16, ef_construction=200, metric="l2", seed=100)`; capacity grows on `add`, and `resize(n)` preallocates it |
| `HnswIndex(..., random_seed=s)` | `HnswIndex(..., seed=s)` |
| `kmeans = RaBitQKMeans(dim, k)`<br>`kmeans.train(x)`<br>`index.build(x, kmeans.centroids, kmeans.assignments)` | `index.train(x)`<br>`index.add(x)` |
| `index.build(x, centroids, cluster_ids)` | `index.set_centroids(centroids)`<br>`index.add(x, cluster_ids=cluster_ids)` |
| `index.add(x, cluster_ids, num_threads, fast_quantization)` | `index.add(x, cluster_ids=..., num_threads=..., fast_quantization=...)` |
| `index.is_built` | `index.is_trained`, or `len(index) > 0` for "holds vectors" |

Except for `dim` and IVF-RaBitQ's `num_clusters`, constructor arguments are
keyword-only in 0.8, and `nbits` has no default. HNSW-RaBitQ used to default to
`nbits=8`; the old form keeps that default.

`train` uses the settings in `index.cp`: 25 iterations, at most 256 sampled
vectors per centroid, spherical k-means for `metric="ip"`, and the index seed.
See [IVF-RaBitQ training](index/ivf.md#training-centroids-in-python) for the
details and for `set_centroids`. The [clustering classes](clustering.md) still
work if you want to run k-means yourself.

### Behavior changes

- **Thread defaults.** `train` and `add` now default to every available CPU
  (`num_threads=0`), capped by `OMP_NUM_THREADS` since 0.8.1; 0.7 `build` and
  `add` defaulted to one thread. This applies to every `add` call that does not
  pass `num_threads`, including 0.7 code such as `index.add(x)`, without a
  warning. `search` still defaults to one thread, and the deprecated `build`
  keeps its default of 1.
- **HNSW-RaBitQ builds are not reproducible by default.** With more than one
  thread, `add` links points in a nondeterministic order. Pass `num_threads=1`
  to `add` for the same graph on every run. With one thread and the same seed,
  `set_centroids` followed by `add` produces exactly the index that 0.7 `build`
  produced from the same centroids and cluster IDs.
- **HNSW-RaBitQ `search` limits `k` to the stored point count**
  (`num_stored_points`) instead of the capacity, since capacity now grows
  geometrically. A `k` between the two used to return padding rows and now
  raises `ValueError`, as it already did for IVF-RaBitQ.
- **HNSW-RaBitQ capacity grows.** `add` resizes the index, at least doubling it,
  instead of raising `ValueError` when the batch does not fit. This also applies
  to `add` on indexes created in the old form, and to the deprecated `build`,
  which no longer rejects more rows than `max_elements`.
- **State errors.** Calling `add`, `search`, `remove`, `reserve` or `save`
  before training raises `RuntimeError("IvfIndex must be trained or loaded
  before ...")`; the 0.7 message said "built". A trained index without vectors
  cannot be saved. `repr` shows `is_trained` instead of `is_built`.
- **Seeds.** `IvfIndex` still draws a random rotation seed by default, and now
  reports it as `index.seed`; pass it back as `seed=` to reproduce an index.
  Indexes loaded from files report `seed=None`.
- **HNSW-RaBitQ `num_clusters`** reports the configured count (16 by default)
  before training, where 0.7 reported 0 before `build`.
- An `IvfIndex` created with `max_elements` is filled by `build`; `train` and
  `set_centroids` raise `RuntimeError` for it. Create the index in the 0.8 form
  to train it.

## SymphonyQG

| 0.7 | 0.8 |
| --- | --- |
| `SymqgIndex(dim, max_degree, metric, quantization_bits)` | `SymqgIndex(dim, max_degree=..., metric=..., nbits=...)`; every argument after `dim` is keyword-only |
| `quantization_bits=0` (raw vectors), `4`, `8` | `nbits=32` (raw vectors, the default), `4`, `8` |
| `index.build(x, ef_construction, num_threads, init)` | `SymqgIndex(dim, ef_construction=..., init=..., seed=...)`<br>`index.add(x, num_threads=...)` |
| `index.search_batch(q, k, ef)` | `index.search(q, k, ef)`; `ef` is now optional |
| `index.quantization_bits` | `index.nbits` (32 instead of 0 for raw vectors) |
| `index.is_built` | `len(index) > 0` |

`train(x)` exists for a uniform workflow but only checks `x` as `add` would,
and `is_trained` is always `True`. `add` builds the graph from all vectors and can run only once;
a second `add`, or `add` on a loaded index, raises `RuntimeError`, as Faiss's
NSG index does. The deprecated `build` keeps its 0.7 behavior: it replaces any
existing graph and, unless you passed `seed`, draws a new random seed each time.

`search` now treats `ef` like HNSW-RaBitQ: an omitted or zero `ef` uses
`max(k, 10)`, and a smaller one is raised to `k`. In 0.7, both raised
`ValueError`. `add` uses every CPU by default, while the deprecated `build`
keeps one thread. The seed now also drives graph construction: without `seed`,
each index draws one random seed and reports it as `index.seed`; 0.7 drew
separate random seeds for the rotation and the graph. Before vectors are added,
`search` and `save` raise `RuntimeError("SymqgIndex has no vectors; ...")`.

## Files and C++

Saved files, the `max_elements` property, and `search` results for the same
index are unchanged. Files written by 0.7 load as trained indexes.

For C++, the existing constructors and `construct` are unchanged; see
[IVF-RaBitQ](index/ivf.md#training-centroids-in-c) and
[HNSW-RaBitQ](index/hnsw.md#training-centroids-in-c) for the new training entry
points. The one C++ behavior change is that saving an index without stored
points now throws `std::logic_error`.
