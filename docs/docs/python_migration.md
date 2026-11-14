# Migrating Python code from 0.7 to 0.8

In 0.8, `IvfIndex`, `HnswIndex` and `SymqgIndex` follow a `train` → `add` →
`search` workflow, as Faiss indexes do; since 0.8.2 the first `add` trains, so
construct → `add` → `search` is enough, and `train` remains for training on a
sample. IVF-RaBitQ and HNSW-RaBitQ learn their own centroids, so you no longer
run k-means and pass the centroids and assignments to `build`; SymphonyQG has
nothing to train and builds its graph in `add`. Code written for 0.7
still runs in 0.8. Each old form raises a `DeprecationWarning` at your call site
and will be removed in **0.9**.

If your code already uses the 0.8 forms, go to
[From 0.8 to 0.9](#from-08-to-09): 0.9 renames some of them, and 0.8.5 accepts
the new names.

Unless an item below says otherwise, a 0.7 call that raises a deprecation
warning keeps its 0.7 behavior. Calls that are valid in both versions, such as
`add(x)` and `search`, follow the 0.8 rules.

## IVF-RaBitQ and HNSW-RaBitQ

### Old and new calls

| 0.7 | 0.8 |
| --- | --- |
| `IvfIndex(dim, max_elements, num_clusters, nbits, metric, initializer)` | `IvfIndex(dim, num_clusters, nbits=..., metric=..., routing=..., seed=None)` (`initializer=` before 0.8.5); storage grows on `add` |
| `IvfIndex(dim=..., max_elements=..., num_clusters=..., nbits=...)` | the same, without `max_elements` |
| `HnswIndex(dim, max_elements, M=16, ef_construction=200, nbits=8, metric="l2", random_seed=100)` | `HnswIndex(dim, nbits=..., num_clusters=16, M=16, ef_construction=200, metric="l2", seed=100)`; capacity grows on `add`, and `reserve(n)` preallocates it (`resize(n)` before 0.8.5) |
| `HnswIndex(..., random_seed=s)` | `HnswIndex(..., seed=s)` |
| `kmeans = RaBitQKMeans(dim, k)`<br>`kmeans.train(x)`<br>`index.build(x, kmeans.centroids, kmeans.assignments)` | `index.add(x)`, which trains on `x` first (since 0.8.2; in 0.8.0 and 0.8.1, call `index.train(x)` first) |
| `index.build(x, centroids, cluster_ids)` | `index.set_centroids(centroids)`<br>`index.add(x, cluster_ids=cluster_ids)` |
| `index.add(x, cluster_ids, num_threads, fast_quantization)` | `index.add(x, cluster_ids=..., num_threads=..., fast_quantization=...)` |
| `index.is_built` | `index.is_trained`, or `len(index) > 0` for "holds vectors" |

Except for `dim` and IVF-RaBitQ's `num_clusters`, constructor arguments are
keyword-only in 0.8. Since 0.8.2, `nbits` defaults to 7 (0.8.0 and 0.8.1 required
it). HNSW-RaBitQ used to default to `nbits=8`; the old form keeps that default.

`train` uses the settings in `index.cp`: 10 iterations (25 before 0.8.2), at most 256 sampled
vectors per centroid, spherical k-means for `metric="ip"`, and the index seed.
See [IVF-RaBitQ training](index/ivf.md#training-centroids-in-python) for the
details and for `set_centroids`. The [clustering classes](clustering.md) still
work if you want to run k-means yourself.

### Behavior changes

- **Thread defaults.** `train` and `add` now default to every available CPU
  (`num_threads=None`, the same as `0`), capped by `OMP_NUM_THREADS` since 0.8.1; 0.7 `build` and
  `add` defaulted to one thread. Since 0.8.5, IVF-RaBitQ `set_centroids` follows
  the same default and takes `num_threads`; before, it always used every CPU and
  ignored `OMP_NUM_THREADS`. This applies to every `add` call that does not
  pass `num_threads`, including 0.7 code such as `index.add(x)`, without a
  warning. `search` still defaults to one thread, and the deprecated `build`
  keeps its default of 1.
- **HNSW-RaBitQ builds are not reproducible by default.** With more than one
  thread, `add` links points in a nondeterministic order. Pass `num_threads=1`
  to `add` for the same graph on every run. With one thread and the same seed,
  `set_centroids` followed by `add` produces exactly the index that 0.7 `build`
  produced from the same centroids and cluster IDs.
- **IVF-RaBitQ HNSW routing builds are not reproducible by default.** 0.7 `build`
  built the routing graph with its own thread count, one by default. With more
  than one thread, `set_centroids`, `train` and a training `add` build it in a
  nondeterministic order; pass `num_threads=1` (to `set_centroids` since 0.8.5)
  and a fixed `seed` for the same graph on every run.
- **`k` above the point count.** 0.8.0 and 0.8.1 raised `ValueError` when `k`
  exceeded `num_stored_points` (HNSW-RaBitQ) or the point count (IVF-RaBitQ,
  SymphonyQG). Since 0.8.2, every index accepts any `k >= 1` and fills missing
  slots with ID `2**32 - 1` and distance `inf`; test `np.isfinite(distances)`.
- **HNSW-RaBitQ capacity grows.** `add` resizes the index, at least doubling it,
  instead of raising `ValueError` when the batch does not fit. This also applies
  to `add` on indexes created in the old form, and to the deprecated `build`,
  which no longer rejects more rows than `max_elements`.
- **State errors.** Calling `add` with `cluster_ids`, `search`, `remove`,
  `reserve_cluster` (`reserve` before 0.8.5) or `save` before training raises `RuntimeError("IvfIndex must be
  trained or loaded before ...")`; the 0.7 message said "built". Since 0.8.2,
  `add` without `cluster_ids` trains first. A trained index without vectors
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

`search` now treats its breadth (`ef_search`; `ef` before 0.8.5) like HNSW-RaBitQ:
omitted, it uses
`max(k, 64)` (`max(k, 10)` before 0.8.2), and a smaller one is raised to `k`. In 0.7, both raised
`ValueError`. `add` uses every CPU by default, while the deprecated `build`
keeps one thread. The seed now also drives graph construction: without `seed`,
each index draws one random seed and reports it as `index.seed`; 0.7 drew
separate random seeds for the rotation and the graph. Before vectors are added,
`search` and `save` raise `RuntimeError("SymqgIndex has no vectors; ...")`.

## Files and C++

Saved files are unchanged, and so is the `max_elements` property up to 0.8.4 (see
[From 0.8 to 0.9](#from-08-to-09)). `HnswIndex.search`
without a search breadth uses `max(k, 64)` since 0.8.2 (`max(k, 10)` before), which changes
its results and latency; pass the breadth as the third argument to keep the old window. Files written by 0.7
load as trained indexes, except empty, untrained HNSW-RaBitQ files, which need
`train` before `add`.

For C++, the existing constructors and `construct` are unchanged; see
[IVF-RaBitQ](index/ivf.md#training-centroids-in-c) and
[HNSW-RaBitQ](index/hnsw.md#training-centroids-in-c) for the new training entry
points. C++ behavior changes: since 0.8.0, saving an index without stored points
throws `std::logic_error`; since 0.8.2, IVF and SymphonyQG searches pad `k` above
the point count instead of throwing, and `default_training_parameters` runs 10
iterations. See the [0.8.2 notes](releases/0.8.2.md#c-changes).

## From 0.8 to 0.9

0.9 gives the three index classes one name for each concept
([#232](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/232)). Since 0.8.5 the
0.8 form in the first column raises a `DeprecationWarning` at your call site; 0.9
removes it.

| 0.8 | Use instead | Works since |
| --- | --- | --- |
| `IvfIndex.max_elements`, `SymqgIndex.max_elements` | `num_stored_points` | Every 0.8 release. The two are equal, except on an `IvfIndex` created in the 0.7 form before `build`. |
| `HnswIndex.max_elements` | `HnswIndex.capacity` | 0.8.5 |
| `HnswIndex.resize(n)` | `HnswIndex.reserve(n)`, which only grows the capacity; `resize` could also shrink it | 0.8.5 |
| `IvfIndex.reserve(cluster_id, num_points)` | `IvfIndex.reserve_cluster(cluster_id, num_points)` | 0.8.5 |
| `IvfIndex(initializer=...)`, `index.initializer` | `IvfIndex(routing=...)`, `index.routing`, with the same values | 0.8.5 |
| `search(queries, k, ef=...)` on `HnswIndex` and `SymqgIndex` | `search(queries, k, ef_search=...)` | 0.8.5; passing the breadth by position works in every release |
| `search(queries, k, nprobe, high_accuracy, num_threads)`, `search(queries, k, ef, num_threads)` | `high_accuracy=` and `num_threads=` by keyword; only `nprobe` or `ef_search` stays positional | Every 0.8 release |
| `nprobe=0`, `ef=0` for the default | `None`, or omit the argument; 0.9 raises `ValueError` for 0, while `num_threads=0` keeps meaning every available CPU | 0.8.2 |
| `HnswIndex(seed=s)` with `s` above `2**32 - 1` | A seed in `[0, 2**32 - 1]`, the range of the other two classes | Every 0.8 release |

The k-means classes take the names of the index parameters
([#233](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/233)), as keywords and as
attributes of `cp`:

| 0.8 | Use instead | Works since |
| --- | --- | --- |
| `QGKMeans(quantization_bits=0 \| 4 \| 8)` | `nbits=32 \| 4 \| 8`: 32, not 0, keeps raw vectors | 0.8.5 |
| `QGKMeans(graph_degree=...)` | `max_degree=...` | 0.8.5 |
| `QGKMeans(ef_build=...)` | `ef_construction=...` | 0.8.5 |
| `final_assignment=FinalAssignmentMode.Approximate \| .Exact`, also as `index.cp.final_assignment` on `IvfIndex` and `HnswIndex` | `final_assignment='approximate' \| 'exact'`; `'approximate'` also replaces the alias `FinalAssignmentMode.SymphonyQG` | 0.8.5 |
| `QGKMeansIterationStats`, `RaBitQKMeansIterationStats` | `KMeansIterationStats`, the one class both always were; since 0.8.5 it also reports that name | 0.8.5 |

Passing a `FinalAssignmentMode` member does not warn by itself; the warning comes where
one of the three old names is imported or read from `rabitqlib`, including through
`from rabitqlib import *`.

Until 0.9, `cp.final_assignment` still reads back as a `FinalAssignmentMode` member,
whose `.name` is `'Approximate'` or `'Exact'`; 0.9 returns `'approximate'` or
`'exact'`. Reading it does not warn. Code that reads it on both releases can
normalize the value:

```python
mode = cp.final_assignment
mode = mode if isinstance(mode, str) else mode.name.lower()
```

Code that must also run on releases before 0.8.5 needs a version check for the rows
marked 0.8.5; the other replacements work on every release named.

Three changes come without a warning in 0.9. `HnswIndex()` without `seed=` draws a seed,
as `IvfIndex` and `SymqgIndex` already do, instead of using 100: pass `seed=` (and
`num_threads=1` to `add`) if you rely on a reproducible build. `repr(index)` prints
`capacity=` and `routing=` where it prints `max_elements=` and `initializer=`. And
`cp.final_assignment` reads back as a string, as described above.
