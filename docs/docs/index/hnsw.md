# HNSW-RaBitQ

C++ callers include `rabitqlib/index/hnsw_index.hpp` and link `rabitqlib::index`.
[HNSW](https://arxiv.org/abs/1603.09320) is a popular graph-based index.
HNSW-RaBitQ combines HNSW graph traversal with RaBitQ quantization. Under
comparable quantization settings, HNSW-RaBitQ typically uses more memory
than IVF-RaBitQ because it stores graph structures in addition to quantized
vectors. The actual comparison depends on vector dimension, bit width, graph
degree, and index parameters.

## Minimal workflow

### Python

Run the [Quick start](../quick_start.md#build-and-search-an-ivf-index) to prepare
`data` and `queries`, then use those same arrays with HNSW-RaBitQ:

```python
from rabitqlib import HnswIndex

index = HnswIndex(data.shape[1], M=16, ef_construction=100, seed=7)  # nbits=7 by default
index.add(data, num_threads=1)  # trains 16 centroids, then links; a seed and 1 thread are reproducible
ids, distances = index.search(queries, k=10)  # ef_search defaults to max(k, 64)
index.save("hnsw.index")
loaded = HnswIndex.load("hnsw.index")
```

The first `add` trains an untrained index on the added vectors, as for
[IVF-RaBitQ](ivf.md#training-centroids-in-python); with fewer rows than
`num_clusters`, it trains one centroid per row, and `num_clusters` then reports
the row count. An empty first `add` raises `RuntimeError`.
`train` and `set_centroids` work as for [IVF-RaBitQ](ivf.md#training-centroids-in-python),
with `num_clusters` centroids (default 16) and always RaBitQKMeans, but
`set_centroids` takes no `num_threads`; `index.cp`
starts from the same defaults with the index `seed`. The first `add` into the
empty index links vectors with their exact coordinates, like the 0.7 `build`;
later adds link through the stored codes (see [Recall after `add`](#recall-after-add)),
so add most vectors in the first call.

### C++

`rabitqlib::HnswIndex` has the names, defaults and behaviour of the Python class.
Here `data` and `queries` are row-major `std::vector<float>` buffers with
dimension `dim`.

```cpp
#include <vector>
#include <rabitqlib/index/hnsw_index.hpp>

rabitqlib::HnswOptions options;  // set the fields by name; nbits is 7 by default
options.ef_construction = 100;
options.seed = 7;
rabitqlib::HnswIndex index(dim, options);
rabitqlib::AddOptions add_options;
add_options.num_threads = 1;  // with the seed, a reproducible graph
index.add(data.data(), data.size() / dim, add_options);  // trains 16 centroids, then links

const size_t num_queries = queries.size() / dim;
std::vector<rabitqlib::Id> ids(num_queries * 10);
std::vector<float> distances(num_queries * 10);
index.search(queries.data(), num_queries, 10, ids.data(), distances.data());
index.save("hnsw.index");
auto loaded = rabitqlib::HnswIndex::load("hnsw.index");
```

For build setup, see [C++ integration](../cpp_integration.md).

## Parameters and limits

| Parameter | Contract |
| --- | --- |
| `dim` | Original input dimension, 64–65,536; the index pads internally. |
| `nbits` | 1–9 total bits including the sign bit; defaults to 7. Raw-vector storage is not supported. |
| `num_clusters` | Default 16: centroids `train` or the first `add` learns, or `set_centroids` supplies; the first `add` caps it at its row count. |
| `M`, `ef_construction` | Graph connectivity and construction search budget; defaults are 16 and 200, and the index reports both. `M` must be at least 2 and is capped at 10,000; `ef_construction` is raised to at least `M`. The base layer keeps up to `2 * M` edges per node. |
| `metric`, `seed` | `metric` defaults to L2 (Python `"l2"`); inner product and [cosine](usage.md#the-cosine-metric) are also supported. `seed` is the graph and rotation seed in `[0, 2**32 - 1]` and the initial `cp.seed`. The default (Python `None`, an empty C++ `std::optional`) draws one at random (it was 100 before 0.9); `seed` reports it. |
| `capacity` | The current capacity, which `add` grows geometrically (at least doubling) and `reserve` can preallocate; in Python they replace `max_elements` and `resize`, which 0.9 removed. |
| `k`, `ef_search` | Result count, from 1 to 2^31, and search window. `k` may exceed the stored point count; missing results are padded with ID `-1` (C++ `kNoId`) and `+inf`. An omitted `ef_search` (Python `None`) uses `max(k, 64)`, an explicit 0 raises `ValueError` (C++ `std::invalid_argument`), a smaller value is raised to at least `k`, and a value above the stored point count is clamped to it. It replaces `ef`, which 0.9 removed. |

In C++, the construction settings are fields of `HnswOptions`, and `ef_search`
and `num_threads` fields of `GraphSearchOptions`.

`seed` controls graph levels and the FHT/Kac rotation (using its low 32 bits).
With identical inputs and parameters, single-threaded construction is reproducible
within the same build environment. A seed is drawn unless one is given, and
`train` and `add` default to every available CPU, so **default builds are not
reproducible**: pass a seed to the constructor and one thread to `add` when you
need the same graph again. Different toolchains or SIMD backends can
still change results. Loading restores the saved rotation.

Results use squared L2 or `1 - dot` estimates, ordered from smallest to largest.
IDs are `int64` (C++ `rabitqlib::Id`), and missing results hold ID `-1` and distance
`+inf`. See [result handling](../quick_start.md#read-the-results)
and the shared [input, count and concurrency contracts](usage.md).

## Index construction

The graph is built by incrementally inserting raw data vectors. Raw vectors
are used to construct and prune graph links, but the completed index stores
the links, centroids, cluster IDs, labels, RaBitQ codes, and factors rather
than a copy of the raw dataset. Total bit widths from 1 through 9 are
supported.

`add` is synchronous. Keep its input buffers valid for the entire call; the
index retains no pointers to them after it returns or throws.

To build from centroids computed elsewhere, for example by
[RaBitQKMeans](../clustering.md#rabitqkmeans), set them and pass each vector's
cluster:

```cpp
rabitqlib::HnswOptions options;
options.num_clusters = num_clusters;
rabitqlib::HnswIndex index(dim, options);
index.set_centroids(centroids.data());
rabitqlib::AddOptions add_options;
add_options.cluster_ids = cluster_ids.data();
index.add(data.data(), num_points, add_options);
```

- **centroids**: `num_clusters` centroids in the original input coordinates; the examples use 16.
- **cluster_ids**: One entry per vector, each in `[0, num_clusters)`.
- **num_threads**: Worker count; no value, the default, uses every available CPU, and `0` also auto-selects.
- **fast_quantization**: Use calibrated scaling instead of per-vector scale search; see [encoding paths](../rabitq/quantizer.md#implementations).

### Training centroids in C++

`HnswIndex` trains as the Python class does. `cp()` returns the training
settings; include `rabitqlib/clustering/rabitqkmeans.hpp` to change them:

```cpp
#include <rabitqlib/clustering/rabitqkmeans.hpp>
#include <rabitqlib/index/hnsw_index.hpp>

rabitqlib::HnswIndex index(dim);
index.cp().niter = 20;
index.train(sample.data(), num_samples);  // no thread count: cp().num_threads
// Or supply centroids: index.set_centroids(centroids.data());
index.add(data.data(), num_points);
```

- `train`, `set_centroids` and the first `add` follow the Python rules above.
  Training always uses [RaBitQKMeans](../clustering.md#rabitqkmeans); points are
  assigned by `add`, not by training.
- The first `add` into an empty graph links points with their exact vectors. If
  it fails, the graph is left empty. Centroids set by `train` or `set_centroids`
  are kept, and a first `add` that trained leaves the index untrained. Later adds link
  through the stored codes; see
  [how added points are linked](#how-added-points-are-linked).
- Arguments outside their range throw `std::invalid_argument` (Python
  `ValueError`), and a call in the wrong state throws `std::logic_error` (Python
  `RuntimeError`), with the Python messages except where those name Python values
  such as `None`.
- `is_trained()` reports whether centroids are set. A trained index without
  points pads every `search` result slot with `kNoId` and `+inf`; `save` throws
  `std::logic_error` until the index stores at least one point. Earlier
  releases saved an untrained empty index; such files still load, and without a
  recorded seed they need `train` or `set_centroids` before `add`.

## Querying

```cpp
void HnswIndex::search(
    const float* queries,
    size_t num_queries,
    size_t k,
    Id* ids,
    float* distances,
    const GraphSearchOptions& options = {}
) const;
```

- **queries**: `num_queries` query vectors in the original input coordinates.
- **k**: The number of nearest neighbors to search.
- **ids**, **distances**: Output buffers of `num_queries * k` values; `distances` may be `nullptr`.
- **options.ef_search**: The size of the candidate set for searching the base layer.
- **options.num_threads**: Number of threads to use (default 1). Each query is processed by one thread.

The search window belongs to each call, so concurrent searches can use
different `ef_search` values and separate outputs while the index is unchanged.
See the shared [concurrency contract](usage.md#threading-and-file-paths).

## Updating an index

A trained index supports `add()`, `reserve()` and `remove()` without the original
data. Every loaded index is trained, except one loaded from a file saved before
training (see below). `save` always writes index format 2; see
[persistence](#persistence).

```c++
Id HnswIndex::add(const float* data, size_t num_points, const AddOptions& options = {});

void HnswIndex::reserve(size_t num_points);
```

- **data**, **num_points**: `num_points` new vectors in the original coordinates. Vector `i` receives
  the ID `num_stored_points() + i`, so IDs stay dense; `add` returns the first.
- **options.cluster_ids**: The cluster of each new vector, in `[0, num_clusters)`. When it
  is `nullptr`, each vector goes to its nearest centroid, chosen the same way a
  query is routed.
- **options.fast_quantization**, **options.num_threads**: As in [index construction](#index-construction).
  With more than one thread the links vary between runs; one thread keeps them
  deterministic. A point keeps the ID `num_stored_points() + i` whatever order the
  threads finish in, because the slots are reserved as one block before the inserts start.
- **num_points** of `reserve`: The capacity to preallocate; a smaller value changes nothing.

In Python:

```python
new_ids = index.add(vectors)                       # route to the nearest centroids
# Alternatively, choose the clusters instead of the call above:
# new_ids = index.add(vectors, cluster_ids=labels)
```

Updates require exclusive access; see the
[concurrency contract](usage.md#threading-and-file-paths).

`add` prepares the result IDs, quantized vectors, and link-list allocations before
changing the graph or point count. A failure during preparation leaves those
unchanged. A failure while linking a later batch can leave it counted in
`num_stored_points()` with only some points reachable through the graph. The index
remains safe to search, save, load, and destroy; inspect `num_stored_points()` before
retrying, or rebuild to restore full reachability.

### Capacity

See the shared [index count contract](usage.md#index-counts). Removed
records still occupy capacity.

`add` grows the capacity first when the batch does not fit, to at least
double it, and `reserve(num_points)` preallocates it; `reserve` never shrinks the
capacity. Reserving capacity avoids reallocating and copying the whole base-layer record storage during `add`.
It does not make insertion allocation-free: batch scratch, upper-layer link lists,
and label-map entries still require allocations, and the label map may rehash.

Growing the capacity rebuilds the base-layer storage and copies it, so it needs memory for
both copies while it runs and costs time proportional to the whole index. It
invalidates every pointer into the index. The per-node upper-layer link lists are
not copied, only their pointer table.

### Recall after `add`

The centroids and the rotation are fixed once the index is trained, so `add`
never retrains them. If the added vectors come from a different distribution, or
the index grows many times beyond its original size, recall can drop. Rebuild
from the original data with new centroids when recall matters more than the cost
of a rebuild.

### Removing points

```c++
size_t HnswIndex::remove(const Id* ids, size_t num_ids);
```

- **ids**, **num_ids**: IDs of the points to remove, as returned by `add` and
  `search`. Every ID must be in `[0, num_stored_points())`; nothing is removed if one is not.
  `remove` returns how many points were newly removed, so repeating an ID or a
  call is safe.

In Python:

```python
removed = index.remove(new_ids[:10])   # IDs in [0, num_stored_points)
```

A removed point keeps its codes, its links, and its place in the graph. Search
still walks through it and `add` may link new points to it, which keeps the graph
connected, as in hnswlib. Search never returns it, so a query can get fewer than
`k` results once points are removed: the missing slots hold ID `-1` (C++ `kNoId`)
with an infinite distance. Removed points still count in
`num_stored_points()` and cannot be restored. Removal requires the same
exclusive access as other updates.

#### Choosing `ef_search` after removals

Removed points still take slots in the `ef_search` candidate set, so a search with
`ef_search` close to `k` finds fewer live points as removals accumulate. Search uses
`max(k, 64)` when `ef_search` is omitted; an explicit value is raised to at least
`k`, and both are clamped to the stored point count.
To keep about the same number of live candidates as before the removals, scale
the effective window by the share of live points (search clamps the result to
`num_stored_points()`):

```text
max(ef_search, k) * num_stored_points() / (num_stored_points() - removed points)
```

Callers that track how many points they removed can apply this directly.

## Persistence

### How removal is stored

`remove` sets the high bit of the point's stored cluster ID. Setting centroids and
`load` limit the number of clusters to 2^31, so no real cluster ID has that bit, and
every reader of the cluster ID masks it off. The mark lives in the base layer, so
it survives `save`, `load`, and capacity growth, and a file without removals
carries no marks.

In the layout written before 0.9, a file that has removals cannot be read by
releases before 0.5.2. Releases 0.3.7
through 0.5.1 check every cluster ID on `load` and reject the file as invalid
instead of returning removed points. Releases before 0.3.7 do not check cluster
IDs, so do not open such a file with them.

`save` writes [index format 2](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/formats.md#index-format-2), which also records the construction seed, so
`seed` survives a reload. Releases up to 0.8 cannot read these files; every earlier
file still loads, without a recorded seed, and saving it writes format 2.
The C++ API of earlier releases could also save an index before training it. Such a
file loads untrained, with `num_clusters` 16; without a recorded seed its first `add`
cannot train it, so call `train` or `set_centroids` before `add`.

New HNSW-RaBitQ indexes pad dimensions to multiples of 32 for all supported bit widths.
Loading preserves the saved padded dimension and rotation, so older indexes
using multiples of 64 remain readable. Older library versions cannot load new
files whose padded dimension is not a multiple of 64.

## Implementation details

### Construction

During construction, we first rotate the centroids and then insert each element one by one. For each element:

1. Update the graph structure (edges) by searching with raw vectors and pruning.
2. Quantize the rotated vector and store its quantization code.

### Data layout

Each indexed element is stored in the following layout:

```
[number of edges]
[edges]
[cluster ID]       (its high bit marks a removed point, see How removal is stored)
[external label]
[BinData (1-bit * padded_dim + factors)]
[ExData (ex-bits * padded_dim + factors)]
```

Here `padded_dim` is the stored rotated dimension, which may exceed the original
input `dim`. Loading preserves the index's saved padded dimension.

### Search algorithm

We first pre-process the query:

1. Rotate the raw query vector.
2. Compute distances between the rotated query and all rotated centroids.
3. Encapsulate the query into `SplitSingleQuery<float>` for subsequent search.

#### Upper Layers

In the upper layers of HNSW-RaBitQ, we compute the 1-bit estimated distance (using `BinData`) to quickly locate the entry point for the next layer.

#### Base Layer

In the base layer, we apply an adaptive re-ranking strategy:

- **search_pool**: A distance-sorted exploration window with capacity
  `min(max(ef_search, k), num_stored_points())`, containing both expanded and
  unexpanded candidates.
- **nearest_candidates**: Current best k candidates.

Repeat while the exploration window has an unexpanded candidate:

1. Select the nearest unexpanded element `e` and mark it expanded. It remains
   in the window unless displaced by a later insertion.
2. Visit all unvisited neighbors of `e`.
3. For each neighbor:
   - Compute the 1-bit lower-bound distance (along with 1-bit estimated distance) using `BinData`.
   - For a nonremoved neighbor, if `nearest_candidates` has fewer than k elements
     or the neighbor's lower bound is smaller than the worst result's estimated distance:
     1. Refine the distance estimate using `ExData` when extra bits are present.
     2. Update `nearest_candidates`.
   - Insert the neighbor into `search_pool` only if the window has space or
     its (possibly refined) estimated distance is no greater than the worst
     distance in the window. Inserting into a full window displaces its last entry.

The search stops when no unexpanded candidates remain in the window, even though
the window still contains entries. `ef_search` controls this exploration budget
separately from the k result count; increasing it allows a wider search at
additional cost, rather than requiring traversal of the entire graph.

A point hidden by `remove` never enters `nearest_candidates`, so search never returns it,
but it can still enter `search_pool` under the same window admission rule,
allowing search to walk through it. A removed entry point keeps the estimate
already computed for it (the full-bits one when nbits > 1). A removed neighbor
is not refined and uses its 1-bit estimate for window admission.

### How added points are linked

The first `add` into an empty graph borrows the caller's vectors and links points
using exact distances between them. That pointer is not retained, so later adds
cannot use it, and instead score candidates with the estimator the search already
uses: the vector being compared against is rebuilt from its stored RaBitQ code,
sign bit and extra bits together, and the other side is scored straight from its
codes. This is the same
approach quantized SymphonyQG construction takes.

Graph quality therefore depends on how a point arrived.

Save/load preserves the existing graph and rotation, but not the random generator
state used to choose levels for future insertions. Adding the same vectors to an
original index and a loaded copy can therefore produce different graphs and
search results.
