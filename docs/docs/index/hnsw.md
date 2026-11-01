# HNSW-RaBitQ

C++ callers include `rabitqlib/index/hnsw/hnsw.hpp` and link `rabitqlib::index`.
[HNSW](https://arxiv.org/abs/1603.09320) is a popular graph-based index.
HNSW-RaBitQ combines HNSW graph traversal with RaBitQ quantization. Under
comparable quantization settings, HNSW-RaBitQ typically uses more memory
than IVF-RaBitQ because it stores graph structures in addition to quantized
vectors. The actual comparison depends on vector dimension, bit width, graph
degree, and index parameters.

## Minimal workflow

### Python

Run the [Quick start](../quick_start.md#build-and-search-an-ivf-index) to prepare
`data`, `queries` and `clustering`, then use those same arrays with HNSW-RaBitQ:

```python
from rabitqlib import HnswIndex

index = HnswIndex(
    dim=data.shape[1], max_elements=len(data), M=16, ef_construction=100, nbits=4,
)
index.build(data, clustering.centroids, clustering.assignments)
ids, distances = index.search(queries, k=10, ef=100)
index.save("hnsw.index")
loaded = HnswIndex.load("hnsw.index")
```

### C++

Link `rabitqlib::index`. Here `data`, `centroids` and `queries` are row-major
`std::vector<float>` buffers with original dimension `dim`; `cluster_ids` is a
`std::vector<rabitqlib::PID>` with one centroid ID per data row. Supply at least
ten points for this example.

```cpp
#include <vector>
#include <rabitqlib/index/hnsw/hnsw.hpp>

// Constructor order: capacity, dimension, total bits, M, ef_construction.
rabitqlib::hnsw::HierarchicalNSW index(data.size() / dim, dim, 4, 16, 100);
index.construct(
    centroids.size() / dim, centroids.data(), data.size() / dim, data.data(),
    cluster_ids.data(), 1
);
auto results = index.search(queries.data(), queries.size() / dim, 10, 100, 1);
index.save("hnsw.index");
rabitqlib::hnsw::HierarchicalNSW loaded;
loaded.load("hnsw.index");
```

C++ results are per-query vectors of `(distance, ID)` pairs. For build setup,
see [C++ integration](../cpp_integration.md).

## Parameters and limits

| Parameter | Contract |
| --- | --- |
| `dim` | Original input dimension; Python supports 64–65,536 and pads internally. |
| `max_elements` | Capacity; build accepts 1 through this many rows. Reserve extra capacity or call `resize` before adding beyond it. |
| `nbits` / C++ `total_bits` | 1–9 total bits including the sign bit; Python defaults to 8. Raw-vector storage is not supported. |
| `M`, `ef_construction` | Graph connectivity and construction search budget; Python defaults are 16 and 200. `M` must be at least 2 and is capped at 10,000; `ef_construction` is raised to at least `M`. |
| `metric`, `random_seed` | Python defaults are `"l2"` and 100; `"ip"` is also supported. |
| `k`, `ef` | Positive result count and search window. Python requires `k <= max_elements`; omitted/zero `ef` uses `max(k, 10)`, positive `ef` is raised to at least `k`, and `ef` above the stored point count is clamped to it. |

`random_seed` controls graph levels and the FHT/Kac rotation (using its low 32 bits).
With identical inputs and parameters, single-threaded construction is reproducible
within the same build environment. Parallel insertion order and different toolchains
or SIMD backends can still change results. Loading restores the saved rotation.

Results use squared L2 or `1 - dot` estimates, ordered from smallest to largest.
Python fills missing results with ID `2**32 - 1` and distance `+inf`; C++ returns
only the available pairs. See [result handling](../quick_start.md#read-the-results)
and the shared [input, count and concurrency contracts](usage.md).

## Index construction

The graph is built by incrementally inserting raw data vectors. Raw vectors
are used to construct and prune graph links, but the completed index stores
the links, centroids, cluster IDs, labels, RaBitQ codes, and factors rather
than a copy of the raw dataset. Total bit widths from 1 through 9 are
supported.

`construct()` is synchronous. Keep its input buffers valid for the entire call;
the index retains no pointers to them after it returns or throws. The borrowed
input pointer and search candidate helpers are private implementation details;
see the [C++ migration guide](../modular_migration.md#public-and-implementation-interfaces).

```cpp
void HierarchicalNSW::construct(
    size_t num_clusters,
    const float* centroids,
    size_t num_points,
    const float* data,
    PID* cluster_ids,
    size_t num_threads = std::numeric_limits<size_t>::max(),
    bool fast_quantization = false
);
```

- **data**: Vectors in the original input coordinates.
- **num_points**: The number of data vectors.
- **centroids**: Centroids computed by [RaBitQKMeans](../clustering.md#rabitqkmeans); the examples use `num_clusters = 16`.
- **cluster_ids**: Array of length `num_points`; every entry must be in the range `[0, num_clusters)`.
- **num_threads**: Number of threads to use (default: every available thread; `0` also auto-selects).
- **fast_quantization**: Use calibrated scaling instead of per-vector scale search; see [encoding paths](../rabitq/quantizer.md#implementations).

### Training centroids in C++

Instead of passing centroids to `construct`, train or set them on an empty
index, then add the points:

```cpp
#include <rabitqlib/index/hnsw/hnsw.hpp>
#include <rabitqlib/index/training.hpp>

rabitqlib::hnsw::HierarchicalNSW index(max_elements, dim, 4, 16, 200);
index.train(16, num_points, data.data(),
            rabitqlib::default_training_parameters(rabitqlib::METRIC_L2, 100));
// Or supply centroids: index.set_centroids(num_clusters, centroids.data());
index.add(data.data(), num_points);
```

- `train(num_clusters, num_points, data, cp)` trains centroids with
  [RaBitQKMeans](../clustering.md#rabitqkmeans); `num_points` must be at least
  `num_clusters`. Points are assigned by `add`, not by training. See the
  [IVF-RaBitQ defaults](ivf.md#training-centroids-in-c) for
  `default_training_parameters`.
- The first `add` into an empty graph links points with their exact vectors, as
  `construct` does; with one thread it produces the same index as `construct`
  with the same centroids and cluster IDs. If it fails, the graph is left empty
  and the centroids are kept. Later adds link through the stored codes; see
  [how added points are linked](#how-added-points-are-linked).
- `train` and `set_centroids` replace the centroids only while the index stores
  no points, removed ones included; otherwise they throw `std::logic_error`.
  A failed call leaves the index unchanged.
- `is_trained()` reports whether centroids are set. Before that, `add` throws
  `std::logic_error`; searching an empty index, trained or not, returns no results.
- `save` throws `std::logic_error` for an index without stored points, trained
  or not. Earlier releases saved an untrained empty index; such files still load.

## Querying

```cpp
std::vector<std::vector<std::pair<float, PID>>> HierarchicalNSW::search(
    const float* queries,
    size_t num_queries,
    size_t k,
    size_t ef_search,
    size_t num_threads
);
```

- **queries**: Query vectors in the original input coordinates.
- **num_queries**: The number of query vectors.
- **k**: The number of nearest neighbors to search.
- **ef_search**: The size of the candidate set for searching HNSW-RaBitQ base layer.
- **num_threads**: Number of threads to use. Each query is processed by one thread.

The search window belongs to each call, so concurrent C++ searches can use
different `ef_search` values and separate outputs while the index is unchanged.
Python follows the shared [concurrency contract](usage.md#threading-and-file-paths).

## Updating an index

A constructed or loaded index supports `add()`, `resize()`,
and `remove()` without the original data; C++ also adds to a
[trained](#training-centroids-in-c) index. Adding points and resizing preserve
the index file format; files with removals have
[compatibility restrictions](#how-removal-is-stored).

```c++
std::vector<PID> HierarchicalNSW::add(
    const float* data,
    size_t num_points,
    const PID* cluster_ids = nullptr,
    bool fast_quantization = false,
    size_t num_threads = 1
);

void HierarchicalNSW::resize(size_t new_max_elements);
```

- **data**, **num_points**: `num_points` new vectors in the original coordinates. Vector `i` receives
  the label `num_stored_points() + i`, so labels stay dense, and `add` returns them.
- **cluster_ids**: The cluster of each new vector, in `[0, num_clusters)`. When it
  is `nullptr`, each vector goes to its nearest centroid, chosen the same way a
  query is routed.
- **fast_quantization**, **num_threads**: Same meaning as in `construct`, but `add` defaults
  to one thread, which keeps the resulting links deterministic; pass `0` to auto-select. A point keeps the label
  `num_stored_points() + i` whatever order the threads finish in, because the slots are
  reserved as one block before the inserts start.
- **new_max_elements**: The new capacity. It must be at least `num_stored_points()`.
  In C++, initialize the index through its parameterized constructor or `load`
  before calling `resize`.

In Python:

```python
# Grow before adding; preserve any capacity already available.
index.resize(max(index.max_elements, index.num_stored_points + len(vectors)))
new_ids = index.add(vectors)                       # route to the nearest centroids
# Alternatively, choose the clusters instead of the call above:
# new_ids = index.add(vectors, cluster_ids=labels)
```

Updates require exclusive access; see the
[concurrency contract](usage.md#threading-and-file-paths).

`add` prepares the result IDs, quantized vectors, and link-list allocations before
changing the graph or point count. A failure during preparation leaves those
unchanged. A failure during graph linking can leave the batch counted in
`num_stored_points()` with only some points reachable through the graph. The index
remains safe to search, save, load, and destroy; inspect `num_stored_points()` before
retrying, or rebuild to restore full reachability.

### Capacity

See the shared [index count contract](usage.md#index-counts). Removed
records still occupy capacity.

`add` never grows the index: it rejects a whole batch whose size exceeds
`max_elements() - num_stored_points()`, and `resize` is the only way to raise
that. Reserving capacity avoids reallocating and copying the whole base-layer record storage during `add`.
It does not make insertion allocation-free: batch scratch, upper-layer link lists,
and label-map entries still require allocations, and the label map may rehash.

`resize` rebuilds the base-layer storage and copies it, so it needs memory for
both copies while it runs and costs time proportional to the whole index. It
invalidates every pointer into the index. The per-node upper-layer link lists are
not copied, only their pointer table.

### Recall after `add`

The centroids and the rotation are fixed when the index is constructed, so `add`
never retrains them. If the added vectors come from a different distribution, or
the index grows many times beyond its original size, recall can drop. Rebuild
from the original data with new centroids when recall matters more than the cost
of a rebuild.

### Removing points

```c++
size_t HierarchicalNSW::remove(const PID* labels, size_t num_points);
```

- **labels**, **num_points**: `num_points` labels of the points to remove, as returned by `add` and
  `search`. Every label must be in the index; nothing is removed if one is not.
  `remove` returns how many points were newly removed, so repeating a label or a
  call is safe.

In Python:

```python
removed = index.remove(ids)   # ids in [0, num_stored_points)
```

A removed point keeps its codes, its links, and its place in the graph. Search
still walks through it and `add` may link new points to it, which keeps the graph
connected, as in hnswlib. Search never returns it, so a query can get fewer than
`k` results once points are removed: in Python the missing slots hold `kPidMax`
(`2**32 - 1`) with an infinite distance. Removed points still count in
`num_stored_points()` and cannot be restored. Removal requires the same
exclusive access as other updates.

#### Choosing `ef` after removals

Removed points still take slots in the `ef_search` candidate set, so a search with
`ef_search` close to `k` finds fewer live points as removals accumulate. C++ search
uses `max(ef_search, k)`, clamped to `num_stored_points()`. Python uses `max(k, 10)`
when `ef` is omitted or zero; an explicit positive `ef` is also raised to at least `k`,
and both are clamped to the stored point count.
To keep about the same number of live candidates as before the removals, scale
the effective window by the share of live points (search clamps the result to
`num_stored_points()`):

```text
max(ef_search, k) * num_stored_points() / (num_stored_points() - removed points)
```

Callers that track how many points they removed can apply this directly.

## Persistence

### How removal is stored

`remove` sets the high bit of the point's stored cluster ID. `construct` and `load`
limit the number of clusters to 2^31, so no real cluster ID has that bit, and
every reader of the cluster ID masks it off. The mark lives in the base layer, so
it survives `save`, `load`, and `resize`, and a file without removals is
unchanged.

In C++, `cluster_id_of(internal_id)` returns a stored record's cluster with the
mark masked off; removed points keep their cluster. It takes an internal ID from 0
through `num_stored_points() - 1`, which matches the label only for points added
in order. It throws `std::logic_error` on a default-constructed index that has
not been loaded, and `std::invalid_argument` for an ID not below
`num_stored_points()`, including any ID before `construct`. Python does not
expose it.

A file that has removals cannot be read by earlier releases. Releases 0.3.7
through 0.5.1 check every cluster ID on `load` and reject the file as invalid
instead of returning removed points. Releases before 0.3.7 do not check cluster
IDs, so do not open such a file with them.

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

`construct` borrows the caller's vectors and links points using exact distances
between them. That pointer is not retained, so `add` cannot use it, and instead
scores candidates with the estimator the search already uses: the vector being
compared against is rebuilt from its stored RaBitQ code, sign bit and extra bits
together, and the other side is scored straight from its codes. This is the same
approach quantized SymphonyQG construction takes.

Graph quality therefore depends on how a point arrived.

Save/load preserves the existing graph and rotation, but not the random generator
state used to choose levels for future insertions. Adding the same vectors to an
original index and a loaded copy can therefore produce different graphs and
search results.
