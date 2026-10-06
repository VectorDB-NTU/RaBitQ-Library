# HNSW RaBitQ
[HNSW](https://arxiv.org/abs/1603.09320) is a popular graph-based index. Under
comparable quantization settings, HNSW RaBitQ typically uses more memory
than IVF RaBitQ because it stores graph structures in addition to quantized
vectors. The actual comparison depends on vector dimension, bit width, graph
degree, and index parameters.
This document describes how the library integrates HNSW with RaBitQ to support efficient vector search.

## Index Construction

The graph is built by incrementally inserting raw data vectors. Raw vectors
are used to construct and prune graph links, but the completed index stores
the links, centroids, cluster IDs, labels, RaBitQ codes, and factors rather
than a copy of the raw dataset. Total bit widths from 1 through 9 are
supported.

Users can invoke:

```cpp
HierarchicalNSW::construct(size_t cluster_num,
                          const float* centroids,
                          size_t data_num,
                          const float* data,
                          PID* cluster_ids,
                          size_t num_threads = 0,
                          bool faster = false);
```

- **data**: Pointer to the raw data vectors.
- **data_num**: The number of data vectors.
- **centroids**: Centroids computed by [RaBitQKMeans](../clustering.md#rabitqkmeans); the examples use `cluster_num = 16`.
- **cluster_ids**: Array of length `data_num`; every entry must be in the range `[0, cluster_num)`.
- **num_threads**: Number of threads to use (default: 0, which auto-selects).
- **faster**: If `true`, enables the faster quantizer.


During construction, we first rotate the centroids and then insert each element one by one. For each element:

1. Update the graph structure (edges) by searching with raw vectors and pruning.  
2. Quantize the rotated vector and store its quantization code.


### Data Layout

Each indexed element is stored in the following layout:

```
[number of edges]
[edges]
[cluster ID]       (its high bit marks a removed point, see Removing points)
[external label]
[BinData (1-bit * dim + factors)]
[ExData (ex-bits * dim + factors)]
```

## Querying
Users can invoke:
```cpp
std::vector<std::vector<std::pair<float, PID>>> HierarchicalNSW::search(const float* queries,
                                                                        size_t query_num,
                                                                        size_t TOPK,
                                                                        size_t efSearch,
                                                                        size_t thread_num);
```

- **queries**: Pointer to the raw query vectors.
-  **query_num**: The number of query vectors.
-  **TOPK**: The number of nearest neighbors to search.
-  **efSearch**: The size of the candidate set for searching HNSW base layer.
-  **thread_num**: Number of threads to use. Each query is processed by one thread.

We first pre-process the query:

1. Rotate the raw query vector.  
2. Compute distances between the rotated query and all rotated centroids.  
3. Encapsulate the query into a `query_wrapper` for subsequent search.  

### Upper Layers

In the upper layers of HNSW, we compute the 1-bit estimated distance (using `BinData`) to quickly locate the entry point for the next layer.

### Base Layer

In the base layer, we apply an adaptive re-ranking strategy:

- **candidate_set**: Elements to be visited.  
- **boundedKNN**: Current best TOPK candidates.

Repeat until `candidate_set` is empty:

1. Extract the nearest element `e` from `candidate_set`.  
2. Visit all unvisited neighbors of `e`.  
3. For each neighbor:
   - Compute the 1-bit lower-bound distance (along with 1-bit estimated distance) using `BinData`.  
   - If `boundedKNN` has fewer than TOPK elements, or if the 1-bit lower-bound is smaller than the full-bits estimated distance of the current farthest element in `boundedKNN`:
     1. Refine the distance estimate using `ExData` to obtain the full-bits estimated distance.  
     2. Update `boundedKNN`.  
   - Insert the neighbor into `candidate_set` with its (possibly refined) estimated distance.  

The search terminates when `candidate_set` is empty.

A point hidden by `remove` never enters `boundedKNN`, so search never returns it,
but it still enters `candidate_set` and the search keeps walking through it. A
removed entry point keeps the estimate already computed for it (the full-bits one
when nbits > 1). A removed neighbor is not refined and enters `candidate_set` with
its 1-bit estimate.

## Updating an Index

Starting with 0.5.2, a constructed or loaded index supports `add()`, `resize()`,
and `remove()` without the original data. Adding points and resizing preserve
the index file format; files with removals have
[compatibility restrictions](#how-removal-is-stored).

```c++
std::vector<PID> HierarchicalNSW::add(
    const float* data,
    size_t n,
    const PID* cluster_ids = nullptr,
    bool faster = false,
    size_t num_threads = 1
);

void HierarchicalNSW::resize(size_t new_max_elements);
```

- **data**, **n**: `n` new vectors in the original coordinates. Vector `i` receives
  the label `num_points() + i`, so labels stay dense, and `add` returns them.
- **cluster_ids**: The cluster of each new vector, in `[0, num_clusters)`. When it
  is `nullptr`, each vector goes to its nearest centroid, chosen the same way a
  query is routed.
- **faster**, **num_threads**: Same meaning as in `construct`, but `add` defaults
  to one thread; pass `0` to auto-select. A point keeps the label
  `num_points() + i` whatever order the threads finish in, because the slots are
  reserved as one block before the inserts start.
- **new_max_elements**: The new capacity. It must be at least `num_points()`.
  In C++, initialize the index through its parameterized constructor or `load`
  before calling `resize`.

In Python:

```python
new_ids = index.add(vectors)                       # route to the nearest centroids
# Alternatively, choose the clusters instead of the call above:
# new_ids = index.add(vectors, cluster_ids=labels)
index.resize(index.max_elements + 100_000)
```

It is not safe to call `add` or `resize` while another thread searches the same
index.

`add` prepares the result IDs, quantized vectors, and link-list allocations before
changing the graph or point count. A failure during preparation leaves those
unchanged. A failure during graph linking can leave the batch counted in
`num_points()` with only some points reachable through the graph. The index
remains safe to search, save, load, and destroy; inspect `num_points()` before
retrying, or rebuild to restore full reachability.

### Capacity

`add` never grows the index: it throws once `num_points()` reaches
`max_elements()`, and `resize` is the only way to raise that. An index built with
spare capacity absorbs points with no reallocation at all, so sizing
`max_elements` ahead of time is the cheap path.

`resize` rebuilds the base-layer storage and copies it, so it needs memory for
both copies while it runs and costs time proportional to the whole index. It
invalidates every pointer into the index. The per-node upper-layer link lists are
not copied, only their pointer table.

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

### Recall after `add`

The centroids and the rotation are fixed when the index is constructed, so `add`
never retrains them. If the added vectors come from a different distribution, or
the index grows many times beyond its original size, recall can drop. Rebuild
from the original data with new centroids when recall matters more than the cost
of a rebuild.

### Removing points

```c++
size_t HierarchicalNSW::remove(const PID* labels, size_t n);
```

- **labels**, **n**: Labels of the points to remove, as returned by `add` and
  `search`. Every label must be in the index; nothing is removed if one is not.
  `remove` returns how many points were newly removed, so repeating a label or a
  call is safe.

In Python:

```python
removed = index.remove(ids)   # ids in [0, num_points)
```

A removed point keeps its codes, its links, and its place in the graph. Search
still walks through it and `add` may link new points to it, which keeps the graph
connected, as in hnswlib. Search never returns it, so a query can get fewer than
`k` results once points are removed: in Python the missing slots hold `kPidMax`
(`2**32 - 1`) with an infinite distance. Removed points still count in
`num_points()` and cannot be restored. It is not safe to call `remove` while
another thread searches or adds to the same index.

#### Choosing `ef` after removals

Removed points still take slots in the `efSearch` candidate set, so a search with
`efSearch` close to `k` finds fewer live points as removals accumulate. `search`
uses the `efSearch` it is given. To keep about the same number of live candidates
as before the removals, scale it by the share of live points:

```text
efSearch * num_points() / (num_points() - removed points)
```

Callers that track how many points they removed can apply this directly.

#### How removal is stored

`remove` sets the high bit of the point's stored cluster ID. `construct` and `load`
limit the number of clusters to 2^31, so no real cluster ID has that bit, and
every reader of the cluster ID masks it off. The mark lives in the base layer, so
it survives `save`, `load`, and `resize`, and a file without removals is
unchanged.

A file that has removals cannot be read by earlier releases. Releases 0.3.7
through 0.5.1 check every cluster ID on `load` and reject the file as invalid
instead of returning removed points. Releases before 0.3.7 do not check cluster
IDs, so do not open such a file with them.
