# HNSW + RaBitQ
[HNSW](https://arxiv.org/abs/1603.09320) is a popular graph-based index. Under
comparable quantization settings, HNSW + RaBitQ typically uses more memory
than IVF + RaBitQ because it stores graph structures in addition to quantized
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
- **centroids**: Centroids computed by K-means clustering on the raw data vectors (we recommend `cluster_num = 16`).  
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
[cluster ID]
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
