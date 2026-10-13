# IVF RaBitQ

C++ callers include `rabitqlib/index/ivf/ivf.hpp` and link `rabitqlib::index`.

[IVF](https://dl.acm.org/doi/10.1109/TPAMI.2010.57) is a classical
clustering-based ANN method. This implementation stores RaBitQ codes and
vector IDs, optionally retaining raw vectors for reranking, and uses
[FastScan](https://dl.acm.org/doi/abs/10.1145/3078971.3078992) to estimate a
batch of distances. Its actual memory, latency, and recall depend on the bit
width, number of clusters, and `nprobe`.

## Minimal workflow

### Python

The [complete Quick Start](../quick_start.md) is the runnable IVF example:
train centroids, call `IvfIndex.build`, search, then save and reload the index.
The update examples below reuse its built `index`.

### C++

Link `rabitqlib::index`. In this snippet, `data`, `centroids` and `queries` are
row-major `std::vector<float>` buffers with original dimension `dim`;
`cluster_ids` is a `std::vector<rabitqlib::PID>` with one centroid ID per data row.
The example requests ten neighbors from five clusters, so supply at least ten
points and five centroids.

```cpp
#include <vector>
#include <rabitqlib/index/ivf/ivf.hpp>

rabitqlib::ivf::IVF index(data.size() / dim, dim, centroids.size() / dim, 4);
index.construct(data.data(), centroids.data(), cluster_ids.data(), false, 1);
const size_t num_queries = queries.size() / dim;
std::vector<rabitqlib::PID> ids(num_queries * 10);
std::vector<float> distances(num_queries * 10);
index.search_batch(queries.data(), num_queries, 10, 5, ids.data(), distances.data());
index.save("ivf.index");
```

See [C++ integration](../cpp_integration.md) for build setup and
[Index Construction](#index-construction) for loading `.fvecs` inputs.

## Parameters and limits

| Parameter | Contract |
| --- | --- |
| `dim` | Original input dimension; Python supports 64–65,536 and pads internally. |
| `max_elements` | Python IVF build requires exactly this many rows; `add` grows storage automatically. C++ calls this constructor argument `num_points`. |
| `num_clusters` | Must match the centroid count; every cluster ID must be in `[0, num_clusters)`. |
| `nbits` / C++ `total_bits` | 1–9 total bits including the sign bit, or 32 for owned raw-vector reranking. |
| `metric` | Python `"l2"` or `"ip"`; defaults to `"l2"`. |
| `initializer` | Defaults to `"auto"`; see [centroid routing](#centroid-routing). |
| `k`, `nprobe` | Positive result count and cluster probe count; Python requires `k <= max_elements`. |
| `high_accuracy` | Python defaults to `None`: HACC for 4–9 bits, standard FastScan otherwise. |

Results contain IDs and squared L2 or `1 - dot` distances, ordered from smallest
to largest. Quantized distances are estimates. Unfilled slots use ID `2**32 - 1`
and distance `+inf`; see [result handling](../quick_start.md#read-the-results).
See the shared [input and concurrency contracts](usage.md).

## Index construction

See the shared [index count contract](usage.md#index-counts).

Use [QGKMeans](../clustering.md#qgkmeans) to partition the input vectors (`.fvecs`).
Build the C++ examples, then run `bin/qgkmeans` to save centroids and cluster IDs.
For small cluster counts, use `bin/rabitqkmeans` for
[RaBitQKMeans](../clustering.md#rabitqkmeans) flat assignment. Both use exact final
assignment. The optional metric defaults to `l2`; append `ip` for normalized input.

For example, split SIFT into 4,096 clusters using squared L2 distance:

```shell
./bin/qgkmeans /data/sift/sift_base.fvecs 4096 \
    /data/sift/sift_centroids_4096_l2.fvecs \
    /data/sift/sift_clusterids_4096_l2.ivecs
```

After files are prepared, load them into memory. The snippet below uses
`const char*` paths `data_file`, `centroids_file`, and `cids_file`. Include the
Eigen aliases and I/O helpers explicitly; `ivf.hpp` does not import them.

```c++
#include <rabitqlib/core/detail/eigen.hpp>
#include <rabitqlib/index/ivf/ivf.hpp>
#include <rabitqlib/index/types.hpp>
#include <rabitqlib/utils/io.hpp>

using PID = rabitqlib::PID;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<PID>;

data_type data;
data_type centroids;
gt_type cids;

rabitqlib::load_vecs<float, data_type>(data_file, data);
rabitqlib::load_vecs<float, data_type>(centroids_file, centroids);
rabitqlib::load_vecs<PID, gt_type>(cids_file, cids);
```

See the [complete indexing example](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/ivf_rabitq_indexing.cpp)
for argument parsing and the surrounding program.

Then initialize an IVF object with the number of data points, vector
dimension, number of clusters, and total bits per dimension. The C++ index
accepts total quantized bit widths from 1 through 9, or `total_bits=32`
(Python `nbits=32`) to store raw float32 vectors for reranking.

```c++
using index_type = rabitqlib::ivf::IVF;

size_t num_points = data.rows();
size_t dim = data.cols();
size_t k = centroids.rows();

index_type ivf(num_points, dim, k, total_bits);
```

### Centroid routing

The optional final C++ constructor argument selects `InitializerType::Auto`,
`Flat`, `FlatRaBitQ`, or `HNSW` from the `rabitqlib::ivf` namespace. Python accepts
`initializer="auto"`, `"flat"`, `"flat_rabitq"`, or `"hnsw"`:

```python
index = IvfIndex(dim, num_points, num_clusters, nbits=4,
                 initializer="flat_rabitq")
print(index.initializer)  # resolved routing type
```

```c++
index_type ivf(num_points, dim, k, total_bits, rabitqlib::METRIC_L2,
               rabitqlib::RotatorType::FhtKacRotator,
               rabitqlib::ivf::InitializerType::FlatRaBitQ);
auto routing = ivf.initializer_type();
```

- `auto` selects `flat` below 5,000 clusters, `flat_rabitq` from 5,000 through
  59,999 clusters, and `hnsw` from 60,000 clusters onward. These defaults are
  practical thresholds; explicitly select a different initializer when it better
  suits your workload.
- `flat` computes exact distances to every centroid.
- `flat_rabitq` scans one-bit centroid codes with FastScan, prunes candidates
  using twice the standard RaBitQ error margin, and recomputes exact distances
  for the retained candidates. Returned distances are exact for those centroids, but
  routing is approximate: pruning can miss a true nearest centroid.
- `hnsw` searches a graph of float32 centroids approximately.

All choices support L2 and inner product. Flat RaBitQ uses the IVF rotation and
padded domain, retains float32 centroids for refinement, and stores its mean and
codes in the main index file. Its code precision is independent of the index's
`nbits`. It trades additional code storage and an O(cluster count) query work
buffer for fewer full-vector distance evaluations. Performance depends on
cluster count, dimension, `nprobe`, and data distribution. Measure recall and
latency on your workload, and use an explicit initializer to override `auto`.

The optional routing argument introduced in 0.6.0 preserves existing constructor
call forms; this does not imply compatibility with all 0.7.0 C++ changes. See the
[migration guide](../modular_migration.md) and rebuild with matching headers and libraries.

Compared with releases before 0.6.0, `auto` changes routing for 5,000–59,999
clusters. The 5,000–19,999 range changes from exact to approximate selection.
To retain the earlier policy when rebuilding, choose `flat` below 20,000 clusters
and `hnsw` otherwise. Loading an old index preserves its historical routing.

Routing is fixed for an index and is also used by `add` when cluster IDs are
omitted. `initializer_type()` (Python `initializer`) reports the resolved type,
including after loading; loading never re-runs the current `auto` policy for a
new-format file.

Finally, call the construct API:

```c++
void IVF::construct(
    const float* data, 
    const float* centroids, 
    const PID* cluster_ids, 
    bool faster = false,
    size_t num_threads = std::numeric_limits<size_t>::max()
);
```

- **data**: Vectors in the original input coordinates.
- **centroids**: One centroid per configured cluster, in the original input coordinates.
- **cluster_ids**: One ID per input vector, in `[0, num_clusters())`.
- **faster**: Use a calibrated scaling factor instead of per-vector scale search; defaults to `false`. See [encoding paths](../rabitq/quantizer.md#implementations).
- **num_threads**: Maximum number of OpenMP threads used to quantize clusters.

For example:

```c++
ivf.construct(data.data(), centroids.data(), cids.data(), true);
```

## Querying

To search a saved index, load it first:

```c++
using index_type = rabitqlib::ivf::IVF;
index_type ivf;
ivf.load(index_file);
```
Search a constructed or loaded index with:

```c++
void IVF::search(
    const float* __restrict__ query, 
    size_t k, 
    size_t nprobe, 
    PID* __restrict__ results,
    float* dists = nullptr
) const;
```

- **query**: Query vector.
- **k**: Top-k.
- **nprobe**: The number of closest clusters to search.
- **results**: Result buffer, size of k.
- **dists**: Optional distance buffer, size of `k`.

FastScan precision is selected automatically: HACC for 4–9 quantized bits, and
standard FastScan for 1–3 bits or raw vectors (`32`). HACC reduces lookup-table
error carried into extra-bit distance estimation; raw reranking computes its
final distances directly from floats. Existing C++ overloads with `use_hacc`
and Python's `high_accuracy=True/False` still allow explicit overrides. Python
uses automatic selection when `high_accuracy` is omitted or `None`.

```cpp
ivf.search(query, k, nprobe, ids, distances);         // Automatic
ivf.search(query, k, nprobe, ids, distances, true);   // Force HACC
ivf.search(query, k, nprobe, ids, distances, false);  // Force standard FastScan
```

The IDs-only overloads accept the same override without the `distances`
argument. In Python, call `index.search(queries, k, nprobe)` for automatic
selection, or pass `high_accuracy=True` / `False` to force either mode.
The querying examples also default to automatic selection; Python's
`--use-hacc` flag forces HACC, and the C++ example accepts an optional final
`true` or `false` argument.

C++ `search_batch` distributes queries across workers.
Input queries are contiguous row-major float32
vectors with shape `(num_queries, dimension())`; output buffers have shape
`(num_queries, k)`.

```cpp
ivf.search_batch(queries, num_queries, k, nprobe, ids, distances,
                 std::nullopt, 4);  // Automatic precision, up to four workers
```

The last two arguments are `std::optional<bool> use_hacc` (default
`std::nullopt`) and `size_t num_threads` (default `1`, using OpenMP for parallel
batches). Pass `true` or `false`
for an explicit precision override. Distances may be `nullptr`. An empty batch
does no work. Input and output buffers must not overlap. Simultaneous searches
need separate output buffers, and must not overlap index updates.
Python's existing `index.search(queries, k, nprobe, num_threads=4)` uses this
batch path with the same output and precision rules.

Concurrent Python searches may use independent `nprobe`, `high_accuracy`, and
`num_threads` values. See the [concurrency contract](usage.md#threading-and-file-paths).

## Updating an index

C++ `IVF` and Python `IvfIndex` support `add()` and `remove()` on a constructed
or loaded index without the original dataset. Updates preserve the file format.

```c++
void IVF::add(
    const float* data,
    size_t n,
    const PID* cluster_ids = nullptr,
    bool faster = false,
    size_t num_threads = std::numeric_limits<size_t>::max()
);

size_t IVF::remove(const PID* ids_to_remove, size_t n);
```

- **data**, **n**: `n` new vectors, quantized like `construct` does. Vector `i`
  receives the PID `num_stored_points() + i`, using the count before the call. Existing
  IDs stay unchanged, and removed IDs are never reused.
- **cluster_ids**: The cluster of each new vector, in `[0, cluster_num)`. When
  it is `nullptr`, vectors use the same centroid routing as queries. Routing uses
  the initializer selected at construction or restored from the index file.
- **faster**, **num_threads**: Same as in `construct`.
- **ids_to_remove**: PIDs to remove. Every PID must be below `num_stored_points()`; nothing is
  removed if one is not. `remove` returns how many were newly removed and can be
  repeated safely.

In Python:

```python
new_ids = index.add(vectors, num_threads=1, fast_quantization=False)
# To choose clusters explicitly, use this instead of the call above:
# new_ids = index.add(vectors, cluster_ids=labels)
removed = index.remove(new_ids[:10])
```

Python `vectors` has shape `(n, index.dim)` and is converted to contiguous float32.
`add()` returns a uint32 array of the new IDs. Optional `cluster_ids` must be a
one-dimensional integer array or sequence with one valid cluster ID per vector;
`remove()` likewise accepts a one-dimensional integer array or sequence of point
IDs and returns the number newly removed. Duplicate IDs count once. Python
`num_threads` defaults to 1; `fast_quantization` defaults to `False`.

After appending, rebuilding with `construct` or `build` requires
`num_stored_points` rows, including removed records. Updates require exclusive
access; see the
[concurrency contract](usage.md#threading-and-file-paths).

### Cost of `add`

`add` builds the grown index next to the old one and swaps it in, so the index is
unchanged if it throws. Every call copies the whole storage, so:

- It needs memory for both copies while it runs.
- Each call has a fixed cost that grows with the size of the whole index, on top of a
  cost per new vector. **Add many vectors per call rather than one at a time.**

On an index of 1,000,000 vectors with 128 dimensions, 1,024 clusters and 1 bit
(8 threads, portable build, Intel i7-12700KF), adding 1,000 vectors in one call took
about 5 ms, and adding the same vectors one call each took about 3 s (2.8 s and 3.7 s
in two runs). For that 33 MB file, a process that had just loaded it held about
65 MB, and peaked at about 100 MB during one `add` of 1,000 vectors.

There is no spare capacity: every `add` moves the data.

### Recall after `add`

The centroids and the rotation are fixed when the index is constructed, so
`add` never retrains them. If the added vectors come from a different distribution,
or the index grows many times beyond its original size, recall at a given
`nprobe` can drop. Rebuild from the original data with new centroids when
recall matters more than the cost of a rebuild.

### How removal is stored

`remove` hides points from search but keeps their storage. Removed points still
count in `num_stored_points()` and cannot be restored. Each nonempty call scans all
stored point IDs, so batch removals when possible.

A removed point has its `f_add` value set to `+inf`. That value is the constant
term of the distance estimate, so the estimated distance and the lower bound of the
point are `+inf` for every bit width, metric, and SIMD backend. The point never
enters the result buffer and never triggers reranking. Any code that estimates
IVF distances must keep this behavior and must never turn `+inf` into NaN.

Search may return fewer than `k` results once points are removed: the missing slots
hold `kPidMax` (`2**32 - 1` in Python) with an infinite distance, as when the probed
clusters hold fewer than `k` points.

The removal marker lives in the ordinary batch data and needs no additional
format change; removal survives `save` and `load`. No file written before
`remove` existed has an infinite `f_add`, so nothing in an old file is
reinterpreted. Release 0.5.0 also excludes removed points when reading its
supported legacy formats. However, current saves always use `RABQIDX1` v2,
including when resaving a loaded legacy index. Older releases cannot read these
files, regardless of whether they contain removals.

## Persistence

Save a constructed index with:

```c++
ivf.save(output_index_file);
```
New files use the `RABQIDX1` magic, version 2, storage flags, an explicit
padded dimension, and a uint32 initializer type (1 = Flat, 2 = Flat RaBitQ,
3 = HNSW) before the index metadata. `Auto` is resolved before saving. Loading
restores the routing type, storage mode, rotation state, and padded dimension.
Raw files also contain the original vectors, so querying does not require an external dataset.

Version-1 `RABQIDX1` files remain readable with their stored padding. Legacy
unversioned quantized files and raw v1 files retain their original 64-dimension
padding. All three historical formats infer routing with the original 20,000
cluster threshold. HNSW routing still requires the accompanying `.hnsw` file.
Flat RaBitQ loads its saved codes without random re-quantization. Saving a loaded
legacy index writes the new header without changing its padded dimension or encoded data. Older library
versions cannot read the new format.

## Implementation details

### Construction

During construction, we process clusters in parallel, rotate their centroids and
vectors, and compute one-bit codes and factors. Quantized mode also computes
`total_bits - 1` extra-bit codes and factors; raw mode stores the original
float32 vectors instead. In raw mode, the index owns a copy of the input, so
the original data can be released after construction.

Construction reuses a rotation buffer of at most 32 vectors per worker. This
temporary storage uses at most `workers * 32 * padded_dim * sizeof(float)` bytes,
independently of cluster sizes.

New indexes pad the rotated dimension to a multiple of 32 for every supported
bit width, including raw mode (32). This applies to both rotator types.
Extra-bit layouts retain full 64-coordinate blocks and store any final
32-coordinate tail compactly.

### Data layout

IVF stores the following data:

```c++
[batch data]    // 1-bit code and factors
[ex_data]       // code for remaining bits, or original float32 vectors
[ids]           // PID of vectors (organized by clusters)
[cluster_lst]   // List of clusters' metadata in IVF
```
In raw mode, `ex_data` holds `4 * num_points * dim` bytes in cluster order,
without padding or extra-bit factors. The one-bit filtering codes and factors
are additional storage. `nbits()` (Python `nbits`) reports 32 in raw mode;
the one-bit filtering code is not included in that value.

### Search algorithm

During search, we rotate the query and select the `nprobe` closest centroids.
FastScan filters candidates using one-bit codes, then reranking uses either the
remaining quantized bits or the stored raw vectors. Raw reranking computes
squared L2 or `1 - dot(query, vector)` in the original coordinates; the search
API is unchanged. Cluster selection and filtering remain approximate in both
modes. Search returns the top `k` results after scanning the selected clusters.
