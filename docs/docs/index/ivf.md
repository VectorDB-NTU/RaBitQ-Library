# IVF-RaBitQ

C++ callers include `rabitqlib/index/ivf_index.hpp` and link `rabitqlib::index`.

[IVF](https://doi.org/10.1109/TPAMI.2010.57) is a classical
clustering-based ANN method. IVF-RaBitQ stores RaBitQ codes and
vector IDs, optionally retaining raw vectors for reranking, and uses
[FastScan](https://dl.acm.org/doi/abs/10.1145/3078971.3078992) to estimate a
batch of distances. Its actual memory, latency, and recall depend on the bit
width, number of clusters, and `nprobe`.

## Minimal workflow

### Python

The [complete Quick Start](../quick_start.md) is the runnable IVF-RaBitQ example:
create an index, `add` the vectors (the first `add` trains the centroids), search,
then save and reload it. [Training](#training-centroids-in-python) describes the
clustering options, and the update examples below reuse the Quick Start's `index`.

### C++

`rabitqlib::IvfIndex` has the names, defaults and behaviour of the Python class.
In this snippet, `data` and `queries` are row-major `std::vector<float>` buffers
with dimension `dim`. As in Python, the first `add` trains the 1,024 centroids on
the added vectors, so supply at least 1,024 of them.

```cpp
#include <vector>
#include <rabitqlib/index/ivf_index.hpp>

rabitqlib::IvfOptions options;  // set the fields by name
options.nbits = 4;
rabitqlib::IvfIndex index(dim, 1024, options);
index.add(data.data(), data.size() / dim);

const size_t num_queries = queries.size() / dim;
std::vector<rabitqlib::Id> ids(num_queries * 10);
std::vector<float> distances(num_queries * 10);
index.search(queries.data(), num_queries, 10, ids.data(), distances.data());
index.save("ivf.index");
auto loaded = rabitqlib::IvfIndex::load("ivf.index");
```

See [C++ integration](../cpp_integration.md) for build setup and
[Index Construction](#index-construction) for building from precomputed
centroids and cluster IDs.

## Parameters and limits

| Parameter | Contract |
| --- | --- |
| `dim` | Original input dimension, 64–65,536; the index pads internally. |
| `num_clusters` | Number of centroids `train` or the first `add` learns, or `set_centroids` supplies; every cluster ID must be in `[0, num_clusters)`. |
| `nbits` | 1–9 total bits including the sign bit, or 32 for owned raw-vector reranking. Defaults to 7. |
| `seed` | Rotation seed in `[0, 2**32 - 1]` and the initial `cp.seed`. The default (Python `None`, an empty C++ `std::optional`) draws one at random; `seed` reports it, and passing it back reproduces the index. |
| `metric` | Python `"l2"`, `"ip"` or `"cosine"` (C++ `METRIC_L2`, `METRIC_IP` or `METRIC_COSINE`); defaults to L2. See [the cosine metric](usage.md#the-cosine-metric). |
| `routing` | Defaults to `"auto"` (C++ `ivf::Routing::Auto`); see [centroid routing](#centroid-routing). It replaces `initializer`, which 0.9 removed. |
| `k`, `nprobe` | Result count, from 1 to 2^31, and positive cluster probe count. `k` may exceed the stored point count; missing results are padded with ID `-1` (C++ `kNoId`) and `+inf`. `nprobe` defaults to `num_clusters / 16`, rounded and at least 1, when omitted (Python `None`); an explicit 0 raises `ValueError` (C++ `std::invalid_argument`). |
| `high_accuracy` | Defaults to automatic selection (Python `None`): HACC for 4–9 bits, standard FastScan otherwise. |

In C++, `nbits`, `seed`, `metric` and `routing` are fields of `IvfOptions`, and
`nprobe`, `high_accuracy` and `num_threads` fields of `IvfSearchOptions`.

Results contain IDs and squared L2 or `1 - dot` distances, ordered from smallest
to largest. Quantized distances are estimates. IDs are `int64` in Python and
`rabitqlib::Id` (`int64_t`) in C++. Unfilled slots use ID `-1` and distance `+inf`;
see [result handling](../quick_start.md#read-the-results).
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
`const char*` paths `data_file`, `centroids_file`, and `cluster_ids_file`. Include the
Eigen aliases and I/O helpers explicitly; `ivf_index.hpp` does not import them.

```c++
#include <rabitqlib/core/detail/eigen.hpp>
#include <rabitqlib/index/ivf_index.hpp>
#include <rabitqlib/index/types.hpp>
#include <rabitqlib/utils/io.hpp>

using PID = rabitqlib::PID;
using data_type = rabitqlib::RowMajorArray<float>;
using cluster_ids_type = rabitqlib::RowMajorArray<PID>;

data_type data;
data_type centroids;
cluster_ids_type cluster_ids;

rabitqlib::load_vecs<float, data_type>(data_file, data);
rabitqlib::load_vecs<float, data_type>(centroids_file, centroids);
rabitqlib::load_vecs<PID, cluster_ids_type>(cluster_ids_file, cluster_ids);
```

See the [complete indexing example](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/ivf_rabitq_indexing.cpp)
for argument parsing and the surrounding program.

Then create the index with the vector dimension, the number of clusters and the
total bits per dimension: 1 through 9, or 32 to store raw float32 vectors for
reranking.

```c++
size_t num_points = data.rows();
size_t dim = data.cols();
size_t num_clusters = centroids.rows();

rabitqlib::IvfOptions options;
options.nbits = total_bits;
rabitqlib::IvfIndex ivf(dim, num_clusters, options);
```

### Centroid routing

`IvfOptions::routing` selects `ivf::Routing::Auto`, `Flat`, `FlatRaBitQ`, or
`HNSW`. Python accepts `routing="auto"`, `"flat"`, `"flat_rabitq"`, or `"hnsw"`:

```python
index = IvfIndex(dim, num_clusters, nbits=4, routing="flat_rabitq")
print(index.routing)  # resolved routing type
```

```c++
options.routing = rabitqlib::ivf::Routing::FlatRaBitQ;
rabitqlib::IvfIndex ivf(dim, num_clusters, options);
auto routing = ivf.routing();  // resolved routing type
```

- `auto` selects `flat` below 5,000 clusters, `flat_rabitq` from 5,000 through
  59,999 clusters, and `hnsw` from 60,000 clusters onward. These defaults are
  practical thresholds; explicitly select a different routing when it better
  suits your workload.
- `flat` computes exact distances to every centroid.
- `flat_rabitq` scans one-bit centroid codes with FastScan, prunes candidates
  using twice the standard RaBitQ error margin, and recomputes exact distances
  for the retained candidates. Returned distances are exact for those centroids, but
  routing is approximate: pruning can miss a true nearest centroid.
- `hnsw` searches a graph of float32 centroids approximately. It suits workloads
  whose `nprobe` is much smaller than the cluster count. Graph construction can
  leave a few centroids unreachable, so points stored in those clusters are not
  returned at any `nprobe`, and an `nprobe` close to the cluster count raises an
  error instead of probing every cluster. Use `flat` or `flat_rabitq` when a
  query should probe most clusters.

All choices support every metric. Flat RaBitQ uses the IVF-RaBitQ rotation and
padded domain, retains float32 centroids for refinement, and stores its mean and
codes in the main index file. Its code precision is independent of the index's
`nbits`. It trades additional code storage and an O(cluster count) query work
buffer for fewer full-vector distance evaluations. Performance depends on
cluster count, dimension, `nprobe`, and data distribution. Measure recall and
latency on your workload, and use an explicit routing to override `auto`.

Compared with releases before 0.6.0, `auto` changes routing for 5,000–59,999
clusters. The 5,000–19,999 range changes from exact to approximate selection.
To retain the earlier policy when rebuilding, choose `flat` below 20,000 clusters
and `hnsw` otherwise. Loading an old index preserves its historical routing.

Routing is fixed for an index and is also used by `add` when cluster IDs are
omitted. `routing()` (Python `routing`) reports the resolved type,
including after loading; loading never re-runs the current `auto` policy for a
new-format file.

Finally, set the centroids and add the vectors with their cluster IDs:

```c++
ivf.set_centroids(centroids.data());
rabitqlib::AddOptions add_options;
add_options.cluster_ids = cluster_ids.data();
add_options.fast_quantization = true;
ivf.add(data.data(), num_points, add_options);
```

- **centroids**: One centroid per configured cluster, in the original input coordinates.
- **cluster_ids**: One ID per input vector, in `[0, num_clusters())`.
- **fast_quantization**: Use a calibrated scaling factor instead of per-vector scale search; defaults to `false`. See [encoding paths](../rabitq/quantizer.md#implementations).
- **num_threads**: Worker count; no value, the default, uses every available CPU.

### Training centroids in Python

`train(x)` learns `num_clusters` centroids from the rows of `x`, which must have
at least `num_clusters` rows, and stores them in the index; `add` then assigns
and quantizes vectors. On an untrained index, the first `add` without
`cluster_ids` calls `train` on the added vectors first, so `train` is needed only
to train on a different sample. Training uses the settings in `index.cp`, a
[`RaBitQKMeansParameters`](../clustering.md#rabitqkmeans) whose defaults differ
from the k-means classes: 10 iterations, at most 256 sampled vectors per
centroid (`max_points_per_centroid`), spherical k-means for `metric="ip"` and `"cosine"`, and
`seed` equal to the index seed. Change fields before training:

```python
index = IvfIndex(dim, 1024, nbits=4, metric="ip")
index.cp.niter = 20
index.train(sample)  # num_threads=None uses index.cp.num_threads (0: all CPUs)
index.add(vectors)
```

- Up to 2,000 clusters, training uses RaBitQKMeans; above that it uses
  [QGKMeans](../clustering.md#qgkmeans), copying the shared fields of `cp` and
  keeping its default graph settings. The threshold is provisional and has not
  been measured.
- `set_centroids(centroids)` uses centroids of shape `(num_clusters, dim)`
  computed elsewhere instead of training. Its keyword `num_threads` sets the
  workers that rotate the centroids and build the routing; only the HNSW routing
  graph varies with it.
- Training or setting centroids again replaces them while the index stores no
  vectors; once it stores any (including removed ones), both raise
  `RuntimeError`. A failed call leaves the index unchanged.
- Before training, `add` with `cluster_ids`, `search`, `remove`, `reserve_cluster` and
  `save` raise `RuntimeError`; `add` without `cluster_ids` trains first. If that
  first `add` fails, the index stays untrained and empty; an empty first `add`
  raises `RuntimeError`. A trained index without
  vectors cannot be saved, and its `search` returns only padded results; add
  vectors first. Loaded indexes are trained; `seed` is `None` for files without
  a recorded seed, such as those from releases up to 0.8.
- `train` and `add` default to every available CPU; `search` defaults to one
  thread.

### Training centroids in C++

`IvfIndex` trains as the Python class does. `cp()` returns the training
settings; include `rabitqlib/clustering/rabitqkmeans.hpp` to change them:

```c++
#include <rabitqlib/clustering/rabitqkmeans.hpp>
#include <rabitqlib/index/ivf_index.hpp>

rabitqlib::IvfOptions options;
options.metric = rabitqlib::METRIC_IP;
options.nbits = 4;
rabitqlib::IvfIndex index(dim, 1024, options);
index.cp().niter = 20;
index.train(sample.data(), num_samples);  // no thread count: cp().num_threads
// Or supply centroids: index.set_centroids(centroids.data());
index.add(data.data(), num_points);
```

- `train`, `set_centroids` and the first `add` follow the Python rules above,
  including the thresholds, the defaults of `cp()` and the state checks.
- Arguments outside their range throw `std::invalid_argument` (Python
  `ValueError`), and a call in the wrong state throws `std::logic_error` (Python
  `RuntimeError`), with the Python messages except where those name Python values
  such as `None`.
- `is_trained()` reports whether centroids are set. A trained index without
  points pads every `search` result slot with `kNoId` and `+inf`; `save` throws
  `std::logic_error` until the index stores at least one point.
- `train` without a thread count and the first `add` without
  `AddOptions::num_threads` train, and build the centroid routing, with
  `cp().num_threads`; `add` quantizes with every available CPU unless
  `num_threads` is set.

## Querying

To search a saved index, load it first:

```c++
auto ivf = rabitqlib::IvfIndex::load(index_file);
```
Search a trained or loaded index with:

```c++
void IvfIndex::search(
    const float* queries,
    size_t num_queries,
    size_t k,
    Id* ids,
    float* distances,
    const IvfSearchOptions& options = {}
) const;
```

- **queries**: `num_queries` contiguous row-major float32 vectors of dimension `dim()`.
- **k**: Results per query.
- **ids**, **distances**: Output buffers of `num_queries * k` values; `distances` may be `nullptr`.
- **options**: `nprobe`, `high_accuracy` and `num_threads`; see below.

FastScan precision is selected automatically: HACC for 4–9 quantized bits, and
standard FastScan for 1–3 bits or raw vectors (`32`). HACC reduces lookup-table
error carried into extra-bit distance estimation; raw reranking computes its
final distances directly from floats. `IvfSearchOptions::high_accuracy` (Python
`high_accuracy=True/False`) overrides the selection; no value (Python `None`)
selects automatically.

```cpp
rabitqlib::IvfSearchOptions options;
options.nprobe = 64;
ivf.search(queries, num_queries, k, ids, distances, options);  // Automatic
options.high_accuracy = true;  // Force HACC
ivf.search(queries, num_queries, k, ids, distances, options);
options.high_accuracy = false;  // Force standard FastScan
ivf.search(queries, num_queries, k, ids, distances, options);
```

In Python, call `index.search(queries, k, nprobe)` for automatic
selection, or pass `high_accuracy=True` / `False` to force either mode.
The querying examples also default to automatic selection; Python's
`--use-hacc` flag forces HACC, and the C++ example accepts an optional final
`true` or `false` argument.

`num_threads` (default 1; 0 selects every available CPU) distributes the
queries across OpenMP workers. An empty batch does no work. Input and output
buffers must not overlap. Simultaneous searches need separate output buffers,
and must not overlap index updates. Python's
`index.search(queries, k, nprobe, num_threads=4)` takes the same settings.

Concurrent Python searches may use independent `nprobe`, `high_accuracy`, and
`num_threads` values. See the [concurrency contract](usage.md#threading-and-file-paths).

## Updating an index

`IvfIndex` supports `add()` and `remove()` on a trained or loaded index without
the original dataset. `save` always writes index format 2; see
[how removal is stored](#how-removal-is-stored).

```c++
Id IvfIndex::add(const float* data, size_t num_points, const AddOptions& options = {});

size_t IvfIndex::remove(const Id* ids, size_t num_ids);

void IvfIndex::reserve_cluster(size_t cluster_id, size_t num_points);
```

- **data**, **num_points**: `num_points` new vectors. Vector `i` receives the ID
  `num_stored_points() + i`, using the count before the call; `add` returns the
  first. Existing IDs stay unchanged, and removed IDs are never reused.
- **options.cluster_ids**: The cluster of each new vector, in `[0, num_clusters)`. When
  it is `nullptr`, each vector goes to its nearest centroid under the index's
  routing type, the one selected at creation or restored from the
  index file. With Flat routing, `add` assigns exactly and queries scan in float,
  so near-ties may resolve differently.
- **options.fast_quantization**, **options.num_threads**: As in [index construction](#index-construction).
- **ids**, **num_ids**: IDs to remove. Every ID must be in `[0, num_stored_points())`; nothing is
  removed if one is not. `remove` returns how many were newly removed and can be
  repeated safely.
- **cluster_id**, **num_points**: `reserve_cluster` preallocates room for cluster `cluster_id`
  to hold `num_points` vectors in total, counting the ones it already holds. It does
  nothing if the cluster already has that room. See [Cost of `add`](#cost-of-add).

In Python:

```python
new_ids = index.add(vectors, num_threads=1, fast_quantization=False)
# To choose clusters explicitly, use this instead of the call above:
# new_ids = index.add(vectors, cluster_ids=labels)
removed = index.remove(new_ids[:10])
index.reserve_cluster(cluster_id, num_points)
```

In Python, `reserve_cluster` replaces `reserve`, which 0.9 removed.

Python `vectors` has shape `(n, index.dim)` and is converted to contiguous float32.
`add()` returns an int64 array of the new IDs. Optional `cluster_ids` must be a
one-dimensional integer array or sequence with one valid cluster ID per vector;
`remove()` likewise accepts a one-dimensional integer array or sequence of point
IDs and returns the number newly removed. Duplicate IDs count once. Python
`num_threads` defaults to `None` (every available CPU) and `fast_quantization` to
`False`; pass both by keyword.

Updates require exclusive access; see the
[concurrency contract](usage.md#threading-and-file-paths).

### Cost of `add`

Each cluster has its own storage, and `add` only touches the clusters that receive
new vectors. Each of those is rebuilt next to its old storage and swapped in, so the
index is unchanged if `add` throws. As a result:

- A call needs extra memory for the clusters it touches while it runs.
- Each call costs time proportional to the sizes of the touched clusters, on top of a
  cost per new vector. Routed vectors usually touch most clusters, so a call can
  still copy most of the index. **Add many vectors per call rather than one at a
  time.**

A rebuilt cluster gets exactly the room it needs. To grow a cluster without
copying it, call `reserve_cluster` first: later `add` calls then write into the reserved
room in place until the cluster outgrows it. The reserved room is memory the index
holds but does not use yet. It is not saved, so a loaded index has none.

Measure update latency and peak memory with your index size, bit width and batch
size; serialized file size alone does not determine peak process memory.

### Recall after `add`

The centroids and the rotation are fixed once the index is trained, so
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
IVF-RaBitQ distances must keep this behavior and must never turn `+inf` into NaN.

Search may return fewer than `k` results once points are removed: the missing slots
hold ID `-1` (C++ `kNoId`) with an infinite distance, as when the probed
clusters hold fewer than `k` points.

The removal marker lives in the ordinary batch data and needs no additional
format change; removal survives `save` and `load`. No file written before
`remove` existed has an infinite `f_add`, so nothing in an old file is
reinterpreted. Release 0.5.0 also excludes removed points when reading its
supported legacy formats. However, current saves always use index format 2,
including when resaving a loaded legacy index. Releases up to 0.8 cannot read these
files, regardless of whether they contain removals.

## Persistence

Save an index that stores at least one point with:

```c++
ivf.save(output_index_file);
```
Files are written in
[index format 2](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/formats.md#index-format-2):
one file with a header, a section table, and sections for the rotation, the routing
data, the codes and the IDs. HNSW routing is a section of that file, so there is no
separate sidecar. `Auto` is resolved before saving. Loading restores the routing type,
storage mode, rotation state, and padded dimension. Raw files also contain the original
vectors, so querying does not require an external dataset. Releases up to 0.8 cannot
read format 2. Indexes of more than 65,536 dimensions cannot be saved.

Files from earlier releases remain readable. `RABQIDX1` version 2 files store their
routing type; version-1 `RABQIDX1` files keep their stored padding; legacy unversioned
quantized files and raw v1 files retain their original 64-dimension padding. These three
older layouts infer routing with the original 20,000 cluster threshold. All of these
layouts keep HNSW routing in the accompanying `.hnsw` file, which loading reads.
Flat RaBitQ loads its saved codes without random re-quantization. Saving a loaded file
writes format 2 without changing its padded dimension or encoded data; a `.hnsw` file
left beside it is not removed and is no longer used.

## Implementation details

### Construction

During construction, we process clusters in parallel, rotate their centroids and
vectors, and compute one-bit codes and factors. Quantized mode also computes
`nbits - 1` extra-bit codes and factors; raw mode stores the original
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

IVF-RaBitQ stores the following data:

```c++
[batch data]        // 1-bit code and factors
[ex_data]           // code for remaining bits, or original float32 vectors
[ids]               // PID of vectors (organized by clusters)
[cluster metadata]  // List of clusters' metadata in IVF-RaBitQ
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
