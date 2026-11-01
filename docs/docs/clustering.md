# Clustering

RaBitQ Library provides two implementations of Lloyd's k-means. Both compute
centroid means from the original float32 data, accumulate in float64, and require
no external clustering package. Choose the class explicitly; neither switches
algorithms based on the cluster count.

For installation, see [Quick start](quick_start.md#install).

You do not need these classes to build an index: `IvfIndex.train` and
`HnswIndex.train` run them for you (see
[IVF-RaBitQ training](index/ivf.md#training-centroids-in-python)). Use them
directly to inspect assignments, reuse one clustering across several indexes,
or pass the result to `set_centroids` and `add(..., cluster_ids=...)`. The
indexes train with their own defaults, including sampling at most 256 vectors
per centroid; the classes below default to training on every vector.

## Choose a method

| Class | Assignment | Recommended use | Cluster count |
| --- | --- | --- | --- |
| [RaBitQKMeans](#rabitqkmeans) | Flat RaBitQ scan | Small cluster counts | `n >= k >= 1` |
| [QGKMeans](#qgkmeans) | SymphonyQG centroid graph | Larger cluster counts | `n >= k > graph_degree` (default 32) |

## Python

Both classes have the same training and result interface:

```python
from rabitqlib import QGKMeans, RaBitQKMeans

kmeans = RaBitQKMeans(d, 16, niter=25, num_threads=8)
# For larger cluster counts, use instead:
# kmeans = QGKMeans(d, 4096, niter=25, num_threads=8)
last_iteration_obj = kmeans.train(x)

centroids = kmeans.centroids      # shape (k, d)
assignments = kmeans.assignments  # shape (n,)
distances = kmeans.distances      # shape (n,)
final_obj = kmeans.final_obj

# Assign other vectors, shape (m, d), to the trained centroids.
query_assignments, query_distances = kmeans.assign(queries)  # shapes (m,), (m,)
```

`x` has shape `(n, d)` and is converted to a C-contiguous float32 array.

## C++

Clustering belongs to `rabitqlib::index`, including the flat RaBitQ method.
Link that target; a core-only installation does not include clustering headers.

```cpp
#include <rabitqlib/clustering/rabitqkmeans.hpp>
#include <rabitqlib/clustering/qgkmeans.hpp>

rabitqlib::rabitqkmeans::RaBitQKMeansParameters parameters;
parameters.niter = 25;
parameters.num_threads = 8;
rabitqlib::rabitqkmeans::RaBitQKMeans kmeans(dimension, 16, parameters);

// For graph assignment, use qgkmeans::QGKMeansParameters and qgkmeans::QGKMeans.
kmeans.train(num_points, data);  // row-major float32 input

const auto& centroids = kmeans.centroids;
const auto& assignments = kmeans.assignments;
const auto& distances = kmeans.distances;
const double final_obj = kmeans.final_obj;

// Caller-owned outputs with num_queries elements each.
std::vector<rabitqlib::PID> query_assignments(num_queries);
std::vector<float> query_distances(num_queries);
kmeans.assign(num_queries, queries, query_assignments.data(), query_distances.data());
```

Complete examples: [RaBitQKMeans](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/rabitqkmeans.cpp)
(`bin/rabitqkmeans`) and [QGKMeans](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/qgkmeans.cpp)
(`bin/qgkmeans`). Run either without arguments for a synthetic demo, or save
clusters for the C++ indexing examples:

```bash
./bin/qgkmeans data.fvecs 4096 centroids.fvecs labels.ivecs
./bin/rabitqkmeans data.fvecs 16 centroids.fvecs labels.ivecs
```

File mode uses exact final assignment and the default training parameters.
Append `ip` for spherical clustering of normalized input; the default is `l2`.
`sample/run_cpp.sh` runs the C++ clustering and indexing workflow without Python.

## Parameters and input

Set Python keyword arguments or fields of the corresponding C++ parameter type:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `niter` | `25` | Maximum clustering iterations. |
| `early_stop_threshold` | `0.0` | Stop from iteration 2 when the absolute relative objective change is at most this value; range `[0, 1]`. Zero stops only on an unchanged objective. |
| `num_threads` | `0` | Use detected available logical CPUs; positive requests are capped at that count. See [threading](index/usage.md#threading-and-file-paths). |
| `spherical` | `false` | Normalize centroids and assign by inner product; input must already be normalized. |
| `seed` | `42` | Seed for centroid selection and rotation. |
| `min_points_per_centroid` | `39` | Warn below this training-points-per-centroid ratio when verbose. |
| `max_points_per_centroid` | `0` | When positive, train on a random sample above `k * max_points_per_centroid` vectors; `0` trains on all vectors. See [sampling](#training-set-sampling). |
| `verbose` | `false` | Print training progress and warnings. |
| `final_assignment` | `Approximate` | Use the selected method for final labels, or `Exact` for nearest-centroid labels. |

`num_threads` sets an upper limit, not a fixed worker count for every stage.
QGKMeans may use fewer workers to build small centroid graphs. That additional
limit does not apply to training-vector assignment, which uses the thread budget
after capping it to the available CPUs.

Training requires `64 <= d <= 65536`, finite coordinates, and
`abs(x) <= sqrt(sqrt(FLT_MAX) / (64 * d))`; scale larger inputs before training.
Cluster-count requirements are listed above. Training does not normalize or
mutate the input; provide normalized vectors for spherical mode.

Empty clusters are reseeded from distant points; duplicate data can still leave
empty clusters in the final assignment. Thread scheduling can affect graph
construction and floating-point reductions, so a fixed seed does not guarantee
identical multithreaded results.

### Training-set sampling

Sampling is disabled by default. When `max_points_per_centroid` is positive
and `n > k * max_points_per_centroid`, training draws
`k * max_points_per_centroid` distinct vectors uniformly at random, as Faiss
does; Faiss enables this by default with 256. Centroid initialization and all
Lloyd iterations use only this sample. The final centroids then assign all `n`
vectors with the selected `final_assignment` mode, as [`assign`](#assign-new-vectors)
does, so `assignments` and `distances` keep shape `(n,)` and `assignments` can be
passed directly to an index's `add(..., cluster_ids=...)`. C++ callers that need
only the centroids can call `train_centroids(n, x)` instead of `train`, which
skips this final assignment and leaves `assignments`, `distances` and
`final_obj` empty or zero.

`seed` selects the sample through a random stream separate from centroid
initialization. Training that does not sample, because the parameter is `0` or
`n` is at most `k * max_points_per_centroid`, returns the same results as
before this parameter existed. The selected vectors depend on the
C++ standard library's random distributions, so the same seed can choose a
different sample on another platform.

`obj` and `iteration_stats` (and the value returned by Python `train()`) are
computed on the sampled vectors, while `distances` and `final_obj` cover all `n`
vectors. Their values are not directly comparable. Validation still checks
every input vector, including those outside the sample.

Sampling reduces iteration work but raises peak memory. Training copies the
sample into a contiguous buffer of `k * max_points_per_centroid * d` float32
values, 4 GiB for `k = 4096`, `max_points_per_centroid = 256` and `d = 960`,
and briefly holds `n` 8-byte row indices to select it, in addition to the
caller's input. The final assignment still processes all `n` vectors.

## Method details

### RaBitQKMeans

The flat assignment design is inspired by
[SuperKMeans](https://github.com/cwida/SuperKMeans). Training points are rotated
and encoded once around a fixed dataset mean. Each centroid becomes a query
that scans all packed point batches, and each point receives the centroid with
the best estimated distance. The model keeps this mean of the training points
(of the sample, when training samples), and [`assign`](#assign-new-vectors)
encodes new vectors around it.

Each iteration compares every training point with every centroid. The batch size of 32
is not a cluster-count limit. This method uses one-bit point codes throughout;
it has no graph parameters or `quantization_bits` setting.

### QGKMeans

QGKMeans builds a SymphonyQG graph over the centroids and uses previous
assignments as search hints. Its additional parameters are:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `quantization_bits` | `0` | Graph centroid storage: raw, or `4`/`8` bits. |
| `graph_degree` | `32` | A positive multiple of 32, strictly smaller than `k`. |
| `ef_build`, `ef_search` | `240`, `16` | Positive graph construction and search window sizes. |
| `graph_build_iterations` | `1` | PiPNN initialization followed by one refinement; values ≥2 request that many graph passes, ending in refinement. Independent of `niter`. |

The seed also controls fallback neighbors; PiPNN uses fixed internal seeds.
`QGAssigner` in `qgkmeans.hpp` supports repeated vector-to-centroid assignment
without the clustering loop. To assign vectors with trained QGKMeans centroids,
use [`assign`](#assign-new-vectors).

## Results and final assignment

Python result properties are `None` until a successful fit; a failed fit
preserves the previous successful results.

`distances` contains distances to the returned **assigned** centroids, computed
from the original float32 vectors and centroids. For squared L2, `final_obj` is
their sum (WCSS); spherical mode sums `1 - inner_product`. `obj` and
`iteration_stats` record objectives before each centroid update; Python
`train()` returns the last such objective.

`FinalAssignmentMode.Approximate` (default) uses the selected class's assignment
algorithm. The legacy name `FinalAssignmentMode.SymphonyQG` remains an alias for
`Approximate`. Labels need not identify the exact nearest centroids, but reported
distances and WCSS remain exact for those labels. Without sampling, quantized
QGKMeans final assignment retains a point's last-iteration label when it is
closer. After sampling, most points have no such label, so every point uses the
same graph search. Approximate training objectives need not decrease monotonically.

To require exact nearest-centroid **final** assignments with either class:

```python
from rabitqlib import FinalAssignmentMode, RaBitQKMeans

kmeans = RaBitQKMeans(d, 16, final_assignment=FinalAssignmentMode.Exact)
kmeans.train(x)
```

In C++, set `parameters.final_assignment` to
`rabitqlib::rabitqkmeans::FinalAssignmentMode::Exact` or
`rabitqlib::qgkmeans::FinalAssignmentMode::Exact`; both names refer to the same
enum. Training still uses the selected approximate assignment algorithm.

Exact final assignment screens all centroids with blocked GEMM and refines
possible winners in float64; ties select the smallest centroid index. Returned
distances remain float32. Work is proportional to `n * k * d`; large coordinate
offsets or near ties can require more refinement.

## Assign new vectors

After a successful fit, `assign(x)` returns `(assignments, distances)` for an
`(m, d)` array: uint32 centroid IDs and float32 distances to those centroids,
each of shape `(m,)`. It uses `final_assignment` and the other parameters of the
last successful fit and checks coordinates as training does; `m` may be `0`.
Before the first successful fit, it raises `RuntimeError`. The C++ method
`assign(n, x, assignments, distances)` writes to caller-owned buffers. It throws
`std::runtime_error` until `train()` succeeds, including when `centroids` were
written directly, and `std::invalid_argument` for invalid `x` or `centroids`.

`assign` uses no labels or search hints from training, so each vector's
result is independent of the other vectors in the call. Approximate
RaBitQKMeans encodes vectors around the stored training mean. Approximate
QGKMeans builds a graph over the centroids on first use and reuses it;
C++ callers that change `centroids` or graph parameters cause one rebuild on
the next call. Each call still validates the `k * d` centroid coordinates.

For the training vectors, `assign` reproduces `assignments` and `distances`
with `FinalAssignmentMode.Exact`, for RaBitQKMeans, and whenever training
sampled the input, because the sampled final pass is the same assignment.
Unsampled approximate QGKMeans training also uses previous labels as graph
search hints, so its final labels can differ.

## Compare with Faiss

The [comparison script](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/python/compare_with_faiss.py)
compares both methods with FAISS and requires `faiss-cpu`. Defaults are one
million 500-dimensional vectors, 4,000 centroids, and up to 25 iterations.
Load data with `--input data.npy`, or `--input data.bin --d D` for raw row-major
float32 data. Choose a thread count within the available hardware limit.

```bash
python sample/python/compare_with_faiss.py --threads 48
```

All three methods receive the same full dataset, iteration limit, cluster count,
thread budget, and seed values; FAISS subsampling is disabled. This command runs
one fit per method with seed 42. Add `--repeats 4` for four fits per method with
rotating order, or `--num-seeds N` to use consecutive seeds. Different initializers
mean identical seeds need not produce identical centroids.

Only `train()` is timed, including the library methods' default approximate
final assignment. The script reports actual iterations, timing medians/ranges,
exhaustive nearest-centroid objectives, and both library methods' returned-label
objectives and label mismatches. Equal-distance ties can produce mismatches
without changing the objective. `--spherical` normalizes the input copy and
compares sums of `1 - inner_product`.

Separate worker processes isolate OpenMP runtimes and share a temporary float32
input file; allow about `4 * n * d` bytes of temporary disk space. Input staging,
process startup, and exhaustive evaluation are outside training time. For Linux
CPU placement, set `OMP_PLACES=threads OMP_PROC_BIND=close`; on NUMA systems,
use `numactl --cpunodebind=0 --membind=0` with a thread count that fits the node.
