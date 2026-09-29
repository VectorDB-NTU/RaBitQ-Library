# Clustering

RaBitQ Library provides two implementations of Lloyd's k-means. Both compute
centroid means from the original float32 data, accumulate in float64, and require
no external clustering package. Choose the class explicitly; neither switches
algorithms based on the cluster count.

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
```

`x` has shape `(n, d)` and is converted to a C-contiguous float32 array.

## C++

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
`example.sh` runs the C++ clustering and indexing workflow without Python.

## Parameters and input

Set Python keyword arguments or fields of the corresponding C++ parameter type:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `niter` | `25` | Maximum clustering iterations. |
| `early_stop_threshold` | `0.0` | Stop from iteration 2 when the absolute relative objective change is at most this value; range `[0, 1]`. Zero stops only on an unchanged objective. |
| `num_threads` | `0` | Use detected hardware threads; positive requests are capped at that count. |
| `spherical` | `false` | Normalize centroids and assign by inner product; input must already be normalized. |
| `seed` | `42` | Seed for centroid selection and rotation. |
| `min_points_per_centroid` | `39` | Warn below this training-points-per-centroid ratio when verbose. |
| `verbose` | `false` | Print training progress and warnings. |
| `final_assignment` | `Approximate` | Use the selected method for final labels, or `Exact` for nearest-centroid labels. |

Training requires `64 <= d <= 65536`, finite coordinates, and
`abs(x) <= sqrt(sqrt(FLT_MAX) / (64 * d))`; scale larger inputs before training.
Cluster-count requirements are listed above. Training does not normalize or
mutate the input; provide normalized vectors for spherical mode.

Empty clusters are reseeded from distant points; duplicate data can still leave
empty clusters in the final assignment. Thread scheduling can affect graph
construction and floating-point reductions, so a fixed seed does not guarantee
identical multithreaded results.

## Method details

### RaBitQKMeans

The flat assignment design is inspired by
[SuperKMeans](https://github.com/cwida/SuperKMeans). Training points are rotated
and encoded once around a fixed dataset mean. Each centroid becomes a query
that scans all packed point batches, and each point receives the centroid with
the best estimated distance.

Each iteration compares all `n * k` point-centroid pairs. The batch size of 32
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
without the clustering loop.

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
distances and WCSS remain exact for those labels. Quantized QGKMeans final
assignment retains a previous label when it is closer. Approximate training
objectives need not decrease monotonically.

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

## Compare with FAISS

The [comparison script](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/python/compare_with_faiss.py)
compares both methods with FAISS and requires `faiss-cpu`. Defaults are one
million 256-dimensional vectors, 4,000 centroids, and up to 25 iterations.
Load data with `--input data.npy`, or `--input data.bin --d D` for raw row-major
float32 data. Choose a thread count within the available hardware limit.

```bash
python sample/python/compare_with_faiss.py --threads 48 --num-seeds 2 --repeats 2
```

All three methods receive the same full dataset, iteration limit, cluster count,
thread budget, and seed values; FAISS subsampling is disabled. This command runs
four fits per method with rotating order. Different initializers mean identical
seeds need not produce identical centroids.

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
