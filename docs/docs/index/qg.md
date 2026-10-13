# SymphonyQG

[SymphonyQG](https://dl.acm.org/doi/abs/10.1145/3709730) combines the graph search
of [QG](https://medium.com/@masajiro.iwasaki/fusion-of-graph-based-indexing-and-product-quantization-for-ann-search-7d1f0336d0d0),
from [NGT](https://github.com/yahoojapan/NGT), with batched RaBitQ distance estimation.
FastScan estimates neighbor distances; visited vertices are scored using stored
raw vectors or quantized codes.

The [original paper](https://dl.acm.org/doi/abs/10.1145/3709730) uses raw vectors for
precise candidate distances during traversal. This library's optional 4/8-bit
storage and PiPNN initialization are implementation extensions; the paper's
reported performance should not be attributed to these configurations.

## Minimal workflow

### Python

Use `data` and `queries` from the [Quick start](../quick_start.md), or supply your
own float32 arrays. SymphonyQG builds directly from vectors and needs no clustering
step. The degree-32 example requires at least 33 data points.

```python
from rabitqlib import SymqgIndex

# data and queries are float32 arrays with shape (count, dim).
index = SymqgIndex(dim=data.shape[1], max_degree=32, metric="l2", quantization_bits=0)
index.build(data, ef_construction=200, num_threads=1, init="pipnn")
index.save("qg_example.index")

loaded = SymqgIndex.load("qg_example.index")
ids, distances = loaded.search_batch(queries, k=10, ef=100, num_threads=1)
```

`search()` accepts the same arguments and is equivalent to `search_batch()`.

### C++

Link `rabitqlib::index`. Here `data` is a row-major `std::vector<float>` with
`rows * cols` values, and `query` contains `cols` values; use at least 33 rows.
See [C++ integration](../cpp_integration.md) for build setup.

```cpp
#include <vector>
#include <rabitqlib/index/symqg/qg.hpp>
#include <rabitqlib/index/symqg/qg_builder.hpp>

using namespace rabitqlib::symqg;

// data contains rows * cols floats.
QuantizedGraph<float> qg(rows, cols, 32);
{
    QGBuilder builder(qg, 200, data.data(), 1, QGInitialization::PiPNN);
    builder.build();
} // release builder scratch before saving
qg.set_ef(100);
std::vector<rabitqlib::PID> ids(10);
std::vector<float> distances(10);
qg.search(query.data(), 10, ids.data(), distances.data());
qg.save("qg_example.index");
```

## Parameters and limits

| Parameter | Contract |
| --- | --- |
| `dim` | Original input dimension; Python supports 64–65,536 and pads internally. |
| `max_degree` | Required positive multiple of 32, strictly smaller than the point count. |
| `quantization_bits` | 0 for owned raw vectors (default), or 4/8 for quantized storage. |
| `metric` | Python `"l2"` or `"ip"`; defaults to `"l2"`. |
| `ef_construction` | Required positive construction search window. |
| `init` | Python `"pipnn"` (default) or `"random"`. |
| `k`, `ef` | Python requires `1 <= k <= num_points` and `ef >= k`. |
| `num_threads` | Python defaults to 1; 0 selects available logical CPUs. |

Both Python search names return `(ids, distances)` arrays. Distances are squared
L2 or `1 - dot`, ordered from smallest to largest; quantized storage returns
estimates. SymphonyQG has no incremental add/remove API. See shared
[input, count and concurrency contracts](usage.md).

## Index construction

See the shared [index count contract](usage.md#index-counts). C++
`num_vertices()` returns the configured graph size, even before building.

[PiPNN](https://dl.acm.org/doi/abs/10.1145/3770855.3817891) is the default initializer
for fast indexing, followed by one SymphonyQG refinement iteration. Random
initialization remains available and uses three iterations by default.

Include `rabitqlib/index/symqg/qg.hpp` and `qg_builder.hpp` from the same directory,
and link `rabitqlib::index`. The C++ API provides
`rabitqlib::symqg::QuantizedGraph<float>` and `QGBuilder`.
Only float vectors are supported; C++17 declarations may omit `<float>` through
class template argument deduction:

```cpp
QuantizedGraph<float>(
    size_t num, size_t dim, size_t max_deg,
    MetricType metric_type = METRIC_L2,
    RotatorType rotator_type = RotatorType::FhtKacRotator,
    size_t quantization_bits = 0, uint32_t seed = std::random_device{}()
);
QGBuilder(
    QuantizedGraph<float>& index, uint32_t ef_build, const float* data,
    size_t num_threads = std::numeric_limits<size_t>::max(),
    QGInitialization init = QGInitialization::PiPNN,
    uint32_t seed = std::random_device{}(), bool cache_vectors = false
);
```

`data` contains `num * dim` floats; `max_deg` has Python's `max_degree`
constraints. C++ defaults to all available threads. `build()` runs one refinement
for PiPNN or three passes for `QGInitialization::Random`; `build(n)` requests
`n >= 2` passes. The older overload with a numeric fifth argument uses that value
as the random-initialization seed.

### Input lifetime and rebuilds

The caller can release input vectors after the `QGBuilder` constructor returns.
Python retains the caller's array during `build`. Complete construction before
querying or saving.

`builder.reset(data)` reinitializes vectors of the same shape and reuses workspace.
Input may be released after reset returns; rebuild before querying or saving.

### Vector cache

`cache_vectors=true` retains transformed vectors and query factors
(`4 * num * padded_dim` bytes plus factors). [QGKMeans](../clustering.md#qgkmeans) enables
this for its centroid graph; the cache is not persisted.

### Random seeds

The graph seed controls FHT/Kac rotation; MatrixRotator samples its own matrix.
The builder seed controls random initialization and fallback edges. PiPNN uses
fixed internal seeds.

## Querying

C++ search accepts one vector in the original input dimension and writes `k` IDs
and distances:

```cpp
QuantizedGraph<float> qg;
qg.load("qg_example.index");
qg.set_ef(100);

std::vector<rabitqlib::PID> ids(10);
std::vector<float> distances(10);
qg.search(query.data(), 10, ids.data(), distances.data());
```

C++ `search_batch()` accepts contiguous row-major batches
and reuses scratch storage within each worker, preserving the results of
independent `search()` calls:

```cpp
// queries contains num_queries * qg.dimension() floats in the original dimension.
std::vector<rabitqlib::PID> batch_ids(num_queries * 10);
std::vector<float> batch_distances(num_queries * 10);
qg.search_batch(
    queries.data(), num_queries, 10, batch_ids.data(), batch_distances.data(), 4
);
```

The C++ signature is `search_batch(queries, num_queries, k, ids, distances,
num_threads=1)`. Set `ef >= k` with `set_ef()` before searching. Both output
buffers are required and contain `num_queries * k` elements. Empty C++ batches
do no work and may use null buffers. Inputs and outputs must not overlap.
Concurrent searches need separate outputs; do not modify the index or call
`set_ef()` while searches are running.

Python `search_batch(queries, k, ef, num_threads=1)` and `search()` accept
the same two-dimensional query array. Both return `(ids, distances)` arrays
of shape `(num_queries, k)`, including empty batches. Both APIs default to one worker; `num_threads=0`
selects the available logical CPU count, capped by the number of queries.
Queries use the original dimension, including when the internal rotation pads it.

Python searches use per-call `ef` values. For access rules, see the
[concurrency contract](usage.md#threading-and-file-paths).

For C++, use
`search_batch_with_ef(queries, num_queries, k, ids, distances, ef, num_threads=1)`
to run concurrent calls with independent windows:

```cpp
qg.search_batch_with_ef(
    queries.data(), num_queries, 10, batch_ids.data(), batch_distances.data(), 100, 4
);
```

This method does not read or change the default set by `set_ef()`. The existing
`search()`, `search_batch()`, and `search_with_scratch()` retain that default.
All concurrent searches require separate outputs and an unchanged index.

See `sample/cpp/symqg_indexing.cpp`, `sample/cpp/symqg_querying.cpp`, and their
Python counterparts for complete examples.

## Persistence

All three modes pad the rotated domain to a multiple of 32. Loading an older
index preserves its saved padded dimension, including historical multiples of 64.
Older library versions that enforce 64-dimension padding cannot load new files
whose padded dimension is not a multiple of 64.

## Implementation details

Quantized vectors share a rotated global centroid. Refinement temporarily
reconstructs source vectors and estimates distances to stored target codes;
reverse edges are rescored because these estimates are directional. Raw refinement
uses owned raw vectors. No full reconstructed-vector cache is retained by default.

### Data layout

Each row contains:

```text
[Raw vector or packed 4/8-bit code + factors]
[One-bit neighbor codes + factors]
[Neighbor IDs]
```

Neighbor codes use FastScan batches of 32, which determines the degree alignment.
Quantized storage reduces each vector's size but not its neighborhood codes;
those codes and temporary refinement pools still consume substantial memory.
