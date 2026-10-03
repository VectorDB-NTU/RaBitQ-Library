# QG + RaBitQ (SymphonyQG)

[SymphonyQG](https://dl.acm.org/doi/abs/10.1145/3709730) combines the graph search
of [QG](https://medium.com/@masajiro.iwasaki/fusion-of-graph-based-indexing-and-product-quantization-for-ann-search-7d1f0336d0d0),
from [NGT](https://github.com/yahoojapan/NGT), with batched RaBitQ distance estimation.
FastScan estimates neighbor distances; visited vertices are scored using stored
raw vectors or quantized codes.

Set `quantization_bits=0` for raw vectors (default), or `4`/`8` for QG-quant.
Quantized vectors share a rotated global centroid. Refinement temporarily
reconstructs source vectors and estimates distances to stored target codes;
reverse edges are rescored because these estimates are directional. Raw refinement
uses owned raw vectors. No full reconstructed-vector cache is retained by default.

## Index Construction

[PiPNN](https://dl.acm.org/doi/abs/10.1145/3770855.3817891) is the default initializer
for fast indexing, followed by one SymphonyQG refinement iteration. Random
initialization remains available and uses three iterations by default.

### Python

```python
from rabitqlib import SymqgIndex

# data and queries are float32 arrays with shape (count, dim).
index = SymqgIndex(dim=data.shape[1], max_degree=32, metric="l2", quantization_bits=0)
index.build(data, ef_construction=200, num_threads=32, init="pipnn")
index.save("qg_example.index")

loaded = SymqgIndex.load("qg_example.index")
ids, distances = loaded.search_batch(queries, k=10, ef=100, num_threads=1)
```

This example uses `search_batch()`, which requires 0.5.2 or newer. On earlier
versions, use `search()` with the same arguments.

`init` defaults to `"pipnn"`; use `"random"` for random initialization.
Supported metrics are `"l2"` and `"ip"`. `max_degree` must be a multiple of 32
and smaller than the point count. `ef_construction` controls the build search
window; `ef` controls the query search window. Python defaults to one thread.

### C++

The C++ API provides `rabitqlib::symqg::QuantizedGraph<float>` and `QGBuilder`.
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

`builder.reset(data)` reinitializes vectors of the same shape and reuses workspace.
Input may be released after reset returns; rebuild before querying or saving.
`cache_vectors=true` retains transformed vectors and query factors
(`4 * num * padded_dim` bytes plus factors). [QGKMeans](../clustering.md#qgkmeans) enables
this for its centroid graph; the cache is not persisted. The graph seed controls
rotation, and the builder seed controls random initialization and fallback edges.
PiPNN uses fixed internal seeds.

```cpp
using namespace rabitqlib::symqg;

// data contains rows * cols floats.
QuantizedGraph<float> qg(rows, cols, 32);
{
    QGBuilder builder(qg, 200, data.data(), 32, QGInitialization::PiPNN);
    builder.build();
} // release builder scratch before saving
qg.save("qg_example.index");
```

The caller can release input vectors after the `QGBuilder` constructor returns.
Python retains the caller's array during `build`. Complete the build before
querying or saving. Initialization does not change query-distance conventions or
save/load formats.

### Data Layout

Each row contains:

```text
[Raw vector or packed 4/8-bit code + factors]
[One-bit neighbor codes + factors]
[Neighbor IDs]
```

Neighbor codes use FastScan batches of 32, which determines the degree alignment.
Quantized storage reduces each vector's size but not its neighborhood codes;
those codes and temporary refinement pools still consume substantial memory.

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

Starting with 0.5.2, C++ `search_batch()` accepts contiguous row-major batches
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

Python provides `search_batch(queries, k, ef, num_threads=1)` starting with 0.5.2.
The existing `search()` accepts the same two-dimensional array and remains an equivalent
entry point. Both return `(ids, distances)` arrays of shape `(num_queries, k)`,
including empty batches. Both APIs default to one worker; `num_threads=0`
selects the available hardware thread count, capped by the number of queries.
Queries use the original dimension, including when the internal rotation pads it.

See `sample/cpp/symqg_indexing.cpp`, `sample/cpp/symqg_querying.cpp`, and their
Python counterparts for complete examples.
