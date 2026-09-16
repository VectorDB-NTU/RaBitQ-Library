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
uses owned raw vectors. No full reconstructed-vector cache is retained.

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
ids, distances = loaded.search(queries, k=10, ef=100, num_threads=1)
```

`init` defaults to `"pipnn"`; use `"random"` for random initialization.
Supported metrics are `"l2"` and `"ip"`. `max_degree` must be a multiple of 32
and smaller than the point count. `ef_construction` controls the build search
window; `ef` controls the query search window. Python defaults to one thread.

### C++

The C++ API uses the float-only `rabitqlib::symqg::QuantizedGraph` and `QGBuilder`.
`QuantizedGraph` is a non-template class; use `QuantizedGraph` instead of
`QuantizedGraph<float>`. Its implementation is compiled in `src/index/qg.cpp`:

```cpp
QuantizedGraph(
    size_t num, size_t dim, size_t max_deg,
    MetricType metric_type = METRIC_L2,
    RotatorType rotator_type = RotatorType::FhtKacRotator,
    size_t quantization_bits = 0
);
QGBuilder(
    QuantizedGraph& index, uint32_t ef_build, const float* data,
    size_t num_threads = std::numeric_limits<size_t>::max(),
    QGInitialization init = QGInitialization::PiPNN
);
```

`data` contains `num * dim` floats; `max_deg` has the same constraints as Python's
`max_degree`. C++ defaults to all available threads. Pass
`QGInitialization::Random` as the final builder argument to use random initialization.
The builder handles initialization internally.

```cpp
using namespace rabitqlib::symqg;

// data contains rows * cols floats.
QuantizedGraph qg(rows, cols, 32);
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
QuantizedGraph qg;
qg.load("qg_example.index");
qg.set_ef(100);

std::vector<rabitqlib::PID> ids(10);
std::vector<float> distances(10);
qg.search(query.data(), 10, ids.data(), distances.data());
```

See `sample/cpp/symqg_indexing.cpp`, `sample/cpp/symqg_querying.cpp`, and their
Python counterparts for complete examples.
