# SymphonyQG

[SymphonyQG](https://dl.acm.org/doi/abs/10.1145/3709730) combines the graph search
of [QG](https://medium.com/@masajiro.iwasaki/fusion-of-graph-based-indexing-and-product-quantization-for-ann-search-7d1f0336d0d0),
from [NGT](https://github.com/NGT-labs/NGT), with batched RaBitQ distance estimation.
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
index = SymqgIndex(data.shape[1], max_degree=32, ef_construction=200, init="pipnn")
ids = index.add(data)  # builds the whole graph; ids are the row numbers
index.save("qg_example.index")

loaded = SymqgIndex.load("qg_example.index")
ids, distances = loaded.search(queries, k=10, ef_search=100)
```

SymphonyQG follows the same `train` → `add` → `search` workflow as the other
indexes, but has nothing to train: `train(x)` only checks `x` as `add` would,
and `is_trained` is always `True`. `add` builds the graph from every vector at once
and can be called only once, as in Faiss's NSG index; a second call, or `add` on
a loaded index, raises `RuntimeError`. To index more vectors, create a new index
and add all of them. Point IDs are the row numbers of the added data. Files
from releases up to 0.8 do not store the construction settings, so an index
loaded from one reports `ef_construction`, `init` and `seed` as `None`.

### C++

`rabitqlib::SymqgIndex` has the names, defaults and behaviour of the Python
class. Here `data` and `queries` are row-major `std::vector<float>` buffers with
dimension `dim`; use at least 33 data rows. See
[C++ integration](../cpp_integration.md) for build setup.

```cpp
#include <vector>
#include <rabitqlib/index/symqg_index.hpp>

rabitqlib::SymqgIndex index(dim);  // max_degree 32, ef_construction 200, PiPNN
index.add(data.data(), data.size() / dim);  // builds the whole graph
index.save("qg_example.index");

auto loaded = rabitqlib::SymqgIndex::load("qg_example.index");
const size_t num_queries = queries.size() / dim;
std::vector<rabitqlib::Id> ids(num_queries * 10);
std::vector<float> distances(num_queries * 10);
rabitqlib::GraphSearchOptions options;
options.ef_search = 100;
loaded.search(queries.data(), num_queries, 10, ids.data(), distances.data(), options);
```

`add` throws `std::logic_error` (Python `RuntimeError`) on a built or loaded
index, and `train` only checks its input.

## Parameters and limits

| Parameter | Contract |
| --- | --- |
| `dim` | Original input dimension, 64–65,536; the index pads internally. |
| `nbits` | 32 (default) for owned raw vectors, or 4/8 for quantized storage. |
| `max_degree` | Default 32: a positive multiple of 32, strictly smaller than the point count. It is a hard degree bound, unlike HNSW-RaBitQ's `M`. |
| `ef_construction` | Default 200: positive construction search window used by `add`. |
| `init` | `"pipnn"` (default) or `"random"`; C++ `symqg::QGInitialization::PiPNN` or `Random`. |
| `metric` | L2 (default), inner product or [cosine](usage.md#the-cosine-metric): Python `"l2"`, `"ip"` or `"cosine"`. |
| `seed` | Rotation and graph-construction seed in `[0, 2**32 - 1]`. The default (Python `None`, an empty C++ `std::optional`) draws one at random; `seed` reports it, and passing it back with one thread rebuilds the same graph from the same data. |
| `k`, `ef_search` | `k` is from 1 to 2^31 and may exceed the point count; slots the graph cannot fill are padded with ID `-1` (C++ `kNoId`) and distance `+inf`. An omitted `ef_search` (Python `None`) uses `max(k, 64)`, an explicit 0 raises `ValueError` (C++ `std::invalid_argument`), and smaller values are raised to `k`, as for HNSW-RaBitQ. It replaces `ef`, which 0.9 removed. |
| `num_threads` | `add` defaults to every available logical CPU, capped by `OMP_NUM_THREADS`, and `search` to 1; Python `None` and C++ 0 for `add` select the available CPUs. |

In C++, the construction settings are fields of `SymqgOptions`, and `ef_search`
and `num_threads` fields of `GraphSearchOptions`; `add` takes the thread count.
Search returns IDs and distances (Python `(ids, distances)` arrays). Distances are squared
L2 or `1 - dot`, ordered from smallest to largest; quantized storage returns
estimates. SymphonyQG has no incremental add/remove API. See shared
[input, count and concurrency contracts](usage.md).

## Index construction

See the shared [index count contract](usage.md#index-counts). `num_points()` is 0
until `add` builds the graph.

[PiPNN](https://dl.acm.org/doi/abs/10.1145/3770855.3817891) is the default initializer
for fast indexing, followed by one SymphonyQG refinement iteration. Random
initialization remains available and uses three iterations by default.

Include `rabitqlib/index/symqg_index.hpp` and link `rabitqlib::index`:

```cpp
rabitqlib::SymqgOptions options;  // set the fields by name
options.nbits = 4;
options.init = rabitqlib::symqg::QGInitialization::Random;
options.seed = 42;
rabitqlib::SymqgIndex index(dim, options);
index.add(data.data(), num_points, /*num_threads=*/1);
```

`add` builds a graph of exactly `num_points` vertices, so `data` contains
`num_points * dim` floats; there is no spare capacity for later appends. It
runs one refinement for PiPNN or three passes for random initialization. If
`add` throws, the index stays without a graph and a later `add` can build one.
The input may be released after `add` returns.

### Random seeds

The seed controls the FHT/Kac rotation, which is the same on every platform, and
random initialization and fallback edges. PiPNN uses fixed internal seeds.

## Querying

C++ search accepts contiguous row-major queries in the original input dimension
and writes `num_queries * k` IDs and distances:

```cpp
void SymqgIndex::search(
    const float* queries,
    size_t num_queries,
    size_t k,
    Id* ids,
    float* distances,
    const GraphSearchOptions& options = {}
) const;
```

```cpp
auto qg = rabitqlib::SymqgIndex::load("qg_example.index");
rabitqlib::GraphSearchOptions options;
options.ef_search = 100;
options.num_threads = 4;
std::vector<rabitqlib::Id> ids(num_queries * 10);
std::vector<float> distances(num_queries * 10);
qg.search(queries.data(), num_queries, 10, ids.data(), distances.data(), options);
```

`distances` may be `nullptr`. Workers reuse scratch storage across their
queries. Empty batches do no work. Inputs and outputs must not overlap.
Concurrent searches may use different `ef_search` values but need separate
outputs and an unchanged index.

Python `search(queries, k, ef_search=None, *, num_threads=None)` accepts a two-dimensional
query array and returns `(ids, distances)` arrays of shape `(num_queries, k)`,
including empty batches. It defaults to one worker; `num_threads=0` selects the
available logical CPU count, capped by `OMP_NUM_THREADS` and by the number of queries.
Queries use the original dimension, including when the internal rotation pads it.

Python searches also use per-call `ef_search` values. For access rules, see the
[concurrency contract](usage.md#threading-and-file-paths).

See `sample/cpp/symqg_indexing.cpp`, `sample/cpp/symqg_querying.cpp`, and their
Python counterparts for complete examples.

## Persistence

`save` writes [index format 2](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/formats.md#index-format-2), which also records the seed, `ef_construction`
and `init`, so they survive a reload. Releases up to 0.8 cannot read these files; every
earlier file still loads, without these settings, and saving it writes format 2.

All three modes pad the rotated domain to a multiple of 32. Loading an older
index preserves its saved padded dimension, including historical multiples of 64.

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
