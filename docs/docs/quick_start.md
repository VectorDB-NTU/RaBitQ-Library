# Quick Start

RaBitQ Library provides Python bindings for complete vector-search indexes and
a C++17 API for both indexes and low-level quantization.

## Requirements

- An x86-64 CPU supported by the selected kernels: most paths accept either
  AVX2 with FMA or AVX-512F/BW/DQ with FMA
- Python 3.9 or newer for the Python package
- A C++17 compiler with OpenMP support
- CMake 3.15 or newer for Python builds, or CMake 3.10 or newer for C++ builds

Most SIMD entry points select AVX-512 kernels when AVX-512F, AVX-512BW, and
AVX-512DQ are detected; otherwise they use AVX2 when AVX2 and FMA are
available. AVX-512 VPOPCNTDQ enables additional popcount kernels. The HNSW
AVX-512 core path also checks for AVX2 and FMA, and otherwise uses its AVX2
path when available. AVX-512 translation units are compiled with FMA enabled.

## Python

### Install

The PyPI package currently builds the native extension during installation.
On Ubuntu or Debian, install the build tools first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev
python -m pip install --upgrade pip
python -m pip install rabitqlib
```

To install the current development version instead:

```bash
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library
python -m pip install .
```

### Build and search an IVF index

The following complete example uses deterministic synthetic data and does not
require a dataset download:

```python
import numpy as np
from rabitqlib import IvfIndex

rng = np.random.default_rng(42)
data = rng.standard_normal((500, 64)).astype(np.float32)
queries = rng.standard_normal((5, 64)).astype(np.float32)

# Assign vectors to five clusters and calculate their centroids.
cluster_ids = (np.arange(len(data)) % 5).astype(np.uint32)
centroids = np.stack(
    [data[cluster_ids == cluster].mean(axis=0) for cluster in range(5)]
).astype(np.float32)

index = IvfIndex(
    dim=64,
    max_elements=len(data),
    num_clusters=5,
    nbits=4,
    metric="l2",
)
index.build(data, centroids, cluster_ids)

ids, distances = index.search(queries, k=10, nprobe=5)
print(ids.shape, distances.shape)  # (5, 10) (5, 10)
print(ids[0])
```

The `metric` argument accepts `"l2"` and `"ip"` (also spelled
`"innerproduct"`). To search by cosine similarity, normalize database and
query vectors first and use `metric="ip"`.

Python bindings are also available for `HnswIndex` and `SymqgIndex`. The
[Python examples](https://github.com/VectorDB-NTU/RaBitQ-Library/tree/main/sample/python)
cover construction, querying, and index persistence.

## C++

Clone the repository and build the library and examples:

```bash
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Example executables are written to `bin/`. Their source demonstrates complete
indexing and querying workflows:

- [IVF + RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/ivf_rabitq_indexing.cpp)
- [HNSW + RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/hnsw_rabitq_indexing.cpp)
- [SymphonyQG](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/symqg_indexing.cpp)
- [Low-level quantization](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/quantizer.cpp)

### Run the C++ tests

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

GoogleTest is downloaded during test configuration.

## Next steps

- Learn how the [RaBitQ quantizer](rabitq/rabitq.md) works.
- Select an index: [IVF](index/ivf.md), [HNSW](index/hnsw.md), or
  [SymphonyQG](index/qg.md).
- Review the
  [contribution workflow](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/CONTRIBUTING.md)
  before opening a pull request.
