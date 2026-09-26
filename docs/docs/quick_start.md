# Quick Start

RaBitQ Library provides Python bindings for complete vector-search indexes and
a C++17 API for both indexes and low-level quantization.

## Requirements

| Platform | CPU baseline | Python wheel targets |
| --- | --- | --- |
| Linux x86-64 | AVX2 and FMA | CPython 3.11–3.14 |
| Linux ARM64 (AArch64) | NEON | CPython 3.11–3.14 |
| Windows x86-64 | AVX2 and FMA | CPython 3.11–3.14 |
| macOS 14+ ARM64 (Apple Silicon) | NEON | CPython 3.11–3.14 |

Source builds require a C++17 compiler, OpenMP, and CMake 3.20 or newer.
Windows uses MSVC (Visual Studio 2026 with the Desktop development with C++
workload and CMake 4.2+ for its generator). Apple Silicon uses AppleClang and
an external OpenMP runtime such as Homebrew `libomp`.

Linux ARM64 source builds and repaired wheels run on native AArch64 CI.
Linux ARM64 wheels carry `manylinux_2_27_aarch64` and
`manylinux_2_28_aarch64` tags.

<details>
<summary>CPU dispatch details</summary>

On x86-64, optional AVX-512 kernels require AVX2, FMA, and AVX-512F/BW/DQ;
MSVC builds also require AVX-512VL/CD. Popcount-specific kernels additionally
require AVX-512 VPOPCNTDQ. Detection checks CPU features and OS support for
the required register state. Missing AVX-512 features select the AVX2 backend.

On ARM64, the build excludes x86 kernels and uses NEON and portable scalar
implementations. Standard FastScan requires AVX2/FMA, a supported AVX-512
backend, or NEON; scalar fallbacks do not remove that requirement.

</details>

## Python

### Install

```bash
python -m pip install rabitqlib
```

Wheels target the platforms above and require no compiler or CMake.
Linux ARM64 and macOS ARM64 wheels bundle OpenMP, so wheel users do not
need a separate OpenMP installation.
Release wheels disable native CPU tuning and select supported kernels at runtime.

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

For raw-vector reranking, use `nbits=32` in the constructor above. The index
copies the original float32 vectors instead of storing extra-bit codes, while
retaining one-bit codes for filtering. Build, search, and save/load use the
same APIs; the original data can be released after construction.

IVF selects FastScan precision automatically: HACC for 4–9-bit codes, standard
FastScan for 1–3-bit codes and raw vectors. To override that choice:

```python
ids, distances = index.search(queries, k=10, nprobe=5, high_accuracy=True)
ids, distances = index.search(queries, k=10, nprobe=5, high_accuracy=False)
# Omit high_accuracy, or pass None, to use automatic selection.
```

See the [IVF guide](index/ivf.md) for storage costs and persistence compatibility.

The `metric` argument accepts `"l2"` and `"ip"` (also spelled
`"innerproduct"`). To search by cosine similarity, normalize database and
query vectors first and use `metric="ip"`.

Python bindings are also available for `HnswIndex` and `SymqgIndex`. The
[Python examples](https://github.com/VectorDB-NTU/RaBitQ-Library/tree/main/sample/python)
cover construction, querying, and index persistence.

<details>
<summary>Build the Python bindings from source</summary>

Source builds require Python 3.11 or newer, a C++17 compiler, CMake 3.20 or
newer, and OpenMP. To install the current development version on Ubuntu or
Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library
python -m pip install .
```

On Apple Silicon, use a native ARM64 Python environment. From the repository root:

```bash
brew install cmake ninja libomp
python -m pip install . -Ccmake.define.OpenMP_ROOT="$(brew --prefix libomp)"
```

On Windows, install the build tools listed above, then run `python -m pip install .`
from the repository root. For a portable source build on any supported platform,
also pass `-Ccmake.define.RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF`.

</details>

## C++

On Linux, clone the repository and build the library and examples:

```bash
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

CMake enables native CPU tuning by default where supported by the compiler.
For portable binaries within a supported OS and architecture, configure with
`-DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF`; runtime kernel selection remains active.
See the [platform-specific build commands](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/tests/README.md#quick-start)
for Linux ARM64, Windows, and Apple Silicon.

Example executables are written to `bin/`. Their source demonstrates complete
indexing and querying workflows:

- [IVF + RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/ivf_rabitq_indexing.cpp)
- [HNSW + RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/hnsw_rabitq_indexing.cpp)
- [SymphonyQG](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/symqg_indexing.cpp)
- [Low-level quantization](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/quantizer.cpp)

### Run the C++ tests on Linux

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
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
