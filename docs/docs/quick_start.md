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
python -m pip install "rabitqlib>=0.5.2"
```

The clustering examples require 0.5.0 or newer; the IVF update example requires
0.5.1 or newer. HNSW `add()`/`resize()`/`remove()` require 0.5.2 or newer; see
the [HNSW update guide](index/hnsw.md#updating-an-index). Python
`SymqgIndex.search_batch()` also requires 0.5.2 or newer. For unreleased changes,
[install from a checkout](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/CONTRIBUTING.md#python-changes).

Wheels target the platforms above and require no compiler or CMake.
Linux ARM64 and macOS ARM64 wheels bundle OpenMP, so wheel users do not
need a separate OpenMP installation.
Release wheels disable native CPU tuning and select supported kernels at runtime.

### Build and search an IVF index

The following complete example uses deterministic synthetic data and does not
require a dataset download:

```python
import numpy as np
from rabitqlib import FinalAssignmentMode, IvfIndex, RaBitQKMeans

rng = np.random.default_rng(42)
data = rng.standard_normal((500, 64)).astype(np.float32)
queries = rng.standard_normal((5, 64)).astype(np.float32)

clustering = RaBitQKMeans(
    64, 5, num_threads=2, final_assignment=FinalAssignmentMode.Exact
)
clustering.train(data)

index = IvfIndex(
    dim=64,
    max_elements=len(data),
    num_clusters=5,
    nbits=4,
    metric="l2",
)
index.build(data, clustering.centroids, clustering.assignments)

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

### Add and remove IVF vectors

Starting with 0.5.1, a built or loaded IVF index can accept new vectors and exclude
existing vectors from search without the original dataset:

```python
new_vectors = rng.standard_normal((50, 64)).astype(np.float32)
new_ids = index.add(new_vectors)  # automatic cluster routing; ids 500..549
removed = index.remove(new_ids[:10])  # returns 10; storage is retained
```

Batch additions because each `add()` copies the index storage. Removed IDs are
not reused, and both additions and removals survive save/load. See the
[IVF update guide](index/ivf.md#updating-an-index) for costs and limits.

The `metric` argument accepts `"l2"` and `"ip"` (also spelled
`"innerproduct"`). To search by cosine similarity, normalize database and
query vectors first and use `metric="ip"`.

### Add and remove HNSW vectors

With 0.5.2 or newer, reuse `data`, `queries`, and `clustering` from the IVF
example above to build and update an HNSW index:

```python
from rabitqlib import HnswIndex

hnsw = HnswIndex(
    dim=64, max_elements=len(data), M=16, ef_construction=100, nbits=4,
)
hnsw.build(data, clustering.centroids, clustering.assignments)

# HNSW needs explicit capacity growth before adding beyond max_elements.
new_vectors = rng.standard_normal((50, 64)).astype(np.float32)
hnsw.resize(hnsw.max_elements + len(new_vectors))
new_ids = hnsw.add(new_vectors)  # automatic nearest-centroid routing
removed = hnsw.remove(new_ids[:10])  # returns 10
ids, distances = hnsw.search(queries, k=10, ef=100)
```

Removed points remain in the graph and count toward capacity, but never appear
in results. Additions and removals survive save/load; files containing HNSW
removals require 0.5.2 or newer. Do not update an index while it is being searched.
See the [HNSW update guide](index/hnsw.md#updating-an-index) for recall and capacity
considerations.

See the [Python examples](https://github.com/VectorDB-NTU/RaBitQ-Library/tree/main/sample/python)
for IVF, HNSW, and SymphonyQG. For clustering, choose
[RaBitQKMeans](clustering.md#rabitqkmeans) for flat assignment, recommended for small cluster
counts, or [QGKMeans](clustering.md#qgkmeans) for graph assignment.

### Threading and file paths

The current source checkout releases the Python GIL during native index search,
construction, updates, and file I/O. The following contract applies to
`IvfIndex`, `HnswIndex`, and `SymqgIndex`:

| Operation on the same index | Concurrent access |
| --- | --- |
| `search`, `search_batch`, `save`, and property reads | May run together |
| `build`, `add`, `remove`, and `resize`, where available | Require exclusive access |

A conflicting call immediately raises
`RuntimeError("Index is busy: conflicting operation in progress")`.
It does not wait or queue an update. Access protection includes Python input
conversion and is released when the operation returns or raises an exception.
Different index objects can operate independently. Concurrent saves must use
different output paths, including any sidecar files.

Search parameters such as `ef`, `nprobe`, precision, and `num_threads` belong to
each call. When using a Python thread pool, consider `num_threads=1` to avoid
starting multiple native worker pools. Keep input arrays and any shared backing
storage unchanged for the entire call; do not resize or modify them from another
thread. Returned result arrays own their storage.

These access checks belong to the Python wrappers. C++ callers must synchronize
index updates themselves and use separate search output buffers.

`num_threads=0` selects the detected available logical CPU count. Positive values
set an upper limit, capped at that count; small workloads may use fewer workers.
Python index methods default to one thread; clustering defaults to `0`.
On Linux, CPU detection accounts for affinity and OpenMP binding. Apply binding
before starting the program.

Index save/load paths are UTF-8 strings on Windows and native path bytes on POSIX
in C++; Python paths are Unicode strings on all platforms.

### Build the Python bindings from source

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

### C++ examples

Example executables are written to `bin/`. Their source demonstrates complete
indexing and querying workflows:

- [IVF RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/ivf_rabitq_indexing.cpp)
- [HNSW RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/hnsw_rabitq_indexing.cpp)
- [SymphonyQG](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/symqg_indexing.cpp)
- [Low-level quantization](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/quantizer.cpp)

The low-level quantization example is provided as source and is not currently a
CMake target. For the GIST benchmark workflow, see
[`example.sh`](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/example.sh).

### Use in another C++ project

The C++ API and ABI are still evolving. For reproducible builds, pin a release
or commit and include RaBitQ-Library as a Git submodule:

```bash
git submodule add https://github.com/VectorDB-NTU/RaBitQ-Library.git third_party/rabitqlib
git submodule update --init --recursive
git -C third_party/rabitqlib checkout <release-or-commit>
git add third_party/rabitqlib
```

Add the library and link its namespaced target in your `CMakeLists.txt`:

```cmake
set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/rabitqlib)
target_link_libraries(my_program PRIVATE rabitqlib::rabitqlib)
```

Update the pinned revision deliberately when adopting upstream changes:

```bash
git -C third_party/rabitqlib fetch
git -C third_party/rabitqlib checkout <release-or-commit>
git add third_party/rabitqlib
```

### Install the C++ library

Installation is useful for package managers, container images, and shared server
environments. Disable native optimization for use on other CPUs:

```bash
cmake -S . -B build \
  -DRABITQ_BUILD_SAMPLES=OFF \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --parallel
cmake --install build
```

Consume the installed package with:

```cmake
find_package(rabitqlib CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE rabitqlib::rabitqlib)
```

For a non-system prefix, point CMake to the installation:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build --parallel
```

Both submodule and installed-package integration require OpenMP on the consuming
system. The [downstream consumer test](https://github.com/VectorDB-NTU/RaBitQ-Library/tree/main/tests/consumer)
provides a complete installed-package example.

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
