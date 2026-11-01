# C++ integration

These component targets and the stable core v1 API are introduced in 0.7.0.
First [get the source](installation.md#get-the-source); see the
[migration guide](modular_migration.md).

## Choose a component

For a stable, storage-neutral interface, start with [core API v1](rabitq/core_api.md)
and the `sample/cpp/stable_core` example. It accepts caller-owned code buffers and
workspace and keeps factors separate from codes. It does not rotate inputs;
callers provide data, centroids and queries in a shared working coordinate system.
Advanced templates have separate contracts; the v1 guarantees do not cover them.

Use `rabitqlib::core` for encoding, single-vector estimation and rotation, or
`rabitqlib::fastscan` for batch scoring, without ANN implementations. Eigen remains bundled;
OpenMP is optional for these components and enabled by default.
The C++ libraries are static; `BUILD_SHARED_LIBS` does not change their type.

| Target | Functionality |
| --- | --- |
| `rabitqlib::core` | Encoding, single-vector query/estimation, rotation, packing and essential SIMD |
| `rabitqlib::fastscan` | Batch layout/packing/encoding, query/LUT/estimation and FastScan/HACC SIMD; depends on core |
| `rabitqlib::index` | IVF-RaBitQ, HNSW-RaBitQ, SymphonyQG and clustering; depends on FastScan |

The default C++ build includes all three targets. For numerical components only,
see [core](#core-only-c-integration) or [FastScan](#fastscan-integration).

## Build the library

On Linux, build the library and examples from the root of that checkout:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

CMake enables native CPU tuning by default where supported by the compiler.
For portable binaries within a supported OS and architecture, configure with
`-DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF`; runtime kernel selection remains active.
See the [platform-specific build commands](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/tests/README.md#quick-start)
for Linux ARM64, Windows, and Apple Silicon.

## C++ examples

ANN example executables are written to `bin/`. Their source demonstrates complete
indexing and querying workflows:

- [IVF-RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/ivf_rabitq_indexing.cpp)
- [HNSW-RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/hnsw_rabitq_indexing.cpp)
- [SymphonyQG](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/symqg_indexing.cpp)
- [Core quantization and distance estimation](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/cpp/core/main.cpp)

The core example is built as the `core_example` CMake target. For the GIST benchmark workflow, see
[`sample/run_cpp.sh`](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/sample/run_cpp.sh).

## Use in another C++ project

The [core quantization v1 API](rabitq/core_api.md) has a C++ source-compatibility contract.
Other C++ APIs may evolve, and no cross-version C++ binary ABI is promised.
Pin the `v0.8.1` release and include RaBitQ-Library as a Git submodule:

```bash
git submodule add https://github.com/VectorDB-NTU/RaBitQ-Library.git third_party/rabitqlib
git submodule update --init --recursive
git -C third_party/rabitqlib checkout v0.8.1
git add third_party/rabitqlib
```

Add the library and link its namespaced target in your `CMakeLists.txt`:

```cmake
set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/rabitqlib)
target_link_libraries(my_program PRIVATE rabitqlib::index)
```

Update the pinned revision deliberately when adopting upstream changes:

```bash
git -C third_party/rabitqlib fetch
git -C third_party/rabitqlib checkout <new-version-or-commit>
git add third_party/rabitqlib
```

## Install the C++ library

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
target_link_libraries(my_program PRIVATE rabitqlib::index)
```

For a non-system prefix, point CMake to the installation:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build --parallel
```

Full-library submodule and installed-package integration require OpenMP on the consuming
system. The [downstream consumer test](https://github.com/VectorDB-NTU/RaBitQ-Library/tree/main/tests/consumer)
provides a complete installed-package example.

## Core-only C++ integration

To build and install only core:

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_INDEX=OFF -DRABITQ_BUILD_FASTSCAN=OFF \
  -DRABITQ_USE_OPENMP=OFF \
  -DRABITQ_BUILD_SAMPLES=OFF \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/rabitqlib-core"
cmake --build build-core --parallel
cmake --install build-core
```

This configuration does not discover or link OpenMP. Legacy parallel batch helpers run
serially, including when a caller requests more workers; v1 numerical calls remain serial.
Enable FastScan with `RABITQ_BUILD_FASTSCAN=ON` if needed. ANN and Python builds currently
require `RABITQ_USE_OPENMP=ON`. Use one consistent OpenMP choice for all RaBitQ headers
and linked targets within an application.

Use a fresh installation prefix when changing between full and core-only installations.
Pure core mode omits FastScan/index targets and their examples/tests; enabling Python bindings
in that mode is a configuration error. Core tests are available with `RABITQ_BUILD_TESTS=ON`.

### Headers

Use canonical module headers in new code:

| Headers | Owning module |
| --- | --- |
| `core/encoder.hpp`, `core/data_layout.hpp`, `core/pack_excode.hpp` | Per-vector encoding and layouts |
| `fastscan/encoder.hpp`, `fastscan/data_layout.hpp`, `fastscan/pack_codes.hpp` | All 32-vector batch layouts, adapters and packing |
| `core/query.hpp`, `core/estimator.hpp`, `core/rotator.hpp` | Core single-vector query, estimation and rotation |
| `fastscan/query.hpp`, `fastscan/estimator.hpp`, `fastscan/fastscan.hpp` | Batch scoring |

Paths are relative to `rabitqlib/`. Old forwarding headers and the full-library alias
are removed in 0.7.0; follow the [migration guide](modular_migration.md).
Installations contain only enabled module headers and their dependencies. In particular,
core-only installations do not carry FastScan, ANN, clustering or vendored hnswlib headers.

### Link core

Installed package:

```cmake
find_package(rabitqlib CONFIG REQUIRED COMPONENTS core)
target_link_libraries(my_program PRIVATE rabitqlib::core)
```

This also works against a full installation without linking the index archive. For ANN
use, request `COMPONENTS index` and link `rabitqlib::index`.
For changes to older targets and headers, see the [migration guide](modular_migration.md).

Source-tree integration:

```cmake
set(RABITQ_BUILD_INDEX OFF CACHE BOOL "" FORCE)
set(RABITQ_BUILD_FASTSCAN OFF CACHE BOOL "" FORCE)
set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(RABITQ_ENABLE_NATIVE_OPTIMIZATION OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/rabitqlib)
target_link_libraries(my_program PRIVATE rabitqlib::core)
```

For FetchContent, the same options apply. This example consumes a local checkout,
which is also the configuration exercised by the integration tests:

```cmake
include(FetchContent)
set(RABITQ_BUILD_INDEX OFF CACHE BOOL "" FORCE)
set(RABITQ_BUILD_FASTSCAN OFF CACHE BOOL "" FORCE)
set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(RABITQ_ENABLE_NATIVE_OPTIMIZATION OFF CACHE BOOL "" FORCE)
FetchContent_Declare(rabitqlib SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rabitqlib")
FetchContent_MakeAvailable(rabitqlib)
target_link_libraries(my_program PRIVATE rabitqlib::core)
```

To fetch remotely, replace `SOURCE_DIR` with `GIT_REPOSITORY` and a pinned `GIT_TAG`.
The local test does not validate network access or any package-manager integration.

### Run the core example

The standalone example in `sample/cpp/core`
rotates a 65-dimensional vector and query using the same seeded rotator, allocates
buffers using the existing data maps, encodes four-bit RaBitQ and estimates squared L2
distance relative to a zero centroid. Query state borrows its rotated input; keep that
input alive through estimation. All internal buffers use the rotator's padded dimension.
No ANN index is constructed.

```bash
cmake -S sample/cpp/core -B build-core-example \
  -DCMAKE_PREFIX_PATH="$HOME/.local/rabitqlib-core"
cmake --build build-core-example
ctest --no-tests=error --test-dir build-core-example --output-on-failure
```

With project samples enabled, the same program is built as `core_example` under
`build-core/sample/cpp/` (under `Release/` there for multi-configuration generators).

## FastScan integration

For batch scoring without ANN, configure `RABITQ_BUILD_INDEX=OFF` and
`RABITQ_BUILD_FASTSCAN=ON`, then consume the installed package with:

```cmake
find_package(rabitqlib CONFIG REQUIRED COMPONENTS fastscan)
target_link_libraries(my_program PRIVATE rabitqlib::fastscan)
```

The standalone FastScan example in `sample/cpp/fastscan` of the same source checkout
encodes 49 vectors in two batches and scores them with HACC and extra-bit refinement,
for both L2 and IP with a nonzero centroid. It reuses storage for 32 physical lanes
and consumes only the 17 logical results of the final batch. Build it with the same
commands as the core example, substituting `sample/cpp/fastscan` and its installation
prefix. The in-tree sample target is `fastscan_example`.

Index builds require FastScan; requesting `RABITQ_BUILD_INDEX=ON` with FastScan
disabled fails configuration.

## Python bindings

Python bindings are optional (`RABITQ_BUILD_PYTHON_BINDINGS=ON`) and require indexes. The
existing `_rabitqlib` extension has a `rabitqlib::python` build-tree alias; build the
`_rabitqlib` target or install the wheel and use `import rabitqlib`. It is a Python
MODULE, not a C++ link library, and is not exported as a C++ `find_package` component.

## Run the C++ tests on Linux

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build --parallel
ctest --no-tests=error --test-dir build --output-on-failure
```

GoogleTest is downloaded during test configuration.
