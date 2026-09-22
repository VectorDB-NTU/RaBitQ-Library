# RaBitQ Tests

This directory contains the C++ unit and integration tests and the Python
binding tests for RaBitQ Library.

## Prerequisites

- CMake 3.20 or newer for the `ctest --test-dir` commands below (4.2 or newer
  for the Visual Studio 2026 generator)
- A C++17 compiler with OpenMP support (GCC, Clang, or Visual Studio 2026)
- An x86-64 CPU supported by RaBitQ's AVX2 or AVX-512 runtime dispatch
- Git and network access during the first configuration so CMake can download
  GoogleTest 1.14.0

On Windows, install Visual Studio 2026 with the Desktop development with C++
workload. Configure with its x64 generator:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DRABITQ_BUILD_TESTS=ON -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

On Ubuntu or Debian, install the required build tools with:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake libomp-dev
```

## Building and Running Tests

### Quick Start

From the project root directory:

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The combined test executable is also available as `build/tests/rabitq_tests`.

Native CPU tuning is on by default for local GCC/Clang builds. Pass
`-DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF` for portable test binaries that may
run on a different CPU from the build machine.

### Building without Tests

By default, tests are **not built**, while the C++ samples are. To build only
the library:

```bash
cmake -S . -B build \
    -DRABITQ_BUILD_SAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```


## Test Structure

```
tests/
├── .gitignore
├── CMakeLists.txt                 # C++ targets and CTest discovery
├── README.md
├── common/                        # Shared C++ test utilities
│   ├── test_data.hpp
│   ├── test_data.cpp
│   └── test_helpers.hpp
├── integration/
│   └── bit_pack_unpack_test.cpp
├── python/
│   ├── conftest.py
│   ├── test_hnsw.py
│   ├── test_import.py
│   ├── test_ivf.py
│   └── test_symqg.py
└── unit/rabitqlib/utils/
    ├── cpu_features_test.cpp
    ├── rotator_test.cpp
    ├── space_test.cpp
    └── visited_set_test.cpp
```

CMake discovers C++ files matching `*_test.cpp` under `unit/` and
`integration/`. The Python tests are run separately with `python -m pytest`.
