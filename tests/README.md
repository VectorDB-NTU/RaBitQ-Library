# RaBitQ Tests

This directory contains the C++ unit and integration tests and the Python
binding tests for RaBitQ Library.

## Prerequisites

- CMake 3.15 or newer
- A GCC- or Clang-compatible C++17 compiler with OpenMP support
- An x86-64 CPU supported by RaBitQ's AVX2 or AVX-512 runtime dispatch
- Network access during the first configuration so CMake can download
  GoogleTest 1.14.0

The current CMake configuration uses GCC/Clang command-line options and does
not provide a supported MSVC build path.

On Ubuntu or Debian, install the required build tools with:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev
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

Release builds use native CPU tuning by default. Pass
`-DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF` when the resulting test binary must
run on a different AVX2- or AVX-512-capable machine.

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
