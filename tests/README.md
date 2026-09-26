# RaBitQ Tests

This directory contains the C++ unit and integration tests and the Python
binding tests for RaBitQ Library.

## Prerequisites

- CMake 3.20 or newer for the `ctest --test-dir` commands below (4.2 or newer
  for the Visual Studio 2026 generator)
- A C++17 compiler with OpenMP support (GCC, Clang, or Visual Studio 2026)
- An x86-64 CPU with AVX2 and FMA (AVX-512 optional), or an ARM64 CPU on Linux or macOS
- Git and network access during the first configuration so CMake can download
  GoogleTest 1.14.0

On Windows, install Visual Studio 2026 with the Desktop development with C++
workload. Use MSVC for all C++ targets, including SIMD kernels.

On Ubuntu or Debian, install the required build tools with:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake libomp-dev
```

## C++ tests

### Quick start

Run from the repository root. All configurations disable native tuning to
exercise the portable runtime-dispatch build.

#### Windows (PowerShell)

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DRABITQ_BUILD_TESTS=ON -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

#### Linux

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

#### Linux ARM64

Use a native AArch64 host with CMake, Ninja, a C++17 compiler, and OpenMP.
From the repository root:

```bash
test "$(uname -m)" = aarch64
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_TESTS=ON -DRABITQ_BUILD_SAMPLES=OFF \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build --parallel 2
OMP_NUM_THREADS=2 ctest --test-dir build --output-on-failure
```

Native Linux ARM64 CI runs the C++ suite and installed CMake consumer. Its wheel
job builds and tests repaired CPython 3.11–3.14 wheels. Set
`RABITQ_TEST_WHEEL=1` after installing a repaired wheel locally to check the
AArch64 extension and bundled OpenMP runtime. AVX backend tests skip on ARM.

#### macOS ARM64

Use a native ARM64 project Python environment. Install CMake, Ninja and the existing
OpenMP dependency (AppleClang does not ship an OpenMP runtime):

```bash
brew install cmake ninja libomp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_TESTS=ON -DRABITQ_BUILD_SAMPLES=OFF \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
  -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build --parallel
OMP_NUM_THREADS=2 ctest --test-dir build --output-on-failure
```

An environment providing `llvm-openmp` can use its prefix as `OpenMP_ROOT` instead.
CMake excludes AVX source files on ARM64. Reference tests run through generic and
NEON kernels; explicit AVX backend tests are guarded or skipped.

For Python source builds, pass `-Ccmake.define.OpenMP_ROOT=<libomp-prefix>` as well
as the native-optimization setting below. Release CI uses native `macos-14` runners,
builds CPython 3.11–3.14 ARM64 wheels, bundles OpenMP with delocate, then installs and
runs the full Python suite for each wheel. The minimum wheel OS target is macOS 14.
Set `RABITQ_TEST_WHEEL=1` when testing a repaired wheel locally to also check its
ARM64 extension and bundled OpenMP loader path.

The combined executable is `build/tests/Release/rabitq_tests.exe` on Windows
and `build/tests/rabitq_tests` on Linux/macOS. To run a subset, add `-R <pattern>` to
the CTest command.

AVX-512 tests skip when the CPU or OS lacks the required features. Tests using
`/dev/full` skip on Windows; native POSIX path tests compile only on their
applicable platforms. Passing on an AVX2 machine does not verify AVX-512 execution.

## Python tests

Use an activated [project environment](../CONTRIBUTING.md#python-environment).
Rebuild and install the extension before testing C++ changes. These commands
work in PowerShell and Bash:

```text
python -m pip install ".[test]" -Ccmake.define.RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
python -c "import rabitqlib._rabitqlib as ext; print(ext.__file__)"
python -m pytest tests/python -ra -q
```

Confirm the printed extension path belongs to the intended environment. The
suite includes all three indexes, quantization, persistence, and Unicode paths.

The example integration tests additionally need `faiss-cpu`. Wheel CI installs it
as a test dependency and runs Faiss clustering and RaBitQ indexing/querying in
separate subprocesses, with guards against importing both libraries together.
These tests cover L2/IP and the saved clustering file used by the
[Python examples](../sample/python/README.md). They skip the Faiss-dependent cases
when Faiss is absent from a local test environment.

## Installed CMake package test

After building the library, verify that another project can consume its installed
headers and library. Use an absolute path for `<install-prefix>`:

```text
cmake --install build --config Release --prefix "<install-prefix>"
cmake -S tests/consumer -B build-consumer "-DCMAKE_PREFIX_PATH=<install-prefix>"
cmake --build build-consumer --config Release
ctest --test-dir build-consumer -C Release --output-on-failure
```

On Windows, add `-G "Visual Studio 18 2026" -A x64` to the consumer configure command.

## Test structure

- `unit/`: C++ index, quantization, FastScan, and utility tests
- `integration/`: packing compatibility tests
- `common/`: shared C++ test utilities
- `python/`: Python binding tests and persistence fixtures
- `consumer/`: installed CMake package test

CMake discovers `*_test.cpp` under `unit/` and `integration/`. Python and
consumer tests run separately using the commands above.
