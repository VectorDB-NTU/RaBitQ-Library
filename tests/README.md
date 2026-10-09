# RaBitQ Tests

This directory contains the C++ unit and integration tests and the Python
binding tests for RaBitQ Library.

Persistence tests include [frozen release-produced indexes](fixtures/persistence/README.md)
and search results, in addition to current round trips. C++ and installed-wheel
tests load the same fixtures, compare search results, and verify save/load migration
and rejection of truncated or unsupported formats.

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
as the native-optimization setting below. Release CI uses native `macos-15` runners,
builds CPython 3.11–3.14 ARM64 wheels, bundles OpenMP with delocate, then installs and
runs the core Python suite for each wheel (excluding the separately tested FAISS
comparison and C++ clustering examples). The minimum wheel OS target is macOS 14.
The wheel build compiles Homebrew's OpenMP source with that deployment target via
`scripts/build-macos-openmp.sh`; the runner's prebuilt OpenMP bottle may require a
newer macOS. The installed-wheel check verifies both the extension and bundled
runtime support macOS 14.
Set `RABITQ_TEST_WHEEL=1` when testing a repaired wheel locally to also check its
ARM64 extension and bundled OpenMP loader path.

The combined executable is `build/tests/Release/rabitq_tests.exe` on Windows
and `build/tests/rabitq_tests` on Linux/macOS. To run a subset, add `-R <pattern>` to
the CTest command.

AVX-512 tests skip when the CPU or OS lacks the required features. Tests using
`/dev/full` skip on Windows; native POSIX path tests compile only on their
applicable platforms. Passing on an AVX2 machine does not verify AVX-512 execution.

## CI coverage

The installed-consumer job runs the full portable Linux Release suite. The Ubuntu
native build runs SIMD and clustering smoke tests; Windows, ARM64, and ASan/UBSan
jobs retain their full C++ suites. The sanitizer job uses Debug with
`-O1 -g -fno-optimize-sibling-calls`, retaining assertions, frame pointers, leak
checking, and both ASan/UBSan while avoiding unoptimized clustering loops.

The portable Linux job also reuses its test binary under pinned
[Intel SDE](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html)
CPU models: Haswell (`-hsw`, AVX2 only) and Ice Lake (`-icl`, AVX-512 including
VPOPCNTDQ). Kernel reference tests and small clustering/index tests exercise real
runtime dispatch under emulation. `RABITQ_TEST_CPU` checks the emulated features
and prints them; an incorrect CPU model fails instead of silently losing coverage.
These are correctness checks, not hardware performance measurements.

Source-install Python jobs check imports, index persistence, and both clustering
APIs. Repaired-wheel jobs run the full core Python suite. Large-dimension Python
cases check representative binding/persistence paths; C++ retains the exhaustive
clustering dimension/storage matrix.

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

The [Python example tests](python/test_examples.py) cover RaBitQKMeans and
QGKMeans clustering, L2/IP indexing, saved cluster files, and queries without
importing FAISS. The [FAISS comparison tests](python/test_compare_with_faiss.py)
cover both clustering methods and skip FAISS-dependent cases when `faiss-cpu`
is absent. CI runs them in a dedicated Linux job; wheel tests do not install FAISS.

The [C++ clustering example tests](python/test_cpp_clustering_examples.py) check
saved vectors, exact labels, and compatibility with the C++ index builders.
The Ubuntu C++ job runs them against its sample binaries, with
`RABITQ_REQUIRE_CPP_EXAMPLES=1` so missing executables fail. Wheel jobs exclude
these tests. Locally, build the examples first; missing executables otherwise skip.

## Installed CMake package test

After building the library, verify that another project can consume its installed
headers and library. The consumer trains QGKMeans and RaBitQKMeans and checks
final assignments, distances, and objective against a scalar L2 reference.
Use an absolute path for `<install-prefix>`:

```text
cmake --install build --config Release --prefix "<install-prefix>"
cmake -S tests/consumer -B build-consumer "-DCMAKE_PREFIX_PATH=<install-prefix>"
cmake --build build-consumer --config Release
ctest --test-dir build-consumer -C Release --output-on-failure
```

On Windows, add `-G "Visual Studio 18 2026" -A x64` to the consumer configure command.

The consumer also checks evaluated compile options for unwanted transitive build
policy. Only OpenMP compiler options, the C++17 requirement, and required MSVC
header definitions are propagated; optimization, native tuning, and warnings
stay private to first-party targets. Sanitized builds additionally propagate
instrumentation and runtime link options: Eigen's public inline allocation code
depends on the sanitizer mode and must stay consistent with the core. Pass
`-DRABITQ_EXPECT_SANITIZERS=ON` when testing such an installation with GCC/Clang.

Test source-tree consumption through `add_subdirectory` with the same consumer:

```text
cmake -S tests/consumer -B build-source-consumer -DCMAKE_BUILD_TYPE=Release -DRABITQ_SOURCE_DIR="<absolute-repository-path>" -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=ON
cmake --build build-source-consumer --config Release
ctest --test-dir build-source-consumer -C Release --output-on-failure
```

CI checks both source-tree and installed native builds, portable installations,
and an installed sanitizer build. Native tuning retains its existing default
(`ON`) for first-party targets; disabling propagation does not make an already
native-compiled library portable. Consumers choose their own optimization
flags, including for inline code in the public headers.

## Test structure

- `unit/`: C++ index, quantization, FastScan, and utility tests
- `integration/`: packing compatibility tests
- `common/`: shared C++ test utilities
- `python/`: Python binding tests and persistence fixtures
- `consumer/`: installed CMake package test

CMake discovers `*_test.cpp` under `unit/` and `integration/`. Python and
consumer tests run separately using the commands above.
