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

Linux CI additionally selects Clang 23 explicitly for portable builds, full C++
tests, installed consumers, and core/FastScan integrations with OpenMP ON/OFF.
This is a tested development toolchain, not a new minimum compiler requirement;
see [compiler and tool versions](../docs/development.md#compiler-and-tool-versions).

OpenMP is optional for core/FastScan tests: configure `RABITQ_BUILD_INDEX=OFF` and
`RABITQ_USE_OPENMP=OFF`. ANN/Python testing retains the full prerequisites above.

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

Use an activated [project environment](../.github/CONTRIBUTING.md#python-environment).
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

## Core-only CMake package tests

Configure with `RABITQ_BUILD_INDEX=OFF`, `RABITQ_BUILD_FASTSCAN=OFF` and
`RABITQ_BUILD_TESTS=ON` to run core references without batch-scoring or index sources.
Enable FastScan with indexes disabled to add the batch-scoring references. The full configuration
continues to run all tests, including index, clustering, persistence and allocation failures.

The external consumer runner copies its CMake project and the standalone core example
to separate working directories, then tests both core and FastScan with
`add_subdirectory`, local-source `FetchContent`, and relocated installed packages.
It also runs each producer's reference tests (downloading the existing GoogleTest dependency). Pass absolute paths:

```text
cmake -DRABITQ_SOURCE_DIR=<checkout> -DWORK_DIR=<external-working-directory> -P tests/core_consumer/run.cmake
```

On macOS, also pass `-DOpenMP_ROOT=<OpenMP-prefix>`. On Windows, set
`CMAKE_GENERATOR=Visual Studio 18 2026` and `CMAKE_GENERATOR_PLATFORM=x64` in the
environment. The runner builds Release with two jobs and disables native tuning.
It compiles module headers with higher-module headers deliberately made unavailable,
checks that retired targets and disabled component headers are absent, and compiles every
installed first-party header independently. It checks actual compilation
units with generators supporting compilation databases,
and both archive and executable symbols when a non-MSVC `nm` is available. It also
checks that core-only Python, indexes without FastScan, and missing required
FastScan/index components fail configuration.
CI runs this workflow on Linux x86-64, Linux ARM64, macOS ARM64 and Windows x86-64.

Core and FastScan consumption must also work from a full installation; the standalone
`sample/cpp/core` and `sample/cpp/fastscan` projects test their respective targets. The original
`tests/consumer` exercises `rabitqlib::index` and checks that HNSW-RaBitQ headers compile with
the IVF-RaBitQ initializer header deliberately unavailable and index value types compile with
Eigen deliberately unavailable. Numerical consumers forbid the removed `defines.hpp`.
All FastScan batch layout/encoding coverage lives under `unit/rabitqlib/fastscan/`;
index ID and nearest-neighbor helper coverage lives under `unit/rabitqlib/index/`.
`CoreModuleBoundary` also rejects format-specific declarations/constants in core and
numerical utilities, complementing include-path and symbol checks.
Frozen persistence fixture tooling
still targets its pinned historical release and is not migrated to current targets.

## Stable core API conformance and allocations

Run the external consumer workflow with `-DRABITQ_USE_OPENMP=OFF` to additionally
forbid OpenMP package discovery, poison `omp.h` in consumer include paths and check
for runtime symbols/compiler flags. The default ON workflow is also retained. Both
variants compile and execute the installed standalone core benchmark smoke tests.
CI covers both choices on its existing platform matrix; local results do not imply
that remote jobs have already passed.

The normal suite includes `CoreApi.*`, the frozen `CoreApiV1Consumer`, and separate
`core_api_no_alloc_dispatch` / `core_api_no_alloc_scalar` processes. The latter force
and count cold-call encoding, layout conversion, query preparation and estimation;
Linux additionally wraps malloc/calloc/realloc/aligned allocation. The scalar process
substitutes CPU predicates so the exact event heap cannot be skipped by SIMD success.

The external module runner also builds the unchanged v1 consumer for each integration
method. Its stable header is tested with Eigen and record-layout headers deliberately
unavailable. The suite is independent of ANN targets and remains enabled in pure core.
See [the fixture policy](fixtures/core_api_v1/README.md).

## Test structure

- `unit/`: C++ index, quantization, FastScan, and utility tests
- `common/`: shared C++ test utilities
- `python/`: Python binding tests and persistence fixtures
- `consumer/`: installed CMake package test

CMake discovers `*_test.cpp` under `unit/`. Python and
consumer tests run separately using the commands above.

## Coverage ownership and test size

C++ reference tests own exhaustive bit-width, SIMD, packing, rounding and algorithm
boundary matrices. Python tests own NumPy conversions, argument/default forwarding,
returned arrays, GIL/access guards and representative end-to-end index operations.
Keep a Python construction/search/persistence case for every exported storage width and
both metrics, but do not cross every width with every original dimension and initializer.

`test_portable_indexes.py` separates supported storage from representative padded-domain
boundaries. IVF-RaBitQ mutation/batching/routing matrices use sign-only, extra-bit and raw-vector
storage; full width coverage remains in the corresponding C++ reference tests. The
SymphonyQG batch test checks both Python names on one built index per storage/metric pair,
including strided/empty queries and bitwise distance equality.

Do not trim frozen compatibility fixtures, independent numerical oracles, failure
injection or concurrency tests just because their inputs resemble another test's inputs.
Preserve public wrapper calls when consolidating lower-level tests; a backend-only test
does not validate wrapper dispatch. Parameter combinations should correspond to a distinct
branch or regression, not multiply unrelated axes by default. Keep the existing CI entry
points and platform-specific checks.

## Initialization and construction-failure regressions

`CoreStaticInitialization` calls the stable core API from a namespace-scope
initializer before `main()`. It also runs in the external numerical consumer
workflow. `ModulesStaticInitialization` covers the enabled FastScan/index dispatch
entry points in the same setting.

`construction_failure_test` isolates allocation replacement from the normal suite.
It checks OpenMP construction errors with one/four requested threads and, on Linux,
wraps C allocations to sweep HNSW-RaBitQ constructor failures and verify ownership. Run
these under ASan/UBSan as well as Release; do not disable allocator-mismatch checking.
`compact_codes_failure_test` covers cold scratch allocation and growth failures in
serial and OpenMP encoding, including the public FastScan batch entry points.
