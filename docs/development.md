# Development reference

Start with [Contributing to RaBitQ](../.github/CONTRIBUTING.md) for setup and your first PR.
This reference covers tool configuration and changes to library internals.
For multi-step agent work, use the optional [task workflow and templates](agent-workflow/README.md)
for goals, execution records, handoffs, and independent review.
Run commands from the repository root in your activated
[project environment](../.github/CONTRIBUTING.md#python-environment), after installing
the [source-build prerequisites](../.github/CONTRIBUTING.md#setup-for-code-changes).
Refresh APT package lists with `sudo apt-get update` before installing tools.

## Compiler and tool versions

Library builds require C++17; the LLVM 23 development tools below do not impose a
Clang 23 minimum on downstream users. CMake respects the caller's compiler choice.
OpenMP is required for ANN indexes and Python bindings, and optional for core/FastScan.

Linux CI explicitly builds and tests with Clang 23, including installed consumers
and OpenMP ON/OFF numerical builds, alongside the existing GCC jobs. Native ARM64,
AppleClang and MSVC checks remain separate. Older GCC/Clang minimum versions have
not yet been established by a dedicated compatibility matrix.

Formatting and static analysis use LLVM 23 as the shared local/CI baseline. Keep
this major version fixed rather than following unversioned distribution defaults.
The LLVM APT release branch supplies updated builds within that major version;
CI prints the exact tool versions, rather than promising a fixed patch build.
Upgrading them requires reviewing format differences and diagnostics together with
the scripts and CI configuration. The Ubuntu CI setup lives in
[setup-clang](../.github/actions/setup-clang/action.yml).

## C++ formatting

RaBitQ uses the repository's `.clang-format` configuration and clang-format
23. After enabling the [LLVM 23 package repository](https://apt.llvm.org/)
for your distribution, install that version on Ubuntu or Debian with:

```bash
sudo apt-get install clang-format-23
```

Format all project-maintained C and C++ files:

```bash
./scripts/apply-format.sh
```

Verify formatting without changing files:

```bash
./scripts/check-format.sh
```

The scripts intentionally exclude vendored code in `include/rabitqlib/third/`
and the frozen compatibility fixture `tests/fixtures/core_api_v1/consumer.cpp`.
To use a nonstandard executable name, set `CLANG_FORMAT`; it must still identify
itself as clang-format 23. On macOS, use Bash 4+ on PATH for the shell wrappers.
Apple SDKs newer than clang-tidy 23 may require a matching newer analyzer; set
`RUN_CLANG_TIDY` to its runner and report that version separately from the Linux
clang-tidy 23 baseline.

clangd embeds its own formatter, so use clangd 23 in editors such as VS Code
if format-on-save must exactly match CI. If another clangd version is required,
disable format-on-save and run the repository scripts before submitting.

To format only lines changed in the staged files, use:

```bash
./scripts/format-changed.sh --staged
```

The complete-file formatter remains useful before the initial formatting pass
or after changing `.clang-format`; CI always checks complete files.

## Static analysis

clang-tidy performs semantic checks and is kept separate from clang-format.
The required baseline contains focused correctness, portability, and
performance checks. Install the pinned analyzer and the dependencies needed to
configure the core library and Python bindings:

```bash
sudo apt-get install clang-23 clang-tidy-23 libomp-23-dev cmake ninja-build
python -m pip install "numpy>=1.23" "pybind11>=2.12"
```

Then configure the same core library and Python binding targets analyzed by CI
and run the check with the same compiler used by CMake:

```bash
export pybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake -S . -B build-tidy -G Ninja \
    -DCMAKE_CXX_COMPILER=clang++-23 \
    -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
    -DRABITQ_BUILD_SAMPLES=OFF \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_BUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$pybind11_DIR"
CXX=clang++-23 ./scripts/check-tidy.sh build-tidy
```

Use a fresh build directory when changing compilers. `CMAKE_CXX_COMPILER`
selects the compiler explicitly; `CXX` on the check command must match it.
Other supported compilers can still build the library; this command selects the
development analysis baseline only.

The wrapper supplies clang-tidy with that compiler's standard-library include
paths and reports diagnostics only for first-party files. Vendored Eigen and
hnswlib are excluded: `rabitqlib/third/` includes are marked as system headers
for analysis, including nested headers. The header filter alone is insufficient
for analyzer reports whose call path includes first-party code. This does not
change production compiler flags. New checks should be added incrementally after
their existing first-party findings are fixed.

Focused clang-tidy checks on affected code are sufficient during iteration.
For core library or binding changes, run the full check above before merging.
Tests and examples are intentionally excluded from CI clang-tidy; use focused
local analysis for substantial changes to their logic. Report whether validation
was focused or complete.

### Include dependency reports

The optional include dependency check uses clang-tidy's
`misc-include-cleaner` to report missing and unused includes. Run it locally when
reviewing include changes. Use LLVM 23 to match the script's default analyzer
and the shared development baseline. After enabling
[LLVM's APT repository](https://apt.llvm.org/) for your distribution, install
the matching compiler, analysis and OpenMP packages:

```bash
sudo apt-get update
sudo apt-get install clang-23 clang-tidy-23 libomp-23-dev cmake ninja-build
export CLANG_TIDY=clang-tidy-23
export CXX=clang++-23
```

Configure and run the same check locally (use a fresh build directory when
changing compilers). Clang needs its matching OpenMP development package;
for LLVM 23, that is `libomp-23-dev`. An older installed
`libomp-dev` alone may not be found by Clang 23:

```bash
cmake -S . -B build-includes -G Ninja \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++-23}" \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DRABITQ_BUILD_SAMPLES=ON \
  -DRABITQ_BUILD_TESTS=ON \
  -DRABITQ_BUILD_BENCHMARKS=ON \
  -DRABITQ_BUILD_PYTHON_BINDINGS=ON \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
  -DCMAKE_BUILD_TYPE=Release
./scripts/check-includes.sh build-includes
```

The script checks every configured first-party source, including bindings, samples,
tests and benchmarks. Disable optional targets above for a smaller check; skipped
files are listed explicitly. Installed-package consumers and the historical fixture
generator have separate build configurations and must be checked in those contexts.
The historical generator requires the pinned release headers described in
[its README](../tests/fixtures/persistence/README.md).

Headers are checked as main files with `-x c++-header`, preserving `#pragma once`
for mutually including headers. Include Cleaner does not report findings in included
headers, so a source-only pass is insufficient. Public headers inherit configured
component flags; numerical headers also run with OpenMP disabled. Private headers
inherit the flags of their configured source consumers through local includes,
covering the enabled ISA variants without maintaining a second backend list.
Run a separate AArch64 compilation database to cover NEON; an x86 check lists those
files as skipped. New untracked files are included, and tracked files deleted from
the working tree are skipped. `INCLUDE_JOBS` controls parallelism (default: 2), and
`PYTHON` can select the existing project interpreter for the driver.

Vendored files are excluded. The check ignores suggestions to include Eigen/hnswlib
implementation headers behind their public entry points, and pybind11's internal
`detail/common.h` behind `PYBIND11_MODULE`. These vendor headers lack the export
annotations needed by Include Cleaner. First-party export annotations describe
intentional API re-exports; `keep` annotations retain configuration-only includes
and explicit header-compilation tests. Neither is a general unused-include waiver.

The script fails for include findings, duplicate includes or analyzer errors and
does not modify files. Review suggestions across supported configurations before
applying them. Preserve public re-exports, frozen source consumers and conditional
includes. Format changes and rebuild/test affected targets after fixes. Findings can
vary between LLVM releases; this check is independent of the general clang-tidy job.

### Focused static analysis

Use a temporary subset of the compilation database with the same wrapper to
retain its compiler include paths and vendored-header exclusions. After the
[static-analysis configuration](#static-analysis) (which creates `build-tidy`),
run this example from the repository root using the
existing project Python environment:

```bash
CXX=clang++-23 python - <<'PY'
import json
from pathlib import Path
import subprocess
import tempfile

selected = {Path("src/utils/cpu_features.cpp").resolve()}
database = json.loads(Path("build-tidy/compile_commands.json").read_text())
entries = [
    entry for entry in database
    if (Path(entry["directory"]) / entry["file"]).resolve() in selected
]
found = {(Path(entry["directory"]) / entry["file"]).resolve() for entry in entries}
if found != selected:
    raise SystemExit("Selected files are missing from the compilation database")
with tempfile.TemporaryDirectory(prefix="rabitq-tidy-") as subset:
    Path(subset, "compile_commands.json").write_text(json.dumps(entries))
    subprocess.run(["./scripts/check-tidy.sh", subset], check=True)
PY
```

Replace `selected` with the affected `.cpp` paths. For a changed header, select
translation units that include it directly or transitively, covering affected
template instantiations and ISA variants. Configure any required test/sample
targets first if they are absent from the database. Do not analyze a header as
a standalone translation unit. If the consumer set is unclear, run the full
check. Replace `CXX=clang++-23` in the example if you configured with another compiler.
The wrapper's success message covers only the selected database entries in this
mode; it does not mean the full pre-merge check passed.

## Python formatting and linting

Python sources, examples, and tests use Ruff 0.16.1:

```bash
python -m pip install "ruff==0.16.1"
./scripts/check-python.sh
```

To apply Python formatting and safe automatic lint fixes before running the
check:

```bash
ruff check --fix python_bindings sample/python tests/python benchmarks/hotpaths benchmarks/plot_homepage.py
ruff format python_bindings sample/python tests/python benchmarks/hotpaths benchmarks/plot_homepage.py
```

## Shell scripts

Run ShellCheck after changing a contributor or automation script:

```bash
sudo apt-get install shellcheck
shellcheck scripts/*.sh sample/run_cpp.sh sample/run_python.sh
```

This matches CI's script set. Also pass any changed shell scripts outside
`scripts/` to `shellcheck` explicitly.

## Performance and compatibility

- Use fixed-width integer types for serialized values and persisted index data.
- Preserve existing public headers, aliases, and index formats unless a change
  is explicitly documented as breaking.
- Add backend-independent tests when introducing or changing SIMD kernels.
- Keep scalar, NEON, AVX2, and AVX-512 implementations behaviorally equivalent.
- Benchmark allocations or algorithm changes in search and quantization hot
  paths, and substantiate performance claims with benchmarks. Run correctness
  tests first and include commands, dataset, CPU, compiler, ISA, thread count,
  latency/throughput, and quality results in the pull request. Correctness fixes
  may omit benchmarks with a documented justification; do not make unmeasured
  performance claims.
- Preserve estimated-distance semantics. Before comparing recall, state the
  metric, workload, and acceptable recall tolerance, using repeated baseline
  measurements where relevant. Keep inputs, seeds, and search parameters fixed.
  Evaluate distance-estimate numerical error separately with justified numerical
  tolerances; deterministic SIMD backends need not produce bitwise-identical
  floating-point results. Report observed differences. Agree on intentional
  speed–quality tradeoffs before implementation; measurement tolerance does not
  authorize a quality reduction.
- Avoid unrelated refactoring or formatting in performance-sensitive changes.

## Implementation recipes

### Validation, ownership, and numerical preparation

Keep validation at the public operation boundary and reuse only invariants established
for that operation. Buffer contents and per-record factors may change between calls:
validate them again even when reusing a query or workspace.

| Boundary | Ownership and checks | Internal numerical work |
| --- | --- | --- |
| Python conversion | `ensure_*` returns a normalized NumPy owner retained across GIL release; `checked_cluster_ids` validates every ID while making the owned PID copy required by builders | Native calls borrow those arrays only within their established lifetime |
| Stable core v1 | Callers own codes, factors, queries and workspace; each operation checks its buffers and ranges | Prepared queries retain validated dimensions, bit width, metric and kernel selection; conversion checks both layouts and each logical code before writing output |
| FastScan LUT preparation | `Lut` owns prepared bytes; constructor/reset check shape once, then check borrowed scratch and resize owned storage | Private `build_lut` performs preparation with the established shape and scratch; reset reuses capacity |
| Query correction | `BatchQuery` owns its LUT and scalar corrections; `SplitBatchQuery` additionally borrows the rotated query for reranking | Complete LUT reset before updating the query's metric and correction metadata |

Keep the v1 `compute` checks for each code view, finite factors, unpacked code values,
and finite output. Keep scratch APIs allocation-bounded as documented; pointer reuse
alone is not permission to skip validation. The FastScan scratch pointers are borrowed
only during preparation and must never become retained query storage. Test rejected
resets, scratch reuse, and changed records alongside normal numerical parity.

### Add or change a SIMD kernel

1. Define the backend-neutral entry point under the owning module's `detail/` headers.
2. Update all applicable declarations and AVX2/AVX-512 implementations in `src/simd/` (core),
   `src/fastscan/` (FastScan), or `src/index/` (indexes).
3. Register selection and unsupported-CPU behavior in `src/simd/dispatch.cpp` when dispatched
   (batch scoring uses `src/fastscan/fastscan_dispatch.cpp`, HNSW search uses
   `src/index/hnsw_dispatch.cpp`).
4. If adding a translation unit, put it in the correct ISA source group and flags in
   `CMakeLists.txt`.
5. Add differential tests against a scalar or simple reference, including boundary dimensions,
   degenerate inputs, and every supported bit width.

Never execute a high-ISA implementation merely to test whether that ISA is supported; detection
must happen in generic code first.

#### Dispatch conventions and coverage

- Keep core backend declarations in `include/rabitqlib/core/detail/simd/`, FastScan
  declarations in `fastscan/detail/`, and HNSW declarations in `index/detail/`. Keep
  core implementations in `src/simd/*_{generic,neon,avx2,avx512}.cpp` and FastScan
  implementations in `src/fastscan/`. HNSW keeps its existing
  `src/index/` implementations and compatibility namespaces.
- Select a cached function pointer through the private shared resolver in
  `src/simd/dispatch_resolver.hpp`. Core, FastScan and HNSW have separate dispatch translation units.
  The x86 order is AVX-512, AVX2/FMA, then the portable generic implementation. ARM64
  selects NEON for raw float distances, FastScan, packed-code dot products (1–8 extra
  bits), masked sums, binary warmup, HNSW search, and FHT/Kac rotation including
  sign flipping, query-bit transposition, uint8/uint16 scalar quantization, and
  FastScan LUT construction. Extra-code packing retains a portable scalar implementation.
  Cache selection in thread-safe function-local statics, including tables; public calls must
  work before `main()` regardless of translation-unit initialization order. Public wrappers
  do not repeat feature checks.
  Standard FastScan accumulation requires AVX2/FMA, AVX-512, or NEON and throws
  `std::runtime_error` when no supported SIMD backend is available. Its scalar kernel
  is retained as a correctness reference only, never as a runtime fallback.
- Preserve stricter predicates: population-count kernels need AVX512_VPOPCNTDQ, and HNSW's
  AVX-512 core variant also needs AVX2 for its warmup implementation. Do not infer support
  from a backend name or from `__AVX*__` macros in a public header.
  All AVX-512 backends require AVX2, which GCC/Clang also enable with `-mavx512f`.
  MSVC's `/arch:AVX512` additionally requires AVX512VL and AVX512CD; keep these checks
  synchronized with the compiler flags. This optional target follows the compiler-group
  approach documented in [NumPy's MSVC compatibility rules](https://numpy.org/doc/1.26/reference/simd/build-options.html#on-x86-microsoft-visual-c-c).
  Missing group features select the AVX2 fallback; they do not raise the library's minimum
  CPU requirements. Native tuning defaults to on for local builds and must be explicitly
  disabled for distributed builds; release wheels already set it to off.
- FastScan selects one immutable private `Backend` in `src/fastscan/fastscan_backend.hpp`
  and `fastscan_dispatch.cpp`. LUT production, HACC conversion, accumulation and batch
  correction must be chosen as a compatible group. Standard LUTs share consecutive
  16-entry tables; HACC has paired 32-byte planes (AVX2/generic/NEON) or grouped 64-byte
  planes (AVX-512). These are transient query layouts, not persisted-code formats.
  Do not select producer and consumer independently when adding a backend.
- Dispatch once at a generic operation boundary. Warm wrappers cache their selected
  function pointer; keep feature selection and backend-table access in cold initialization.
  A backend may directly compose an accumulator only when its LUT layout is established.
  AVX-512 batch correction does so. Explicit AVX2/generic correction entry points also
  accept host-prepared queries and intentionally retain host accumulation dispatch;
  their function name alone does not specify the LUT format. Estimator templates require
  explicit accumulator arguments so new call sites must make this choice deliberately.
  Never bypass public dimension checks, unsupported-CPU errors or initialization safety.
- Keep performance regression workloads in [component hot-path benchmarks](../benchmarks/hotpaths/README.md).
  Validate numerical results first, then inspect generated code and alternate comparable
  baseline/candidate runs. Keep failed experiments and quality results. Fusion, inlining,
  scratch reuse and dispatch changes need measurements across relevant vector sizes;
  a short-vector improvement does not justify applying it to wider vectors.
- Put calculations and scratch-storage helpers outside the dispatcher. Shared implementation
  headers use internal linkage so independently compiled backends retain their own bodies.
  The AVX FHT intrinsics in `src/simd/fht_kernels.hpp` use the private kernel namespace
  for the same reason. Rotation uses this implementation on GCC, Clang, and MSVC;
  no assembler is required. Keep the FFHT attribution and MIT license in the
  intrinsics header intact.
- Pass ordinary pointers, sizes, and library-owned query state across ISA boundaries. Never
  pass Eigen matrix/packet objects between backends. The private matrix implementation
  header includes Eigen under a namespace selected by each backend translation unit.
  Otherwise, Eigen emits identically named out-of-line template helpers, which the linker
  can merge across incompatible ISA builds.
  Keep this isolation when adding matrix kernels; do not modify the vendor snapshot.
- Preserve existing public names and namespaces, including legacy functions with `_avx` in
  their name that now dispatch at runtime. The explicit `_avx2` and `_avx512` entry points
  are for selected kernels and capability-guarded backend tests.

The current first-party kernel audit is summarized below. Dispatching an index operation
covers its arithmetic kernels, not every scalar loop in construction and search.

| Area | Dispatch coverage / deliberate boundary |
| --- | --- |
| Raw float distances, norms, packed-code products | Runtime AVX2/AVX-512/NEON selection; portable generic fallback |
| Quantizer rescale search | SIMD bounded search; the certified scalar event sweep remains the fallback |
| Integer scalar quantization, extra-code packing, transpose, sign masks | Existing runtime selection; byte layouts and rounding rules are unchanged |
| Standard and high-accuracy FastScan | Runtime selection, including NEON float LUT construction; HACC LUT byte conversion remains generic on ARM |
| IVF and float SymphonyQG batch correction | Complete estimator runs in the selected backend; non-float template paths remain generic |
| FHT/Kac rotation | Complete rotation and scaling run in selected ISA translation units; shared AVX intrinsics preserve the FFHT butterfly order |
| Float matrix rotation and PiPNN construction | Matrix products, row norms, and lower-triangle pairwise distances use isolated matrix backends |
| RaBitQKMeans and QGKMeans centroid updates | Indexed float32 rows accumulate into float64 sums through the matrix backends, preserving per-cluster point order without additional workspace |
| HNSW search and IVF centroid routing | Cached HNSW search selection; centroid routing uses the common raw-distance dispatcher |
| Quantization orchestration, reconstruction, non-float utilities | Template/control code remains generic; no blanket native tuning or reduction-order rewrite |
| Graph scheduling, candidate queues, I/O, allocation, random initialization | Generic control code; IVF one-bit candidate insertion stays in a small compiled function to avoid inlining-induced register spills; thread scheduling and seeds remain caller-owned |
| Example KMeans training | Explicit RaBitQKMeans flat or QGKMeans graph assignment; FAISS is an optional comparison dependency only |

Portable wheels disable `RABITQ_ENABLE_NATIVE_OPTIMIZATION`. ARM64 builds exclude all
x86 source groups and use standard AArch64 NEON intrinsics, without Apple-only APIs.
macOS and Linux ARM64 have native C++ and installed-wheel CI. The reference suites
also exercise portable scalar kernels on x86, including byte-for-byte packing parity
with the selected x86 backend.

### Module build boundaries

Core must not depend on FastScan scoring or index headers, objects or symbols. FastScan
must not depend on indexes. The canonical headers are under `core/`, `fastscan/` and
`index/{ivf,hnsw,symqg}/`. Old `quantization/`, `utils/rotator.hpp` and generic
`index/{query,lut,estimator}.hpp` forwarders have been removed. Migrate consumers to
their owning module; see the [migration guide](../docs/docs/modular_migration.md).
Core-only installs exclude FastScan/index/clustering headers and vendored hnswlib.
Installed detail headers support templates and are not public APIs. Internal-only
non-template dispatch declarations stay in `src/`.
`defines.hpp` has been removed. Keep metrics in `core/metric.hpp`, scalar quantizer,
rotation and quantization configuration in `core/types.hpp`, Eigen aliases in `core/detail/eigen.hpp`, and ANN values
in `index/types.hpp`. Do not introduce an umbrella that couples them again.
IVF initializer choices live in `index/ivf/types.hpp`. IVF/HNSW/SymphonyQG public index headers
contain declarations and small accessors; their operations compile in
`src/index/{ivf,hnsw,qg}.cpp`. Backend translation units include HNSW's private
`src/index/hnsw_search_impl.hpp` to retain per-ISA template instantiation.
The installed consumer compiles and links each index header independently while poisoning
encoding, scoring, rotation implementation and vendor headers to enforce this boundary.
All CMake components export 64-byte Eigen alignment limits so native and generic
translation units share compatible storage and TLS allocation rules. Retain these
definitions in installed exports; SIMD backends keep their private Eigen namespaces.
Both two-factor and three-factor batch layouts/encoders belong to
`fastscan/{data_layout,encoder}.hpp`. Index-only nearest-neighbor helpers
belong in `index/detail/`, not generic numerical `utils/space.hpp`.
FastScan's 32-vector packing, inverse packing, batch size and permutation are owned by
`fastscan/pack_codes.hpp`; the batch encoder adapter lives in `fastscan/detail/encoder.hpp`.
Core retains per-vector packing and encoding loops independent of that batch format.
`utils/packed.hpp` supplies generic unaligned byte access for both numerical layouts and
index records. The `CoreModuleBoundary` test rejects format-specific declarations copied
into core even if they introduce no upward header include.

Keep core's and FastScan's architecture source groups explicit in CMake. The NEON LUT
implementation belongs to `fastscan_neon.cpp`, while scalar-query preprocessing belongs
to `query_neon.cpp`. All dispatchers share the private resolver and CPU predicates.
Eigen remains bundled. `RABITQ_USE_OPENMP=OFF` removes OpenMP from core/FastScan builds;
ANN still requires it. Guard optional numerical OpenMP calls/pragmas with the exported
`RABITQ_USE_OPENMP` definition via `utils/openmp.hpp`; do not provide global fake `omp_*`
functions. Test both configurations, including installed consumers. Python is a MODULE extension with an optional
build-tree alias; it is not a C++ link library or a C++ package component.
Validate these boundaries with the
[external-consumer checks](../tests/README.md#core-only-cmake-package-tests).

### Stable core API changes

`core/api.hpp` is the source-stable float32 v1 boundary; the implementation is
`src/core/api.cpp`. See [its contract](../docs/docs/rabitq/core_api.md). Keep the frozen
consumer, API unit tests and both `core_api_no_alloc_*` processes in the validation
matrix. The scalar allocation process forces cached dispatch onto the exact event
fallback. Do not replace that fallback with an approximate winner to avoid allocations.
Legacy and caller-backed paths share factor/search implementations. Validate numerical
parity against a saved pre-change build when changing those shared implementations.

### Change quantization or packing

Check all of these together:

- `core/encoder.hpp` and `core/detail/rabitq_impl.hpp`;
- `core/data_layout.hpp` and `core/pack_excode.hpp`;
- SIMD pack, inner-product, and FastScan kernels;
- query-side factors in `core/query.hpp` and `fastscan/query.hpp`;
- estimators in `core/estimator.hpp` and `fastscan/estimator.hpp`;
- IVF, HNSW, and SymphonyQG consumers;
- C++ tests for factor finiteness, reconstruction, sign convention, pack/unpack, and estimates.

Test both `METRIC_L2` and `METRIC_IP` where supported. In IVF and HNSW, total bits are represented
as one sign bit plus `ex_bits`; accepted total bit counts are 1 through 9. SymphonyQG raw storage is
`quantization_bits == 0`; its quantized storage currently accepts only 4 or 8 bits.

### Change an index API

Update the C++ declaration/implementation, the corresponding file in `python_bindings/`, Python
tests, samples, and the relevant page under `docs/docs/index/`. Preserve public signatures where
possible; add a forwarding overload when evolving an API compatibly.

### Change persistence

Do not silently reinterpret an old file. Add a magic/version discriminator, use fixed-width
serialized fields for new formats, validate sizes before allocation, check every read, and retain a
compatibility test fixture or an explicit rejection path. IVF, HNSW, and SymphonyQG have separate
formats and must each be reviewed. SymphonyQG includes a versioned quantized format plus a legacy
raw-format fallback; preserve both unless a breaking change is explicitly requested.

IVF `remove` stores a removed point as `f_add = +inf` inside the existing batch data, so it needs no
discriminator. No index written before `remove` existed has an infinite `f_add`, so nothing in an old
file is reinterpreted. In the legacy file formats readable by release 0.5.0, that reader also
excludes removed points. This compatibility of the removal marker does not extend to the current
writer's `RABQIDX1` v2 format: older releases cannot read current saves, even when resaving a loaded
legacy index. Keep this convention when touching IVF code:

- Every estimator and scan path must turn `f_add = +inf` into an infinite estimated distance and an
  infinite lower bound, and never into NaN. A NaN distance would enter the result buffer.
- `fastscan::unpack_codes` must stay the exact inverse of `pack_codes`; `IVF::add` uses it to refill
  the partial last batch of a cluster.
- `IvfRemoveTest.ExcludesRemovedPointsOnEveryScanPath`, `IvfRemoveTest.SurvivesSaveLoadAndLaterAdds`,
  and `FastScanPackingTest.UnpackInvertsPackIncludingTailsAndDoesNotOverrun` guard these rules.

HNSW `remove` sets the high bit (`kRemovedMask`) of the stored cluster ID. `construct` and `load`
cap the number of clusters at 2^31, so no index written before `remove` existed has that bit set and
nothing in an old file is reinterpreted. A file that has removals is rejected by releases 0.3.7
through 0.5.1, whose `load` range-checks every cluster ID; that rejection is the compatibility path.
This mark is unrelated to the `SearchBuffer` ID marker. Keep these rules when touching HNSW code:

- Read cluster IDs only through `get_clusterid_by_internalid`, which masks the bit. Estimators,
  routing, and the `add` reconstruction index centroids with it.
- A removed point stays in the graph: base-layer search still inserts it into `candidate_set`, and
  `add` may link to it. Only the result buffer skips it, including when it is the entry point.
- `load` masks the bit before range-checking the cluster ID.
- `HnswRemoveTest.SearchReachesLivePointsThroughRemovedOnes`,
  `HnswRemoveTest.SurvivesSaveLoadResizeAndLaterAdds`, and
  `HnswRemoveTest.LoadRejectsAnOutOfRangeClusterEvenWithTheMark` guard these rules.

### Change Python bindings

Shared NumPy and string conversion helpers live in `python_bindings/bindings_common.hpp`. Register
index-specific APIs in their own binding file and export public classes from
`python_bindings/__init__.py`. Validate rank, dimensionality, state, and parameter ranges before
entering the core. Be deliberate about `py::array::forcecast`: it permits dtype/layout copies and
must not be used where callers expect in-place mutation or pointer identity.
