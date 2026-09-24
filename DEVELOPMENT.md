# Development reference

Start with [Contributing to RaBitQ](CONTRIBUTING.md) for setup and your first PR.
This reference covers tool configuration and changes to library internals.
Run commands from the repository root in your activated
[project environment](CONTRIBUTING.md#python-environment), after installing
the [source-build prerequisites](CONTRIBUTING.md#setup-for-code-changes).
Refresh APT package lists with `sudo apt-get update` before installing tools.

## C++ formatting

RaBitQ uses the repository's `.clang-format` configuration and clang-format
15. Install that version on Ubuntu or Debian with:

```bash
sudo apt-get install clang-format-15
```

Format all project-maintained C and C++ files:

```bash
./scripts/apply-format.sh
```

Verify formatting without changing files:

```bash
./scripts/check-format.sh
```

The scripts intentionally exclude vendored code in `include/rabitqlib/third/`.
To use a nonstandard executable name, set `CLANG_FORMAT`; it must still identify
itself as clang-format 15. On macOS, use Bash 4+ on PATH for the shell wrappers.
Apple SDKs newer than clang-tidy 15 may require a matching newer analyzer; set
`RUN_CLANG_TIDY` to its runner and report that version separately from the Linux
clang-tidy 15 baseline.

clangd embeds its own formatter, so use clangd 15 in editors such as VS Code
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
configure every first-party target:

```bash
sudo apt-get install clang-tidy-15 libomp-15-dev cmake ninja-build
python -m pip install "numpy>=1.23" "pybind11>=2.12"
```

Then configure the same core library and Python binding targets analyzed by CI
and run the check with the same compiler used by CMake:

```bash
export pybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake -S . -B build-tidy -G Ninja \
    -DCMAKE_CXX_COMPILER=c++ \
    -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
    -DRABITQ_BUILD_SAMPLES=OFF \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_BUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$pybind11_DIR"
CXX=c++ ./scripts/check-tidy.sh build-tidy
```

Use a fresh build directory when changing compilers. `CMAKE_CXX_COMPILER`
selects the compiler explicitly; `CXX=c++` on the check command must match it.

The wrapper supplies clang-tidy with that compiler's standard-library include
paths and reports diagnostics only for first-party files. Vendored Eigen and
hnswlib are excluded. New checks should be added incrementally after their
existing first-party findings are fixed.

Focused clang-tidy checks on affected code are sufficient during iteration.
For first-party C++ changes, run the full check above before merging and report
whether validation was focused or complete.

### Include dependency reports

The **Include cleaner (advisory)** job uses clang-tidy's `misc-include-cleaner`
check to report missing and unused includes. CI uses `ubuntu-latest` and
unversioned distribution packages:

```bash
sudo apt-get update
sudo apt-get install clang-tidy clang libomp-dev cmake ninja-build
```

Clang-tidy 17 or newer is required. On older distributions such as Ubuntu 22.04,
use [LLVM's APT repository](https://apt.llvm.org/) to install a newer release.
For example, install LLVM 22 and its matching analysis and OpenMP packages:

```bash
wget https://apt.llvm.org/llvm.sh
sudo bash llvm.sh 22
sudo apt-get install clang-tidy-22 libomp-22-dev cmake ninja-build
export CLANG_TIDY=clang-tidy-22
export CXX=clang++-22
```

Configure and run the same check locally (use a fresh build directory when
changing compilers). Clang needs its matching OpenMP development package;
for the LLVM 22 example above, that is `libomp-22-dev`. An older installed
`libomp-dev` alone may not be found by Clang 22:

```bash
cmake -S . -B build-includes -G Ninja \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DRABITQ_BUILD_SAMPLES=OFF \
  -DRABITQ_BUILD_TESTS=OFF \
  -DRABITQ_BUILD_PYTHON_BINDINGS=OFF \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
  -DCMAKE_BUILD_TYPE=Release
./scripts/check-includes.sh build-includes
```

The script checks library sources using their compilation database and checks
headers as main files, since this clang-tidy check does not report findings in
included headers. Private headers are checked with AVX2 and AVX-512 flags. New
untracked files are included, and tracked files deleted from the working tree are skipped.
Vendored files are excluded. The script also ignores suggestions to include
Eigen and hnswlib implementation headers behind their existing public headers;
these vendor snapshots lack the export annotations needed by include-cleaner.
`INCLUDE_JOBS` controls parallelism (default: 2).
The script fails for findings or analyzer errors; CI keeps this step advisory
and uploads the `include-report` artifact without modifying files.

Review suggestions before applying them, especially for templates and public
forwarding headers. For a source file, automatic fixes can be applied with:

```bash
"${CLANG_TIDY:-clang-tidy}" -p build-includes \
  --config='{CheckOptions: {misc-include-cleaner.IgnoreHeaders: "rabitqlib/third/Eigen/src/.*;rabitqlib/third/hnswlib/(hnswalg|space_l2)[.]h"}}' \
  --checks='-*,misc-include-cleaner' --fix src/simd/dispatch.cpp
```

Review the diff, run formatting, and rebuild and test affected code after fixes.
Findings can vary between LLVM releases. This include check is independent of
the existing general clang-tidy job.

### Focused static analysis

Use a temporary subset of the compilation database with the same wrapper to
retain its compiler include paths and vendored-header exclusions. After the
[static-analysis configuration](#static-analysis) (which creates `build-tidy`),
run this example from the repository root using the
existing project Python environment:

```bash
CXX=c++ python - <<'PY'
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
check. Replace `CXX=c++` in the example if you configured with another compiler.
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
ruff check --fix python python_bindings sample/python tests/python
ruff format python python_bindings sample/python tests/python
```

## Shell scripts

Run ShellCheck after changing a contributor or automation script:

```bash
sudo apt-get install shellcheck
shellcheck scripts/*.sh
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

### Add or change a SIMD kernel

1. Define the backend-neutral entry point under `include/rabitqlib/simd/`.
2. Update all applicable declarations and AVX2/AVX-512 implementations in `src/simd/` or
   `src/index/`.
3. Register selection and unsupported-CPU behavior in `src/simd/dispatch.cpp` when dispatched.
4. If adding a translation unit, put it in the correct ISA source group and flags in
   `CMakeLists.txt`.
5. Add differential tests against a scalar or simple reference, including boundary dimensions,
   degenerate inputs, and every supported bit width.

Never execute a high-ISA implementation merely to test whether that ISA is supported; detection
must happen in generic code first.

#### Dispatch conventions and coverage

- Keep backend-neutral declarations in `include/rabitqlib/simd/*_dispatch.hpp` and
  implementations in `src/simd/*_{generic,neon,avx2,avx512}.cpp`. HNSW keeps its existing
  `src/index/` implementations and compatibility namespaces.
- Select a cached function pointer through `resolve_kernel` in `src/simd/dispatch.cpp`.
  The x86 order is AVX-512, AVX2/FMA, then the portable generic implementation. ARM64
  selects NEON for raw float distances, FastScan, packed-code dot products (1–8 extra
  bits), masked sums, binary warmup, HNSW search, and FHT/Kac rotation including
  sign flipping, query-bit transposition, uint8/uint16 scalar quantization, and
  FastScan LUT construction. Extra-code packing retains a portable scalar implementation.
  Public wrappers do not repeat feature checks.
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
| HNSW search and IVF centroid routing | Cached HNSW search selection; centroid routing uses the common raw-distance dispatcher |
| Quantization orchestration, reconstruction, non-float utilities | Template/control code remains generic; no blanket native tuning or reduction-order rewrite |
| Graph scheduling, candidate queues, I/O, allocation, random initialization | Generic control code; IVF one-bit candidate insertion stays in a small compiled function to avoid inlining-induced register spills; thread scheduling and seeds remain caller-owned |
| Example KMeans training | Uses external FAISS, whose build and dispatch are independent of this package |

Portable wheels disable `RABITQ_ENABLE_NATIVE_OPTIMIZATION`. ARM64 builds exclude all
x86 source groups and use standard AArch64 NEON intrinsics, without Apple-only APIs.
macOS ARM64 has native C++ and installed-wheel CI. Linux ARM64 still requires separate
platform verification. The reference suites also exercise portable scalar kernels on x86, including
byte-for-byte packing parity with the selected x86 backend.

### Change quantization or packing

Check all of these together:

- `quantization/rabitq.hpp` and `quantization/rabitq_impl.hpp`;
- `quantization/data_layout.hpp` and `quantization/pack_excode.hpp`;
- SIMD pack, inner-product, and FastScan kernels;
- query-side factors in `index/query.hpp`;
- estimators in `index/estimator.hpp`;
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

### Change Python bindings

Shared NumPy and string conversion helpers live in `python_bindings/bindings_common.hpp`. Register
index-specific APIs in their own binding file and export public classes from
`python_bindings/__init__.py`. Validate rank, dimensionality, state, and parameter ranges before
entering the core. Be deliberate about `py::array::forcecast`: it permits dtype/layout copies and
must not be used where callers expect in-place mutation or pointer identity.
