# Development reference

Start with [Contributing to RaBitQ](../.github/CONTRIBUTING.md) for setup and your first PR.
Read the sections relevant to your change; this reference is not a checklist to run in full.
Required checks by change type live in [AGENTS.md](../AGENTS.md#verification-by-change-type).
For multi-step agent work, use the optional [task workflow and templates](agent-workflow/README.md)
for goals, execution records, handoffs, and independent review.
Run commands from the repository root in your activated
[project environment](../.github/CONTRIBUTING.md#python-environment), after installing
the [source-build prerequisites](../.github/CONTRIBUTING.md#setup-for-code-changes).
Refresh APT package lists with `sudo apt-get update` before installing tools.

## Overview

### Find the relevant workflow

| Task | Start here |
| --- | --- |
| Set up development tools | [Compiler and tool versions](#compiler-and-tool-versions) |
| Format or lint changes | [C++](#c-formatting), [Python](#python-formatting-and-linting), [shell scripts](#shell-scripts), [static analysis](#static-analysis) |
| Change module boundaries or the stable API | [Module build boundaries](#module-build-boundaries), [stable core API](#stable-core-api-changes) |
| Change numerical kernels or layouts | [SIMD recipe](#add-or-change-a-simd-kernel), [quantization and packing](#change-quantization-or-packing), [dimensions](#rotation-and-padded-dimensions), [dispatch reference](#dispatch-conventions-and-coverage) |
| Change index or Python APIs | [Index API](#change-an-index-api), [Python bindings and type checks](#change-python-bindings) |
| Change persistence or object state | [Persistence](#change-persistence), [validation and ownership](#validation-ownership-and-numerical-preparation), [exception safety](#exception-safe-state-updates) |
| Check naming conventions | [Indexes](#index-variable-names), [numerical modules](#numerical-module-variable-names) |
| Evaluate performance or prepare a release | [Performance and compatibility](#performance-and-compatibility), [downstream checks](#downstream-compatibility-validation) |

For the required checks by change type, see the
[verification matrix](../AGENTS.md#verification-by-change-type).

### Project and code map

RaBitQ-Library is a C++17 library with Python bindings (`rabitqlib`) for compact vector
quantization and approximate nearest-neighbor search. It provides one-bit and multi-bit RaBitQ
encoding and IVF, HNSW, and SymphonyQG indexes. It targets x86-64 with runtime AVX2/AVX-512
dispatch and Linux/macOS ARM64 with NEON and portable scalar kernels. Native Linux AArch64
C++ and installed-wheel CI validate that platform.
RaBitQ rotates vectors and quantizes residuals relative to centroids; compact codes and correction
factors estimate L2 distance or inner product.

| Path | Responsibility |
| --- | --- |
| `include/rabitqlib/core/api.hpp` | Stable float32 v1 API; code/factors separate, caller workspace, source compatibility |
| `include/rabitqlib/core/` | Per-vector encoding, query/estimation, rotation and per-vector byte layouts |
| `include/rabitqlib/fastscan/` | All 32-vector batch layouts, encoding adapters, packing, LUT/estimation and FastScan/HACC |
| `include/rabitqlib/core/types.hpp`, `core/detail/eigen.hpp` | Numerical configuration/types and internal Eigen aliases, respectively |
| `include/rabitqlib/index/types.hpp` | Point IDs and ANN candidate values; no Eigen dependency |
| `include/rabitqlib/clustering/` | RaBitQKMeans flat assignment, QGKMeans graph assignment, and shared Lloyd training |
| `include/rabitqlib/index/{ivf,hnsw,symqg}/` | Index construction, persistence, and search |
| `include/rabitqlib/{core,fastscan,index}/detail/` | Module-owned implementation headers; no public stability promise |
| `src/core/` | Compiled stable core v1 API implementation |
| `src/simd/` | Core kernels, dispatch, and the private shared resolver |
| `src/fastscan/` | FastScan kernels, batch estimators, and paired backend selection |
| `src/index/` | Compiled IVF/HNSW/SymphonyQG operations and private search kernels |
| `src/utils/cpu_features.cpp` | Generic CPU feature detection |
| `include/rabitqlib/utils/` | Shared allocation, I/O and numerical helpers; index buffers live in `index/detail/` |
| `python_bindings/` | pybind11 extension and index wrappers |
| `sample/cpp/`, `sample/python/` | Usage examples |
| `tests/unit/`, `tests/python/` | C++ reference/unit tests and Python boundary/integration tests |
| `docs/docs/` | MkDocs site sources |

## Development checks

### Compiler and tool versions

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

### C++ formatting

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

### Python formatting and linting

Python sources, examples, and tests use Ruff 0.16.1:

```bash
python -m pip install "ruff==0.16.1"
./scripts/check-python.sh
```

To apply Python formatting and safe automatic lint fixes before running the
check:

```bash
ruff check --fix python_bindings sample/python tests/python tests/test_mypy.py tests/test_pyright.py benchmarks
ruff format python_bindings sample/python tests/python tests/test_mypy.py tests/test_pyright.py benchmarks
```

### Shell scripts

Run ShellCheck after changing a contributor or automation script:

```bash
sudo apt-get install shellcheck
shellcheck scripts/*.sh sample/run_cpp.sh sample/run_python.sh
```

This matches CI's script set. Also pass any changed shell scripts outside
`scripts/` to `shellcheck` explicitly.

### Static analysis

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

#### Include dependency reports

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

#### Focused static analysis

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

## Implementation recipes

### Validation, ownership, and numerical preparation

Keep validation at the public operation boundary and reuse only invariants established
for that operation. Buffer contents and per-record factors may change between calls:
validate them again even when reusing a query or workspace.

**Python conversion**

`ensure_*` returns a normalized NumPy owner retained across GIL release;
`checked_cluster_ids` validates every ID while making the owned PID copy required by
builders.

Native calls borrow those arrays only within their established lifetime.

**Stable core v1**

Callers own codes, factors, queries and workspace; each operation checks its buffers and
ranges.

Prepared queries retain validated dimensions, bit width, metric and kernel selection;
conversion checks both layouts and each logical code before writing output.

**FastScan LUT preparation**

`Lut` owns prepared bytes; constructor/reset check shape once, then check borrowed scratch
and resize owned storage.

Private `build_lut` performs preparation with the established shape and scratch; reset
reuses capacity.

**Query correction**

`BatchQuery` owns its LUT and scalar corrections; `SplitBatchQuery` additionally borrows the
rotated query for reranking.

Complete LUT reset before updating the query's metric and correction metadata.

Keep the v1 `compute` checks for each code view, finite factors, unpacked code values,
and finite output. Keep scratch APIs allocation-bounded as documented; pointer reuse
alone is not permission to skip validation. The FastScan scratch pointers are borrowed
only during preparation and must never become retained query storage. Test rejected
resets, scratch reuse, and changed records alongside normal numerical parity.

### Exception-safe state updates

Keep owned storage and its dimensions, masks, pointers and correction metadata
consistent when allocation or copying throws. For query/LUT, rotator and visited-set
assignment, perform potentially throwing storage work before publishing metadata.
When several owned containers must change together, prepare a complete replacement
and commit it with a non-throwing move or swap. Do not replace these assignments
with memberwise defaults without checking failure ordering and borrowed pointers.

Index construction can have a different failure contract: an early validation
failure may retain the old index, while a later allocation failure may invalidate
it. Python wrapper state must reflect the native state after either outcome;
IVF uses stored count, not live count, because all points may have been removed.
Do not promise rollback for every index mutation. Test the affected object's
documented post-failure state and subsequent reuse when changing these paths.

### Reuse numerical helpers without changing contracts

Share an existing calculation in its owning module rather than copying it into
each caller or ISA backend. QG batch correction lives in
`fastscan/detail/qg_estimator.hpp`; callers explicitly select the accumulator
compatible with the prepared LUT. Scalar word-interleaved binary query scoring
is shared through `utils/warmup_space.hpp`; its layout is distinct from the
512-coordinate warmup layout. Retain translation-unit-local linkage for shared
implementation bodies compiled under different ISA flags.

Prefer standard value copying and existing size/layout helpers over new wrappers
or cached values derivable from authoritative state. Preserve arithmetic order,
precision and ownership while consolidating code. In templated numerical paths,
retain the computation type (`T`) for norms and correction intermediates; do not
narrow a supported double path to float. Verify supported non-float paths as well
as dispatched float kernels when changing shared formulas.

### Add or change a SIMD kernel

1. Define the backend-neutral entry point under the owning module's `detail/` headers.
2. Update all applicable declarations and generic/NEON/AVX2/AVX-512 implementations in `src/simd/` (core),
   `src/fastscan/` (FastScan), or `src/index/` (indexes).
3. Register selection and unsupported-CPU behavior in `src/simd/dispatch.cpp` when dispatched
   (batch scoring uses `src/fastscan/fastscan_dispatch.cpp`, HNSW-RaBitQ search uses
   `src/index/hnsw_dispatch.cpp`).
4. If adding a translation unit, put it in the correct ISA source group and flags in
   `CMakeLists.txt`.
5. Add differential tests against a scalar or simple reference, including boundary dimensions,
   degenerate inputs, and every supported bit width.

Never execute a high-ISA implementation merely to test whether that ISA is supported; detection
must happen in generic code first.

Startup tests must exercise calls before `main()` and compare against independent
expected values. Use nonzero inputs and sentinel outputs so skipped work or a
constant-zero result cannot pass; finiteness alone is not a numerical oracle.
See `tests/core_consumer/core_static_init.cpp` and
`tests/dispatch/modules_static_init.cpp`. Use `ctest --no-tests=error` in validation
commands, including filtered and installed-consumer runs, so an empty selection
fails instead of appearing to validate the change.

### Stable core API changes

`core/api.hpp` is the source-stable float32 v1 boundary; the implementation is
`src/core/api.cpp`. See [its contract](../docs/docs/rabitq/core_api.md). Keep the frozen
consumer, API unit tests and both `core_api_no_alloc_*` processes in the validation
matrix. The scalar allocation process forces cached dispatch onto the exact event
fallback. Do not replace that fallback with an approximate winner to avoid allocations.
Legacy and caller-backed paths share factor/search implementations. Validate numerical
parity against a saved pre-change build when changing those shared implementations.

Keep `tests/fixtures/core_api_v1/consumer.cpp` unchanged when extending v1; add tests.
Changes to v1 names, signatures, aggregate fields, enum values, buffer/code semantics
or allocation guarantees require an explicit API-major decision. The stable header
must not expose Eigen, record maps, ANN or ISA-specific types. Workspace APIs must
never silently allocate on the heap or spawn worker threads.

### Change quantization or packing

Preserve the [rotation/padding](#rotation-and-padded-dimensions) and
[distance conventions](#distance-conventions). For shared layout, packing, factor or
distance-semantic changes, audit every producer, backend and consumer sharing the
contract. For a local implementation fix, identify its affected paths from the map
below and check those paths; broaden the audit if the shared contract changes.

- `core/encoder.hpp` and `core/detail/rabitq_impl.hpp`;
- `core/data_layout.hpp` and `core/pack_excode.hpp`;
- SIMD pack, inner-product, and FastScan kernels;
- query-side factors in `core/query.hpp` and `fastscan/query.hpp`;
- estimators in `core/estimator.hpp` and `fastscan/estimator.hpp`;
- IVF-RaBitQ, HNSW-RaBitQ, and SymphonyQG consumers;
- C++ tests for factor finiteness, reconstruction, sign convention, pack/unpack, and estimates.

Estimator and scan changes must also preserve [removal semantics](#removal-markers-and-search),
even when no file format changes.

#### Layout invariants

- `BatchDataMap`, `ExDataMap`, `BinDataMap`, and their const variants define raw-buffer sizes and
  offsets. Do not duplicate those calculations. Audit all producers and consumers when changing
  offsets, alignment, factor order, or code packing.
- FastScan batches contain 32 vectors, including physical storage for tail batches. Distinguish
  logical point count from batch capacity.
- Retain explicit zero-residual sign-convention tests and factor-finiteness, reconstruction,
  pack/unpack, and estimation coverage. Test both `METRIC_L2` and `METRIC_IP` where supported.

#### IVF and HNSW bit widths

IVF/HNSW quantized total bits are one sign bit plus `ex_bits`, with totals 1 through 9.
IVF also accepts `bits == 32`: one-bit filtering plus owned original float32 vectors
in place of extra-bit codes.

#### IVF formats and initialization

New IVF files have a magic/version header, an explicit padded dimension, and a resolved initializer type (v2); v1 files infer the historical
Flat/HNSW choice at 20,000 clusters, independently of current auto selection.
New IVF auto routing uses Flat below 5,000 clusters, Flat RaBitQ below 60,000,
and HNSW otherwise. Flat RaBitQ persists its mean and codes in the main file;
HNSW retains its sidecar. Legacy quantized files and raw v1 files infer
64-dimension padding.

#### Search defaults

IVF search defaults to HACC for 4–9 bits and standard FastScan for 1–3 bits or raw storage.

#### SymphonyQG storage

SymphonyQG supports raw storage (`quantization_bits == 0`) and quantized storage at 4 or 8 bits.

### Change an index API

Update the C++ declaration/implementation, the corresponding file in `python_bindings/`, Python
type stubs, tests, samples, and the relevant page under `docs/docs/index/`. Preserve public signatures where
possible; add a forwarding overload when evolving an API compatibly. Index construction
and buffer changes must also preserve the [ID marker](#search-buffer-id-marker) and
[borrowed-input lifetimes](#ownership-during-construction).

### Filesystem paths

Filesystem path strings are UTF-8 on Windows and native path bytes on POSIX. Use
`io_impl::filesystem_path` from `utils/path.hpp` before opening files or querying
filesystem metadata, including IVF's HNSW sidecar. It converts UTF-8 to native Unicode
on Windows and preserves existing path bytes on Linux/macOS.

For path-only changes, check path conversion and the affected I/O operations; the
format audit below applies when serialized data or its interpretation changes.
Python path callbacks also follow the [binding rules](#change-python-bindings).

### Change persistence

Use the [path conversion helper](#filesystem-paths) for file access. Preserve
[removal markers](#removal-markers-and-search) and [stored rotation/dimensions](#rotation-and-padded-dimensions).

Frozen release-produced indexes and reference searches live in `tests/fixtures/persistence/`;
keep their provenance and hashes, and do not regenerate them with the current writer.

Do not silently reinterpret an old file. Add a magic/version discriminator, use fixed-width
serialized fields for new formats, validate sizes before allocation, check every read, and retain a
compatibility test fixture or an explicit rejection path. Review each affected format;
changes to shared serialization helpers require auditing every format using them.
IVF-RaBitQ, HNSW-RaBitQ, and SymphonyQG have separate formats. SymphonyQG includes a versioned quantized format plus a legacy
raw-format fallback; preserve both unless a breaking change is explicitly requested.

### Change Python bindings

Shared NumPy and string conversion helpers live in `python_bindings/bindings_common.hpp`. Register
index-specific APIs in their own binding file and export public classes from
`python_bindings/__init__.py`. Validate rank, dimensionality, state, and parameter ranges before
entering the core. Be deliberate about `py::array::forcecast`: it permits dtype/layout copies and
must not be used where callers expect in-place mutation or pointer identity.

C-contiguous NumPy storage is not necessarily aligned even to `alignof(T)`.
`ensure_2d_array` copies misaligned inputs into an aligned owner; one-dimensional
ID conversion uses packed reads in `checked_cluster_ids` and `checked_removal_ids`
before narrowing to owned PID storage. Reuse these helpers rather than directly
dereferencing a possibly misaligned typed pointer. Removal IDs are checked against
stored count, including removed records, rather than live count.

Resolve filesystem-path conversion callbacks (`__fspath__`) before acquiring
`IndexAccess` guards. Keep NumPy array conversion, including `__array__` callbacks,
inside the guard's lifetime so concurrent conflicting operations are rejected during
conversion. Acquire and destroy guards with the GIL held, release the GIL only inside
their lifetime, and keep converted owners alive until reacquisition.

Python wrappers use `IndexAccess` for concurrent reads and fail-fast exclusive updates.
Extract native pointers before releasing the GIL and keep search parameters local to
each call. C++ callers synchronize index mutations themselves.

Keep `python_bindings/__init__.pyi` and runtime docstrings synchronized with public
signatures, defaults and accepted inputs. For changes to Python API signatures,
type declarations or accepted-input contracts,
rebuild/install in the existing environment and run the
[editor and type checks](../tests/README.md#python-editor-support):

```bash
python -m pytest tests/python/test_editor_support.py tests/test_mypy.py tests/test_pyright.py -q
```

These checks require the `test,typing` extras and Node 24 as described there.
The mypy and Pyright tests live outside `tests/python` and are not selected by a
plain `python -m pytest` run with the repository's default `testpaths`.

## Naming conventions

### Index variable names

Use the same name for the same role across IVF, HNSW and SymphonyQG. Choose the
role and unit first; similar-looking parameters can have different contracts.
These conventions apply to new internal variables and compatible renaming work.

| Concept | Preferred names |
| --- | --- |
| Dimensions and counts | `dim`, `padded_dim`, `num_points`, `num_queries`, `num_clusters` |
| Index state | `max_elements_`, `num_stored_points_`, `num_points()` |
| Identity and outputs | `internal_id`, `node_id`, `label`, `cluster_id`, `result_ids`, `distances` |
| Search | `candidate_set`, `search_pool`, `result_pool`, `ef_search`, `ef_construction`, `nprobe` |
| Storage | `*_offset_`; record/link byte sizes named after the stored object |
| Parallelism and encoding | `num_threads`, `num_workers`, `fast_quantization` |

#### Names and units

**Dimensions**

`dim` is the original dimension at an index boundary; `padded_dim` is the rotated, padded
dimension. A lower-level helper's `dim` describes its own input shape.

**Counts**

`num_points` for an input batch or a builder's point set; `num_queries` and `num_clusters`
for those input counts. A partial encoding batch uses `batch_size`.

**Index state**

`max_elements_` for configured logical capacity; HNSW uses `num_stored_points_` for stored
records, including removals. IVF derives stored count from capacity after construction
completes. The public `num_points()` counts live records. QG has no removal and derives both
counts from capacity only after construction completes.

**Point identity**

`internal_id` or `node_id` for graph slots; `label` for a caller-visible HNSW label;
`cluster_id` for centroid routing. Never assume an external label equals an internal slot.

**Search outputs**

`k` for requested result count, `result_ids` for ID-only output arrays, `distances` for
their scores. `results` may hold complete distance/ID records.

**Search state**

`candidate_set` for indexing traversal candidates and `search_pool` for querying traversal
candidates, `result_pool` for retained results (HNSW retains `nearest_candidates` for query
results and `top_candidates` for construction/pruning candidates), `query_wrapper` for query
preprocessing objects, `visited` for visited state.

**Search windows**

`ef_search` for a per-call graph window, `default_ef_search_` for a stored default, and
`ef_construction` for graph-building search breadth. IVF's `nprobe` counts clusters and
remains distinct.

**Distances**

`query_centroid_ip` for a dot product, `residual_norm` for a residual norm,
`estimated_distances` for estimated score arrays, `lower_bounds` for pruning bounds, and
`dist_k` for the current result cutoff. Do not conflate squared L2 with L2 or an IP distance
with a dot product.

**Storage**

Name byte sizes after the stored object: HNSW's `level0_record_bytes_` is one complete
level-0 record; `upper_layer_links_bytes_` and `level0_links_bytes_` are one node's
adjacency-list sizes at the respective levels, including their headers. QG uses
`record_bytes_` for one complete record. Storage offsets use `*_offset_`: bytes are
implicit, so do not add a `_bytes` suffix. PiPNN scratch uses `points_buffer_size`,
`distances_buffer_size` and `worker_buffer_size`, with a declaration comment specifying
float-element counts.

**Threads and encoding**

`num_threads` for a requested limit, `num_workers` for the resolved worker count requested
from the runtime (OpenMP may use fewer threads), and `fast_quantization` for the fixed-scale
encoding option.

#### Compatibility and algorithm-specific names

Keep short loop indices and conventional mathematical symbols where their role
is local and unambiguous. HNSW's `M` remains its algorithm-specific configuration;
its upper and base layers use `upper_layer_max_degree_` and
`base_layer_max_degree_`. QG's `max_degree_` is its fixed row degree, a multiple
of 32. These settings are not interchangeable.

Public names and Python keywords are compatibility contracts. For example,
`Cluster::num()`, the protected `Initializer::num_cluster_`, QG's
`degree_bound()`/`set_ef()`, Python's `ef`, and `QGKMeansParameters::ef_build`
retain their existing spellings. C++ parameter names may be clarified without
changing types, order or defaults. Keep declarations, definitions and comments
in agreement; do not rename methods or literals by textual substitution.

Storage-mode parameters also retain their contracts: IVF/HNSW use `nbits()`
with separate internal `ex_bits_`; IVF additionally accepts 32 for raw-vector
reranking. QG uses `quantization_bits()` with 0 for raw storage and 4/8 for packed
vectors. Naming work does not change these values, allocation sizes, numeric
operations, serialized field order, or ownership.

### Numerical module variable names

Apply the same role-based naming to Core, FastScan, clustering and SIMD. Preserve
public and protected names, Python keywords, field order and numerical operations.

- Use `query_wrapper` for prepared query objects. Retain the existing `G_*`
  private correction-field names where they identify mathematical coefficients.
- Distinguish a scalar `estimated_distance` or `lower_bound` from batch outputs
  `estimated_distances` and `lower_bounds`. Keep `ip_x0_qr` where it identifies the
  mathematical intermediate used by refinement.
- Use `lookup_table` for FastScan table pointers and `accumulated_values` for
  integer accumulation outputs. Retain `lut` in compact local kernels where its
  role is clear.
- Clustering's private state uses `dim_`, `num_points_`, `num_clusters_`,
  `num_workers_` and `data_`. Keep public `d`, `k`, `num_threads` and graph parameter
  names compatible. Retain the existing private `niter_` graph iteration name.
- FHT backend arguments use `rotated_data`, `padded_dim`, `trunc_dim` and
  `scale_factor` consistently across declarations, dispatch and implementations.
  A lower-level kernel's `dim` still denotes that kernel's own input shape.

Keep short mathematical symbols, loop indices and SIMD register names when their
meaning is local and clear. Naming cleanup must not alter formulas, arithmetic
order, packed layouts, ISA selection or runtime behavior.

## Technical contracts

### Module build boundaries

CMake exposes `rabitqlib::core`, `rabitqlib::fastscan` (depending on core) and
`rabitqlib::index` (depending on FastScan). No upward header, object or symbol dependencies
are allowed. For core alone disable `RABITQ_BUILD_INDEX` and `RABITQ_BUILD_FASTSCAN`.
Install only enabled components and their dependencies; core-only installs exclude
FastScan/index/clustering and vendored hnswlib. `benchmarks/core/` links only core.
Python is a MODULE extension with build-tree alias `rabitqlib::python` and import name
`rabitqlib`, not a C++ link library or installed C++ component.

The 0.7.0 refactor removed `rabitqlib::rabitqlib`, `defines.hpp`, `quantization/`,
`utils/rotator.hpp` and generic `index/{query,lut,estimator}.hpp` forwarders. Migrate
consumers to canonical `core/`, `fastscan/` and `index/{ivf,hnsw,symqg}/` headers;
do not restore umbrellas/forwarders to bypass migration failures. See the
[migration guide](docs/modular_migration.md).

**Core values**

Metric in `core/metric.hpp`; scalar quantizer, rotation and quantization configuration in
`core/types.hpp`; Eigen aliases in `core/detail/eigen.hpp`. No ANN IDs/candidates, QG batch
adapters or FastScan declarations/layouts in core.

**FastScan batches**

Two-/three-factor `BatchDataMap`/`QGBatchDataMap` layouts and encoders in
`fastscan/{data_layout,encoder}.hpp`; 32-vector packing, inverse packing, batch size and
permutation in `fastscan/pack_codes.hpp`; encoder adapter in `fastscan/detail/encoder.hpp`.
Core retains only per-vector encoding/packing.

**Index helpers**

ANN values in `index/types.hpp`, IVF initializer choices in `index/ivf/types.hpp`,
`exact_nn` in `index/detail/exact_nn.hpp`, scheduling in `index/detail/parallel_for.hpp`.
HNSW must not import IVF for scheduling.

**Packed access**

`utils/packed.hpp` supplies generic unaligned value/array access for numerical layouts and
index records.

IVF/HNSW/SymphonyQG public index headers contain declarations and small accessors;
they must not import encoding, scoring, rotator implementation, search scratch, Eigen
or hnswlib. Operations compile in `src/index/{ivf,hnsw,qg}.cpp`; HNSW's per-ISA templates
live in `src/index/hnsw_search_impl.hpp`. Preserve the advanced `hnsw_quant.hpp`
query-cache header. Installed-consumer tests poison implementation/vendor headers to
check index header independence; `CoreModuleBoundary` also rejects batch declarations
copied into core without an upward include.

Installed detail headers support templates and are not public APIs. Non-template dispatch
declarations needed by those headers also live in the owning module's `detail/` directory.
Backend tables, the shared resolver and implementation-only declarations stay in `src/`.
Keep architecture source groups explicit: NEON LUT code belongs to `fastscan_neon.cpp`,
scalar-query preprocessing to `query_neon.cpp`. Dispatchers share the resolver/CPU predicates.

Eigen remains bundled. All CMake components export 64-byte Eigen alignment limits for
compatible storage/TLS allocation across native and generic translation units; retain
these definitions in installed exports and private Eigen namespaces in SIMD backends.

`RABITQ_USE_OPENMP` defaults to ON and is exported. Core/FastScan support OFF; ANN/Python
require ON. Guard optional numerical runtime calls and pragmas via `utils/openmp.hpp`;
OFF uses serial helpers and installed packages must neither find nor link OpenMP.
Do not provide global fake `omp_*` functions. Test both configurations, including
[installed consumers](../tests/README.md#core-only-cmake-package-tests).

### Rotation and padded dimensions

Indexes quantize and search in the rotated, padded domain. FHT/Kac, RaBitQKMeans, QGKMeans,
SymphonyQG, IVF (all bit widths), and HNSW pad to multiples of 32. Packed extra-bit
and sign codes retain the historical full 64-coordinate blocks and compact final
32-coordinate tails. Binary query planes round their scratch storage up to
whole uint64_t words; 32-coordinate padding keeps HNSW record strides PID-aligned. Callers
provide vectors in the original `dim`, while internal code and quantization buffers
generally use `padded_dim`. A buffer allocated for `dim` must never receive `padded_dim` values.
Centroids, data, and queries compared by the same estimator must be in the same domain.

Rotator state is part of persisted index state. Loading an index must restore the exact rotation
used at construction; generating a new random rotator produces silently incorrect distances.
Honor stored padded dimensions instead of recomputing them from current defaults. IVF `RABQIDX1`
v1 and v2 files explicitly store this field; preserve it for both versions. Only legacy unversioned
quantized IVF files and old raw v1 files omit it and infer historical 64-dimension padding.
Unreleased pad16 files whose stored dimension is not divisible by 32 are rejected.

### Distance conventions

The library supports `METRIC_L2` and `METRIC_IP`. Some internal helpers represent inner-product
distance as `1 - dot_product`, and estimator correction terms differ between metrics. Do not reuse
an L2 norm/correction formula for IP merely because both paths compile. Test ordering and returned
distance semantics separately.

### Search-buffer ID marker

`SearchBuffer` uses the top bit of `PID` as an internal checked marker. Real point IDs therefore
must remain below `buffer::kSearchBufferMaxPointCount`. Do not change `PID`, sentinel values, or
checked-bit manipulation independently.

### Ownership during construction

Some index builders retain non-owning access to input vectors only during synchronous construction.
Do not extend such a pointer's lifetime without introducing explicit ownership. The HNSW Python
wrapper copies cluster IDs before calling the mutable C++ construction API; preserve that boundary
unless the C++ contract becomes const-correct.

### Removal markers and search

Apply the relevant index's rules when changing estimators, scans, graph traversal,
updates or persistence. These markers affect live search as well as saved files.

#### IVF removal

IVF-RaBitQ `remove` stores a removed point as `f_add = +inf` inside the existing batch data, so it needs no
discriminator. No index written before `remove` existed has an infinite `f_add`, so nothing in an old
file is reinterpreted. In the legacy file formats readable by release 0.5.0, that reader also
excludes removed points. This compatibility of the removal marker does not extend to the current
writer's `RABQIDX1` v2 format: older releases cannot read current saves, even when resaving a loaded
legacy index. Keep this convention when touching IVF-RaBitQ code:

- Every estimator and scan path must turn `f_add = +inf` into an infinite estimated distance and an
  infinite lower bound, and never into NaN. A NaN distance would enter the result buffer.
- `fastscan::unpack_codes` must stay the exact inverse of `pack_codes`; `IVF::add` uses it to refill
  the partial last batch of a cluster.
- `IvfRemoveTest.ExcludesRemovedPointsOnEveryScanPath`, `IvfRemoveTest.SurvivesSaveLoadAndLaterAdds`,
  and `FastScanPackingTest.UnpackInvertsPackIncludingTailsAndDoesNotOverrun` guard these rules.

#### HNSW removal

HNSW-RaBitQ `remove` sets the high bit (`kRemovedMask`) of the stored cluster ID. `construct` and `load`
cap the number of clusters at 2^31, so no index written before `remove` existed has that bit set and
nothing in an old file is reinterpreted. A file that has removals is rejected by releases 0.3.7
through 0.5.1, whose `load` range-checks every cluster ID; that rejection is the compatibility path.
This mark is unrelated to the `SearchBuffer` ID marker. Keep these rules when touching HNSW-RaBitQ code:

- Read cluster IDs only through `get_clusterid_by_internalid`, which masks the bit. Estimators,
  routing, and the `add` reconstruction index centroids with it.
- A removed point stays in the graph: base-layer search still inserts it into `candidate_set`, and
  `add` may link to it. Only the result buffer skips it, including when it is the entry point.
- `load` masks the bit before range-checking the cluster ID.
- `HnswRemoveTest.SearchReachesLivePointsThroughRemovedOnes`,
  `HnswRemoveTest.SurvivesSaveLoadResizeAndLaterAdds`, and
  `HnswRemoveTest.LoadRejectsAnOutOfRangeClusterEvenWithTheMark` guard these rules.

### Dispatch conventions and coverage

#### Backend locations

Keep core backend declarations in `include/rabitqlib/core/detail/simd/`, FastScan
declarations in `fastscan/detail/`, and HNSW-RaBitQ declarations in `index/detail/`. Keep
core implementations in `src/simd/*_{generic,neon,avx2,avx512}.cpp` and FastScan
implementations in `src/fastscan/`. HNSW-RaBitQ keeps its existing
`src/index/` implementations and compatibility namespaces.

#### Backend selection and startup

Select a cached function pointer through the private shared resolver in
`src/simd/dispatch_resolver.hpp`. Core, FastScan and HNSW-RaBitQ have separate dispatch translation units.
The x86 order is AVX-512, AVX2/FMA, then the portable generic implementation. ARM64
selects NEON for raw float distances, FastScan, packed-code dot products (1–8 extra
bits), masked sums, binary warmup, HNSW-RaBitQ search, and FHT/Kac rotation including
sign flipping, query-bit transposition, uint8/uint16 scalar quantization, and
FastScan LUT construction. Extra-code packing retains a portable scalar implementation.
Cache selection in thread-safe function-local statics, including tables; public calls must
work before `main()` regardless of translation-unit initialization order. Public wrappers
do not repeat feature checks.
Standard FastScan accumulation requires AVX2/FMA, AVX-512, or NEON and throws
`std::runtime_error` when no supported SIMD backend is available. Its scalar kernel
is retained as a correctness reference only, never as a runtime fallback.

#### CPU predicates and compiler flags

Preserve stricter predicates: population-count kernels need AVX512_VPOPCNTDQ, and HNSW-RaBitQ's
AVX-512 core variant also needs AVX2 for its warmup implementation. Do not infer support
from a backend name or from `__AVX*__` macros in a public header.
All AVX-512 backends require AVX2, which GCC/Clang also enable with `-mavx512f`.
MSVC's `/arch:AVX512` additionally requires AVX512VL and AVX512CD; keep these checks
synchronized with the compiler flags. This optional target follows the compiler-group
approach documented in [NumPy's MSVC compatibility rules](https://numpy.org/doc/1.26/reference/simd/build-options.html#on-x86-microsoft-visual-c-c).
Missing group features select the AVX2 fallback; they do not raise the library's minimum
CPU requirements. Native tuning defaults to on for local builds and must be explicitly
disabled for distributed builds; release wheels already set it to off.

#### Paired LUT production and consumption

FastScan selects one immutable private `Backend` in `src/fastscan/fastscan_backend.hpp`
and `fastscan_dispatch.cpp`. LUT production, HACC conversion, accumulation and batch
correction must be chosen as a compatible group. Standard LUTs share consecutive
16-entry tables; HACC has paired 32-byte planes (AVX2/generic/NEON) or grouped 64-byte
planes (AVX-512). These are transient query layouts, not persisted-code formats.
Do not select producer and consumer independently when adding a backend.

#### Dispatch boundaries

Dispatch once at a generic operation boundary. Warm wrappers cache their selected
function pointer; keep feature selection and backend-table access in cold initialization.
A backend may directly compose an accumulator only when its LUT layout is established.
AVX-512 batch correction does so. Explicit AVX2/generic correction entry points also
accept host-prepared queries and intentionally retain host accumulation dispatch;
their function name alone does not specify the LUT format. Estimator templates require
explicit accumulator arguments so new call sites must make this choice deliberately.
Never bypass public dimension checks, unsupported-CPU errors or initialization safety.

#### Hot-path measurements

Keep performance regression workloads in [component hot-path benchmarks](../benchmarks/hotpaths/README.md).
Validate numerical results first, then inspect generated code and alternate comparable
baseline/candidate runs. Keep failed experiments and quality results. Fusion, inlining,
scratch reuse and dispatch changes need measurements across relevant vector sizes;
a short-vector improvement does not justify applying it to wider vectors.

#### Implementation linkage

Put calculations and scratch-storage helpers outside the dispatcher. Shared implementation
headers use internal linkage so independently compiled backends retain their own bodies.
The AVX FHT intrinsics in `src/simd/fht_kernels.hpp` use the private kernel namespace
for the same reason. Rotation uses this implementation on GCC, Clang, and MSVC;
no assembler is required. Keep the FFHT attribution and MIT license in the
intrinsics header intact.

#### Eigen isolation

Pass ordinary pointers, sizes, and library-owned query state across ISA boundaries. Never
pass Eigen matrix/packet objects between backends. The private matrix implementation
header includes Eigen under a namespace selected by each backend translation unit.
Otherwise, Eigen emits identically named out-of-line template helpers, which the linker
can merge across incompatible ISA builds.
Keep this isolation when adding matrix kernels; do not modify the vendor snapshot.

#### Public entry points

Preserve existing public names and namespaces, including legacy functions with `_avx` in
their name that now dispatch at runtime. The explicit `_avx2` and `_avx512` entry points
are for selected kernels and capability-guarded backend tests.

#### Kernel coverage

The current first-party kernel audit is summarized below. Dispatching an index operation
covers its arithmetic kernels, not every scalar loop in construction and search.

**Raw float distances, norms, packed-code products**

Runtime AVX2/AVX-512/NEON selection; portable generic fallback.

**Quantizer rescale search**

SIMD bounded search; the certified scalar event sweep remains the fallback.

**Integer scalar quantization, extra-code packing, transpose, sign masks**

Existing runtime selection; byte layouts and rounding rules are unchanged.

**Standard and high-accuracy FastScan**

Runtime selection, including NEON float LUT construction; HACC LUT byte conversion remains
generic on ARM.

**IVF-RaBitQ and float SymphonyQG batch correction**

Complete estimator runs in the selected backend; non-float template paths remain generic.

**FHT/Kac rotation**

Complete rotation and scaling run in selected ISA translation units; shared AVX intrinsics
preserve the FFHT butterfly order.

**Float matrix rotation and PiPNN construction**

Matrix products, row norms, and lower-triangle pairwise distances use isolated matrix
backends.

**RaBitQKMeans and QGKMeans centroid updates**

Indexed float32 rows accumulate into float64 sums through the matrix backends, preserving
per-cluster point order without additional workspace.

**HNSW-RaBitQ search and IVF-RaBitQ centroid routing**

Cached HNSW search selection; centroid routing uses the common raw-distance dispatcher.

**Quantization orchestration, reconstruction, non-float utilities**

Template/control code remains generic; no blanket native tuning or reduction-order rewrite.

**Graph scheduling, candidate queues, I/O, allocation, random initialization**

Generic control code; IVF-RaBitQ one-bit candidate insertion stays in a small compiled
function to avoid inlining-induced register spills; thread scheduling and seeds remain
caller-owned.

**Example KMeans training**

Explicit RaBitQKMeans flat or QGKMeans graph assignment; FAISS is an optional comparison
dependency only.

#### Portable builds and test coverage

Portable wheels disable `RABITQ_ENABLE_NATIVE_OPTIMIZATION`. ARM64 builds exclude all
x86 source groups and use standard AArch64 NEON intrinsics, without Apple-only APIs.
macOS and Linux ARM64 have native C++ and installed-wheel CI. The reference suites
also exercise portable scalar kernels on x86, including byte-for-byte packing parity
with the selected x86 backend.

A portable build on the current CPU does not establish AVX2 coverage. For affected
backends, report which actually ran and which remain untested; use capability-guarded
backend tests or suitable hardware. Semantic kernel changes require an independent
reference and coverage of all implementations, packing boundaries and supported bits
when shared formulas, layouts or intended semantics change. A backend-local fix that
restores the existing contract needs independent-reference coverage of the corrected
implementation and its affected dimensions/bit widths; still audit the other backends
and consumers sharing that contract. If shared code changes, cover every affected
implementation. Keep portable builds and report unavailable backend coverage.
Keep portable reference tests on ARM; guard only explicit x86 calls. Synchronize ISA
source groups/flags with all three dispatchers and `src/utils/cpu_features.cpp`.

## Performance and release validation

### Performance and compatibility

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

### Downstream compatibility validation

Before a release, check the real downstream integrations on Linux x86-64.
Pin each downstream to a tag or commit and verify it uses the candidate library.
These checks are currently manual and are not enforced by release CI.

| Downstream | Minimum checks |
| --- | --- |
| [txtai](docs/integrations/txtai.md) | Through its `rabitq` adapter: IVF/HNSW build, query and save/load; IVF append/delete. |
| [zvec](docs/integrations/zvec.md) | Build its RaBitQ components with candidate core/FastScan; exercise IVF/HNSW build, query and save/load. |

Use small, fixed inputs to check IDs, score conventions, recall against exact
search, and save/load consistency. Set quality thresholds and numerical tolerances
before running, and compare with the previous supported library version.
For updates, check that added vectors are retrievable and deleted IDs stay absent
after save/load. Follow the operations supported by the pinned downstream adapter.

Record library/downstream revisions, platform, commands, results and untested scope
in the release PR or issue. Investigate failures and clearly identify skipped checks.
See [release preparation](maintaining.md#publishing-a-release).
