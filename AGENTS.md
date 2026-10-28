# AGENTS.md

Repository guidance for coding agents and contributors. Follow more specific directory instructions
where present. Keep this file current when the paths, commands, or contracts below change.

## Project and code map

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

See [README.md](README.md) for usage, [docs/development.md](docs/development.md) for tool setup and
[implementation recipes](docs/development.md#implementation-recipes), and
[tests/README.md](tests/README.md) for build and test commands.

CMake exposes `rabitqlib::core`, `rabitqlib::fastscan` (depending on core), and
`rabitqlib::index` (depending on FastScan). The old `rabitqlib::rabitqlib` target and
forwarding header paths have been removed in the 0.7.0 breaking refactor.
Optional Python bindings expose the build-tree MODULE alias
`rabitqlib::python` and keep `import rabitqlib`. For pure core, disable both
`RABITQ_BUILD_INDEX` and `RABITQ_BUILD_FASTSCAN`. Canonical core headers must not include
FastScan or index headers; canonical FastScan headers must not include index headers.
Install only enabled components and their required header dependencies. Do not restore
umbrella targets/forwarders to bypass migration failures. Shared index scheduling belongs
in `index/detail/parallel_for.hpp`; HNSW must not import IVF to access this utility.
The former `defines.hpp` is removed: include metric, scalar-quantizer, Eigen or index
types explicitly. Core must not expose ANN IDs/candidates or QG batch adapters.
`BatchDataMap`, `QGBatchDataMap`, their encoders and batch packing/permutation belong
to FastScan, including their uses from IVF, clustering and SymphonyQG. Core must not
declare `fastscan::` functions/constants or embed these layouts. Generic unaligned
value/array access lives in `utils/packed.hpp`. `exact_nn` belongs to
`index/detail/exact_nn.hpp`.
See [the migration contract](docs/docs/modular_migration.md).

IVF/HNSW/SymphonyQG public index headers must not import encoding, scoring, rotator
implementation, search scratch, Eigen, or hnswlib headers. Their non-template operations
live in `src/index/{ivf,hnsw,qg}.cpp`;
HNSW's per-ISA search template definitions live in `src/index/hnsw_search_impl.hpp`.
Keep lightweight numerical values in `core/types.hpp` and IVF initializer choices in
`index/ivf/types.hpp`. Preserve the explicit advanced `hnsw_quant.hpp` query-cache header.

Core/FastScan support `RABITQ_USE_OPENMP=OFF`; ANN/Python require ON. The option defaults
to ON and is exported to consumers. Optional numerical OpenMP use must include
`utils/openmp.hpp` and guard both runtime calls and pragmas. Disabled builds use serial
helpers and their installed package must neither find nor link OpenMP. The standalone
core benchmark lives in `benchmarks/core/` and links only core.

The v1 contract is documented in [Stable Core API](docs/docs/rabitq/core_api.md).
Keep `tests/fixtures/core_api_v1/consumer.cpp` unchanged when extending v1; add tests
instead of editing away compatibility failures. Changes to v1 names, signatures,
aggregate fields, enum values, buffer/code semantics or allocation guarantees require
an explicit API-major decision. The stable header must not expose Eigen, record maps,
ANN or ISA-specific types. Keep the existing algorithm and exact fallback authoritative;
workspace APIs must never silently fall back to heap allocation or spawn worker threads.

## Required working boundaries

- Inspect relevant code and tests first. For multi-step work, state brief, verifiable success
  criteria. Make the minimum complete change and preserve existing user edits.
- Proceed with scoped local edits and verification. Ask before adding or upgrading production
  dependencies, breaking public APIs or persisted formats, deleting user data, or publishing
  changes unless the user has already authorized that action. Surface material ambiguity before
  choosing behavior or architecture.
- Do not commit, push, create branches, or rewrite Git history unless explicitly requested.
- Avoid unrelated cleanup and generated-file edits unless the project workflow requires them.
  Never expose credentials or private data in code, logs, or responses.
- Reuse the existing project Python environment for builds, installs, and tests. Do not use system
  or user-global package installation or create/replace an environment without authorization.
  Keep machine-specific interpreter paths and environment names in personal agent configuration.
- Report what changed, the checks actually run, and any failures or unavailable checks. Do not
  claim completion while an in-scope verification failure has a safe, available fix.

## Verification by change type

Run the smallest relevant check first, then broaden for shared contracts and regression risk.
Apply all matching rows; these are conditional requirements, not a checklist for every edit.
Run tests in the foreground. Use the existing project interpreter wherever commands say `python`.
Tool versions and installation instructions are maintained in [docs/development.md](docs/development.md).

| Change | Required verification |
| --- | --- |
| Repository Markdown, including this file | Review accuracy, local links, and `git diff --check`; no code build required |
| MkDocs content, configuration, or dependencies | Above, plus `python -m mkdocs build --strict --config-file docs/mkdocs.yml` |
| First-party C++ | `./scripts/check-format.sh` and affected build/tests; core/binding changes require full `./scripts/check-tidy.sh build-tidy` before merging (focused checks suffice during iteration). Tests/examples are excluded from CI clang-tidy; use focused local analysis for substantial logic changes. |
| Python sources | `./scripts/check-python.sh` and affected Python tests via `python -m pytest` |
| Bindings or Python-visible C++ behavior | C++ and Python checks above; rebuild/install the package in the existing environment before testing, then verify the imported package/extension paths and that the extension includes the current changes |
| SIMD, metrics, packing, layouts, or index formats | Before building, audit every backend and consumer sharing the contract; build and run relevant reference/compatibility tests with `RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF` |
| Hot-path algorithm/allocation changes or performance claims | Correctness tests first, then a representative benchmark reporting dataset, CPU, compiler, ISA, threads, latency/throughput, and quality; a correctness fix may omit benchmarks with a documented justification, without making an unmeasured performance claim |
| Examples | Build or run the affected example |
| Shell scripts | ShellCheck on affected scripts; see contributor instructions |
| Build or packaging configuration | Exercise the affected build/package workflow and its tests |

Use the C++ configure/build/CTest commands in [tests/README.md](tests/README.md#quick-start).
Run the full relevant C++ or Python suite for changes affecting shared code or multiple indexes.
If a required tool, dependency, or supported CPU is unavailable, report the blocker and the checks
still needed; do not silently skip it or replace the user's environment.
When a check fails, determine whether the change caused it. Fix introduced failures; report
unrelated baseline failures without expanding scope. Preserve user edits when comparing baselines.
For focused static analysis, use the recipe in [docs/development.md](docs/development.md#focused-static-analysis).

Behavior changes require focused regression coverage. Python-visible core changes normally need
both an underlying C++ test and a Python boundary test; Python-only changes need Python coverage.
Keep tests small, deterministic, and independent of external datasets. Test precise error types
and messages, packing boundaries, degenerate inputs, and persistence metadata/search round trips
where relevant. SIMD tests must use dispatch or explicit CPU capability guards.

## Code conventions

Required:

- Follow `.clang-format`, `.clang-tidy`, `.editorconfig`, and surrounding naming conventions:
  `PascalCase` types, `snake_case` functions/variables/members, and `kPascalCase` constants.
- Across indexes, use the [index naming glossary](docs/development.md#index-variable-names)
  for shared concepts. Distinguish capacity, stored/live counts, internal IDs/external labels,
  original/padded dimensions, and storage sizes/offsets. Storage offsets default to bytes;
  keep `*_offset_` names without a redundant `_bytes` suffix. Naming-only work must preserve
  public/protected names, Python keywords, types, defaults, and serialized layouts unless
  a compatibility change is explicitly approved.
- Public API misuse must produce descriptive exceptions rather than `exit()`, `abort()`, or
  unconditional diagnostics. Assertions are for internal invariants. Keep library output opt-in.
- Make ownership and non-owning pointer lifetimes explicit. Wrap legacy owning factory pointers
  immediately in RAII. Do not add manual allocation ownership where RAII containers/deleters fit.
- Preserve public signatures and data formats unless a breaking change is authorized. Use
  fixed-width integer types for new serialized fields and persisted identifiers.
- For index API changes, update affected bindings, tests, examples, and index documentation.
  Follow the linked implementation recipes for SIMD, quantization, persistence, and bindings.

Recommended:

- Prefer existing dependencies and standard-library functionality; avoid speculative abstractions.
- Reuse scratch storage in repeated search/build loops. Consider allocation, copying, cache
  locality, alignment, vectorization, and thread scheduling in hot paths.
- Avoid exposing implementation headers or vendored types through new public APIs.

## Required correctness and compatibility safeguards

### SIMD dispatch and backend parity

- Public/generic code calls centralized dispatch entry points. Keep ISA-specific translation units
  and their flags in `CMakeLists.txt` synchronized with feature predicates in
  `src/simd/dispatch.cpp`, `src/fastscan/fastscan_dispatch.cpp`, `src/index/hnsw_dispatch.cpp` and detection in
  `src/utils/cpu_features.cpp`, including HNSW source groups.
  ARM64 builds must exclude all x86 ISA source groups; ARM kernels use portable AArch64
  NEON intrinsics.
- Use the private shared resolver in `src/simd/dispatch_resolver.hpp` for cached selection.
  Core dispatch lives in `src/simd/dispatch.cpp`; batch scoring dispatch lives in
  `src/fastscan/fastscan_dispatch.cpp`; HNSW dispatch lives in
  `src/index/hnsw_dispatch.cpp` so core never references index search symbols.
  Keep calculations in backend source files; see the dispatch coverage table in
  [docs/development.md](docs/development.md#dispatch-conventions-and-coverage).
- Dispatch caches function pointers in thread-safe function-local statics so calls before `main()`
  are safe regardless of translation-unit initialization order. Detection must use safe generic
  code; never execute a high-ISA kernel to find out whether the CPU supports it.
- Semantic kernel changes must cover every implementation and a backend-independent reference
  test, including dimensions around vector-width/packing boundaries and supported bit widths.
  Keep portable reference tests enabled on ARM; guard only explicit x86 backend calls.
- Native tuning can affect generic code independently of dispatch. Keep native optimization off
  for portable binaries and wheels; runtime dispatch alone does not make a native build portable.
  A portable build tested on the current CPU does not establish AVX2 backend coverage. Report
  which backends actually executed and which remain untested; use capability-guarded backend tests
  or suitable hardware to verify each affected backend.
- Performance improvements must preserve estimated-distance semantics. For recall comparisons,
  keep inputs, seeds, and search parameters fixed. State the metric, workload, and acceptable
  recall tolerance before evaluating results, using repeated baseline measurements where relevant.
  Evaluate distance-estimate numerical error separately with justified numerical tolerances;
  deterministic SIMD backends need not produce bitwise-identical floating-point results. Report
  observed differences. Discuss intentional speed–quality tradeoffs with the user before
  implementing them; do not treat measurement tolerance as permission to reduce quality.

### Quantization and byte layouts

- `BatchDataMap`, `ExDataMap`, `BinDataMap`, and their const variants define raw-buffer sizes and
  offsets. Do not duplicate those calculations. Audit all producers and consumers when changing
  offsets, alignment, factor order, or code packing.
- FastScan batches contain 32 vectors, including physical storage for tail batches. Distinguish
  logical point count from batch capacity.
- Retain explicit zero-residual sign-convention tests and factor-finiteness, reconstruction,
  pack/unpack, and estimation coverage. Test both `METRIC_L2` and `METRIC_IP` where supported.
- IVF/HNSW quantized total bits are one sign bit plus `ex_bits`, with totals 1 through 9.
  IVF also accepts `bits == 32`: one-bit filtering plus owned original float32 vectors
  in place of extra-bit codes. New IVF files have a magic/version header, an explicit
  padded dimension, and a resolved initializer type (v2); v1 files infer the historical
  Flat/HNSW choice at 20,000 clusters, independently of current auto selection.
  New IVF auto routing uses Flat below 5,000 clusters, Flat RaBitQ below 60,000,
  and HNSW otherwise. Flat RaBitQ persists its mean and codes in the main file;
  HNSW retains its sidecar. Legacy quantized files and raw v1 files infer
  64-dimension padding.
  IVF search defaults to HACC for 4–9 bits and standard FastScan for 1–3 bits or raw storage.
  SymphonyQG supports
  raw storage (`quantization_bits == 0`) and quantized storage at 4 or 8 bits.

### Persistence and Python boundaries

Filesystem path strings are UTF-8 on Windows and native path bytes on POSIX. Use
`io_impl::filesystem_path` from `utils/path.hpp` before opening files or querying
filesystem metadata, including IVF's HNSW sidecar. It converts UTF-8 to native Unicode
on Windows and preserves existing path bytes on Linux/macOS.

Never silently reinterpret an old index file. Format changes require a magic/version discriminator,
validated sizes before allocation, checked reads, and a compatibility fixture or explicit rejection
path. Review each affected index format; preserve SymphonyQG's versioned quantized format and legacy
raw fallback unless a breaking change is authorized.

Frozen release-produced indexes and reference searches live in `tests/fixtures/persistence/`;
keep their provenance and hashes, and do not regenerate them with the current writer.

Use `python_bindings/bindings_common.hpp` for shared conversions. Validate array rank, dimensions,
index state, and parameter ranges before entering the core. `py::array::forcecast` permits copies;
do not use it where callers expect in-place mutation or pointer identity.

Python index wrappers use `IndexAccess` for concurrent reads and fail-fast exclusive updates.
Acquire and destroy its guards with the GIL held; release the GIL only inside the guard's
lifetime, after extracting native pointers, and retain Python array owners until it is reacquired.
Keep search parameters local to each call. C++ callers remain responsible for synchronizing
index mutations.

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

### Vendored code

`include/rabitqlib/third/` contains imported code. Do not reformat, refactor, or include it
in project-wide lint fixes unless the task explicitly targets the vendor snapshot. Keep
its attribution and license text intact.
