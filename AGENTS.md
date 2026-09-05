# AGENTS.md

Repository guidance for coding agents and contributors. Follow more specific directory instructions
where present. Keep this file current when the paths, commands, or contracts below change.

## Project and code map

RaBitQ-Library is a C++17 library with Python bindings (`rabitqlib`) for compact vector
quantization and approximate nearest-neighbor search. It provides one-bit and multi-bit RaBitQ
encoding and IVF, HNSW, and SymphonyQG indexes. It targets x86-64 with runtime AVX2/AVX-512 dispatch.
RaBitQ rotates vectors and quantizes residuals relative to centroids; compact codes and correction
factors estimate L2 distance or inner product.

| Path | Responsibility |
| --- | --- |
| `include/rabitqlib/quantization/` | Encoding, packing, reconstruction, and byte layouts |
| `include/rabitqlib/fastscan/` | FastScan interfaces and high-accuracy scanning |
| `include/rabitqlib/index/{ivf,hnsw,symqg}/` | Index construction, persistence, and search |
| `include/rabitqlib/index/{query,estimator}.hpp` | Query state and distance estimation |
| `include/rabitqlib/simd/`, `src/simd/` | Kernel declarations, implementations, and dispatch |
| `src/index/` | Compiled HNSW search kernels |
| `src/utils/cpu_features.cpp` | Generic x86 feature detection |
| `include/rabitqlib/utils/` | Rotation, allocation, buffers, I/O, and helpers |
| `python_bindings/` | pybind11 extension and index wrappers |
| `sample/cpp/`, `sample/python/` | Usage examples |
| `tests/unit/`, `tests/integration/`, `tests/python/` | C++ and Python tests |
| `docs/docs/` | MkDocs site sources |

See [README.md](README.md) for usage, [CONTRIBUTING.md](CONTRIBUTING.md) for tool setup and
[implementation recipes](CONTRIBUTING.md#implementation-recipes), and
[tests/README.md](tests/README.md) for build and test commands.

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
Tool versions and installation instructions are maintained in [CONTRIBUTING.md](CONTRIBUTING.md).

| Change | Required verification |
| --- | --- |
| Repository Markdown, including this file | Review accuracy, local links, and `git diff --check`; no code build required |
| MkDocs content, configuration, or dependencies | Above, plus `python -m mkdocs build --strict --config-file docs/mkdocs.yml` |
| First-party C++ | `./scripts/check-format.sh` and affected build/tests; focused clang-tidy checks are sufficient during iteration, with the full `./scripts/check-tidy.sh build-tidy` required before merging |
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
For focused static analysis, use the recipe in [CONTRIBUTING.md](CONTRIBUTING.md#focused-static-analysis).

Behavior changes require focused regression coverage. Python-visible core changes normally need
both an underlying C++ test and a Python boundary test; Python-only changes need Python coverage.
Keep tests small, deterministic, and independent of external datasets. Test precise error types
and messages, packing boundaries, degenerate inputs, and persistence metadata/search round trips
where relevant. SIMD tests must use dispatch or explicit CPU capability guards.

## Code conventions

Required:

- Follow `.clang-format`, `.clang-tidy`, `.editorconfig`, and surrounding naming conventions:
  `PascalCase` types, `snake_case` functions/variables/members, and `kPascalCase` constants.
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
  `src/simd/dispatch.cpp` and detection in `src/utils/cpu_features.cpp`, including HNSW source groups.
- Dispatch resolves function pointers during static initialization. Detection must use safe generic
  code; never execute a high-ISA kernel to find out whether the CPU supports it.
- Semantic kernel changes must cover every implementation and a backend-independent reference
  test, including dimensions around vector-width/packing boundaries and supported bit widths.
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
- IVF/HNSW total bits are one sign bit plus `ex_bits`, with totals 1 through 9. SymphonyQG supports
  raw storage (`quantization_bits == 0`) and quantized storage at 4 or 8 bits.

### Persistence and Python boundaries

Never silently reinterpret an old index file. Format changes require a magic/version discriminator,
validated sizes before allocation, checked reads, and a compatibility fixture or explicit rejection
path. Review each affected index format; preserve SymphonyQG's versioned quantized format and legacy
raw fallback unless a breaking change is authorized.

Use `python_bindings/bindings_common.hpp` for shared conversions. Validate array rank, dimensions,
index state, and parameter ranges before entering the core. `py::array::forcecast` permits copies;
do not use it where callers expect in-place mutation or pointer identity.

### Rotation and padded dimensions

Indexes quantize and search in the rotated, padded domain. FHT/Kac rotation pads to a multiple of
64; callers provide vectors in the original `dim`, while internal code and quantization buffers
generally use `padded_dim`. A buffer allocated for `dim` must never receive `padded_dim` values.
Centroids, data, and queries compared by the same estimator must be in the same domain.

Rotator state is part of persisted index state. Loading an index must restore the exact rotation
used at construction; generating a new random rotator produces silently incorrect distances.

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

`include/rabitqlib/third/` and `include/rabitqlib/utils/fht_avx.hpp` are imported code. Do not
reformat, refactor, or include them in project-wide lint fixes unless the task explicitly targets
the vendor snapshot. Keep attribution and license text intact.
