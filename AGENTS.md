# AGENTS.md

Repository instructions for agents and contributors. Follow more specific directory instructions.
Keep paths, commands and contracts current when they change.

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
- Do not claim completion while an in-scope verification failure has a safe, available fix.

Keep updates and the final response proportional to the task: report the material change,
checks actually run, and relevant failures or unavailable checks. Do not recite these
instructions, list unrelated skipped checks, or create plans/logs for routine edits.

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
| Python API signatures, type declarations or accepted-input contracts | Keep `python_bindings/__init__.pyi` and runtime docstrings synchronized; run the [editor/type checks](tests/README.md#python-editor-support), including the explicitly selected mypy and Pyright tests outside `tests/python` |
| SIMD, metrics, packing, layouts, or index formats | Before building, audit every backend and consumer sharing the contract; build and run relevant reference/compatibility tests with `RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF`; run `./scripts/check-isa-symbols.sh` on that build when ISA-flagged sources or headers they include change |
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
Use `ctest --no-tests=error`, including filtered and installed-consumer runs.

## Project and code map

C++17 quantization/ANN library with Python bindings (`rabitqlib`). Components depend in one
direction: `index` → `fastscan` → `core`; core and FastScan support OpenMP OFF, ANN/Python require ON.
Stable float32 v1 API: `include/rabitqlib/core/api.hpp`. Implementations: `src/{core,simd,utils,fastscan,index}/`.
Bindings: `python_bindings/`; tests: `tests/{unit,python}/`; site sources: `docs/docs/`.
See the [full code map](docs/development.md#project-and-code-map) when locating unfamiliar code.

## Code conventions

- Follow `.clang-format`, `.clang-tidy`, `.editorconfig` and surrounding style:
  `PascalCase` types, `snake_case` functions/variables/members, `kPascalCase` constants.
- Preserve public/protected names, Python keywords, signatures, defaults and persisted formats
  unless a compatibility change is authorized. Use fixed-width types for new serialized fields.
- Public misuse raises descriptive exceptions; assertions are for internal invariants.
  Do not use `exit()`, `abort()` or unconditional diagnostics. Library output stays opt-in.
- Make ownership/lifetimes explicit. Wrap owning factory pointers immediately in RAII;
  prefer existing dependencies, standard containers and helpers over speculative abstractions.
- Reuse hot-path scratch; consider allocation, copying, cache locality, alignment, vectorization
  and scheduling. Avoid exposing implementation or vendor headers through new public APIs.
- `include/rabitqlib/third/` is vendored: do not include it in cleanup or lint fixes unless
  explicitly requested; retain attribution/licenses. Frozen compatibility fixtures are not
  regenerated or edited to make current code pass.

## Required reading by task

Before editing an affected area, read its linked sections in `docs/development.md` and follow
those contracts. Read all matching rows; unrelated sections need not be loaded. The links
carry the detailed requirements, not optional background.

| Affected area | Required sections |
| --- | --- |
| Targets, includes, installation, OpenMP | [Module boundaries](docs/development.md#module-build-boundaries) |
| Stable core API or its implementation | [v1 compatibility and allocation guarantees](docs/development.md#stable-core-api-changes) |
| Index naming | [Index names](docs/development.md#index-variable-names) |
| Core/FastScan/clustering/SIMD naming | [Numerical names](docs/development.md#numerical-module-variable-names) |
| Numerical helper refactoring | [Shared helpers](docs/development.md#reuse-numerical-helpers-without-changing-contracts) |
| SIMD, dispatch, arithmetic kernels | [Kernel recipe](docs/development.md#add-or-change-a-simd-kernel), [dispatch contracts](docs/development.md#dispatch-conventions-and-coverage) |
| Quantization, metrics, packing, record layouts | [Quantization and packing](docs/development.md#change-quantization-or-packing), [dimensions and distance conventions](docs/development.md#rotation-and-padded-dimensions) |
| Estimation, index scanning/search or removal | [Removal markers and search](docs/development.md#removal-markers-and-search), [ID marker](docs/development.md#search-buffer-id-marker) |
| Index API, construction, ownership, buffers | [Index API](docs/development.md#change-an-index-api), [validation](docs/development.md#validation-ownership-and-numerical-preparation) |
| Owned state, copy assignment, reset or allocation (any module) | [Exception safety](docs/development.md#exception-safe-state-updates), [validation and ownership](docs/development.md#validation-ownership-and-numerical-preparation) |
| Persistence formats/readers/writers | [Persistence](docs/development.md#change-persistence), [stored rotation/dimensions](docs/development.md#rotation-and-padded-dimensions) |
| Filesystem paths | [Path conversion](docs/development.md#filesystem-paths); for Python also [binding conversions](docs/development.md#change-python-bindings) |
| Python bindings/API | [Bindings, conversions, GIL and typing](docs/development.md#change-python-bindings) |
| Hot-path changes or performance claims | [Performance and compatibility](docs/development.md#performance-and-compatibility) |
| Release preparation | [Downstream validation](docs/development.md#downstream-compatibility-validation) |
