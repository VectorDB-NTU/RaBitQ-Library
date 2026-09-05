# Contributing to RaBitQ

Thank you for contributing to RaBitQ. Before submitting a pull request, build
the library, run the relevant tests, and check the C++ formatting.

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

The scripts intentionally exclude vendored Eigen code and the imported FFHT
implementation. To use a nonstandard executable name, set `CLANG_FORMAT`; it
must still identify itself as clang-format 15.

clangd embeds its own formatter, so use clangd 15 in editors such as VS Code
if format-on-save must exactly match CI. If another clangd version is required,
disable format-on-save and run the repository scripts before submitting.

To format only lines changed in the staged files, use:

```bash
./scripts/format-changed.sh --staged
```

The complete-file formatter remains useful before the initial formatting pass
or after changing `.clang-format`; CI always checks complete files.

## Optional pre-commit hook

Install [pre-commit](https://pre-commit.com/) and enable the repository hook:

```bash
python -m pip install pre-commit
pre-commit install
```

The hook formats only staged C and C++ files. CI runs the read-only formatting
check over the complete project-maintained source tree.

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
CXX=c++ cmake -S . -B build-tidy -G Ninja \
    -DRABITQ_BUILD_SAMPLES=OFF \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_BUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$pybind11_DIR"
CXX=c++ ./scripts/check-tidy.sh build-tidy
```

The wrapper supplies clang-tidy with that compiler's standard-library include
paths and reports diagnostics only for first-party files. Vendored Eigen,
hnswlib, and the imported FFHT implementation are excluded. New checks should
be added incrementally after their existing first-party findings are fixed.

Focused clang-tidy checks on affected code are sufficient during iteration.
For first-party C++ changes, run the full check above before merging and report
whether validation was focused or complete.

### Focused static analysis

Use a temporary subset of the compilation database with the same wrapper to
retain its compiler include paths and vendored-header exclusions. After the
configuration above, run this example from the repository root using the
existing project Python environment:

```bash
python - <<'PY'
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
check. Keep `CXX` set to the compiler used for configuration when it is not `c++`.
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

## Pull request labels and release notes

GitHub generates categorized release notes from merged pull requests. Before
merging, maintainers should apply the label that best describes the change:

| Label | Release note category |
| --- | --- |
| `enhancement` | Added |
| `bug` | Fixed |
| `documentation` | Documentation |
| `python` | Python |
| `dependencies` | Dependencies |

Pull requests without one of these labels appear under `Other changes`. Pull
requests labeled `duplicate` or `invalid` are omitted from release notes.

## Performance and compatibility

- Use fixed-width integer types for serialized values and persisted index data.
- Preserve existing public headers, aliases, and index formats unless a change
  is explicitly documented as breaking.
- Add backend-independent tests when introducing or changing SIMD kernels.
- Keep scalar, AVX2, and AVX-512 implementations behaviorally equivalent.
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
