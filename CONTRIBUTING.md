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
sudo apt-get install clang-tidy-15 cmake ninja-build
python -m pip install "numpy>=1.23" "pybind11>=2.12"
```

Then configure the same tests and Python bindings analyzed by CI and run the
check with the same compiler used by CMake:

```bash
export pybind11_DIR="$(python -m pybind11 --cmakedir)"
CXX=c++ cmake -S . -B build-tidy -G Ninja \
    -DRABITQ_BUILD_TESTS=ON \
    -DRABITQ_BUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$pybind11_DIR"
CXX=c++ ./scripts/check-tidy.sh build-tidy
```

The wrapper supplies clang-tidy with that compiler's standard-library include
paths and reports diagnostics only for first-party files. Vendored Eigen,
hnswlib, and the imported FFHT implementation are excluded. New checks should
be added incrementally after their existing first-party findings are fixed.

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

## Performance and compatibility

- Use fixed-width integer types for serialized values and persisted index data.
- Preserve existing public headers, aliases, and index formats unless a change
  is explicitly documented as breaking.
- Add backend-independent tests when introducing or changing SIMD kernels.
- Keep scalar, AVX2, and AVX-512 implementations behaviorally equivalent.
- Benchmark allocations or algorithm changes in search and quantization hot
  paths, and include the commands and results in the pull request.
- Avoid unrelated refactoring or formatting in performance-sensitive changes.
