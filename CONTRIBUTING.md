# Contributing to RaBitQ

Documentation improvements, examples, bug reports, and code changes are welcome.
You do not need to understand the quantization internals to make a useful contribution.

## Your first contribution

1. **Choose a small task.** Browse the [starter tasks](#starter-tasks) or
   [existing issues](https://github.com/VectorDB-NTU/RaBitQ-Library/issues).
   Check for an existing issue or PR first. Ask in the issue if the scope is
   unclear; straightforward documentation fixes can go directly to a PR.
2. **Fork and clone the repository**, then create a branch for your change.
   Run the commands in this guide from the repository root.
3. **Follow one setup path below:** [documentation](#documentation-changes),
   [Python](#python-changes), or [C++](#c-changes).
4. **Make a focused change**, run the [matching checks](#before-opening-a-pull-request),
   and open a PR against `main`. Draft PRs are welcome when you need feedback.

## Documentation changes

For repository Markdown such as `README.md` or this guide, review accuracy,
check local links, and run:

```bash
git diff --check
```

No C++ build or formatter is required for prose-only changes.
For the documentation site under `docs/`, use a [Python environment](#python-environment)
and also run:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict --config-file docs/mkdocs.yml
```

Run any examples you add or change, including examples inside Markdown.

## Setup for code changes

Source builds support x86-64 with AVX2/FMA (AVX-512 optional) and macOS ARM64
with NEON. They need CMake 3.20 or newer, a C++17 compiler, and OpenMP.
Windows x86-64 uses Visual Studio 2026; Linux AArch64 needs separate platform
validation. See the [platform-specific build instructions](tests/README.md#quick-start)
for Windows and [Apple Silicon](tests/README.md#macos-arm64).
Documentation-only contributions do not need this hardware or compiler setup.

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake libomp-dev
```

### Python environment

Use Python 3.11 or newer (`python3 --version` to check). On Ubuntu/Debian,
install the matching Python development headers and venv package if needed
(for the distribution Python: `sudo apt-get install python3-dev python3-venv`).
If the distribution Python is older than 3.11, use an installed Python 3.11+
interpreter in place of `python3` below. Activate your existing project environment if you
have one. Otherwise, you can create a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Use that same environment for installs, linting, and tests.

### Python changes

Install the checkout and test dependencies, then run one test to check your setup:

```bash
python -m pip install ".[test]"
python -m pytest tests/python/test_ivf.py::test_search_output_shape -q
```

On Apple Silicon, pass the OpenMP prefix and disable native tuning when building
the extension in the same environment:

```bash
python -m pip install ".[test]" \
    -Ccmake.define.OpenMP_ROOT="$(brew --prefix libomp)" \
    -Ccmake.define.RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
python -m pytest tests/python/test_ivf.py::test_search_output_shape -q
```

This builds the C++ extension, so the source-build prerequisites above apply.
Re-run the install command after changing package sources, bindings, or C++ code;
this is not an editable install. Tests should use the newly installed package.

### C++ changes

On Linux, configure, build, and run the tests:

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON \
    -DRABITQ_BUILD_SAMPLES=OFF -DCMAKE_BUILD_TYPE=Release \
    -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The first configuration needs Git and network access to download GoogleTest.
Increase `--parallel 2` if your machine has enough memory for more compiler jobs.
Native CPU tuning is disabled here to keep generic code portable. On Windows
or Apple Silicon, use the matching commands in [the test guide](tests/README.md#quick-start).

## Before opening a pull request

Keep changes focused and follow the surrounding code style. Add a small,
deterministic regression test when changing behavior. Start with an affected
test during development, then apply every matching row below before submitting.

| Your change | Local checks |
| --- | --- |
| Repository Markdown | Check accuracy and local links; `git diff --check` |
| Documentation site | Above, plus the strict MkDocs build shown above |
| Python sources | [Python lint and formatting](DEVELOPMENT.md#python-formatting-and-linting), plus affected tests with `python -m pytest` |
| First-party C++ | [C++ formatting](DEVELOPMENT.md#c-formatting), affected build/tests, and [static analysis](DEVELOPMENT.md#static-analysis); focused analysis is sufficient during iteration |
| Bindings or Python-visible C++ behavior | Both Python and C++ checks; reinstall before testing and verify package/extension paths as described below |
| Examples | Run the example and its language checks |
| Shell scripts | [ShellCheck](DEVELOPMENT.md#shell-scripts) on affected scripts |

For shared code or changes affecting multiple indexes, run the full relevant
C++ or Python suite. Changes to SIMD, metrics, packing, persistence, hot paths,
or build configuration also need the applicable
[specialized checks](AGENTS.md#verification-by-change-type). Consult the
[implementation recipes](DEVELOPMENT.md#implementation-recipes) before changing
these contracts; preserve public APIs and stored index formats unless a breaking
change has been agreed on.

In your PR, describe:

- The problem and what your change does; link an issue if there is one.
- The commands you ran and their results, including failures or unavailable checks.
- Any compatibility implications or performance measurements relevant to the change.

**CI and merge requirements:** CI selects jobs from the changed files and may
run broader checks than your focused local tests. Before merging a C++ change,
the full [static analysis check](DEVELOPMENT.md#static-analysis) is required.
If a tool or supported CPU is unavailable, explain that in the PR so the
maintainer can help arrange validation. See [CI routing](MAINTAINING.md#checks-selected-by-changed-files)
for details.

## If setup fails

| Problem | What to check |
| --- | --- |
| CMake cannot find OpenMP or a compiler | Check the [platform-specific prerequisites](tests/README.md#prerequisites). On Apple Silicon, install `libomp` and pass its prefix as `OpenMP_ROOT`. Include the compiler version and CMake error when asking for help. |
| Unsupported CPU or architecture | Use x86-64 with AVX2/FMA or macOS ARM64 with NEON. Linux AArch64 still needs platform validation; disabling native tuning alone does not establish support. Report your CPU model in the issue or PR. |
| Tests do not reflect your Python or C++ edits | Re-run the platform-specific install command above in the active environment, then check the import paths below. Re-run the regression test that exercises your change. |
| Formatting or analysis reports the wrong tool version | Use the pinned versions and executable overrides in the [development reference](DEVELOPMENT.md). |

Check which installed package and extension Python imports:

```bash
python -c "import rabitqlib; import rabitqlib._rabitqlib as ext; print(rabitqlib.__file__); print(ext.__file__)"
```

For help, use the appropriate
[issue form](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/new/choose)
and include the command, error output, operating system, and relevant tool versions.

## Starter tasks

These are small contribution ideas, not reserved or assigned issues. Check
the current files and open PRs first; if a task is already complete, choose
another. Each task can be a separate PR.

| Task | Where to start | Done when |
| --- | --- | --- |
| Explain Python search results | [Quick start](docs/docs/quick_start.md), [IVF binding](python_bindings/ivf_bindings.cpp), and [IVF tests](tests/python/test_ivf.py) | A short explanation covers result shapes, how IDs map to input rows, and what L2 distances represent, checked against the implementation. The example runs and the strict docs build passes. |
| Add a self-contained IVF save/load example | [Python examples](sample/python/), [README quick start](README.md#python-quick-start), and [IVF tests](tests/python/test_ivf.py) | A deterministic script builds from synthetic data, saves to a temporary directory, reloads, and checks that search IDs and distances match. It needs no dataset download, cleans up its temporary files, passes Python checks, and is linked from the README. |
| Explain the quick start's cluster assignment | [Quick start](docs/docs/quick_start.md) and [IVF guide](docs/docs/index/ivf.md) | The tutorial explains that its round-robin assignment is for a small runnable example, explains why real IVF workloads use clustering, and links to the existing clustering workflow. The strict docs build passes. |

## Feedback and further reading

[Yutong Gou (@gouyt13)](https://github.com/gouyt13) handles issue triage and
reviews. Review timing depends on availability; keep questions in the relevant
issue or PR so others can learn from the answers.

- [Development reference](DEVELOPMENT.md): formatters, analysis,
  performance requirements, and implementation recipes.
- [Maintainer reference](MAINTAINING.md): CI routing, releases, and PR labels.
- [Roadmap and project contacts](ROADMAP.md).
