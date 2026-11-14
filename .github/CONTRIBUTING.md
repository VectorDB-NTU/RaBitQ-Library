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

Source builds support Linux and Windows x86-64 with AVX2/FMA (AVX-512
optional), and Linux and macOS ARM64 with NEON. They need CMake 3.20 or newer,
a C++17 compiler, and OpenMP for ANN/Python. Core/FastScan can disable OpenMP
with `RABITQ_USE_OPENMP=OFF`; pure core also has scalar kernels for CPUs without
the FastScan SIMD baseline. See [component builds](../docs/docs/cpp_integration.md#core-only-c-integration).
Windows x86-64 uses Visual Studio 2026. See the
[platform-specific build instructions](../tests/README.md#quick-start) for
[Linux ARM64](../tests/README.md#linux-arm64) and
[Apple Silicon](../tests/README.md#macos-arm64).
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
python -m pytest tests/python/test_ivf.py::test_search_output_contract -q
```

On Apple Silicon, pass the OpenMP prefix and disable native tuning when building
the extension in the same environment:

```bash
python -m pip install ".[test]" \
    -Ccmake.define.OpenMP_ROOT="$(brew --prefix libomp)" \
    -Ccmake.define.RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
python -m pytest tests/python/test_ivf.py::test_search_output_contract -q
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
ctest --no-tests=error --test-dir build --output-on-failure
```

The first configuration needs network access to download GoogleTest.
Increase `--parallel 2` if your machine has enough memory for more compiler jobs.
Native CPU tuning is disabled here to keep generic code portable. On Windows
or Apple Silicon, use the matching commands in [the test guide](../tests/README.md#quick-start).

## Before opening a pull request

Keep changes focused and follow the surrounding code style. Add a small,
deterministic regression test when changing behavior. Start with an affected
test during development, then apply every matching row below before submitting.

| Your change | Local checks |
| --- | --- |
| Repository Markdown | Check accuracy and local links; `git diff --check` |
| Documentation site | Above, plus the strict MkDocs build shown above |
| Python sources | [Python lint and formatting](../docs/development.md#python-formatting-and-linting), plus affected tests with `python -m pytest` |
| First-party C++ | [C++ formatting](../docs/development.md#c-formatting), affected build/tests, and [static analysis](../docs/development.md#static-analysis); focused analysis is sufficient during iteration |
| Bindings or Python-visible C++ behavior | Both Python and C++ checks; reinstall before testing and verify package/extension paths as described below |
| Examples | Run the example and its language checks |
| Shell scripts | [ShellCheck](../docs/development.md#shell-scripts) on affected scripts |

For shared code or changes affecting multiple indexes, run the full relevant
C++ or Python suite. Changes to SIMD, metrics, packing, persistence, hot paths,
or build configuration also need the applicable
[specialized checks](../AGENTS.md#verification-by-change-type). Consult the
[implementation recipes](../docs/development.md#implementation-recipes) before changing
these contracts; preserve public APIs and stored index formats unless a breaking
change has been agreed on.

In your PR, describe:

- The problem and what your change does; link an issue if there is one.
- The commands you ran and their results, including failures or unavailable checks.
- Any compatibility implications or performance measurements relevant to the change.

Use a title such as `fix(ivf): correct append metadata` or
`docs: clarify installation`. The prefix automatically selects the release-note
category; see the [title and label mapping](../docs/maintaining.md#pull-request-labels-and-release-notes).

**CI and merge requirements:** CI selects jobs from the changed files and may
run broader checks than your focused local tests. Core library and binding changes
require the full [static analysis check](../docs/development.md#static-analysis) before merging.
Tests and examples are intentionally excluded from CI clang-tidy; use focused
local analysis for substantial logic changes.
For a first contribution, the formatting check and the affected tests are
enough to open a PR. Code PRs must be built and tested locally: CI does not
replace building the code you change and running its affected tests. If another
check needs a tool, compiler or CPU you do not have (for example clang-tidy 23 or
a particular SIMD instruction set), say so in the PR and a maintainer will run it.
See [CI routing](../docs/maintaining.md#checks-selected-by-changed-files)
for details.

You are responsible for every change you submit, including changes produced
with automated tools: verify it yourself, report the results, and respond to
review. Code PRs that were not built and tested locally, or that are updated
without addressing review feedback, are closed.

## If setup fails

| Problem | What to check |
| --- | --- |
| CMake cannot find OpenMP or a compiler | Check the [platform-specific prerequisites](../tests/README.md#prerequisites). On Apple Silicon, install `libomp` and pass its prefix as `OpenMP_ROOT`. Include the compiler version and CMake error when asking for help. |
| Unsupported CPU or architecture | Use Linux or Windows x86-64 with AVX2/FMA, or Linux or macOS ARM64 with NEON. Disabling native tuning does not add an unsupported architecture. Report your CPU model in the issue or PR. |
| Tests do not reflect your Python or C++ edits | Re-run the platform-specific install command above in the active environment, then check the import paths below. Re-run the regression test that exercises your change. |
| Formatting or analysis reports the wrong tool version | Use the pinned versions and executable overrides in the [development reference](../docs/development.md). |

Check which installed package and extension Python imports:

```bash
python -c "import rabitqlib; import rabitqlib._rabitqlib as ext; print(rabitqlib.__file__); print(ext.__file__)"
```

For help, use the appropriate
[issue form](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/new/choose)
and include the command, error output, operating system, and relevant tool versions.

## Starter tasks

Open issues labeled
[good first issue](https://github.com/VectorDB-NTU/RaBitQ-Library/labels/good%20first%20issue)
are small, self-contained tasks that say where to start and when they are done;
[help wanted](https://github.com/VectorDB-NTU/RaBitQ-Library/labels/help%20wanted)
issues are larger. The [roadmap issue](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/220)
lists the goals for the next releases. Comment on an issue before starting so work
is not duplicated. When claiming a code issue, say how you will verify the change
locally, such as the tests or benchmark you will run. A maintainer may unassign an
issue when review feedback goes unanswered.

## Feedback and further reading

[Yutong Gou (@gouyt13)](https://github.com/gouyt13) leads development and
maintains the library, including issue triage, reviews and releases. Review
timing depends on availability; keep questions in the relevant
issue or PR so others can learn from the answers.

- [Development reference](../docs/development.md): formatters, analysis,
  performance requirements, and implementation recipes.
- [Maintainer reference](../docs/maintaining.md): CI routing, releases, and PR labels.

Use [GitHub Issues](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/new/choose)
for bug reports, feature requests, and usage or contribution questions. Check
existing issues first. See [your first contribution](#your-first-contribution)
to get started and [GitHub Releases](https://github.com/VectorDB-NTU/RaBitQ-Library/releases)
for published changes.
