# Maintainer reference

For your first contribution, see [Contributing to RaBitQ](../.github/CONTRIBUTING.md).
This reference describes CI routing and release maintenance.

## Checks selected by changed files

CI starts a small changed-file check on each push or pull request, then runs
only the affected job groups:

| Changes | Checks |
| --- | --- |
| Documentation and Markdown only | Documentation build for `docs/`; no C++ or Python builds; the documentation summary is always emitted |
| Issue forms/configuration and CODEOWNERS | No C++/Python builds |
| `benchmarks/core/`, `benchmarks/hotpaths/` code/build files | Ubuntu benchmark build and bounded smoke tests; C++ formatting or Python lint as applicable, without the wheel/platform matrix for benchmark-only edits |
| `.github/workflows/test.yaml` | C++ checks, ShellCheck, Python quality (including summary-job tests), and benchmark smoke |
| Known root benchmark Python scripts | Python lint and dataset-independent benchmark protocol tests; no index or wheel builds |
| `tests/python/test_benchmark_protocol.py` | Python lint and benchmark protocol tests; no index or wheel builds |
| `.github/workflows/docs.yml` | Documentation build and Python quality checks for workflow regression tests |
| `.github/workflows/pr-labels.yml` | Python quality, including metadata-classification regression tests |
| `.github/release.yml` | Python quality, including release-category compatibility tests |
| `sample/run_cpp.sh`, `sample/run_python.sh` | ShellCheck, including these runners |
| `tests/core_consumer/`, `tests/fixtures/core_api_v1/` | C++ checks; the frozen v1 fixture is excluded from formatting |
| `tests/core_consumer/run.cmake`, `scripts/check-format.sh`, `scripts/check-tidy.sh` | Their usual checks, plus Python quality to run their failure-path regression tests |
| C++ headers or library sources | C++ formatting, clang-tidy, platform tests, sanitizers, consumer build, and Python builds/wheels |
| C++ tests or examples | C++ formatting and platform tests; no clang-tidy. Index/clustering-only sources, tests and examples skip the core/FastScan consumer matrix |
| `tests/allocation/`, `tests/failures/` | C++ formatting and platform tests, including sanitizer and allocation-failure coverage; no Python builds or wheels |
| Compiled Python bindings | C++ formatting and clang-tidy, plus Python builds and wheel tests |
| Python tests | Python lint, source-install smoke tests, and wheel tests |
| `tests/python/test_cpp_clustering_examples.py` | Python lint and C++ checks; examples run in the Ubuntu job |
| `tests/python/conftest.py` | Python lint, C++ checks, and Python builds/wheels |
| `sample/python/` | Python lint, source-install smoke tests, comparison tests, and wheels |
| Python automation under `.github/scripts/` | Python lint (CI routing scripts trigger broader checks) |
| CMake configuration | C++ tests and clang-tidy, plus Python builds/wheels |
| `pyproject.toml` | Python lint, builds, and wheels; no C++ checks |
| Shell scripts | ShellCheck and any check driven by the changed script |

See [CI coverage](../tests/README.md#ci-coverage) for full-suite versus smoke-test
ownership, dedicated FAISS comparisons, and emulated SIMD checks.

The optional include-cleaner report is available through
`./scripts/check-includes.sh` and no longer runs on every C++ change. Shared
headers still trigger broad regression suites; CI does not infer
individual test dependencies from C++ function changes. Unknown paths run all
checks. Retired example paths are recognized when deleted: `example.sh` and
`example_py.sh` select ShellCheck; `python/ivf.py` and `python/utils/io.py` select
Python example checks and wheels. Other unrecognized paths keep the full fallback.
Pushes compare the complete pushed range; pull requests compare against
their base. Renames and deletions are included. Relevant jobs also run if
change detection fails, so a detection error cannot silently waive a check.

Every Linux and macOS job that compiles the library with native tuning disabled uses
`ccache` through CMake's compiler launcher, set up by `.github/actions/setup-ccache`.
Each job saves its cache under a per-configuration key and the commit, and later runs
restore the newest matching cache, so unchanged translation units are not recompiled
between runs. Identical compilations within one job also reuse each other's output.
The native-tuned `Ubuntu` job has no compiler cache: ccache keys `-march=native`
literally, so objects built for one runner's CPU could otherwise be restored on a
runner without those instructions. Windows keeps its existing Visual Studio build
path, which does not support compiler launchers.

The C++ change selection has a separate `cpp_core` output for the core/FastScan
consumer matrix. Paths that only affect ANN indexes or clustering, such as
`src/index/`, index/clustering headers and tests, the installed index consumer and
the index examples, select the other C++ checks but not that matrix.

Numerical reference suites have one owner per compiler/platform and OpenMP mode.
The full C++ jobs own OpenMP ON references; component consumer jobs retain their
build/install/header/symbol checks without rebuilding those suites. OpenMP OFF
runs the combined core/FastScan reference suite in the FastScan producer only.
Across the component and Clang consumer matrices, this reduces producer reference
builds from 20 to 5. The `add_subdirectory` and FetchContent source integrations run
for both OpenMP modes on the GCC Ubuntu partition and on the MSVC core OpenMP ON
partition; the other partitions pass `-DRABITQ_INTEGRATIONS=PACKAGE` and check only
the relocated installed package. This keeps 30 of the former 60 consumer integration
configurations, with every compiler/platform and OpenMP mode still consuming the
installed package.
`C++ static analysis` runs the complete configured library/binding compilation
database in one job, replacing the two shards and their summary job.

Windows wheels build one CPython ABI per runner because a single Windows job
was the longest check in a pull request; the other platforms build their ABIs
sequentially in one job, and every platform keeps its existing summary check.

Release tags and manual wheel builds always run the full wheel build. CI
routing edits run the full relevant matrix on pull requests. After an already
released version reaches `main`, a push containing routing edits avoids repeated
platform builds only when all accompanying files select at most Python quality,
ShellCheck, and documentation checks under the normal path rules. Python lint and
selector tests always run in this case, and any accompanying shell checks are retained. For
example, routing plus `scripts/check-python.sh` runs Python quality and ShellCheck;
routing plus library code, build configuration, wheel-related inputs or unknown
paths keeps the full selection. PR routing checks remain broad. An untagged version on `main` still forces full C++ and Python CI,
even for a subsequent docs-only commit, before automatic publishing can proceed.

## Benchmark smoke coverage

`Benchmark build and smoke` uses one Ubuntu runner with benchmarks enabled,
ordinary unit tests and samples disabled, native optimization disabled, and Python
bindings enabled for the ID-conversion helper. It builds the targets and runs
only their registered small CTest cases, with a 60-second per-test timeout.
No dataset download, timing threshold, or performance comparison is run.

The job also runs for shared library sources/headers, root/CMake build settings,
Python bindings/package configuration, and changes to the Test workflow. Other
unrecognized benchmark paths retain conservative full selection. All benchmark
Python scripts are included in Python lint. Known root Python drivers and their
protocol tests select the existing `Python formatting and linting` job, which
also runs `tests/python/test_benchmark_protocol.py` with NumPy and pytest. These
tests need no native package build, external dataset, or baseline installation.
This is Linux build/functionality coverage, not a performance result or proof of
every SIMD backend executing.

## Documentation and concurrency gates

The documentation workflow runs on every PR and main push. Its changed-file
output selects the strict MkDocs build for documentation changes; manual runs
also build. `Documentation validation` always reports a result, accepting a
skipped build only when change detection succeeded and did not select docs.
Selected build failures, cancellations, and unexpected skips fail this summary.
Only successful push builds can deploy the documentation.

After the updated workflow is merged and the selected-build and intentional-skip
paths have been verified on GitHub, add `Documentation validation` to main's
required status checks, preserving existing contexts and their app IDs. Until
that remote update is applied, the documentation failure is not a required merge
gate. Do not require the conditional `Build documentation` job or a workflow
filtered out by paths: missing check names can leave unrelated PRs pending.

Test and Python workflow concurrency groups use the PR number, so unrelated
forks using the same branch name do not cancel each other. Push runs keep their
commit-specific groups, and release serialization is unchanged.

## Matrix summary checks

`Clang validation`, `Core and FastScan validation`, and `Python validation`
provide stable check names for the corresponding matrices. `Python validation`
covers the Linux Python 3.11 source-install job; the release wheel jobs build and fully
test both the oldest and newest ABIs on pull requests. Windows Python and wheel
checks retain their separate gates. They run even when a dependency fails or is
skipped. A successful matrix passes. A skipped matrix passes only when change
detection succeeded and explicitly selected no checks for that matrix
(`cpp`, `cpp_core` or `python_build`); failure, cancellation,
or an unexpected skip fails the summary. If detection fails, the existing
fallback runs the matrix and its success is required.

`Python validation` also covers the FAISS comparison run in the Linux 3.11 job.
When migrating from the older workflow, remove the obsolete `FAISS comparison script`
required context before dropping its duplicate summary job. Keep `Python validation`
and every other required context with their existing app IDs.

After these checks have appeared and been validated on GitHub, add their exact
names to main's required status checks. Workflow definitions do not update branch
protection automatically. Keep existing protections until that transition is
complete; these summaries do not cover unrelated jobs. Once `Python validation`
is verified, replace the individual required `Python 3.11` and `Python 3.14`
contexts with it: a skipped matrix does not emit those expanded names, which
otherwise leaves documentation-only PRs waiting indefinitely.

## Publishing a release

Before merging a version bump, run the
[downstream compatibility checks](development.md#downstream-compatibility-validation)
and record results and any unresolved failures or missing coverage in the release PR
or issue. These manual checks are not enforced by the automatic publishing workflow.

Update the stable `X.Y.Z` version in both `pyproject.toml` and `CMakeLists.txt`,
add a short README news entry, and merge into `main`. Once the `Test` and
`Python Wheel` workflows succeed for the same commit, `Release wheels` checks
whether that version is newer than the existing release tags. It then builds
and tests the release wheels, pushes `vX.Y.Z` at that tested commit, publishes
to PyPI, and creates a GitHub Release with generated notes and wheel assets.
A later successful commit can release an untagged version if the version-bump
commit failed CI. Ordinary commits with an already released version do not
publish again. Keep published tags unchanged; each release gets a new tag.

The workflow uses the repository token to create tags and the existing `pypi`
environment with Trusted Publishing for PyPI. Repository tag rules and any
required environment approvals still apply. Publishing runs in the same
workflow as tag creation; it does not depend on a bot-created tag triggering
another workflow.

Manual `vX.Y.Z` tag pushes still build and publish, with a package-version
check. Pull requests and **Run workflow** only build wheels. To recover a
partial release, rerun the failed release jobs: existing tags must still point
to the tested commit, and already uploaded PyPI files are skipped.

From the repository root, using your activated project Python environment,
validate automation changes locally with the command below. The workflow contract
tests parse workflows with PyYAML, which the Python quality job installs alongside
Ruff; install it in your environment if it is missing.

```bash
python -m unittest discover -s .github/scripts -p 'test_*.py'
```

## Pull request labels and release notes

GitHub generates categorized release notes from merged pull requests using
[the release configuration](../.github/release.yml). The
[classification workflow](../.github/workflows/pr-labels.yml) reads the current
PR title when a PR is opened, edited, reopened, or marked ready for review:

| Title prefix | Automatic label | Release note category |
| --- | --- | --- |
| `feat:` | `release:added` | Added |
| `fix:` | `release:fixed` | Fixed |
| `perf:` | `release:performance` | Performance |
| `docs:` | `release:documentation` | Documentation |
| `ci:`, `build:` | `release:build` | Build and CI |
| `deps:` | `release:dependencies` | Dependencies |
| `chore:`, `refactor:`, `test:`, `style:` | `release:maintenance` | Maintenance |

Use lowercase prefixes and a nonempty summary, for example
`fix(ivf): update only affected clusters`. A scope and a `!` marker before the
colon are optional. The marker does not change the category or authorize a
breaking change; compatibility decisions still follow the project rules.
For dependency-only changes, use `deps:` rather than `chore(deps):`.

The workflow creates its repository labels as needed and updates only the
listed `release:*` labels. Renaming a PR replaces its old automatic category;
other labels are preserved. Unrecognized titles produce a workflow warning and
clear any previous automatic category. They do not block a contribution.
Before merging, check the title and resulting label, especially for mixed changes.

Manual labels remain supported: `enhancement`, `bug`, `performance`,
`documentation`, `build`/`ci`, `python`, and `dependencies`. They map to Added,
Fixed, Performance, Documentation, Build and CI, Python, and Dependencies,
respectively. If labels match several categories, the first matching category
in the release configuration wins; keep manual and automatic labels consistent.
PRs without a matching label appear under `Other changes`. Labels `duplicate`
or `invalid` exclude a PR from release notes.

The classifier uses `pull_request_target` so fork PRs can be labeled. It reads
GitHub metadata only and never checks out or executes PR code. Keep PR titles
as data; never interpolate them into executable workflow expressions.
Local classifier tests execute the workflow's JavaScript with Node.js and mocked
GitHub APIs through the automation unittest command above.

The workflow becomes active after merging into the default branch. It does not
backfill historical PRs or update published releases. Before a version-bump PR
is merged, review the upcoming PR labels and prepare any user-facing highlights:
automatic categories organize the PR list but do not write a release summary.
