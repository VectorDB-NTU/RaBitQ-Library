# Maintainer reference

For your first contribution, see [Contributing to RaBitQ](CONTRIBUTING.md).
This reference describes CI routing and release maintenance.

## Checks selected by changed files

CI starts a small changed-file check on each push or pull request, then runs
only the affected job groups:

| Changes | Checks |
| --- | --- |
| Documentation and Markdown only | Documentation workflow for `docs/`; no C++ or Python builds |
| C++ headers or library sources | C++ formatting, clang-tidy, platform tests, sanitizers, consumer build, and Python builds/wheels |
| C++ tests or examples | C++ formatting and platform tests; no clang-tidy |
| Compiled Python bindings | C++ formatting and clang-tidy, plus Python checks and wheel tests |
| Python tests | Python checks and wheel tests |
| Python scripts or examples | Python lint |
| CMake configuration | C++ tests and clang-tidy, plus Python builds/wheels |
| `pyproject.toml` | Python lint, builds, and wheels; no C++ checks |
| Shell scripts | ShellCheck and any check driven by the changed script |

The optional include-cleaner report is available through
`./scripts/check-includes.sh` and no longer runs on every C++ change. Shared
headers still trigger broad regression suites; CI does not infer
individual test dependencies from C++ function changes. Unknown paths run all
checks. Pushes compare the complete pushed range; pull requests compare against
their base. Renames and deletions are included. Relevant jobs also run if
change detection fails, so a detection error cannot silently waive a check.

Release tags and manual wheel builds always run the full wheel build. CI
routing edits run the full relevant matrix on pull requests. After an already
released version reaches `main`, a push changing only the routing files and
Markdown reruns Python lint and selector tests without repeating platform
builds. An untagged version on `main` still forces full C++ and Python CI,
even for a subsequent docs-only commit, before automatic publishing can proceed.

## Publishing a release

Update the stable `X.Y.Z` version in both `pyproject.toml` and `CMakeLists.txt`,
add a short README news entry, and merge into `main`. Once the `Test` and
`Python Wheel` workflows succeed for the same commit, `Release wheels` checks
whether that version is newer than the existing release tags. It then builds
and tests the release wheels, pushes `vX.Y.Z` at that tested commit, publishes
to PyPI, and creates a GitHub Release with generated notes and wheel assets.
A later successful commit can release an untagged version if the version-bump
commit failed CI. Ordinary commits with an already released version do not
publish again.

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
validate automation changes locally with:

```bash
python -m unittest discover -s .github/scripts -p 'test_*.py'
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
