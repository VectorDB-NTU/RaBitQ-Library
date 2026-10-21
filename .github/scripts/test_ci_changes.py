"""Regression checks for CI selection and release gating."""

import subprocess
import unittest
from unittest.mock import patch

from ci_changes import GROUPS, classify, detect


class ChangeSelectionTest(unittest.TestCase):
    def test_benchmark_and_workflow_self_coverage(self):
        cases = {
            ".github/workflows/test.yaml": {
                "cpp",
                "cpp_format",
                "cpp_tidy",
                "shell",
                "python_quality",
                "benchmarks",
            },
            "benchmarks/core/main.cpp": {"benchmarks", "cpp_format"},
            "benchmarks/core/CMakeLists.txt": {"benchmarks"},
            "benchmarks/hotpaths/fastscan.cpp": {"benchmarks", "cpp_format"},
            "benchmarks/hotpaths/common.hpp": {"benchmarks", "cpp_format"},
            "benchmarks/hotpaths/ids.py": {"benchmarks", "python_quality"},
            "benchmarks/hotpaths/compare.py": {"benchmarks", "python_quality"},
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(classify([path]), expected)
        for path in (
            "include/rabitqlib/core/api.hpp",
            "src/core/api.cpp",
            "python_bindings/bindings_common.hpp",
            "pyproject.toml",
            "CMakeLists.txt",
            "cmake/rabitqOpenMP.cmake",
        ):
            with self.subTest(dependency=path):
                self.assertIn("benchmarks", classify([path]))
        self.assertEqual(classify(["benchmarks/core/README.md"]), set())
        self.assertEqual(classify(["benchmarks/new_kernel.cpp"]), set(GROUPS))

    def test_docs_only_skip_builds(self):
        self.assertEqual(
            classify(
                [
                    "README.md",
                    ".github/CONTRIBUTING.md",
                    "docs/docs/index/ivf.md",
                    "docs/mkdocs.yml",
                    "docs/requirements.txt",
                    "docs/docs/assets/img/plot.png",
                    ".github/workflows/docs.yml",
                ]
            ),
            set(),
        )

    def test_known_support_paths_do_not_select_unrelated_builds(self):
        cases = {
            ".github/ISSUE_TEMPLATE/config.yml": set(),
            ".github/ISSUE_TEMPLATE/bug_report.yml": set(),
            ".github/CODEOWNERS": set(),
            "benchmarks/plot_homepage.py": {"python_quality"},
            "sample/run_cpp.sh": {"shell"},
            "sample/run_python.sh": {"shell"},
            "example.sh": {"shell"},  # Deleted paths still occur in a push diff.
            "example_py.sh": {"shell"},
            "python/ivf.py": {"python_quality", "python_build", "wheels"},
            "python/utils/io.py": {"python_quality", "python_build", "wheels"},
            "tests/core_consumer/run.cmake": {"cpp"},
            "tests/fixtures/core_api_v1/consumer.cpp": {"cpp"},
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(classify([path]), expected)

    def test_support_paths_do_not_mask_source_or_unknown_changes(self):
        support = [".github/ISSUE_TEMPLATE/config.yml", "benchmarks/plot_homepage.py"]
        self.assertEqual(
            classify(support + ["src/core/api.cpp"]),
            {
                "benchmarks",
                "cpp",
                "cpp_format",
                "cpp_tidy",
                "python_build",
                "python_quality",
                "wheels",
            },
        )
        self.assertEqual(classify(support + ["unknown-build-input"]), set(GROUPS))
        self.assertEqual(classify(["benchmarks/new_kernel.cpp"]), set(GROUPS))

    def test_shared_code_runs_cpp_and_python(self):
        for path in ("include/rabitqlib/index/ivf/ivf.hpp", "src/simd/dispatch.cpp"):
            with self.subTest(path=path):
                self.assertEqual(
                    classify([path]),
                    {
                        "cpp",
                        "cpp_format",
                        "cpp_tidy",
                        "python_build",
                        "wheels",
                        "benchmarks",
                    },
                )

    def test_language_specific_changes(self):
        cases = {
            "tests/unit/rabitqlib/index/ivf_test.cpp": {"cpp", "cpp_format"},
            "tests/dispatch/fastscan_no_simd_test.cpp": {"cpp", "cpp_format"},
            "sample/cpp/ivf_rabitq_querying.cpp": {"cpp", "cpp_format"},
            "python_bindings/ivf_bindings.cpp": {
                "benchmarks",
                "cpp_format",
                "cpp_tidy",
                "python_build",
                "wheels",
            },
            "python_bindings/__init__.py": {
                "python_quality",
                "python_build",
                "wheels",
                "benchmarks",
            },
            "python_bindings/licenses/LLVM-OpenMP.txt": {
                "python_build",
                "wheels",
                "benchmarks",
            },
            "tests/python/test_cpp_clustering_examples.py": {"cpp", "python_quality"},
            "tests/python/conftest.py": {
                "cpp",
                "python_quality",
                "python_build",
                "wheels",
            },
            "tests/python/test_ivf.py": {"python_quality", "python_build", "wheels"},
            "tests/python/fixtures/ivf_legacy_4bit.index": {
                "python_build",
                "wheels",
            },
            "sample/python/ivf_rabitq_indexing.py": {
                "python_quality",
                "python_build",
                "wheels",
            },
            "sample/python/clustering_cli/ivf.py": {
                "python_quality",
                "python_build",
                "wheels",
            },
            "sample/python/clustering_cli/utils/io.py": {
                "python_quality",
                "python_build",
                "wheels",
            },
            ".github/scripts/prepare_release.py": {"python_quality"},
            "scripts/check-python.sh": {"shell", "python_quality"},
            "scripts/build-macos-openmp.sh": {"shell", "wheels"},
            "scripts/check-tidy.sh": {"shell", "cpp_tidy"},
            "scripts/check-format.sh": {"shell", "cpp_format"},
            "scripts/check-includes.sh": {"shell"},
            ".clang-format": {"cpp_format"},
            ".clang-tidy": {"cpp_tidy"},
            ".github/workflows/test.yaml": {
                "cpp",
                "cpp_format",
                "cpp_tidy",
                "python_quality",
                "shell",
                "benchmarks",
            },
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(classify([path]), expected)

    def test_build_configuration_selects_affected_checks(self):
        self.assertEqual(
            classify(["pyproject.toml"]),
            {"python_quality", "python_build", "wheels", "benchmarks"},
        )
        for path in ("CMakeLists.txt", "cmake/RaBitQLibConfig.cmake.in"):
            with self.subTest(path=path):
                self.assertEqual(
                    classify([path]),
                    {"cpp", "cpp_tidy", "python_build", "wheels", "benchmarks"},
                )

    def test_mixed_changes_and_renamed_source(self):
        self.assertIn("cpp", classify(["docs/new.md", "src/deleted.cpp"]))
        self.assertEqual(
            classify(["sample/python/tool.py", "tests/unit/example.cpp"]),
            {"cpp", "cpp_format", "python_quality", "python_build", "wheels"},
        )

    def test_unknown_inputs_and_filter_edits_run_everything(self):
        for path in (
            "new-build.config",
            ".github/scripts/ci_changes.py",
            ".github/workflows/changes.yml",
            ".github/scripts/test_ci_changes.py",
        ):
            self.assertEqual(classify([path]), set(GROUPS))

    def test_pr_uses_complete_base_diff(self):
        with patch(
            "ci_changes.git", side_effect=["merge-base", "README.md\0src/changed.cpp\0"]
        ) as git:
            result = detect(
                "pull_request", {"pull_request": {"base": {"sha": "base"}}}, "42/merge"
            )
        self.assertIn("cpp", result)
        self.assertEqual(git.call_args_list[0].args, ("merge-base", "base", "HEAD"))
        self.assertEqual(
            git.call_args_list[1].args,
            ("diff", "--name-only", "--no-renames", "-z", "merge-base", "HEAD"),
        )

    def test_push_compares_whole_pushed_range(self):
        with patch("ci_changes.git", return_value="tests/python/test_ivf.py\0") as git:
            result = detect("push", {"before": "old-tip"}, "master")
        self.assertIn("python_build", result)
        git.assert_called_once_with(
            "diff", "--name-only", "--no-renames", "-z", "old-tip", "HEAD"
        )

    def test_unreleased_version_forces_full_ci_even_for_docs(self):
        for tags, expected in (("", set(GROUPS)), ("v0.3.3", set())):
            with (
                patch("ci_changes.git", side_effect=["README.md\0", tags]),
                patch(
                    "ci_changes.Path.read_text",
                    return_value='[project]\nversion = "0.3.3"',
                ),
            ):
                self.assertEqual(detect("push", {"before": "base"}, "main"), expected)

    def test_ci_routing_changes_run_full_pr_ci_but_not_repeat_on_tagged_main(self):
        paths = [
            ".github/scripts/ci_changes.py",
            ".github/scripts/test_ci_changes.py",
            ".github/workflows/changes.yml",
            ".github/workflows/test.yaml",
            "docs/maintaining.md",
        ]
        changed = "\0".join(paths) + "\0"
        with patch("ci_changes.git", side_effect=["merge-base", changed]):
            self.assertEqual(
                detect(
                    "pull_request",
                    {"pull_request": {"base": {"sha": "base"}}},
                    "42/merge",
                ),
                set(GROUPS),
            )
        for changed_paths, tags, expected in (
            (paths, "v0.3.3", {"python_quality"}),
            (paths, "", set(GROUPS)),
            (paths + ["src/simd/dispatch.cpp"], "v0.3.3", set(GROUPS)),
        ):
            with self.subTest(changed_paths=changed_paths, tags=tags):
                with (
                    patch(
                        "ci_changes.git",
                        side_effect=["\0".join(changed_paths) + "\0", tags],
                    ),
                    patch(
                        "ci_changes.Path.read_text",
                        return_value='[project]\nversion = "0.3.3"',
                    ),
                ):
                    self.assertEqual(
                        detect("push", {"before": "base"}, "main"), expected
                    )

    def test_tagged_main_routing_changes_preserve_lightweight_companion_checks(self):
        # Complete changed-file set from PR #148, including its lint helper.
        routing = [
            ".github/scripts/ci_changes.py",
            ".github/scripts/test_ci_changes.py",
            ".github/workflows/test.yaml",
            "docs/development.md",
            "docs/maintaining.md",
        ]
        for companions, expected in (
            (["scripts/check-python.sh"], {"python_quality", "shell"}),
            (["sample/run_cpp.sh"], {"python_quality", "shell"}),
            (["benchmarks/plot_homepage.py"], {"python_quality"}),
            ([".github/ISSUE_TEMPLATE/config.yml"], {"python_quality"}),
            (["src/core/api.cpp"], set(GROUPS)),
            (["python_bindings/__init__.py"], set(GROUPS)),
            (["scripts/build-macos-openmp.sh"], set(GROUPS)),
            (["scripts/check-tidy.sh"], set(GROUPS)),
            (["benchmarks/core/main.cpp"], set(GROUPS)),
            (["CMakeLists.txt"], set(GROUPS)),
            (["pyproject.toml"], set(GROUPS)),
            (["unknown-build-input"], set(GROUPS)),
            (["scripts/check-python.sh", "src/core/api.cpp"], set(GROUPS)),
            (["python/ivf.py", "sample/python/clustering_cli/ivf.py"], set(GROUPS)),
        ):
            changed = "\0".join(routing + companions) + "\0"
            for event, ref, tags, result in (
                ("push", "main", "v0.7.0", expected),
                ("push", "main", "", set(GROUPS)),
                ("push", "master", None, set(GROUPS)),
                ("pull_request", "148/merge", None, set(GROUPS)),
            ):
                replies = (
                    ["merge-base", changed] if event == "pull_request" else [changed]
                )
                if tags is not None:
                    replies.append(tags)
                payload = (
                    {"pull_request": {"base": {"sha": "base"}}}
                    if event == "pull_request"
                    else {"before": "base"}
                )
                with self.subTest(
                    companions=companions, event=event, ref=ref, tags=tags
                ):
                    with (
                        patch("ci_changes.git", side_effect=replies),
                        patch(
                            "ci_changes.Path.read_text",
                            return_value='[project]\nversion = "0.7.0"',
                        ),
                    ):
                        self.assertEqual(detect(event, payload, ref), result)

    def test_release_events_and_new_branches_run_everything(self):
        for event, payload, ref in (
            ("workflow_dispatch", {}, "main"),
            ("workflow_run", {}, "main"),
            ("push", {"ref": "refs/tags/v0.3.3"}, "v0.3.3"),
            ("push", {"before": "0" * 40}, "main"),
        ):
            self.assertEqual(detect(event, payload, ref), set(GROUPS))

    def test_diff_failure_is_not_treated_as_docs_only(self):
        with patch(
            "ci_changes.git", side_effect=subprocess.CalledProcessError(1, "git")
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                detect("push", {"before": "missing"}, "main")


if __name__ == "__main__":
    unittest.main()
