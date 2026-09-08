"""Regression checks for CI selection and release gating."""

import subprocess
import unittest
from unittest.mock import patch

from ci_changes import GROUPS, classify, detect


class ChangeSelectionTest(unittest.TestCase):
    def test_docs_only_skip_builds(self):
        self.assertEqual(
            classify(
                [
                    "README.md",
                    "docs/docs/index/ivf.md",
                    "docs/mkdocs.yml",
                    "docs/requirements.txt",
                    "docs/docs/assets/img/plot.png",
                    ".github/workflows/docs.yml",
                ]
            ),
            set(),
        )

    def test_shared_code_runs_cpp_and_python(self):
        for path in ("include/rabitqlib/index/ivf/ivf.hpp", "src/simd/dispatch.cpp"):
            with self.subTest(path=path):
                self.assertEqual(classify([path]), {"cpp", "python_build", "wheels"})

    def test_language_specific_changes(self):
        cases = {
            "tests/unit/rabitqlib/index/ivf_test.cpp": {"cpp"},
            "sample/cpp/ivf_rabitq_querying.cpp": {"cpp"},
            "python_bindings/ivf_bindings.cpp": {
                "cpp",
                "python_quality",
                "python_build",
                "wheels",
            },
            "python_bindings/__init__.py": {"python_quality", "python_build", "wheels"},
            "tests/python/test_ivf.py": {"python_quality", "python_build", "wheels"},
            "tests/python/fixtures/ivf_legacy_4bit.index": {
                "python_quality",
                "python_build",
                "wheels",
            },
            "sample/python/ivf_rabitq_indexing.py": {"python_quality"},
            ".github/scripts/prepare_release.py": {"python_quality"},
            "scripts/check-python.sh": {"shell", "python_quality"},
            "scripts/check-tidy.sh": {"shell", "cpp"},
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(classify([path]), expected)

    def test_build_configuration_checks_both_languages(self):
        for path in (
            "pyproject.toml",
            "CMakeLists.txt",
            "cmake/RaBitQLibConfig.cmake.in",
        ):
            self.assertEqual(
                classify([path]), {"cpp", "python_quality", "python_build", "wheels"}
            )

    def test_mixed_changes_and_renamed_source(self):
        self.assertIn("cpp", classify(["docs/new.md", "src/deleted.cpp"]))
        self.assertEqual(
            classify(["sample/python/tool.py", "tests/unit/example.cpp"]),
            {"cpp", "python_quality"},
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
