"""Exercise release decisions without creating tags or calling publishing APIs."""

import copy
import json
import unittest
from unittest.mock import patch

from prepare_release import prepare, stable_version

SHA = "a" * 40
REPO = "VectorDB-NTU/RaBitQ-Library"
EVENT = {
    "workflow_run": {
        "event": "push",
        "head_branch": "main",
        "head_repository": {"full_name": REPO},
        "conclusion": "success",
        "head_sha": SHA,
    }
}


class ReleaseTest(unittest.TestCase):
    def run_prepare(
        self,
        *,
        event_name="workflow_run",
        event=None,
        tags="v0.3.2",
        ci="success",
        tag_sha=SHA,
        releases=None,
        ref="main",
        version="0.3.3",
    ):
        def fake_command(*args):
            if args[:2] == ("git", "rev-parse"):
                return SHA
            if args[:2] == ("git", "tag"):
                return tags
            if args[:2] == ("git", "rev-list"):
                return tag_sha
            if "/actions/workflows/" in args[-1]:
                conclusion = (
                    ci.get(args[-1].split("/workflows/")[1].split("/")[0])
                    if isinstance(ci, dict)
                    else ci
                )
                runs = (
                    []
                    if conclusion == "missing"
                    else [
                        {
                            "head_branch": "main",
                            "run_number": 2,
                            "conclusion": conclusion,
                        },
                        {
                            "head_branch": "main",
                            "run_number": 1,
                            "conclusion": "success",
                        },
                    ]
                )
                return json.dumps({"workflow_runs": runs})
            if "/releases?" in args[-1]:
                return json.dumps([releases or []])
            raise AssertionError(args)

        with (
            patch("prepare_release.command", side_effect=fake_command),
            patch(
                "prepare_release.Path.read_text",
                side_effect=[
                    f'[project]\nversion = "{version}"',
                    f"project(RaBitQLib VERSION {version} LANGUAGES CXX)",
                ],
            ),
        ):
            return prepare(event_name, event or EVENT, REPO, ref)

    def test_new_version_after_ci(self):
        result = self.run_prepare()
        self.assertEqual(
            result, {"sha": SHA, "tag": "v0.3.3", "build": "true", "publish": "true"}
        )

    def test_incomplete_or_failed_ci(self):
        for ci in (None, "failure", "cancelled", "missing"):
            with self.subTest(ci=ci):
                result = self.run_prepare(ci=ci)
                self.assertEqual(result["build"], "false")
                self.assertEqual(result["publish"], "false")

    def test_both_workflows_must_pass(self):
        result = self.run_prepare(ci={"test.yaml": "success", "python.yml": None})
        self.assertEqual(result["publish"], "false")

    def test_only_trusted_main_pushes(self):
        for key, value in (
            ("event", "pull_request"),
            ("head_branch", "feature"),
            ("head_repository", {"full_name": "someone/fork"}),
            ("conclusion", "failure"),
            ("head_sha", "b" * 40),
        ):
            with self.subTest(key=key):
                event = copy.deepcopy(EVENT)
                event["workflow_run"][key] = value
                self.assertEqual(self.run_prepare(event=event)["publish"], "false")

    def test_no_republication_or_downgrade(self):
        self.assertEqual(
            self.run_prepare(tags="v0.3.3", tag_sha="b" * 40)["publish"], "false"
        )
        self.assertEqual(self.run_prepare(tags="v0.4.0")["publish"], "false")
        release = {"tag_name": "v0.3.3", "draft": False}
        self.assertEqual(
            self.run_prepare(tags="v0.3.3", releases=[release])["publish"], "false"
        )

    def test_partial_release_can_resume(self):
        self.assertEqual(self.run_prepare(tags="v0.3.3")["publish"], "true")

    def test_manual_tag_release(self):
        self.assertEqual(
            self.run_prepare(event_name="push", ref="v0.3.3")["publish"], "true"
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            self.run_prepare(event_name="push", ref="v0.3.2")

    def test_pr_and_dispatch_build_without_publishing(self):
        for event in ("pull_request", "workflow_dispatch"):
            result = self.run_prepare(event_name=event)
            self.assertEqual(result["build"], "true")
            self.assertEqual(result["publish"], "false")

    def test_version_ordering(self):
        self.assertGreater(stable_version("0.3.10"), stable_version("0.3.9"))
        with self.assertRaisesRegex(ValueError, "stable X.Y.Z"):
            stable_version("0.3.3rc1")

    def test_versions_must_match(self):
        with (
            patch("prepare_release.command", return_value=SHA),
            patch(
                "prepare_release.Path.read_text",
                side_effect=[
                    '[project]\nversion = "0.3.3"',
                    "project(RaBitQLib VERSION 0.3.2 LANGUAGES CXX)",
                ],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "versions must match"):
                prepare("workflow_dispatch", {}, REPO, "main")


if __name__ == "__main__":
    unittest.main()
