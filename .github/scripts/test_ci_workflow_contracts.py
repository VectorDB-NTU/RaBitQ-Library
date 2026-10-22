"""Check lightweight CI coverage and workflow scheduling contracts."""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ci_changes import GROUPS, classify

ROOT = Path(__file__).resolve().parents[2]


class WorkflowContractsTest(unittest.TestCase):
    def test_benchmark_support_changes_keep_protocol_coverage_without_builds(self):
        for path in sorted((ROOT / "benchmarks").glob("*.py")):
            with self.subTest(path=path.name):
                self.assertEqual(
                    classify([str(path.relative_to(ROOT))]), {"python_quality"}
                )
        self.assertEqual(
            classify(["tests/python/test_benchmark_protocol.py"]), {"python_quality"}
        )
        self.assertEqual(classify(["benchmarks/unknown.py"]), set(GROUPS))
        self.assertIn(
            "cpp",
            classify(["benchmarks/build_indexes.py", "src/core/api.cpp"]),
        )
        workflow = (ROOT / ".github/workflows/python.yml").read_text()
        quality = workflow.split("  python-quality:\n", 1)[1].split(
            "  python-build:\n", 1
        )[0]
        self.assertIn("tests/python/test_benchmark_protocol.py", quality)
        self.assertIn("numpy", quality)
        self.assertIn("pytest", quality)
        self.assertNotIn("pip install .", quality)

    def test_python_checker_covers_every_benchmark_python_file(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "ruff"
            log = Path(directory) / "calls.jsonl"
            helper.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print('ruff 0.16.1')\n"
                "else:\n"
                "    with open(os.environ['RUFF_AUDIT_LOG'], 'a') as stream:\n"
                "        stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            helper.chmod(0o755)
            subprocess.run(
                ["bash", str(ROOT / "scripts/check-python.sh")],
                env=dict(os.environ, RUFF=str(helper), RUFF_AUDIT_LOG=str(log)),
                check=True,
                capture_output=True,
                text=True,
            )
            calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        for call in calls:
            covered = set()
            for argument in call:
                path = ROOT / argument
                if path.is_dir():
                    covered.update(path.rglob("*.py"))
                elif path.suffix == ".py":
                    covered.add(path)
            self.assertLessEqual(set((ROOT / "benchmarks").rglob("*.py")), covered)

    def test_concurrency_groups_distinguish_forks_and_reuse_one_pr(self):
        def group(template, number, sha, workflow):
            context = {
                "github.event.pull_request.number": number,
                "github.head_ref": "main",
                "github.sha": sha,
                "github.workflow": workflow,
            }

            def expand(match):
                return str(
                    next(
                        context[key.strip()]
                        for key in match.group(1).split("||")
                        if context[key.strip()]
                    )
                )

            return re.sub(r"\$\{\{(.*?)\}\}", expand, template)

        for filename in ("test.yaml", "python.yml"):
            text = (ROOT / ".github/workflows" / filename).read_text()
            template = re.search(r"^  group: (.+)$", text, re.MULTILINE).group(1)
            with self.subTest(workflow=filename):
                self.assertNotEqual(
                    group(template, 201, "a", filename),
                    group(template, 202, "b", filename),
                )
                self.assertEqual(
                    group(template, 201, "a", filename),
                    group(template, 201, "b", filename),
                )
                self.assertNotEqual(
                    group(template, 0, "a", filename),
                    group(template, 0, "b", filename),
                )

    def test_documentation_summary_is_available_for_non_doc_prs(self):
        text = (ROOT / ".github/workflows/docs.yml").read_text()
        triggers = text.split("concurrency:", 1)[0]
        self.assertNotIn("paths:", triggers)
        self.assertNotIn("paths-ignore:", triggers)
        self.assertIn("  pull_request:", triggers)
        self.assertIn("  workflow_dispatch:", triggers)
        self.assertIn("uses: ./.github/workflows/changes.yml", text)
        build = text.split("  build:\n", 1)[1].split("  docs-result:\n", 1)[0]
        self.assertIn("needs: changes", build)
        self.assertIn("needs.changes.outputs.docs == 'true'", build)
        self.assertIn("needs.changes.result != 'success'", build)
        self.assertEqual(classify(["docs/docs/quick_start.md"]), {"docs"})


if __name__ == "__main__":
    unittest.main()
