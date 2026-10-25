"""Execute the matrix verdict commands with success, skip and failure inputs."""

import os
import re
import subprocess
import textwrap
import unittest
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


class MatrixSummaryTest(unittest.TestCase):
    def test_matrix_verdicts_and_dependencies(self):
        for filename, job, dependency, output, selection_var in (
            ("test.yaml", "clang-result", "clang", "cpp", "CPP_SELECTED"),
            (
                "test.yaml",
                "core-consumer-result",
                "core-consumer",
                "cpp",
                "CPP_SELECTED",
            ),
            (
                "python.yml",
                "python-result",
                "python-build",
                "python_build",
                "PYTHON_SELECTED",
            ),
            ("docs.yml", "docs-result", "build", "docs", "DOCS_SELECTED"),
        ):
            workflow = (WORKFLOWS / filename).read_text()
            block = re.search(
                rf"^  {job}:\n(.*?)(?=^  [\w-]+:|\Z)",
                workflow,
                re.MULTILINE | re.DOTALL,
            ).group(1)
            self.assertIn(f"needs: [changes, {dependency}]", block)
            self.assertIn("if: ${{ always() }}", block)
            self.assertIn(f"MATRIX_RESULT: ${{{{ needs.{dependency}.result }}}}", block)
            self.assertIn("CHANGES_RESULT: ${{ needs.changes.result }}", block)
            self.assertIn(
                f"{selection_var}: ${{{{ needs.changes.outputs.{output} }}}}", block
            )
            command = textwrap.dedent(block.split("        run: |\n", 1)[1])
            for detection in ("success", "failure", "cancelled", "skipped"):
                for selected in ("true", "false", ""):
                    for matrix in ("success", "failure", "cancelled", "skipped", ""):
                        expected = matrix == "success" or (
                            detection == "success"
                            and selected == "false"
                            and matrix == "skipped"
                        )
                        with self.subTest(
                            job=job,
                            detection=detection,
                            selected=selected,
                            matrix=matrix,
                        ):
                            result = subprocess.run(
                                ["bash", "-e", "-c", command],
                                env=dict(
                                    os.environ,
                                    CHANGES_RESULT=detection,
                                    **{selection_var: selected},
                                    MATRIX_RESULT=matrix,
                                ),
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                            self.assertEqual(result.returncode == 0, expected)


if __name__ == "__main__":
    unittest.main()
