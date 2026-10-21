"""Execute the matrix verdict commands with success, skip and failure inputs."""

import os
import re
import subprocess
import textwrap
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "test.yaml"


class MatrixSummaryTest(unittest.TestCase):
    def test_matrix_verdicts_and_dependencies(self):
        workflow = WORKFLOW.read_text()
        for job, dependency in (
            ("clang-result", "clang"),
            ("core-consumer-result", "core-consumer"),
        ):
            block = re.search(
                rf"^  {job}:\n(.*?)(?=^  [\w-]+:|\Z)",
                workflow,
                re.MULTILINE | re.DOTALL,
            ).group(1)
            self.assertIn(f"needs: [changes, {dependency}]", block)
            self.assertIn("if: ${{ always() }}", block)
            self.assertIn(f"MATRIX_RESULT: ${{{{ needs.{dependency}.result }}}}", block)
            self.assertIn("CHANGES_RESULT: ${{ needs.changes.result }}", block)
            self.assertIn("CPP_SELECTED: ${{ needs.changes.outputs.cpp }}", block)
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
                                    CPP_SELECTED=selected,
                                    MATRIX_RESULT=matrix,
                                ),
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                            self.assertEqual(result.returncode == 0, expected)


if __name__ == "__main__":
    unittest.main()
