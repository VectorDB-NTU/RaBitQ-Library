"""Execute the matrix verdict commands with success, skip and failure inputs."""

import os
import subprocess
import unittest

from workflow_data import job, load_workflow, needs, steps


class MatrixSummaryTest(unittest.TestCase):
    def test_matrix_verdicts_and_dependencies(self):
        for filename, job_name, dependency, output, selection_var in (
            ("test.yaml", "clang-result", "clang", "cpp", "CPP_SELECTED"),
            (
                "test.yaml",
                "core-consumer-result",
                "core-consumer",
                "cpp_core",
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
            summary = job(load_workflow(filename), job_name)
            self.assertEqual(needs(summary), ["changes", dependency])
            self.assertEqual(summary["if"], "${{ always() }}")
            (verdict,) = steps(summary)
            self.assertEqual(
                verdict["env"],
                {
                    "CHANGES_RESULT": "${{ needs.changes.result }}",
                    selection_var: f"${{{{ needs.changes.outputs.{output} }}}}",
                    "MATRIX_RESULT": f"${{{{ needs.{dependency}.result }}}}",
                },
            )
            command = verdict["run"]
            for detection in ("success", "failure", "cancelled", "skipped"):
                for selected in ("true", "false", ""):
                    for matrix in ("success", "failure", "cancelled", "skipped", ""):
                        expected = matrix == "success" or (
                            detection == "success"
                            and selected == "false"
                            and matrix == "skipped"
                        )
                        with self.subTest(
                            job=job_name,
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
