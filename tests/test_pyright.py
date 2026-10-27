"""Explicit Pyright checks; run with the test and typing extras plus Node 24."""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/python/typing/public_api.py"


def run_pyright(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--project",
            str(ROOT / "pyproject.toml"),
            "--pythonpath",
            sys.executable,
            "--warnings",
            "--outputjson",
            *(str(path) for path in paths),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pyright_public_api_and_tests():
    result = run_pyright()
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["summary"]["filesAnalyzed"] == 3, result.stdout
    assert report["summary"]["errorCount"] == 0, result.stdout
    assert report["summary"]["warningCount"] == 0, result.stdout


def test_pyright_rejects_invalid_calls(tmp_path: Path):
    source = FIXTURE.read_text()
    expected_lines = {
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if "# type: ignore[" in line
    }
    assert expected_lines
    invalid = tmp_path / "invalid.py"
    invalid.write_text(re.sub(r"  # type: ignore\[[^]]+\]", "", source))
    result = run_pyright(invalid)
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["summary"]["filesAnalyzed"] == 1, result.stdout
    assert report["summary"]["warningCount"] == 0, result.stdout
    actual_lines = set()
    for diagnostic in report["generalDiagnostics"]:
        assert diagnostic["severity"] == "error", diagnostic
        assert Path(diagnostic["file"]).resolve() == invalid.resolve(), diagnostic
        actual_lines.add(diagnostic["range"]["start"]["line"] + 1)
    assert actual_lines == expected_lines, result.stdout
