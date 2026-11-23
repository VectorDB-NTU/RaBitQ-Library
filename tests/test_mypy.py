"""Static public usage checked against an installed package with mypy."""

import re
import subprocess
import sys
from pathlib import Path


def test_static_public_usage(tmp_path):
    fixture = Path(__file__).parent / "python" / "typing" / "public_api.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "cache"),
            str(fixture),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Removing ignores must expose the deliberately invalid calls, proving the
    # package is typed rather than silently treated as Any.
    invalid = tmp_path / "invalid.py"
    invalid.write_text(re.sub(r"  # type: ignore\[[^]]+\]", "", fixture.read_text()))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "cache"),
            str(invalid),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    expected_lines = {
        number
        for number, line in enumerate(fixture.read_text().splitlines(), start=1)
        if "# type: ignore[" in line
    }
    error_lines = {
        int(number)
        for number in re.findall(
            rf"{re.escape(str(invalid))}:(\d+): error:", result.stdout
        )
    }
    assert error_lines == expected_lines, result.stdout
