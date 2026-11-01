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


# Each 0.7 call form the compatibility layer accepts is marked deprecated in the
# stubs, so editors flag it while it still type-checks.
DEPRECATED_USAGE = """\
import numpy as np
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

x = np.zeros((80, 64), dtype=np.float32)
labels = np.zeros(80, dtype=np.uint32)
ivf = IvfIndex(dim=64, max_elements=80, num_clusters=1, nbits=4, metric="ip")
hnsw = HnswIndex(dim=64, max_elements=80, M=16, ef_construction=200, nbits=4)
seeded = HnswIndex(64, nbits=4, random_seed=5)
ivf.build(x, x[:1], labels)
hnsw.build(x, x[:1], labels)
ivf.add(x, labels)
flag = ivf.is_built
graph = SymqgIndex(64, 32, "l2", 4)
alias = SymqgIndex(64, quantization_bits=0)
graph.build(x, 40)
graph.search_batch(x, 5, 32)
bits = graph.quantization_bits
built = graph.is_built
"""


def test_deprecated_forms_are_flagged(tmp_path):
    usage = tmp_path / "deprecated_usage.py"
    usage.write_text(DEPRECATED_USAGE)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--enable-error-code",
            "deprecated",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "cache"),
            str(usage),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    flagged = {
        int(number)
        for number in re.findall(
            rf"{re.escape(str(usage))}:(\d+): error: .*\[deprecated\]", result.stdout
        )
    }
    assert flagged == set(range(6, 19)), result.stdout
