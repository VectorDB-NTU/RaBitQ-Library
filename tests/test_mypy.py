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


# Each 0.7 call form the compatibility layer accepts, and each 0.8 spelling that 0.9
# renames, is marked deprecated in the stubs, so editors flag it while it still
# type-checks. Deprecations that depend on a value (a breadth of 0, a seed above
# 2**32 - 1) cannot be declared and are documented in the docstrings only. The two
# old statistics class names are plain aliases, which PEP 702 cannot mark; only the
# runtime warns for them.
DEPRECATED_USAGE = """\
import numpy as np
import rabitqlib
from rabitqlib import HnswIndex, IvfIndex, QGKMeans, SymqgIndex

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
routed = IvfIndex(64, 1, initializer="flat")
strategy = routed.initializer
stored = routed.max_elements
routed.reserve(0, 8)
routed.search(x, 5, 1, None)
capacity = seeded.max_elements
seeded.resize(128)
seeded.search(x, 5, ef=32)
seeded.search(x, 5, 32, 1)
count = graph.max_elements
graph.search(x, 5, ef=32)
graph.search(x, 5, 32, 1)
kmeans = QGKMeans(64, 40, quantization_bits=4)
kmeans = QGKMeans(64, 40, graph_degree=32)
kmeans = QGKMeans(64, 40, ef_build=100)
kmeans.cp.quantization_bits = 4
kmeans.cp.graph_degree = 32
kmeans.cp.ef_build = 100
mode = rabitqlib.FinalAssignmentMode.Exact
storage = kmeans.cp.quantization_bits
degree = kmeans.cp.graph_degree
breadth = kmeans.cp.ef_build
"""

# The forms 0.9 keeps: none may be flagged as deprecated.
CURRENT_USAGE = """\
import numpy as np
from rabitqlib import HnswIndex, IvfIndex, QGKMeans, RaBitQKMeans, SymqgIndex

x = np.zeros((80, 64), dtype=np.float32)
ivf = IvfIndex(64, 1, nbits=4, routing="flat")
strategy = ivf.routing
stored = ivf.num_stored_points
ivf.reserve_cluster(0, 8)
ivf.search(x, 5, 1, high_accuracy=None, num_threads=1)
hnsw = HnswIndex(64, nbits=4, seed=5)
capacity = hnsw.capacity
hnsw.reserve(128)
hnsw.search(x, 5, 32, num_threads=1)
hnsw.search(x, 5, ef_search=32)
graph = SymqgIndex(64)
graph.search(x, 5, ef_search=32, num_threads=1)
kmeans = QGKMeans(
    64, 40, nbits=4, max_degree=32, ef_construction=100, final_assignment="exact"
)
kmeans.cp.nbits = 8
kmeans.cp.max_degree = 32
kmeans.cp.ef_construction = 100
kmeans.cp.final_assignment = "approximate"
width = kmeans.cp.nbits
flat = RaBitQKMeans(64, 2, final_assignment="exact")
"""


def deprecated_lines(tmp_path, source):
    """Line numbers mypy flags as deprecated in source, and its whole output."""
    usage = tmp_path / "usage.py"
    usage.write_text(source)
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
    return flagged, result.stdout


def test_deprecated_forms_are_flagged(tmp_path):
    flagged, output = deprecated_lines(tmp_path, DEPRECATED_USAGE)
    old_forms = set(range(7, 42))
    # mypy 2.4 reports assignments to a deprecated settable property, not reads of
    # it: the last three lines, which read QGKMeansParameters attributes, may go
    # unreported. Each of those attributes is also assigned above.
    settable_reads = {39, 40, 41}
    assert old_forms - settable_reads <= flagged <= old_forms, output


def test_current_forms_are_not_flagged(tmp_path):
    flagged, output = deprecated_lines(tmp_path, CURRENT_USAGE)
    assert not flagged, output
    assert "Success: no issues found" in output, output
