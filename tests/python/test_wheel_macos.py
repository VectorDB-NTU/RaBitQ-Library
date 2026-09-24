"""Check the installed, repaired macOS wheel, including its OpenMP runtime."""

import os
import platform
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RABITQ_TEST_WHEEL") != "1" or platform.system() != "Darwin",
    reason="Run with RABITQ_TEST_WHEEL=1 after installing a repaired macOS wheel",
)
def test_native_arm64_wheel_bundles_openmp():
    import rabitqlib._rabitqlib as extension

    assert platform.machine() == "arm64"
    path = Path(extension.__file__).resolve()
    assert "site-packages" in path.parts
    assert (
        subprocess.check_output(["lipo", "-archs", str(path)], text=True).strip()
        == "arm64"
    )
    assert (path.parent / "licenses" / "LLVM-OpenMP.txt").is_file()
    dependencies = subprocess.check_output(["otool", "-L", str(path)], text=True)
    omp = [
        line.strip().split(" ")[0]
        for line in dependencies.splitlines()[1:]
        if "libomp" in line
    ]
    assert len(omp) == 1, dependencies
    assert omp[0].startswith("@loader_path/"), dependencies
    runtime = path.parent / omp[0].removeprefix("@loader_path/")
    assert runtime.is_file()
    assert (
        subprocess.check_output(["lipo", "-archs", str(runtime)], text=True).strip()
        == "arm64"
    )
