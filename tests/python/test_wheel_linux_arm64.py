"""Check the installed, repaired Linux ARM64 wheel and its OpenMP runtime."""

import os
import platform
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RABITQ_TEST_WHEEL") != "1"
    or platform.system() != "Linux"
    or platform.machine() != "aarch64",
    reason="Run with RABITQ_TEST_WHEEL=1 after installing a repaired Linux ARM64 wheel",
)
def test_native_aarch64_wheel_bundles_openmp():
    import rabitqlib._rabitqlib as extension

    path = Path(extension.__file__).resolve()
    assert "site-packages" in path.parts
    with path.open("rb") as binary:
        header = binary.read(20)
    assert header[:4] == b"\x7fELF"
    assert int.from_bytes(header[18:20], "little") == 183  # EM_AARCH64

    dependencies = subprocess.check_output(["ldd", str(path)], text=True)
    openmp = [line for line in dependencies.splitlines() if "libgomp" in line]
    assert len(openmp) == 1, dependencies
    assert "=>" in openmp[0], dependencies
    runtime = Path(openmp[0].split("=>", 1)[1].split("(", 1)[0].strip())
    assert "site-packages" in runtime.parts, dependencies
    assert runtime.is_file(), dependencies
