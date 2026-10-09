"""Installed Linux packages retain notices for the release wheel's GNU runtime."""

import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux GNU OpenMP packaging")
def test_linux_package_includes_gnu_openmp_notices():
    import rabitqlib

    licenses = Path(rabitqlib.__file__).parent / "licenses"
    for filename, text in (
        ("GNU-OpenMP.txt", "GNU OpenMP runtime (libgomp)"),
        ("GPL-3.0.txt", "GNU GENERAL PUBLIC LICENSE"),
        ("GCC-Runtime-Library-Exception.txt", "GCC RUNTIME LIBRARY EXCEPTION"),
    ):
        assert text in (licenses / filename).read_text(encoding="utf-8")
