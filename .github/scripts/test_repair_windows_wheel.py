"""Exercise runtime selection and fail-closed wheel repair without Windows tools."""

import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import repair_windows_wheel as repair


def metadata(toolset=(14, 51), version=None):
    return json.dumps([toolset, version]).encode()


def read_metadata(data):
    toolset, version = json.loads(data)
    return tuple(toolset), tuple(version) if version is not None else None


class WindowsWheelRepairTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.installation = self.root / "Visual Studio"
        mock = patch.object(repair, "pe_details", side_effect=read_metadata)
        mock.start()
        self.addCleanup(mock.stop)

    def runtimes(self, folder, version, arch="x64", openmp=True):
        root = self.installation / "VC/Redist/MSVC" / folder / arch
        crt = root / "Microsoft.VC143.CRT"
        crt.mkdir(parents=True)
        (crt / "msvcp140.dll").write_bytes(metadata(version=version))
        omp = root / "Microsoft.VC143.OpenMP"
        if openmp:
            omp.mkdir()
            (omp / "vcomp140.dll").write_bytes(metadata(version=version))
        return crt, omp

    def wheel(self, path, dlls):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("rabitqlib/_rabitqlib.cp311-win_amd64.pyd", metadata())
            for name, version in dlls.items():
                archive.writestr("rabitqlib.libs/" + name, metadata(version=version))

    def test_selects_matching_x64_runtime_not_older_or_wrong_architecture(self):
        self.runtimes("14.40", (14, 40, 33810, 0))
        expected = self.runtimes("14.51", (14, 51, 36231, 0))
        self.runtimes("14.99", (14, 99, 0, 0), arch="x86")
        self.assertEqual(
            repair.find_runtime_dirs(self.installation, (14, 51)), expected
        )

    def test_checks_actual_file_version_instead_of_directory_name(self):
        self.runtimes("14.99", (14, 40, 33810, 0))
        with self.assertRaisesRegex(RuntimeError, "No x64 VS runtime"):
            repair.find_runtime_dirs(self.installation, (14, 51))

    def test_requires_openmp_from_the_same_redist_tree(self):
        self.runtimes("14.51", (14, 51, 36231, 0), openmp=False)
        self.runtimes("14.40", (14, 40, 33810, 0))
        with self.assertRaisesRegex(RuntimeError, "No x64 VS runtime"):
            repair.find_runtime_dirs(self.installation, (14, 51))

    def test_rejects_observed_mixed_runtime_wheel(self):
        wheel = self.root / "mixed.whl"
        self.wheel(
            wheel,
            {
                "msvcp140-old.dll": (14, 40, 33810, 0),
                "msvcp140_1-new.dll": (14, 51, 36231, 0),
                "vcomp140-new.dll": (14, 51, 36231, 0),
            },
        )
        with self.assertRaisesRegex(RuntimeError, "older than toolset"):
            repair.verify_runtime_versions(wheel, repair.wheel_toolset(wheel))

    def test_rejects_missing_or_unversioned_runtime(self):
        wheel = self.root / "missing.whl"
        for dlls in ({}, {"msvcp140-hash.dll": (14, 51, 36231, 0)}):
            self.wheel(wheel, dlls)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                repair.verify_runtime_versions(wheel, (14, 51))
        self.wheel(wheel, {"msvcp140-hash.dll": None})
        with self.assertRaisesRegex(RuntimeError, "no file-version"):
            repair.verify_runtime_versions(wheel, (14, 51))

    def test_repair_prioritizes_selected_directories_and_checks_output(self):
        paths = self.runtimes("14.51", (14, 51, 36231, 0))
        source = self.root / "source.whl"
        destination = self.root / "output"
        destination.mkdir()
        self.wheel(source, {})

        def run(command, *, check):
            self.assertTrue(check)
            self.assertEqual(command[1:4], ["-m", "delvewheel", "repair"])
            self.assertEqual(
                command[command.index("--add-path") + 1],
                os.pathsep.join(map(str, paths)),
            )
            self.wheel(
                destination / source.name,
                {
                    "msvcp140-hash.dll": (14, 51, 36231, 0),
                    "vcomp140-hash.dll": (14, 51, 36231, 0),
                },
            )

        with patch.object(repair.subprocess, "run", side_effect=run):
            repair.repair(source, destination, self.installation)

        def bad_output(*args, **kwargs):
            self.wheel(
                destination / source.name, {"msvcp140-old.dll": (14, 40, 33810, 0)}
            )

        with patch.object(repair.subprocess, "run", side_effect=bad_output):
            with self.assertRaisesRegex(RuntimeError, "older than toolset"):
                repair.repair(source, destination, self.installation)

        with patch.object(
            repair.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, "repair"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                repair.repair(source, destination, self.installation)


if __name__ == "__main__":
    unittest.main()
