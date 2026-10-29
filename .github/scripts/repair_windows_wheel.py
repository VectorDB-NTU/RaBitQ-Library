"""Repair x64 wheels using VS runtimes, then reject older bundled runtimes."""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

RUNTIME_PREFIXES = ("msvcp140", "vcruntime140", "vcomp140", "concrt140")


def pe_details(data):
    # pefile is installed by the existing delvewheel before-build hook.
    import pefile

    with pefile.PE(data=data) as image:
        if image.FILE_HEADER.Machine != 0x8664:
            raise RuntimeError("Windows wheel repair requires x64 PE binaries")
        toolset = (
            image.OPTIONAL_HEADER.MajorLinkerVersion,
            image.OPTIONAL_HEADER.MinorLinkerVersion,
        )
        version = None
        if getattr(image, "VS_FIXEDFILEINFO", None):
            info = image.VS_FIXEDFILEINFO[0]
            version = (
                info.FileVersionMS >> 16,
                info.FileVersionMS & 0xFFFF,
                info.FileVersionLS >> 16,
                info.FileVersionLS & 0xFFFF,
            )
        return toolset, version


def runtime_version(data):
    _, version = pe_details(data)
    if version is None:
        raise RuntimeError("Runtime DLL has no file-version resource")
    return version


def wheel_toolset(wheel):
    with zipfile.ZipFile(wheel) as archive:
        extensions = [
            name
            for name in archive.namelist()
            if Path(name).name.startswith("_rabitqlib.") and name.endswith(".pyd")
        ]
        if len(extensions) != 1:
            raise RuntimeError("Expected exactly one _rabitqlib extension in the wheel")
        toolset, _ = pe_details(archive.read(extensions[0]))
        return toolset


def find_runtime_dirs(installation, minimum):
    candidates = []
    root = installation / "VC" / "Redist" / "MSVC"
    for crt in root.glob("*/x64/Microsoft.VC*.CRT"):
        main_dll = crt / "msvcp140.dll"
        if not main_dll.is_file():
            continue
        version = runtime_version(main_dll.read_bytes())
        if version[:2] < minimum:
            continue
        for openmp in crt.parent.glob("Microsoft.VC*.OpenMP"):
            omp_dll = openmp / "vcomp140.dll"
            if (
                omp_dll.is_file()
                and runtime_version(omp_dll.read_bytes())[:2] >= minimum
            ):
                candidates.append((version, crt, openmp))
    if not candidates:
        raise RuntimeError(f"No x64 VS runtime directories match toolset {minimum}")
    _, crt, openmp = max(candidates, key=lambda candidate: candidate[0])
    return crt, openmp


def verify_runtime_versions(wheel, minimum):
    found = set()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            filename = Path(name).name.lower()
            if not filename.endswith(".dll") or not filename.startswith(
                RUNTIME_PREFIXES
            ):
                continue
            version = runtime_version(archive.read(name))
            print(f"Bundled {name}: {'.'.join(map(str, version))}", flush=True)
            if version[:2] < minimum:
                raise RuntimeError(
                    f"Bundled {name} runtime {version} is older than toolset {minimum}"
                )
            found.add(filename.split("-", 1)[0].removesuffix(".dll"))
    if not {"msvcp140", "vcomp140"} <= found:
        raise RuntimeError("Repaired wheel is missing its MSVC or OpenMP runtime")


def repair(wheel, destination, installation):
    minimum = wheel_toolset(wheel)
    runtime_dirs = find_runtime_dirs(installation, minimum)
    print(
        f"Wheel toolset: {minimum}; runtime search directories: {runtime_dirs}",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "delvewheel",
            "repair",
            "--add-path",
            os.pathsep.join(map(str, runtime_dirs)),
            "-w",
            str(destination),
            str(wheel),
        ],
        check=True,
    )
    verify_runtime_versions(destination / wheel.name, minimum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    vswhere = (
        Path(os.environ["ProgramFiles(x86)"])
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    installation = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
            "-utf8",
        ],
        encoding="utf-8-sig",
    ).strip()
    if not installation:
        raise RuntimeError("No Visual Studio C++ installation found")
    repair(args.wheel, args.destination, Path(installation))


if __name__ == "__main__":
    main()
