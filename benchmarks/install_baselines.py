"""Install the benchmark libraries in the current Python 3.13 environment (Ubuntu x86-64)."""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGES = (
    "cmake==3.28.3",
    "ninja",
    "numpy==2.5.2",
    "scikit-build-core==1.0.3",
    "pybind11==3.1.0",
    "swig==4.5.0",
    "packaging",
    "matplotlib==3.11.2",
    "gdown==6.4.1",
    "h5py==3.16.0",
)
SOURCES = {
    "faiss-1.15.0": (
        "https://codeload.github.com/facebookresearch/faiss/tar.gz/refs/tags/v1.15.0"
    ),
    "superkmeans-0.2.0": (
        "https://files.pythonhosted.org/packages/b4/0b/ed67af34f0ed106226538fd62c540400f6bb44fdbe5a4e1d6a9a603ace53/superkmeans-0.2.0.tar.gz"
    ),
}

VERIFY = r"""
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

expected = json.loads(sys.argv[1])
installed = {}
for package, module_name, library in (
    ("rabitqlib", "rabitqlib._rabitqlib", None),
    ("faiss-cpu", "faiss", "libfaiss.so"),
    ("superkmeans", "superkmeans._superkmeans", None),
):
    version = importlib.metadata.version(package)
    if version != expected[package]:
        raise RuntimeError(f"unexpected {package} version: {version}")
    module = importlib.import_module(module_name)
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"{package} imported outside the current environment: {path}")
    binary = path.parent / library if library else path
    linked = subprocess.check_output(["ldd", str(binary)], text=True)
    if "not found" in linked or "libgomp.so" not in linked:
        raise RuntimeError(f"missing dependency or GNU OpenMP for {binary}:\n{linked}")
    if package != "rabitqlib" and "libmkl_rt.so" not in linked:
        raise RuntimeError(f"{package} is not linked to MKL")
    installed[package] = {"version": version, "path": str(binary), "ldd": linked}
print(json.dumps(installed, indent=2))
"""


def preflight(compiler):
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("this benchmark installer requires Linux x86-64")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("use the benchmark's existing Python 3.13 environment")
    if sys.prefix == sys.base_prefix and not (Path(sys.prefix) / "conda-meta").is_dir():
        raise RuntimeError("run inside an existing virtualenv or Conda environment")
    missing = [
        tool
        for tool in (compiler, "git", "numactl", "ldd", "curl")
        if not shutil.which(tool)
    ]
    missing += [
        str(path)
        for path in (
            Path("/usr/include/mkl/mkl.h"),
            Path("/usr/lib/x86_64-linux-gnu/libmkl_rt.so"),
            Path("/usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so"),
        )
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing prerequisites: "
            + ", ".join(missing)
            + "\nOn Ubuntu, run:\n  sudo apt-get update\n"
            "  sudo apt-get install build-essential g++-13 git numactl curl "
            "libmkl-dev libmkl-rt libmkl-gnu-thread\n"
            "See the benchmark installation guide for package-source requirements."
        )
    return shutil.which(compiler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--compiler", default="g++-13")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check prerequisites and print steps without installing",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("job count must be positive")
    try:
        compiler = preflight(args.compiler)
    except RuntimeError as error:
        parser.error(str(error))
    work = ROOT / "build/benchmark-baselines"
    if not args.dry_run:
        for name in ("sources", "wheels", "logs"):
            (work / name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    env["MKL_THREADING_LAYER"] = "GNU"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(Path(sys.prefix) / "lib"), env.get("LD_LIBRARY_PATH")) if p
    )
    env["PYTHONNOUSERSITE"] = "1"

    def run(command, log=None):
        print(shlex.join(map(str, command)), flush=True)
        if args.dry_run:
            return
        if log is None:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        else:
            print(f"Build log: {log}", flush=True)
            with log.open("w") as stream:
                try:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=stream,
                        stderr=stream,
                        check=True,
                    )
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(f"build failed; see {log}") from error

    run([sys.executable, "-m", "pip", "install", *PYTHON_PACKAGES])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "docs/requirements.txt"),
        ]
    )
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    rabitq_name = f"rabitqlib-{version}"
    definitions = {
        rabitq_name: {"RABITQ_ENABLE_NATIVE_OPTIMIZATION": "ON"},
        "faiss-1.15.0": {
            "FAISS_ENABLE_MKL": "ON",
            "BLA_VENDOR": "Intel10_64_dyn",
            "BLA_VENDOR_THREADING": "gnu",
            "FAISS_USE_LTO": "OFF",
        },
        "superkmeans-0.2.0": {
            "MKL_DIR": str(ROOT / "benchmarks/mkl"),
            "SKMEANS_PORTABLE": "OFF",
            "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",
            "SKMEANS_MARCH": "native",
            # Match the published wheel's rotation backend; this sdist omits FindFFTW.cmake.
            "SKMEANS_SKIP_FFTW": "ON",
        },
    }
    records = []
    wheels = []
    for name in definitions:
        if name == rabitq_name:
            source = ROOT
            provenance = {}
        else:
            url = SOURCES[name]
            source = work / "sources" / name
            provenance = {"url": url}
            print(f"Source: {url}", flush=True)
            if not args.dry_run:
                archive = work / "sources" / f"{name}.tar.gz"
                if not archive.exists():
                    with urllib.request.urlopen(url, timeout=60) as response:
                        data = response.read()
                    archive.write_bytes(data)
                if not source.exists():
                    with tarfile.open(archive) as package:
                        package.extractall(work / "sources", filter="data")
        short_name = name.split("-")[0]
        flags = {
            "CMAKE_CXX_COMPILER": compiler,
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "CMAKE_BUILD_TYPE": "Release",
            **definitions[name],
        }
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(source),
            "--no-build-isolation",
            "--no-deps",
            f"-Cbuild-dir={source / 'build/benchmark-baselines' / (short_name + '-build')}",
            *[f"-Ccmake.define.{key}={value}" for key, value in flags.items()],
            "-v",
        ]
        log = work / "logs" / f"{short_name}-build.log"
        if args.dry_run:
            run([*command, "-w", str(work / "wheels" / name)], log)
            wheels.append(str(work / "wheels" / f"<{name} wheel>"))
            continue
        # Select only this build's wheel, never an older cached version or Python ABI.
        with tempfile.TemporaryDirectory(dir=work / "wheels", prefix=name + "-") as tmp:
            run([*command, "-w", tmp], log)
            built = list(Path(tmp).glob("*.whl"))
            if len(built) != 1:
                raise RuntimeError(f"expected one wheel from {name}, got {len(built)}")
            wheel = work / "wheels" / built[0].name
            shutil.copyfile(built[0], wheel)
        wheels.append(str(wheel))
        records.append(
            {
                "source": name,
                **provenance,
                "cmake": flags,
                "compiler": subprocess.check_output([compiler, "--version"], text=True),
                "wheel": str(wheel),
            }
        )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            *wheels,
        ]
    )
    if args.dry_run:
        print(
            "Then verify versions, environment paths, MKL/GNU OpenMP links, and write builds.json and installed.json."
        )
        return
    (work / "builds.json").write_text(json.dumps(records, indent=2) + "\n")
    expected = {"rabitqlib": version, "faiss-cpu": "1.15.0", "superkmeans": "0.2.0"}
    verified = subprocess.check_output(
        [sys.executable, "-c", VERIFY, json.dumps(expected)],
        cwd=ROOT,
        env=env,
        text=True,
    )
    (work / "installed.json").write_text(verified)
    activation = ROOT / "benchmarks/env.sh"
    print(
        f"Installed and verified all three libraries in {sys.prefix}.\nBefore running benchmarks: source {shlex.quote(str(activation))}"
    )


if __name__ == "__main__":
    main()
