#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
export CLANG_TIDY="${CLANG_TIDY:-clang-tidy-23}"
include_build="$(realpath "${1:-build-includes}")"
export include_build
export include_root="$repo_root"

"$CLANG_TIDY" --version
if ! "$CLANG_TIDY" --checks='-*,misc-include-cleaner' --list-checks \
    | grep -q 'misc-include-cleaner'; then
    echo 'error: misc-include-cleaner requires clang-tidy 17 or newer' >&2
    exit 1
fi
if [[ ! -f "$include_build/compile_commands.json" ]]; then
    echo "error: configure a compilation database in $include_build first" >&2
    exit 1
fi

# Reuse configured flags for optional dependencies and cross-compilation. Headers
# need to be main files because include-cleaner does not diagnose included files.
"${PYTHON:-python3}" - <<'PYCODE'
import concurrent.futures
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(os.environ["include_root"])
build = Path(os.environ["include_build"])
tidy = os.environ["CLANG_TIDY"]
entries = {}
for entry in json.loads((build / "compile_commands.json").read_text()):
    file = (Path(entry["directory"]) / entry["file"]).resolve()
    if file.is_relative_to(root) and not file.is_relative_to(
        root / "include/rabitqlib/third"
    ):
        entries.setdefault(file, entry)
if not entries:
    sys.exit("error: no first-party sources in the compilation database")

# Public vendor entry points lack complete IWYU export annotations. Keep their
# implementation headers out of callers, including PYBIND11_MODULE's provider.
config = json.dumps(
    {
        "Checks": "-*,misc-include-cleaner,readability-duplicate-include",
        "WarningsAsErrors": "misc-include-cleaner,readability-duplicate-include",
        "CheckOptions": {
            "misc-include-cleaner.IgnoreHeaders": "rabitqlib/third/Eigen/src/.*;"
            "rabitqlib/third/hnswlib/(hnswalg|space_l2)[.]h;"
            "pybind11/detail/common[.]h"
        },
    }
)


def flags_for(entry):
    args = entry.get("arguments") or shlex.split(entry["command"])
    flags = []
    args = iter(args[1:])
    for arg in args:
        if arg == "-o":
            next(args)
        elif (
            arg != "-c"
            and (Path(entry["directory"]) / arg).resolve()
            != (Path(entry["directory"]) / entry["file"]).resolve()
        ):
            flags.append(arg)
    return flags + ["-x", "c++-header"]


def representative(prefix):
    return next((e for f, e in entries.items() if f.is_relative_to(root / prefix)), None)


base = entries.get(root / "src/core/api.cpp")
if base is None:
    sys.exit("error: configure the library targets before checking includes")
# Private headers inherit the ISA flags of their configured source consumers.
# Follow local includes so shared kernel headers get every configured backend.
private_flags = {}
for source, entry in entries.items():
    if not source.is_relative_to(root / "src"):
        continue
    pending, seen = [source], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for include in re.findall(
            r'^\s*#\s*include\s*"([^"]+)"', current.read_text(), re.MULTILINE
        ):
            header = (current.parent / include).resolve()
            if header.is_relative_to(root / "src") and header.is_file():
                private_flags.setdefault(header, set()).add(tuple(flags_for(entry)))
                pending.append(header)
has_index = representative("src/index") is not None
has_fastscan = representative("src/fastscan") is not None
files = (
    subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.cpp",
            "*.cc",
            "*.cxx",
            "*.h",
            "*.hpp",
            ":(exclude)include/rabitqlib/third/**",
        ],
        cwd=root,
    )
    .decode()
    .split("\0")
)
jobs = []
skipped = []
for name in sorted(set(files) - {""}):
    file = root / name
    if not file.is_file():
        continue
    if file.suffix not in {".h", ".hpp"}:
        if file in entries:
            jobs.append((file, "configured", ["-p", str(build)]))
        else:
            skipped.append((name, "not in compilation database"))
        continue
    if (
        name.startswith(
            ("include/rabitqlib/index/", "include/rabitqlib/clustering/", "src/index/")
        )
        and not has_index
        or name.startswith(("include/rabitqlib/fastscan/", "src/fastscan/"))
        and not has_fastscan
    ):
        skipped.append((name, "component disabled"))
        continue
    if name.startswith("src/"):
        contexts = sorted(private_flags.get(file, set()))
        if not contexts:
            skipped.append((name, "no configured source includes this private header"))
        for index, flags in enumerate(contexts):
            jobs.append((file, f"private context {index + 1}", ["--"] + list(flags)))
        continue
    entry = base
    for prefix in ("python_bindings", "tests", "sample", "benchmarks"):
        if name.startswith(prefix + "/"):
            entry = representative(prefix)
            break
    if entry is None:
        skipped.append((name, "owning target not configured"))
        continue
    flags = flags_for(entry)
    jobs.append((file, "header", ["--"] + flags))
    if name.startswith(
        (
            "include/rabitqlib/core/",
            "include/rabitqlib/fastscan/",
            "include/rabitqlib/utils/",
        )
    ):
        serial = [
            f for f in flags if not f.startswith(("-fopenmp", "-DRABITQ_USE_OPENMP="))
        ]
        jobs.append((file, "OpenMP OFF", ["--"] + serial + ["-DRABITQ_USE_OPENMP=0"]))


def check(job):
    file, mode, flags = job
    result = subprocess.run(
        [tidy, "--config=" + config, "--quiet", str(file)] + flags,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return file, mode, result


failed = 0
with concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.environ.get("INCLUDE_JOBS", "2"))
) as pool:
    for file, mode, result in pool.map(check, jobs):
        print(f"Checking {file.relative_to(root)} [{mode}]", flush=True)
        if result.stdout:
            print(result.stdout, end="", flush=True)
        failed += result.returncode != 0
for file, reason in skipped:
    print(f"Skipped {file}: {reason}")
print(
    f"Checked {len({job[0] for job in jobs})} files in {len(jobs)} contexts; "
    f"{failed} failed, {len(skipped)} skipped."
)
sys.exit(1 if failed else 0)
PYCODE
