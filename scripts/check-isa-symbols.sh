#!/usr/bin/env bash

# Fail when an inline or template function is emitted as a weak symbol by
# translation units compiled with different ISA flags and the copies differ.
# The linker keeps one copy for every caller, so a generic caller could run
# AVX code. Usage: check-isa-symbols.sh BUILD_DIR [EXTRA_COMPILE_FLAG...]
# BUILD_DIR needs compile_commands.json; extra flags (for example -O0) are
# appended to every compilation. Requires objdump and nm (GNU binutils).

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
readonly repo_root

if (($# < 1)); then
    echo "usage: $0 BUILD_DIR [EXTRA_COMPILE_FLAG...]" >&2
    exit 2
fi
readonly build_dir="$1"
shift

if [[ ! -f "$build_dir/compile_commands.json" ]]; then
    echo "error: $build_dir/compile_commands.json was not found" >&2
    echo "configure CMake with CMAKE_EXPORT_COMPILE_COMMANDS=ON first" >&2
    exit 1
fi

for tool in nm objdump; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool was not found" >&2
        exit 1
    fi
done

object_dir="$(mktemp -d)"
if [[ -z "$object_dir" || ! -d "$object_dir" ]]; then
    echo "error: could not create a temporary object directory" >&2
    exit 1
fi
trap 'rm -rf -- "$object_dir"' EXIT

"${PYTHON:-python3}" - "$build_dir/compile_commands.json" "$repo_root" "$object_dir" "$@" <<'PYCODE'
import collections
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys

database, repo_root, object_dir, *extra_flags = sys.argv[1:]
source_pattern = re.compile(re.escape(repo_root) + r"/src/.+\.cpp$")
isa_flag = re.compile(r"^(-m(avx|fma|sse|popcnt|bmi|f16c|arch=)|/arch:|-march=)")


def compile_unit(index, entry):
    arguments = entry.get("arguments") or shlex.split(entry["command"])
    arguments = list(arguments)
    output = os.path.join(object_dir, f"{index}.o")
    position = arguments.index("-o")
    arguments[position + 1] = output
    result = subprocess.run(
        arguments + list(extra_flags), cwd=entry["directory"], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"error: compiling {entry['file']} failed:\n{result.stderr}")
    isa = " ".join(sorted(a for a in arguments if isa_flag.match(a))) or "generic"
    return output, isa, os.path.relpath(entry["file"], repo_root)


entries = []
for entry in json.load(open(database)):
    path = os.path.abspath(os.path.join(entry["directory"], entry["file"]))
    if source_pattern.match(path):
        entries.append(dict(entry, file=path))
if not entries:
    sys.exit("error: no first-party library sources in the compilation database")

with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
    units = list(pool.map(lambda item: compile_unit(*item), enumerate(entries)))
if len({isa for _, isa, _ in units}) < 2:
    sys.exit("error: the build has no ISA-specific translation units to compare")


def weak_functions(obj):
    listing = subprocess.run(
        ["nm", "--defined-only", obj], capture_output=True, text=True, check=True
    ).stdout
    return {line.split()[2] for line in listing.splitlines() if line.split()[1:2] == ["W"]}


def function_bodies(obj):
    # Normalize addresses so copies compare by instructions, not placement.
    text = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-r", obj],
        capture_output=True, text=True, check=True,
    ).stdout
    bodies, name, lines = {}, None, []
    for line in text.splitlines():
        if line.startswith("Disassembly of section"):
            if name is not None:
                bodies[name] = "\n".join(lines)
            name, lines = None, []
            continue
        header = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if header:
            if name is not None:
                bodies[name] = "\n".join(lines)
            name, lines = header.group(1), []
            continue
        if name is None or not line.strip():
            continue
        # Weak functions live in their own COMDAT sections, so branch targets are
        # already section-relative. Keep immediates: copies that differ only in a
        # constant are different code. Only placement-dependent names are normalized.
        line = re.sub(r"^\s*[0-9a-f]+:\s*", "", line)
        line = re.sub(r"<[^>]*>", "<>", line)
        line = re.sub(r"\.L\w+", ".L", line)
        line = re.sub(r"(\.(?:rodata|data|bss)[\w.]*)[+-]0x[0-9a-f]+", r"\1+#", line)
        lines.append(line.strip())
    if name is not None:
        bodies[name] = "\n".join(lines)
    return bodies


definitions = collections.defaultdict(list)
for obj, isa, source in units:
    for symbol in weak_functions(obj):
        definitions[symbol].append((obj, isa, source))

shared = {
    symbol: owners
    for symbol, owners in definitions.items()
    if len({isa for _, isa, _ in owners}) > 1
}
needed = sorted({obj for owners in shared.values() for obj, _, _ in owners})
with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
    disassembly = dict(zip(needed, pool.map(function_bodies, needed)))

failures = []
for symbol, owners in sorted(shared.items()):
    copies = {}
    for obj, isa, source in owners:
        copies.setdefault(isa, (disassembly[obj].get(symbol), source))
    if len({body for body, _ in copies.values()}) > 1:
        demangled = subprocess.run(
            ["c++filt", symbol], capture_output=True, text=True
        ).stdout.strip() or symbol
        failures.append((demangled, sorted((isa, source) for isa, (_, source) in copies.items())))

if failures:
    print(f"error: {len(failures)} weak functions differ between ISA builds:", file=sys.stderr)
    for symbol, owners in failures:
        print(f"  {symbol}", file=sys.stderr)
        for isa, source in owners:
            print(f"      [{isa}] {source}", file=sys.stderr)
    sys.exit(1)
print(f"ISA symbol check passed for {len(units)} translation units")
PYCODE
