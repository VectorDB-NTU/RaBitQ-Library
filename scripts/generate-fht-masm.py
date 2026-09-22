#!/usr/bin/env python3
"""Regenerate the Windows MASM FHT kernels from the imported FFHT header.

Requires Clang 22. The checked-in assembly is built with MSVC; Clang is only
needed when regenerating it. Keep the AVX instruction sequences in sync with
include/rabitqlib/utils/fht_avx.hpp.
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include/rabitqlib/utils/fht_avx.hpp"
OUTPUT = ROOT / "src/simd/fht_avx2_masm.asm"
HOT_LOG_SIZES = range(6, 12)
ALL_HELPERS = [(kind, size) for kind in ("float", "double") for size in range(1, 31)]
OTHER_HELPERS = [
    (kind, size)
    for kind, size in ALL_HELPERS
    if kind != "float" or size not in HOT_LOG_SIZES
]


def to_intel(match):
    instruction, operands = match.groups()
    operands = [
        re.sub(r"\((%\d+)\)", r"[\1]", operand).replace("%%ymm", "ymm").replace("$", "")
        for operand in operands.split(", ")
    ]
    return '"' + instruction + " " + ", ".join(reversed(operands)) + r"\n" + '"'


def build_clang_source(directory, helpers, name):
    source = HEADER.read_text()
    source = re.sub(r'"(v\w+) (.*?)\\n"', to_intel, source)
    header = directory / "fht_avx_intel.hpp"
    header.write_text(source)
    wrapper = directory / f"fht_windows_{name}.cpp"
    wrapper.write_text(
        f'#include "{header}"\n'
        + "".join(
            f'extern "C" void rabitq_fht_avx_{kind}_{size}({kind}* values) '
            f"{{ helper_{kind}_{size}(values); }}\n"
            for kind, size in helpers
        )
    )
    return wrapper


def to_masm(source):
    functions = []
    current = None
    for line in source.splitlines():
        if "# -- Begin function " in line:
            current = []
            continue
        if "# -- End function" in line:
            functions.append(current)
            current = None
            continue
        if current is not None:
            current.append(line.split("#", 1)[0].rstrip())

    internal = {}
    for function in functions:
        for line in function:
            for name in re.findall(r'"(\?helper_[^\"]+)"', line):
                if name not in internal:
                    internal[name] = "fht_" + re.match(
                        r"\?([A-Za-z_0-9]+)@@", name
                    ).group(1)

    result = []
    for function in functions:
        started = False
        name = None
        framed = any(line.startswith(".seh_proc") for line in function)
        for line in function:
            for old, new in internal.items():
                line = line.replace(f'"{old}"', new)
            line = line.strip()
            if not line or line.startswith((".globl", ".p2align", ".section")):
                continue
            if line.endswith(":") and not started:
                name = line[:-1]
                result.append(f"{name} PROC" + (" FRAME" if framed else ""))
                started = True
            elif line.startswith(".seh_proc") or line.startswith(".seh_endproc"):
                continue
            elif line.startswith(".seh_pushreg"):
                result.append("    .pushreg " + line.split(None, 1)[1])
            elif line.startswith(".seh_stackalloc"):
                result.append("    .allocstack " + line.split(None, 1)[1])
            elif line.startswith(".seh_savexmm"):
                result.append("    .savexmm128 " + line.split(None, 1)[1])
            elif line == ".seh_endprologue":
                result.append("    .endprolog")
            elif line.startswith(".seh_") or line in ("#APP", "#NO_APP"):
                continue
            elif line.startswith(".LBB") and line.endswith(":"):
                result.append(line[1:])
            elif line.startswith("."):
                raise ValueError(f"Unrecognized assembler directive: {line}")
            else:
                # Clang spells 64-bit immediate moves "movabs" in Intel syntax;
                # MASM uses "mov" for the same instruction.
                line = re.sub(r"^movabs(?=\s)", "mov", line)
                result.append("    " + line.replace(".LBB", "LBB"))
        if not started:
            raise ValueError("Function label missing")
        result.append(f"{name} ENDP")
        result.append("")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", default="clang++-22")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        body = []
        for name, helpers, optimization in (
            ("hot", [("float", size) for size in HOT_LOG_SIZES], "-O3"),
            ("other", OTHER_HELPERS, "-O0"),
        ):
            wrapper = build_clang_source(directory, helpers, name)
            assembly = directory / f"fht_windows_{name}.s"
            subprocess.run(
                [
                    args.clang,
                    "--target=x86_64-pc-windows-msvc",
                    "-mavx",
                    optimization,
                    "-S",
                    "-masm=intel",
                    str(wrapper),
                    "-o",
                    str(assembly),
                ],
                check=True,
            )
            body.extend(to_masm(assembly.read_text()))
    license_lines = HEADER.read_text().splitlines()[:23]
    output = ["; Generated by scripts/generate-fht-masm.py from fht_avx.hpp."]
    output.extend(
        "; " + line.lstrip("/ ") if line.strip() else ";" for line in license_lines
    )
    output.append("")
    output.append(
        "; The six float sizes used by RaBitQ are optimized; other FFHT helpers"
    )
    output.append("; retain the original AVX blocks and loop structure.")
    output.extend(f"PUBLIC rabitq_fht_avx_{kind}_{size}" for kind, size in ALL_HELPERS)
    output.extend(["", ".code", ""])
    output.extend(body)
    output.append("END")
    OUTPUT.write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
