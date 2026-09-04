#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
readonly repo_root
readonly build_dir="${1:-$repo_root/build}"
readonly tidy_runner="${RUN_CLANG_TIDY:-run-clang-tidy-15}"
readonly compiler="${CXX:-c++}"

if [[ ! -f "$build_dir/compile_commands.json" ]]; then
    echo "error: $build_dir/compile_commands.json was not found" >&2
    echo "configure CMake with CMAKE_EXPORT_COMPILE_COMMANDS=ON first" >&2
    exit 1
fi

if ! command -v "$tidy_runner" >/dev/null 2>&1; then
    echo "error: $tidy_runner was not found; install clang-tidy 15" >&2
    exit 1
fi

if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "error: $compiler was not found; set CXX to the compiler used by CMake" >&2
    exit 1
fi

mapfile -t system_include_dirs < <(
    "$compiler" -E -x c++ - -v </dev/null 2>&1 \
        | awk '
            /#include <\.\.\.> search starts here:/ { in_search_path = 1; next }
            /End of search list/ { in_search_path = 0 }
            in_search_path {
                sub(/^ /, "")
                print
            }
        '
)

if ((${#system_include_dirs[@]} == 0)); then
    echo "error: could not determine the system include paths for $compiler" >&2
    exit 1
fi

extra_args=("-extra-arg-before=-Wno-unused-command-line-argument")
if [[ -n "${CLANG_RESOURCE_DIR:-}" ]]; then
    if [[ ! -d "$CLANG_RESOURCE_DIR" ]]; then
        echo "error: CLANG_RESOURCE_DIR is not a directory: $CLANG_RESOURCE_DIR" >&2
        exit 1
    fi
    extra_args+=("-extra-arg-before=-resource-dir=$CLANG_RESOURCE_DIR")
fi
for include_dir in "${system_include_dirs[@]}"; do
    if [[ "$include_dir" == */lib/gcc/*/include ]]; then
        # Keep Clang's intrinsic headers ahead of GCC's, while still making
        # compiler-provided headers such as omp.h available.
        extra_args+=("-extra-arg-before=-idirafter$include_dir")
    else
        extra_args+=("-extra-arg-before=-isystem$include_dir")
    fi
done

mapfile -t first_party_headers < <(
    git -C "$repo_root" ls-files -- '*.h' '*.hpp' \
        ':(exclude)include/rabitqlib/third/**' \
        ':(exclude)include/rabitqlib/utils/fht_avx.hpp'
)

header_filter="$repo_root/("
separator=""
for header in "${first_party_headers[@]}"; do
    escaped_header="${header//./\\.}"
    header_filter+="$separator$escaped_header"
    separator="|"
done
header_filter+=")$"

tidy_output="$(mktemp)"
if [[ -z "$tidy_output" || ! -f "$tidy_output" ]]; then
    echo "error: could not create a temporary clang-tidy output file" >&2
    exit 1
fi
trap 'rm -f -- "$tidy_output"' EXIT

if ! "$tidy_runner" \
    -quiet \
    -p "$build_dir" \
    -header-filter="^$header_filter" \
    "${extra_args[@]}" \
    "^$repo_root/(src|python_bindings|sample|tests)/" \
    >"$tidy_output" 2>&1; then
    cat "$tidy_output" >&2
    exit 1
fi

echo "clang-tidy passed for all configured first-party translation units"
