#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
export CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"
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

check_file() {
    local file="$1"
    local status=0
    local report
    report="$(mktemp)"
    # Isolate this check from the project's general clang-tidy configuration.
    # Vendored snapshots lack export annotations on their public headers.
    # Do not replace those headers with Eigen/hnswlib implementation includes.
    local args=(--config='{CheckOptions: {misc-include-cleaner.IgnoreHeaders: "rabitqlib/third/Eigen/src/.*;rabitqlib/third/hnswlib/(hnswalg|space_l2)[.]h"}}'  --checks='-*,misc-include-cleaner'
        --warnings-as-errors='misc-include-cleaner' --quiet)
    if [[ "$file" == src/*.cpp ]]; then
        "$CLANG_TIDY" "${args[@]}" -p "$include_build" "$file" >"$report" 2>&1 || status=1
    else
        local flags=(-x c++ -std=c++17 -fopenmp -I"$include_root/include"
            -Wno-pragma-once-outside-header)
        if [[ "$file" == src/* ]]; then
            # Private kernels share AVX2 and AVX-512 implementations. Check both
            # preprocessor paths, including the popcount-enabled path.
            if [[ "$file" != *avx512* ]]; then
                "$CLANG_TIDY" "${args[@]}" "$file" -- "${flags[@]}" \
                    -mavx2 -mfma >"$report" 2>&1 || status=1
            fi
            if [[ "$file" != *avx2* ]]; then
                "$CLANG_TIDY" "${args[@]}" "$file" -- "${flags[@]}" \
                    -mavx2 -mfma -mavx512f -mavx512bw -mavx512dq \
                    -mavx512vpopcntdq >>"$report" 2>&1 || status=1
            fi
        else
            "$CLANG_TIDY" "${args[@]}" "$file" -- "${flags[@]}" >"$report" 2>&1 || status=1
        fi
    fi
    echo "Checking $file"
    cat "$report"
    rm -f "$report"
    return "$status"
}
export -f check_file

# The child shell expands its positional argument.
# shellcheck disable=SC2016
git ls-files -z -- 'src/*.cpp' 'src/*.hpp' 'include/rabitqlib/*.hpp' \
    ':(exclude)include/rabitqlib/third/**' \
    ':(exclude)include/rabitqlib/utils/fht_avx.hpp' \
    | xargs -0 -r -n 1 -P "${INCLUDE_JOBS:-2}" bash -c 'check_file "$1"' _
