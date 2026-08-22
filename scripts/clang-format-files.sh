#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
readonly repo_root

if (($# > 0)); then
    printf '%s\0' "$@"
else
    git -C "$repo_root" ls-files -z -- \
        '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hpp' '*.cu' '*.cuh' \
        ':(exclude)include/rabitqlib/third/**' \
        ':(exclude)include/rabitqlib/utils/fht_avx.hpp'
fi | while IFS= read -r -d '' file; do
    if [[ "$file" == /* ]]; then
        file="$(realpath --relative-to="$repo_root" "$file")"
    else
        file="${file#./}"
    fi

    case "$file" in
        ../* | */../* | */..)
            continue
            ;;
        include/rabitqlib/third/* | include/rabitqlib/utils/fht_avx.hpp)
            continue
            ;;
    esac

    if [[ -f "$repo_root/$file" ]]; then
        printf '%s\0' "$repo_root/$file"
    fi
done
