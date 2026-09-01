#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
readonly repo_root
readonly formatter="${CLANG_FORMAT:-clang-format-15}"

if ! command -v "$formatter" >/dev/null 2>&1; then
    echo "error: $formatter was not found; install clang-format 15 or set CLANG_FORMAT" >&2
    exit 1
fi

if [[ "$($formatter --version)" != *"version 15."* ]]; then
    echo "error: RaBitQ formatting requires clang-format 15" >&2
    echo "found: $($formatter --version)" >&2
    exit 1
fi

mapfile -d '' absolute_files < <("$repo_root/scripts/clang-format-files.sh")
files=()
for file in "${absolute_files[@]}"; do
    files+=("${file#"$repo_root/"}")
done

if ((${#files[@]} > 0)); then
    git -C "$repo_root" clang-format --binary "$formatter" "$@" -- "${files[@]}"
fi
