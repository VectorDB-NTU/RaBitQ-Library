#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
readonly repo_root
readonly ruff="${RUFF:-ruff}"

if ! command -v "$ruff" >/dev/null 2>&1; then
    echo "error: $ruff was not found; install Ruff 0.16.1 or set RUFF" >&2
    exit 1
fi

if [[ "$($ruff --version)" != "ruff 0.16.1" ]]; then
    echo "error: RaBitQ Python checks require Ruff 0.16.1" >&2
    echo "found: $($ruff --version)" >&2
    exit 1
fi

cd "$repo_root"
"$ruff" format --check python python_bindings sample/python tests/python
"$ruff" check python python_bindings sample/python tests/python
