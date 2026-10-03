#!/usr/bin/env bash

set -euo pipefail

readonly install_prefix="${1:?OpenMP install prefix is required}"
readonly deployment_target="${MACOSX_DEPLOYMENT_TARGET:?macOS deployment target is required}"
source_dir="$(mktemp -d)"
readonly source_dir
trap 'rm -rf "$source_dir"' EXIT

# Homebrew bottles target the runner OS. Build its checksum-verified source
# ourselves so the bundled runtime supports the wheel's older deployment target.
brew fetch --build-from-source libomp
tar -xf "$(brew --cache --build-from-source libomp)" \
    -C "$source_dir" --strip-components=1
cmake -S "$source_dir/runtimes" -B "$source_dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install_prefix" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DLLVM_ENABLE_RUNTIMES=openmp \
    -DOPENMP_INSTALL_LIBDIR=lib \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
    -DLIBOMP_INSTALL_ALIASES=OFF
cmake --build "$source_dir/build" --parallel "$(sysctl -n hw.logicalcpu)"
cmake --install "$source_dir/build"
