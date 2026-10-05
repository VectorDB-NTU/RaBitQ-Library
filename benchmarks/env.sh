#!/usr/bin/env bash
# Source this file after activating the Python environment used for installation.
rabitq_benchmark_lib="$(python -c 'import sys; print(sys.prefix + "/lib")')" || return
export MKL_THREADING_LAYER=GNU
export LD_LIBRARY_PATH="${rabitq_benchmark_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset rabitq_benchmark_lib
