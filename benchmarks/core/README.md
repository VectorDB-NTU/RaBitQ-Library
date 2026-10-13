# Standalone core benchmark

This C++17 executable links only `rabitqlib::core`. It measures stable v1 encoding,
query preparation, packed scoring and packed-to-unpacked conversion using caller-owned
storage. No ANN implementation, Python environment or dataset download is required.

Build from the checkout:

```bash
cmake -S . -B build-core-bench -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_INDEX=OFF -DRABITQ_BUILD_FASTSCAN=OFF \
  -DRABITQ_BUILD_SAMPLES=OFF -DRABITQ_USE_OPENMP=OFF \
  -DRABITQ_BUILD_BENCHMARKS=ON -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build-core-bench --parallel
ctest --test-dir build-core-bench --output-on-failure
./build-core-bench/benchmarks/core/rabitq_core_benchmark > core-results.csv
```

For an installed package, configure this directory directly:

```bash
cmake -S benchmarks/core -B build-core-bench-installed \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="<core-install-prefix>"
cmake --build build-core-bench-installed --parallel
ctest --test-dir build-core-bench-installed --output-on-failure
./build-core-bench-installed/rabitq_core_benchmark 128 4 4096 64 7 > core-results.csv
```

Multi-configuration generators also need `--config Release` / `-C Release`, and the
executable is inside `Release/`. Positional arguments are dimension, total bits,
number of vectors, number of queries and measured repetitions. Defaults are
`128 4 4096 64 5`. This benchmark's FHT/Kac rotation requires a dimension at least 64
and divisible by 32; bits are 1–9 and at least ten vectors are required. These are
benchmark constraints, not additional restrictions on the stable unpacked API.

The deterministic uniform dataset uses a fixed LCG seed 42; one shared FHT/Kac rotator
is initialized with seed 42 for both data and queries. Codes use SplitPacked, L2 and a
zero centroid. All storage allocation, rotation, initial encoding and query preparation
happen before measurement. Each operation has one untimed warmup, followed by the
requested number of timed passes. Query preparation is timed explicitly in its own row.
The executable is single-threaded even in an OpenMP-enabled build.

The output includes raw time, operations/s, reciprocal throughput in ns/operation and
a checksum for every repetition. Comment-prefixed metadata records parameters, available
ISA capabilities, OpenMP build choice, distance RMSE, maximum absolute distance error
and recall@10 of an exhaustive quantized scan. Ground truth is double-precision L2 on
the actual rotated float32 vectors. There is no ANN graph/pruning or reranking. Recall
therefore measures this quantizer/workload, not an index's search recall or end-to-end QPS.

The scoring pass traverses all encoded vectors for each prepared query, unlike the
earlier single-code cache-resident microbenchmark. Timing excludes top-k selection and
quality computation. Do not compare these operations/s directly with index QPS. Quality
values describe approximation error; they are not tolerances allowing semantic changes.

For comparisons, retain all raw repetitions and record CPU model, compiler/version,
build flags, affinity/SMT policy, OS and dataset parameters beside the CSV. Keep these
fixed and alternate baseline/candidate runs; compare quality before claiming a speedup.
Linux users can pin the executable with `taskset -c <cpu>`. The program reports available
ISA features, not a trace of every selected kernel. CI only smoke-tests functionality;
shared CI timing is not a performance gate.

## Scoring-call overhead

The stable v1 `compute` takes `CodeView` by value. Its thin inline entry forwards
that local view by reference to the private checked implementation, avoiding an
aggregate copy across the compiled call boundary in optimized direct calls.
Validation stays in the checked implementation, and taking the public function's
address remains supported. Use matching headers/library and rebuild consumers
when upgrading. Performance measured for optimized direct calls does not imply
the same gain through function pointers or in unoptimized builds. The checked
interface can cost more than an unchecked inlined estimator.

`rabitq_core_scoring_benchmark` compares the checked v1 call with three benchmark-only
unchecked references: the same kernels/formula inlined into the loop, a separate
translation-unit call taking `CodeView` by value, and a separate call taking it by
reference. These references are not library
APIs and only receive valid encoded inputs. They help distinguish validation overhead
from the combined effects of call boundaries, argument passing and inlining; subtracting
their timings does not isolate one CPU instruction's cost.

```bash
./build-core-bench/benchmarks/core/rabitq_core_scoring_benchmark \
  128 4 packed hot 1000000 7
./build-core-bench/benchmarks/core/rabitq_core_scoring_benchmark \
  768 9 packed random 262144 7 64
./build-core-bench/benchmarks/core/rabitq_core_scoring_benchmark \
  65 4 unpacked sequential 32768 7 64
```

Arguments are dimension, total bits, `packed|unpacked`, `hot|sequential|random`, minimum
iterations, repetitions and optional code/factor working-set target in MiB (default 64).
An optional final `all|v1` argument selects timed paths; `v1` still checks every prototype
against all three references before timing and is useful for a large baseline/candidate matrix.
No arguments select `128 4 packed hot 1000000 5`. Packed dimensions must be multiples
of 32; unpacked dimensions may be any supported dimension. Build without LTO or native
tuning for the intended comparison. The target disables CMake IPO, but callers must
also avoid injecting manual LTO flags.

Hot mode repeatedly scores one code. The other modes use a power-of-two record count
fitting the byte target, with sequential IDs or a fixed shuffled permutation. Metadata
reports actual code/factor bytes and the additional order-array bytes; this is a defined
access pattern, not a promise that accesses miss a particular cache. Each timed pass
visits at least the entire bank, even if the requested minimum iterations is smaller.

Data/query values use an LCG with seed 42; the permutation uses seed 91. At most 32
valid code/factor prototypes are encoded, then replicated at distinct memory addresses.
This models access and call costs, not dataset-dependent ANN quality. Inputs are already
in the numerical working domain; no rotation or top-k retrieval is timed or evaluated.
The metric is L2 with a zero centroid. Prototype scores must match all three references
exactly before timing. Raw timings and double-precision checksums are printed for every
path and repetition; path order rotates and each timed pass has its own warmup.

Compare baseline/candidate binaries with identical harness source, compiler flags and
parameters. Alternate process order, retain all samples, and verify checksums alongside
performance. Record machine/affinity/SMT details as for the main benchmark. The unchecked
paths are useful controls, but their generated code need not match a real downstream
application. Neither these loops nor reciprocal throughput report ANN QPS or isolated
call latency.
