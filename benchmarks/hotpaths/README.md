# Component hot-path benchmarks

These deterministic benchmarks retain the small workloads that exposed modularization
costs. They complement the [stable core benchmark](../core/README.md) and
[full ANN benchmarks](../README.md); their throughput is not end-to-end ANN QPS.

Enable `RABITQ_BUILD_BENCHMARKS=ON` in a Release build. Targets follow enabled components:

| Executable | Dependency | Measurement |
| --- | --- | --- |
| `rabitq_core_hotpaths` | core only | One-bit compact encoding or raw L2 calls |
| `rabitq_fastscan_hotpaths` | FastScan | Cached standard/HACC batch scoring |
| `rabitq_load_hotpaths` | index | HNSW centroid-sidecar load and validation |
| `rabitq_ids_benchmark` module + `ids.py` | Python bindings | Production cluster-ID conversion helper |

No additional production dependencies are introduced. Core/FastScan benchmarks support
OpenMP OFF. Python benchmarks use the existing Python/NumPy/pybind11 environment and
are available only with `RABITQ_BUILD_PYTHON_BINDINGS=ON`. The helper module is not installed
in the package. `ctest` smoke tests verify functionality, never shared-host speed thresholds.
The CI `Benchmark build and smoke` job builds these targets on Ubuntu, including
the Python helper, and runs the registered small cases without external datasets.

## Build and run

```bash
cmake -S . -B build-hotpaths -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_BENCHMARKS=ON -DRABITQ_BUILD_SAMPLES=OFF \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build-hotpaths --parallel
ctest --test-dir build-hotpaths -R 'Hotpath' --output-on-failure
```

Set both `RABITQ_BUILD_INDEX=OFF` and `RABITQ_BUILD_FASTSCAN=OFF` for pure core.
For the Python helper, additionally enable bindings and select your existing interpreter
with `Python_EXECUTABLE` and its `pybind11_DIR`, as described in
[development setup](../../docs/development.md). Multi-configuration generators need
`--config Release`, `ctest -C Release`, and the `Release/` executable subdirectory.

Examples (from the repository root):

```bash
# dim rows threads iterations repetitions mode
./build-hotpaths/benchmarks/hotpaths/rabitq_core_hotpaths 64 4096 1 1024 3 encode
./build-hotpaths/benchmarks/hotpaths/rabitq_core_hotpaths 128 1 1 2000000 5 l2

# dim batches repetitions standard|hacc: one batch contains 32 vectors
./build-hotpaths/benchmarks/hotpaths/rabitq_fastscan_hotpaths 128 50000000 3 hacc

# Prepare once; both versions must load the same file. The writer appends .hnsw.
./build-hotpaths/benchmarks/hotpaths/rabitq_load_hotpaths prepare "$TMPDIR/centroids" 128 16384 1
./build-hotpaths/benchmarks/hotpaths/rabitq_load_hotpaths load "$TMPDIR/centroids" 128 16384 7

python benchmarks/hotpaths/ids.py --module-dir build-hotpaths/python_bindings
```

Use a task-specific writable `TMPDIR`. `prepare` reuses and validates an existing
fixture instead of replacing it. It rejects a file whose centroid contents do not match
the deterministic input. Original production indexes are not benchmark fixtures.

## Timing and correctness contract

- IVF add verifies the expected point counts and samples new points' self-neighbors
  outside the timed call, in addition to comparing output hashes across repetitions.
- Inputs use an LCG with seed 42 (query/centroid seed 91 where applicable), metric L2.
  Dimensions for encoding/scoring are multiples of 32. Use 64, 128 and 960 dimensions;
  encoding rows 32/4096 and threads 1/4 retain the cases from the original investigation.
- Core encoding times the existing serial template at one thread and OpenMP template
  above one thread. It includes scratch reuse and encoder scheduling, excludes input/output
  allocation, and verifies serial/parallel code and factor hashes before timing.
- L2 checks a float64 reference with absolute tolerance `2e-6 * dim`. Results are stored
  per row; this benchmark includes those stores and is not an isolated instruction latency.
- FastScan repeatedly scores one 32-vector batch in cache. It checks generic correction
  within `2e-6 * dim`, reports one-bit filtering RMSE and recall@10 against float64 L2,
  and hashes estimated distances, lower bounds and inner products. The query uses three
  extra bits for preparation, but this benchmark times only the one-bit filter, not reranking.
- Loading times `load` only, including its allocations and validation. Initial constructor,
  destructor, centroid verification and reference search are outside timing. The first
  untimed load warms the filesystem cache. Each subsequent load verifies every centroid
  and hashes the same eight search results.
- Python times conversion with its owned output and Python call overhead, for native
  uint32 and int64 arrays. It excludes input generation and prints the helper's path/hash.
  Its `hash` column is only the returned final ID, not a full-array correctness proof;
  the binding regression tests provide boundary coverage. It does not measure full builds.
- C++ results are hashed outside timing; each case has one untimed warmup. IPO is disabled
  for these targets; do not inject manual LTO flags. Do not run builds/profilers during timing.

Use an identical compiler, Release flags, ISA policy, inputs, affinity and thread placement
for before/after. Native tuning is optional for local experiments and must be identical on
both sides; portable correctness validation uses native OFF. Record CPU, compiler/version,
flags, revisions, OpenMP settings and NUMA policy. Pin memory as well as CPU **before loading**
on NUMA hosts. Available ISA metadata does not prove all backends executed.

## Alternating comparisons and retained evidence

`compare.py` runs commands without a shell, alternates process order, checks matching CSV
output fingerprints, and saves every stdout/stderr file plus status and medians. It requires
a new output directory so earlier failures are not overwritten. For example:

```bash
python benchmarks/hotpaths/compare.py \
  --before './before/benchmarks/hotpaths/rabitq_fastscan_hotpaths 128 50000000 3 hacc' \
  --after './after/benchmarks/hotpaths/rabitq_fastscan_hotpaths 128 50000000 3 hacc' \
  --pairs 3 --output "$TMPDIR/hacc-comparison" \
  --notes 'before/after revisions; CPU; compiler/version; flags; threads=1; CPU/memory binding'
```

Launch the runner under your host's CPU/memory binding tool; children inherit its placement.
The notes are required protocol metadata, not automatic verification of NUMA placement.
Review quality metadata and individual repetitions alongside medians. A hash difference stops
comparison for investigation; it does not prove a numerical bug across different compilers/ISAs.
Repeated matching hashes also do not establish correctness by themselves. Retain failed runs,
report regressions, and do not infer index QPS gains from these microbenchmarks.
