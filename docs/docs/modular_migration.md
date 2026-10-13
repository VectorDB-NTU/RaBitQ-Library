# Migrating to 0.7.0

This guide covers upgrading from **0.6.0 to 0.7.0**. See the
[release overview](releases/0.7.0.md).

The former full-library CMake alias and forwarding header paths are removed.
Version 0.7.0 introduces the stable float32 core API v1, reusing the existing
quantization algorithms and code interpretation. Persisted index formats remain
compatible with 0.6.0. Index count semantics are unified as described below.
Rebuild C++ consumers against 0.7.0; no cross-version C++ binary ABI is promised.

## Choose a component explicitly

```cmake
find_package(rabitqlib CONFIG REQUIRED COMPONENTS index)
target_link_libraries(my_app PRIVATE rabitqlib::index)
```

Replace `rabitqlib::rabitqlib` with the smallest component needed:

| Target | Contains | Dependency |
| --- | --- | --- |
| `rabitqlib::core` | Encoding, single-vector estimation, rotation and packing | Numerical helpers and essential SIMD |
| `rabitqlib::fastscan` | Batch layouts, packing/encoding adapters and FastScan/HACC scoring | core |
| `rabitqlib::index` | IVF, HNSW, SymphonyQG and clustering | fastscan and core |
| `rabitqlib::python` | Existing Python ANN extension; build-tree alias only | index |

The old implementation target `rabitq_headers` is also removed. C++ libraries remain
static. Python still uses `import rabitqlib`; it is not an exported C++ package component.

For a numerical library without ANN or OpenMP:

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_INDEX=OFF -DRABITQ_BUILD_FASTSCAN=OFF \
  -DRABITQ_USE_OPENMP=OFF -DRABITQ_BUILD_SAMPLES=OFF \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
  -DCMAKE_INSTALL_PREFIX="<new-empty-prefix>"
cmake --build build-core --parallel
cmake --install build-core
```

Enable FastScan separately if needed. ANN/Python still require OpenMP. A core-only
installation contains core and its header dependencies, including bundled Eigen;
it does not install FastScan, index, clustering or vendored hnswlib headers. Source-tree
presence of disabled modules does not make them part of the installed component.

Use a fresh install prefix during migration. CMake installation does not remove files
left by an older package, and stale headers can conceal a missing migration. Point each
consumer to the intended prefix and rebuild its CMake cache/objects accordingly.

## Replace old includes

Paths below are relative to `rabitqlib/`. Include the modules actually used instead of
relying on transitive declarations from a former umbrella header.

| Removed header | Replacement |
| --- | --- |
| `quantization/rabitq.hpp` | `core/encoder.hpp` for per-vector encoding; `fastscan/encoder.hpp` for batch encoding; `fastscan/fastscan.hpp` for accumulation/LUT functions |
| `quantization/data_layout.hpp` | `core/data_layout.hpp` for per-vector maps; `fastscan/data_layout.hpp` for `BatchDataMap`, `QGBatchDataMap` and their const variants |
| `quantization/pack_excode.hpp` | `core/pack_excode.hpp` |
| `utils/rotator.hpp` | `core/rotator.hpp` |
| `index/query.hpp` | `core/query.hpp` for single vectors; `fastscan/query.hpp` for batches |
| `index/estimator.hpp` | `core/estimator.hpp` for single vectors; `fastscan/estimator.hpp` for batches |
| `index/lut.hpp` | Use the batch-query interfaces in `fastscan/query.hpp` |
| `defines.hpp` | Include `core/metric.hpp`, `core/types.hpp` or `index/types.hpp` according to the types used; internal Eigen aliases live in `core/detail/eigen.hpp` |
| `core/pack_codes.hpp` | `fastscan/pack_codes.hpp` for 32-vector batch packing/unpacking |

New integrations should prefer [`core/api.hpp`](rabitq/core_api.md) and its versioned,
storage-neutral interface. Migrating an existing low-level caller to canonical headers
does not require changing its code bytes or factors. Batch scoring requires linking
FastScan; advanced core headers do not import that component for convenience.

## Public and implementation interfaces

### Encoding and shared types

The advanced two-factor batch adapter has also moved: include `fastscan/data_layout.hpp`
for `QGBatchDataMap` / `ConstQGBatchDataMap`, and `fastscan/encoder.hpp` for
`quant::quantize_qg_batch`. It is shared by FastScan, clustering and SymphonyQG, so it
does not belong exclusively to the index module. Its code packing, factor order and
record sizes are unchanged. The three-factor `BatchDataMap` / `ConstBatchDataMap` also
belong to `fastscan/data_layout.hpp`. Include `fastscan/encoder.hpp` for
`quantize_one_batch` and `quantize_split_batch`, and link `rabitqlib::fastscan` when
using this format. The fixed batch size, permutation and `pack_codes`/`unpack_codes`
are all in `fastscan/pack_codes.hpp`; no FastScan declarations remain in core.

Core retains single-vector `BinDataMap`, `ExDataMap` and `BaseDataMap`, per-vector
packing and generic encoding loops that do not impose a 32-vector format. Shared
unaligned value/array access resides in `utils/packed.hpp`. The v1 `SplitPacked` layout
is a per-vector sign/extra representation and remains part of core.

`index/types.hpp` supplies `PID`, `kPidMax` and `AnnCandidate` without importing Eigen.
Identifier width, candidate layout and marker semantics are unchanged. The advanced
`exact_nn` helper now requires `index/detail/exact_nn.hpp` instead of `utils/space.hpp`.
Metric definitions move to `core/metric.hpp`; `ScalarQuantizerType` now resides in
`core/types.hpp`. The unused `BIT_ID` macro and internal `LOWBIT` macro are removed.
The new `core/api.hpp` exposes the stable v1 interface; it was not present in 0.6.0.

Only `core/api.hpp`'s documented v1 surface has the new source-stability guarantee.
Other core/fastscan/index module interfaces remain available as advanced interfaces;
they are not implicitly frozen by that promise.

Query metric arguments use `MetricType`. `SplitSingleQuery` no longer accepts
integer arguments; pass `METRIC_L2` or `METRIC_IP`.

### Index headers

IVF, HNSW and SymphonyQG operations now compile into `rabitqlib::index`. Their
main headers no longer import numerical implementations, search scratch, Eigen
or hnswlib. This implementation move preserves method signatures and file formats.

Include the types you use explicitly:

| Types | Header |
| --- | --- |
| `RotatorType`, `quant::RabitqConfig`, scalar quantizer choices | `core/types.hpp` |
| `ivf::InitializerType` | `index/ivf/types.hpp` |
| SymphonyQG query and buffer helpers | `fastscan/query.hpp`, `index/detail/buffer.hpp`, `index/detail/visited_set.hpp` |
| Internal `SearchScratch` | `index/symqg/detail/search_scratch.hpp` |

Explicit includes of `core/rotator.hpp`, `core/encoder.hpp` and
`index/ivf/initializer.hpp` still expose their associated types. HNSW backend
search templates are private in `src/index/hnsw_search_impl.hpp`.

### Removed HNSW members

HNSW no longer exposes `HierarchicalNSW::rawDataPtr_`, `ResultRecord`, `Candidate`,
or `BoundedKNN` as public interfaces. The construction pointer and search helpers
are private implementation details. This is a C++ source compatibility break:
code that accessed those members must stop doing so. Pass construction input through
`construct()` and consume the distance/label pairs returned by `search()`; there are
no public replacements for the internal candidate types. Construction borrows its
input buffers only until the synchronous call returns or throws. This visibility
change does not alter persisted indexes or the Python API.

### Internal headers

Internal headers have explicit ownership:

| Implementation | Location |
| --- | --- |
| Exact encoder scratch and quantization implementation | `core/detail/` |
| Eigen aliases used by numerical implementations | `core/detail/eigen.hpp` |
| Core SIMD declarations | `core/detail/simd/` |
| FastScan SIMD declarations and LUT implementation | `fastscan/detail/` |
| FastScan batch-encoder adapter | `fastscan/detail/encoder.hpp` |
| HNSW dispatch, search buffers, visited sets and shared scheduling | `index/detail/` |
| Non-template dispatcher implementation and private resolver | `src/simd/`, `src/fastscan/` and `src/index/` |

The old `simd/*` public paths, `fastscan/{simd,estimator_dispatch,lut}.hpp`,
`quantization/rabitq_impl.hpp` and index-specific `utils/{buffer,hashset,visited_*}.hpp`
paths are removed. Callers using these internals should migrate to supported module
operations; direct use of `detail/` binds a caller to that implementation revision.
Some detail headers must be installed because public templates include them. Installation
does not turn them into a stable extension API.

HNSW and IVF share `index/detail/parallel_for.hpp`. Callers relying on HNSW
to import the IVF initializer must include it explicitly. Scheduling behavior
is unchanged; custom thread-pool and SIMD registration remain unsupported.

## Unified index counts

All three indexes now share the [count contract](index/usage.md#index-counts)
in C++ and Python.

Compared with the 0.6.0 Python API, IVF adds `num_points` and `num_stored_points`,
HNSW adds `num_stored_points`, and SymphonyQG adds `max_elements` and
`num_stored_points`.

**HNSW `num_points` now excludes removed records.** Use `num_stored_points` for
ID bounds, append offsets, and minimum resize capacity. Use `num_points` for a
live count, without subtracting a separate deletion counter.

The IVF constructor keyword `max_elements` still specifies the required initial
build size. SymphonyQG's C++ `num_vertices()` still returns the configured graph
size for builders. Existing index files need no conversion: loading recovers
counts from their removal markers.

## Downstream rollout

Pin 0.7.0 in downstream migration PRs and validate the affected integration
entry points with their maintainers.
Projects that have not migrated can continue pinning their supported old release.

The conformance suite preserves the frozen v1 source consumer and historical index
fixtures. It also verifies component link closure, absent retired targets/headers,
standalone installed headers, and independent add_subdirectory/FetchContent/package
consumers. Historical release notes and the v0.5.2 fixture generator deliberately retain
their old target names: those documents/tools describe or build the historical release.
