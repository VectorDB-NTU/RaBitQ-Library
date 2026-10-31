# Stable core quantization API v1

This interface is introduced in [RaBitQ Library 0.7.0](../releases/0.7.0.md).

Include `rabitqlib/core/api.hpp` and link `rabitqlib::core`. The stable C++17 source
interface lives in `rabitqlib::quant::v1`; it uses float32 inputs, caller-owned buffers
and separate factor values. The public header does not include Eigen, index containers,
record maps, or ISA-specific interfaces.

This interface reuses the existing RaBitQ encoder and full-distance formula. It does
not rotate inputs, create a thread pool, or take ownership of application memory.
These entry points execute serially. Core/FastScan can be built without OpenMP using
`RABITQ_BUILD_INDEX=OFF` and `RABITQ_USE_OPENMP=OFF`; the default full build retains it.

## Minimal example

The call sequence is `requirements` → `encode` → `prepare_query` → `compute`.
This complete example generates 128-dimensional data and a query with a fixed
seed, applies the same seeded [FHT/Kac rotation](rotator.md) to both, and uses
unpacked four-bit codes with a zero centroid. It compares the estimated squared
L2 distance with the true distance computed from the original vectors.
Rotation is explicit caller-side preprocessing; the v1 API does not perform it.

```cpp
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <rabitqlib/core/api.hpp>
#include <rabitqlib/core/rotator.hpp>

int main() {
    namespace api = rabitqlib::quant::v1;
    constexpr size_t kDim = 128;
    constexpr size_t kTotalBits = 4;  // Bits per dimension: 1 sign bit + 3 extra bits.
    std::mt19937 generator(42);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> input(kDim), query(kDim);
    for (size_t i = 0; i < kDim; ++i) {
        input[i] = normal(generator);
        query[i] = normal(generator);
    }

    std::unique_ptr<rabitqlib::Rotator<float>> rotator(rabitqlib::choose_rotator<float>(
        kDim, rabitqlib::RotatorType::FhtKacRotator, 0, 42
    ));
    std::vector<float> rotated_input(rotator->size()), rotated_query(rotator->size());
    rotator->rotate(input.data(), rotated_input.data());
    rotator->rotate(query.data(), rotated_query.data());
    // The zero centroid stays zero under rotation.
    const api::Parameters parameters{rotator->size(), kTotalBits, rabitqlib::METRIC_L2};
    // Each 4-bit code occupies a uint16_t here; SplitPacked stores codes compactly.
    const auto layout = api::CodeLayout::UnpackedU16;
    const auto sizes = api::requirements(parameters, layout);

    // Allocate during setup, with space to satisfy the reported alignment.
    std::vector<std::byte> codes(sizes.code_bytes);
    std::vector<std::byte> scratch(sizes.workspace_bytes + sizes.workspace_alignment - 1);
    void* workspace = scratch.data();
    size_t available = scratch.size();
    std::align(sizes.workspace_alignment, sizes.workspace_bytes, workspace, available);

    const auto factors = api::encode(
        parameters, rotated_input.data(), nullptr,
        {layout, {codes.data(), codes.size()}, {}},
        {workspace, sizes.workspace_bytes}
    );
    // QueryState borrows rotated_query; keep it alive and unchanged through compute().
    const auto state = api::prepare_query(parameters, rotated_query.data(), nullptr);
    const float distance = api::compute(
        state, {layout, {codes.data(), codes.size()}, {}}, factors
    );
    double true_distance = 0.0;
    for (size_t i = 0; i < kDim; ++i) {
        const double difference = static_cast<double>(input[i]) - query[i];
        true_distance += difference * difference;
    }
    std::cout << "Estimated squared L2 distance: " << distance << '\n'
              << "True squared L2 distance: " << true_distance << '\n';
}
```

Link this program to `rabitqlib::core`; see
[C++ integration](../cpp_integration.md#core-only-c-integration) for setup.
The application owns all storage. `encode` uses the workspace only during its
call; scoring needs the codes, factors and live query, but not that workspace.
For a nonzero centroid, rotate it with the same rotator before encoding and
query preparation. Fixed seeds make repeated runs reproducible within a given
toolchain; random-distribution implementations and floating-point reductions may
differ across platforms. This single pair illustrates usage, not a quality benchmark.
Reuse allocated storage for subsequent calls with the same requirements.
For packed codes, use `SplitPacked` and provide both buffers described under
[code layouts and sizes](#code-layouts-and-sizes).

## Stability policy

`RABITQLIB_QUANT_API_VERSION_MAJOR` is `1` and `RABITQLIB_QUANT_API_VERSION_MINOR` is
`0`. API v1.0 is independent of the package's 0.7.0 release version.

### What remains stable

Within API major 1, existing declarations and documented behavior remain
source-compatible. This covers names, signatures, argument order/defaults,
aggregate fields, enum values, supported formats/ranges, code interpretation,
metrics, ownership and allocation guarantees.

Minor revisions may add functions or separate types, but cannot remove or
repurpose v1 declarations, including deprecated ones, or add aggregate fields.
Breaking changes require a new API major and migration guidance.

### Compatibility limits

This is **source compatibility**, not a C ABI or a cross-version C++ binary ABI. Rebuild
consumers when upgrading the library/toolchain. Do not serialize C++ structs or query
state using `sizeof`/`memcpy`. Workspace sizes may change: query them again during setup,
and do not persist or hardcode them. Compiler warnings from future compilers are not
part of the compatibility guarantee.

The stable surface is the v1 namespace, its version macros, and the existing
`MetricType` / `METRIC_L2` / `METRIC_IP` values used by it. Other core templates,
`detail/`, SIMD entry points, rotator implementations, FastScan and ANN
interfaces are not implicitly frozen by including this header. Existing interfaces
remain available through canonical module headers; old forwarding paths are removed as
described in the [migration guide](../modular_migration.md). Advanced interfaces have
operation-specific allocation and threading behavior; the v1 guarantees below do not
automatically apply to them.

## Data and numerical contract

`Parameters` contains:

| Field | Meaning |
| --- | --- |
| `dim` | Number of coordinates actually supplied, at least 2; no implicit padding |
| `total_bits` | 1 through 9, including the sign bit |
| `metric` | `METRIC_L2` or `METRIC_IP` |
| `t_const` | Finite, at most `INT_MAX / 2`; nonpositive selects the existing exact search, positive selects the existing fast scaling algorithm |

Inputs and a non-null centroid contain at least `dim` aligned float32 values. Pass
`nullptr` for an all-zero centroid. Data, query and centroid must use the same transformed
coordinate system. If preserving distances from an original coordinate system, use a
compatible metric-preserving rotation/padding pipeline. The quantizer does not infer
or validate the transform. The quality assumptions of RaBitQ still apply; arbitrary
preprocessing does not acquire a quality guarantee merely by satisfying the API types.

Values must be finite and intermediate computations representable in float32. Use the
usual round-to-nearest floating-point environment. No implicit normalization, NaN repair,
precision-mode change or approximate-search fallback is performed. The exact encoder
keeps its SIMD-certified search and original exact event-sweep fallback. A positive
`t_const` explicitly selects the pre-existing speed/accuracy tradeoff; the API does not
choose it automatically.

`Factors {add, rescale, error}` are the values returned by the existing full-code encoder.
They may be stored in separate columns, inside user records, or passed as values. No
byte offset relative to the code is prescribed. `error` is the existing full-code error
factor, not a guaranteed deterministic lower bound; `compute` does not use it for pruning.

`prepare_query` borrows the supplied float query and computes metric/centroid terms.
It does not quantize or copy the query and needs no workspace. The centroid is read
synchronously and not retained. The query must remain alive and unchanged while any
returned/copied `QueryState` is used. Default-constructed state is invalid until assigned
from `prepare_query`. To change a query or centroid, prepare a new state.

For logical integer codes `u[i]` and `B = total_bits`, the full estimator is:

```text
L2: g_add = squared_norm(query - centroid)
IP: g_add = -dot(query, centroid)

distance = factors.add + g_add
         + factors.rescale * (dot(query, u) - ((2^B - 1) / 2) * sum(query))
```

L2 results estimate squared Euclidean distance and are not clamped to zero. IP results
estimate `1 - dot(data, query)`, not the raw dot product. Codes, factors and query state
must agree on dimension, bit width, metric and centroid; that metadata is not embedded
in the buffers. This is full float-query estimation even at one bit, not the legacy
four-bit query approximation used by `split_single_estdist` filtering.

Different layouts/ISAs can use different float reduction orders. Bit-identical distances
across layouts, compilers or CPUs are not promised. Numerical checks must account for
absolute summands and cancellation, rather than using only a relative error against a
possibly near-zero result. Layout conversion itself preserves every logical integer
code exactly and does not change factors.

## Code layouts and sizes

Call `requirements(parameters, layout)` during setup. It returns code/extra buffer sizes
and the workspace size/alignment. Capacities may be larger; only required prefixes are
accessed. Code buffers support byte alignment, including unaligned fields in user records.
The workspace must satisfy the reported alignment.

| Layout | `codes` | `extra` | Dimension requirement |
| --- | --- | --- | --- |
| `UnpackedU16` | `2 * dim` bytes; one little-endian unsigned 16-bit value per coordinate, in `[0, 2^B)` | Unused | Any `dim >= 2` |
| `SplitPacked` | `dim / 8` bytes of sign codes in the existing full-64/final-32 word order | `dim * (B-1) / 8` bytes in the existing extra-code packing; unused at one bit | Multiple of 32 |

For `SplitPacked`, sign and extra buffers can be anywhere in memory; they need not be
adjacent. No factor/header is appended. Bit/coordinate order matches existing library
pack/unpack kernels. The existing packed convention targets supported little-endian
x86-64/ARM64 systems. `UnpackedU16` byte order is explicitly little-endian, independent
of the alignment of the supplied byte buffer.

`convert_codes` converts either layout to either layout with the same dimension/bit width.
It does not rotate, requantize, pad, or alter factors. Source and destination must not
overlap, even for the same layout. To use an application-specific layout, consume logical
codes or convert explicitly; the API does not interpret arbitrary external bit fields.
Packing/unpacking should normally happen during ingestion or construction, not for every
candidate distance. Packed scoring uses the existing SIMD kernels; unpacked scoring is
a general float32 reference path and is not promised equal throughput.

## Ownership, overlap and allocation

`Buffer` and `ConstBuffer` are non-owning pointer/capacity pairs. Requirements do not
allocate. `encode` and `convert_codes` consume an aligned caller workspace synchronously
and retain no pointers to it. All writable code regions and the workspace must be mutually
disjoint and disjoint from their inputs, using the required ranges rather than surplus
capacity. Read-only input/centroid regions may alias each other. In-place encoding and
conversion are not supported. Unused `extra` buffers are ignored.

On valid successful calls, **requirements, encode, convert_codes, prepare_query and
compute make no heap allocations**, including their first call, larger dimensions within
the supplied workspace, and exact fallback searches. There is no warmup requirement,
TLS growth, hidden vector allocation, or heap fallback for an undersized workspace.
The guarantee assumes supported standard-library/runtime configurations, not instrumentation
that allocates on every operation.

Allocate application storage and initialize any rotation/configuration objects before
entering the hot path. Allocation failures during setup and allocation by exception
machinery for invalid calls are outside this guarantee. Other numerical APIs have their own allocation behavior; for example,
`MatrixRotator` uses temporary storage. Callers may supply their own rotation pipeline.

Concurrent encodes/conversions use separate workspaces and writable outputs. Read-only
codes, factors, parameters and prepared query state can be shared; callers synchronize
mutations and ensure borrowed query lifetimes.

## Errors

| Error type | Conditions |
| --- | --- |
| `std::invalid_argument` | Invalid dimensions/bit width/metric/layout/scaling constant; null/short/misaligned buffers; overlapping writable regions; non-finite input/factors; invalid query state or logical code value |
| `std::length_error` | Dimension/index or required byte counts cannot be represented |
| `std::overflow_error` | Required float32 norms, factors or estimated result are not finite |

Input pointers must reference accessible live storage; capacity claims do not let the
library detect dangling pointers or fictitious allocations. Error categories are stable;
exact diagnostic wording is not. Validation/arithmetic errors leave encoded output and
conversion destination bytes unchanged; workspace contents are unspecified after any call.

## Example and verification

The standalone example (`sample/cpp/stable_core` in the same source checkout)
uses only `api.hpp`, application-owned records/factor columns, a 65-coordinate working
space and an external transform. Its sign/permutation transform demonstrates integration;
it is not a representative random-rotation quality benchmark.

```bash
cmake -S sample/cpp/stable_core -B build-stable-example \
  -DCMAKE_PREFIX_PATH="<core-install-prefix>"
cmake --build build-stable-example
ctest --no-tests=error --test-dir build-stable-example --output-on-failure
```

The frozen v1 consumer (`tests/fixtures/core_api_v1/consumer.cpp`) is compiled
against source-tree and installed packages in CI. The internal workspace resource
uses only caller memory with no upstream allocator. A separate test checks the
FHT rotator's first `rotate` call after construction for allocations.

The unit suite covers bits 1–9, both metrics, dimensions around packing boundaries,
non-padded logical codes, zero residuals, buffer guards, code/factor parity, legacy packed
estimation and independent workspaces on concurrent callers. Separate allocation test
processes exercise normal dispatch and forced scalar fallback without warming the API.
They intercept C++ allocation on all tested platforms and also C allocation entry points
on Linux. Existing index, persistence and Python tests protect the shared implementation.

Zero allocation does not imply zero call overhead. The
[scoring performance reference](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/benchmarks/core/README.md#scoring-call-overhead)
explains the call implementation, comparison protocol and numerical checks for hot,
sequential and random code access. Scoring throughput does not predict end-to-end
ANN query performance.
