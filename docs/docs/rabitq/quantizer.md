# Quantizer

For a new per-vector quantization integration, start with the
[stable core API v1](core_api.md#minimal-example). This page is an advanced
reference for the existing encoding interfaces and five data formats below.

| Your task | Start here |
| --- | --- |
| Encode and score individual vectors in your own storage | [Stable core API v1](core_api.md): caller-owned buffers, separate factors and a source-stability contract |
| Score batches with FastScan | [FastScan integration](../cpp_integration.md#fastscan-integration) and the batch formats below |
| Integrate existing scalar, split-code or record layouts | [Data format reference](#data-format) below; these advanced interfaces are outside the v1 stability guarantee |
| Build a complete vector-search index | [Quick start](../quick_start.md) |

These data formats map floating-point vectors into `uint8` or `uint32`
codes. For the packed extended-code layouts, see
`rabitqlib/core/pack_excode.hpp`.

The examples below focus on layouts and omit rotation. In an index, rotate data,
centroids, and queries with the same rotator first and use its padded dimension.
Single-vector query snippets use `rabitqlib/core/estimator.hpp` and link
`rabitqlib::core`; batch-query snippets use `rabitqlib/fastscan/estimator.hpp` and link
`rabitqlib::fastscan`. Both batch formats use `fastscan/encoder.hpp` and
`fastscan/data_layout.hpp`; their 32-vector packing is in `fastscan/pack_codes.hpp`.
The old `quantization/` and generic `index/` forwarding paths have been removed.
See the [migration guide](../modular_migration.md) and
[modular C++ integration](../cpp_integration.md#core-only-c-integration).

## Implementations

Let $B$ be the total bits per dimension. Encoding produces two parts:

1. A **sign code**: one bit per coordinate.
2. An **extra-bit code**: the remaining $B-1$ bits when $B>1$.

The internal sign encoder is in `core/detail/rabitq_impl.hpp`. Use the public
entry points in `core/encoder.hpp` or `fastscan/encoder.hpp`; implementation
headers are not a stable integration surface.

Choose between two extra-code encoding paths:

The default path searches the configured rescaling interval
for the best quantized direction, following the RaBitQ [SIGMOD paper](https://doi.org/10.1145/3725413)
(Section 3.2.2). The current implementation can use SIMD to evaluate and prune
intervals; a scalar event sweep remains the reference and fallback when the SIMD
search cannot certify a winner. This path preserves the reference tie and rounding rules.

The fast path reuses a calibrated scaling factor instead of searching per vector.
The factor is the average of the default path's scaling factors over sampled
uniformly distributed unit vectors. `faster_config` reads it from a frozen,
precomputed table for padded dimensions that are multiples of 32 up to 65536 and
total bit widths 2 through 9, so the same arguments give the same factor on every
platform, standard library and CPU, at no setup cost. Other dimensions compute the
factor when called from `std::normal_distribution` samples, which differ between
standard libraries (MSVC and libc++ results differ from libstdc++ by several percent)
and, in the last bits, between FMA and non-FMA builds; calibrate once and reuse the
result for the same padded dimension and total bit width. This
reduces encoding work at a possible accuracy cost; evaluate it on your data.

## Data format

### Format 1: Scalar codes

Use scalar codes when your application manages its own storage. The encoder
returns unpacked integer codes, a scale and an offset; packing is separate.

```cpp
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/core/encoder.hpp"

int main() {
    size_t dim = 768;

    std::vector<float> vector(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dist(gen);
    }

    size_t bits = 8;                  // num of bits for total code
    std::vector<uint32_t> code(dim);  // code
    float delta;                      // delta for scalar quantization
    float vl;                         // lower value for scalar quantization

    // scalar quantization
    rabitqlib::quant::quantize_scalar(vector.data(), dim, bits, code.data(), delta, vl);

    // faster version, must init a config struct first
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);
    rabitqlib::quant::quantize_scalar(
        vector.data(), dim, bits, code.data(), delta, vl, config
    );

    // Note that with this interface, the codes are not compactly stored.
    // For packed storage, see
    // `rabitqlib/core/pack_excode.hpp`

    return 0;
}
```

### Format 2: Single-vector scoring

#### Indexing

Encode one vector into unpacked integer codes and three distance-estimation
factors. The application stores the codes and factors separately or together.

```cpp
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "rabitqlib/core/encoder.hpp"

int main() {
    size_t dim = 768;

    std::vector<float> vector(dim);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dist(gen);
        centroid[i] = dist(gen);
    }

    size_t bits = 8;                  // num of bits for total code
    std::vector<uint32_t> code(dim);  // code
    float f_add;                      // factors for estimating similarity
    float f_rescale;                  // factors for estimating similarity
    float f_error;                    // factors for computing error bounds

    rabitqlib::quant::quantize_full_single(
        vector.data(),
        centroid.data(),
        dim,
        bits,
        code.data(),
        f_add,
        f_rescale,
        f_error,
        rabitqlib::METRIC_L2
    );

    // faster version, must init a config struct first
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);
    rabitqlib::quant::quantize_full_single(
        vector.data(),
        centroid.data(),
        dim,
        bits,
        code.data(),
        f_add,
        f_rescale,
        f_error,
        rabitqlib::METRIC_L2,
        config
    );

    // Note that with this interface, the codes are not compactly stored.
    // For packed storage, see
    // `rabitqlib/core/pack_excode.hpp`

    return 0;
}
```

The advanced encoders check a few preconditions and throw
`std::invalid_argument` when they fail: encoders that produce error factors
require `dim` of at least 2 (the error factor divides by `dim - 1`),
`quantize_full_single` requires a code element type that holds `bits` (nine
bits need `uint16_t`), and `SplitSingleQuery` needs a padded dimension that is
a positive multiple of 32. `quantize_scalar` does not check the code element
type, so a type that is too narrow silently produces wrong codes. None of the
per-vector encoders checks that inputs are finite; validate each batch before
encoding, as the stable API and the indexes do.

#### Querying

Score a query using the encoded vector and its factors.
Insert the following before `return 0` in the indexing example above.

```cpp
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
        query[i] = dist(gen);
    }

    // See the Estimation guide for these query factors.
    float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
    float g_k1xsumq = c_1 * std::accumulate(query.begin(), query.end(), 0.F);
    float g_add = rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim);

    float estimated_distance = rabitqlib::quant::full_est_dist(
        code.data(),
        query.data(),
        rabitqlib::excode_ipimpl::ip_fxi,
        dim,
        bits,
        f_add,
        f_rescale,
        g_add,
        g_k1xsumq
    );
    float gt_dist = rabitqlib::euclidean_sqr(query.data(), vector.data(), dim);

    std::cout << "Estimated distance: " << estimated_distance << '\n';
    std::cout << "GT distance: " << gt_dist << '\n';

```

### Format 3: Batch scoring (QG)

#### Indexing

Encode up to 32 vectors for batched one-bit distance estimation with
[FastScan](https://arxiv.org/abs/1704.07355).
SymphonyQG and clustering use the two-factor `QGBatchDataMap` layout from
`rabitqlib/fastscan/data_layout.hpp`. Its encoder lives in
`rabitqlib/fastscan/encoder.hpp`; both are owned by the FastScan component.

```cpp
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"

int main() {
    size_t dim = 768;
    size_t batch_size = 32;  // a batch for FastScan contains 32 vectors

    std::vector<float> vector(dim * batch_size);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < batch_size * dim; ++i) {
        vector[i] = dist(gen);
    }
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = dist(gen);
    }

    std::vector<char> batch_data(rabitqlib::QGBatchDataMap<float>::data_bytes(dim));
    rabitqlib::quant::quantize_qg_batch(
        vector.data(), centroid.data(), batch_size, dim, batch_data.data(),
        rabitqlib::METRIC_L2
    );

    return 0;
}
```

#### Querying

Prepare the query lookup tables once, then score each encoded batch.
Insert the following before `return 0` in the indexing example, reusing its
initialized `batch_data`, `centroid`, `dim`, and `batch_size`.

```cpp

    std::vector<float> query(dim, 1.0F);

    rabitqlib::BatchQuery<float> query_wrapper(query.data(), dim);

    // g_add is the squared L2 distance between the query and the centroid.
    // This standalone example omits rotation, so both operands are raw vectors.
    // In an index that applies rotation, rotate both with the same transformation.
    query_wrapper.set_g_add(
        rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim)
    );

    std::vector<float> estimated_distances(batch_size);  // store the estimated distances

    // QG scoring returns estimates only. Use the IVF-RaBitQ batch format below
    // when error intervals are needed.

    rabitqlib::qg_batch_estdist(
        batch_data.data(), query_wrapper, dim, estimated_distances.data()
    );

```

<a id="format-4-split-single-vector-scoring-hnsw"></a>

### Format 4: Split single-vector scoring (HNSW-RaBitQ)

#### Indexing

Store a vector as separate sign and extra-bit codes, with factors for each.
Score the sign code first, then refine selected candidates using the extra bits.
HNSW-RaBitQ uses this layout.

```cpp
#include <cmath>
#include <random>
#include <vector>

#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/estimator.hpp"

int main() {
    size_t dim = 768;

    std::vector<float> vector(dim);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dist(gen);
        centroid[i] = dist(gen);
    }

    size_t bits = 5;  // num of bits for total code
    // Binary codes and three factors; extended codes and two factors.
    std::vector<char> bin_data(rabitqlib::BinDataMap<float>::data_bytes(dim));
    std::vector<char> ex_data(rabitqlib::ExDataMap<float>::data_bytes(dim, bits - 1));

    rabitqlib::quant::quantize_split_single(
        vector.data(),
        centroid.data(),
        dim,
        bits - 1,
        bin_data.data(),
        ex_data.data(),
        rabitqlib::METRIC_L2
    );

    // use fast implementation for the data format
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);
    rabitqlib::quant::quantize_split_single(
        vector.data(),
        centroid.data(),
        dim,
        bits - 1,
        bin_data.data(),
        ex_data.data(),
        rabitqlib::METRIC_L2,
        config
    );

    return 0;
}
```

#### Querying

Insert this before `return 0` in the indexing example, reusing its data and dimensions.

```cpp
    std::vector<float> query(dim);
    for (auto& value : query) {
        value = dist(gen);
    }

    // Queries use four bits, independently of the data code width.
    rabitqlib::quant::RabitqConfig query_config = rabitqlib::quant::faster_config(
        dim, rabitqlib::SplitSingleQuery<float>::kNumBits
    );

    rabitqlib::SplitSingleQuery<float> query_wrapper(
        query.data(), dim, bits - 1, query_config, rabitqlib::METRIC_L2
    );

    // This standalone example omits rotation, so query and centroid are both raw.
    // If rotation is used, apply the same transformation to both operands.
    float distance_sqr = rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim);
    float distance_norm = std::sqrt(distance_sqr);

    // Compute estimated distances based on binary codes
    float ip_x0_qr;
    float estimated_distance;
    float lower_bound;

    rabitqlib::split_single_estdist(
        bin_data.data(),
        query_wrapper,
        dim,
        ip_x0_qr,
        estimated_distance,
        lower_bound,
        distance_sqr,
        distance_norm
    );

    // the kernel of computing inner product between compact codes and query vectors
    auto ip_func = rabitqlib::select_excode_ipfunc(bits - 1);

    // Compute more accurate distance based on full codes.
    float estimated_distance_ex;
    float lower_bound_ex;
    float ip_x0_qr_ex;

    rabitqlib::split_single_fulldist(
        bin_data.data(),
        ex_data.data(),
        ip_func,
        query_wrapper,
        dim,
        bits - 1,
        estimated_distance_ex,
        lower_bound_ex,
        ip_x0_qr_ex,
        distance_sqr,
        distance_norm
    );
```

<a id="format-5-split-batch-scoring-ivf"></a>

### Format 5: Split batch scoring (IVF-RaBitQ)

#### Indexing

Batch the sign codes for FastScan and store each vector's extra-bit codes
separately. IVF-RaBitQ uses this layout to filter a batch before refining candidates.

```cpp
#include <cmath>
#include <random>
#include <vector>

#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"

int main() {
    size_t dim = 768;
    size_t batch_size = 32;

    std::vector<float> vector(dim * batch_size);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = dist(gen);
    }
    for (size_t i = 0; i < dim * batch_size; ++i) {
        vector[i] = dist(gen);
    }

    size_t bits = 5;  // num of bits for full codes
    // BatchDataMap includes physical storage for all 32 vectors, even in a tail batch.
    std::vector<char> batch_data(rabitqlib::BatchDataMap<float>::data_bytes(dim));
    std::vector<char> ex_data(
        rabitqlib::ExDataMap<float>::data_bytes(dim, bits - 1) * batch_size
    );

    rabitqlib::quant::quantize_split_batch(
        vector.data(),
        centroid.data(),
        batch_size,
        dim,
        bits - 1,
        batch_data.data(),
        ex_data.data(),
        rabitqlib::METRIC_L2
    );

    // use fast implementation for the data format
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);
    rabitqlib::quant::quantize_split_batch(
        vector.data(),
        centroid.data(),
        batch_size,
        dim,
        bits - 1,
        batch_data.data(),
        ex_data.data(),
        rabitqlib::METRIC_L2,
        config
    );

    return 0;
}
```

#### Querying

Insert this before `return 0` in the indexing example, reusing its buffers and dimensions.

```cpp
    std::vector<float> query(dim);
    for (auto& value : query) {
        value = dist(gen);
    }

    // The flag use_hacc controls the precision of FastScan.
    // `use_hacc = false` - each number in LUTs is quantized into 8 bits.
    // `use_hacc = true` - each number in LUTs is quantized into 16 bits.
    // SplitBatchQuery defaults to HACC; this example selects it explicitly.
    // IVF-RaBitQ instead selects HACC for 4-9 total bits and standard FastScan for
    // 1-3 total bits or raw-vector storage, unless the caller overrides it.

    rabitqlib::SplitBatchQuery<float> query_wrapper(
        query.data(), dim, bits - 1, rabitqlib::METRIC_L2, true
    );

    // SplitBatchQuery expects the L2 norm and squares it internally for g_add.
    // This example omits rotation, so query and centroid are both raw vectors.
    float distance_norm =
        std::sqrt(rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim));
    query_wrapper.set_g_add(distance_norm);

    std::vector<float> estimated_distances(batch_size);  // store the estimated distances
    std::vector<float> lower_bounds(batch_size);  // store the lower bound
    std::vector<float> ip_x0_qr(batch_size
    );  // store the intermediate result of inner product

    // For estimates without error intervals, see the QG batch format above.

    rabitqlib::split_batch_estdist(
        batch_data.data(),
        query_wrapper,
        dim,
        estimated_distances.data(),
        lower_bounds.data(),
        ip_x0_qr.data(),
        true
    );

    // the kernel of computing inner product between compact codes and query vectors
    auto ip_func = rabitqlib::select_excode_ipfunc(bits - 1);

    size_t i = 15;
    // Replace this candidate's one-bit estimate with the refined distance.
    estimated_distances[i] = rabitqlib::split_distance_boosting(
        ex_data.data() + (i * rabitqlib::ExDataMap<float>::data_bytes(dim, bits - 1)),
        ip_func,
        query_wrapper,
        dim,
        bits - 1,
        ip_x0_qr[i]
    );

```

## Batch API contracts

Batch query preparation and scoring require a positive dimension divisible by 16.
Index pipelines use multiples of 32, which also satisfy the packed extra-code
requirements. Use the same working dimension for encoding, query preparation and
scoring; none of these batch calls rotates or pads an input vector for you.

Single-batch encoders accept 0 through 32 logical vectors. Zero vectors leave the
output unchanged; more than 32 throws `std::invalid_argument` before writing.
For larger inputs, iterate in chunks of `fastscan::kBatchSize`, as shown in
`sample/cpp/fastscan` in the same checkout. Every batch, including a tail, needs
full `BatchDataMap<T>` or `QGBatchDataMap<T>` storage and 32 score outputs. Read only
its logical results. Buffers belong to the caller; encoders retain no input pointers.
`quantize_one_batch<T, true>` uses OpenMP when enabled and otherwise runs serially.
Allocation exceptions propagate to the caller, including from worker threads; a
failed encoding may leave partial output, so retry or discard that batch.

`BatchQuery<T>` and QG encoding/scoring support `float` and `double`. LUT preparation
uses the same scalar type as the query; float uses SIMD dispatch, while double
uses generic arithmetic without narrowing to float. Byte LUTs still quantize the
query, so double does not make estimates exact. Split batch scoring and
`SplitSingleQuery<float>` use float32 query paths; the latter does not support
`SplitSingleQuery<double>`.

A default `BatchQuery<T>` must be prepared with `reset` before scoring. Its reset
scratch needs **`padded_dim * 4` elements of `T`**, not bytes. Reset borrows query,
scratch and optional precomputed correction only for the call; it owns the resulting
LUT and clears `g_add`. Reuse the scratch between calls, then call `set_g_add` for
the current centroid. `BatchQuery::set_g_add` takes squared L2 distance, or
`1 - dot(query, centroid)` for IP.

`SplitBatchQuery` owns its LUT but borrows the rotated query for later refinement;
keep that vector alive and unchanged. Its `set_g_add` takes the **L2 norm** of
`query - centroid` for either metric, plus `dot(query, centroid)` for IP. All operands
must use the same rotated, padded domain. The standalone FastScan example demonstrates
both metrics with a nonzero centroid.

For repeated queries, default-construct `SplitBatchQuery<T>` and use `reset`.
LUT scratch needs `padded_dim * 4` elements of `T`; HACC also needs that many
`uint16_t` elements (pass `nullptr` in standard mode). Scratch is call-only and
must not overlap other buffers. The query remains borrowed until the next successful
reset or destruction. `ex_bits` must be in `[0, 8]`.

Only first preparation or capacity growth may allocate. Success clears `g_add`
and `g_error`; call `set_g_add` again before scoring. Invalid arguments or allocation
failure preserve the previous state.

Single-vector estimators also have named-result functions (`estimate_split_single`,
`estimate_split_single_full`, `estimate_xy_base`, and `estimate_xy_full`) returning
`distance`, `lower_bound`, and `inner_product` fields instead of taking three output
references. Both forms use the same numerical implementation.

Scoring rejects unprepared queries, mismatched dimensions and mismatched HACC modes
with `std::invalid_argument` before touching output. Pass `prepared.padded_dim()`
and, for split scoring, `prepared.use_hacc()` to reuse the prepared shape. The batch
buffer must still contain codes encoded for that shape: raw buffers carry no metadata
or capacity, so the caller must allocate them using the corresponding data map.
