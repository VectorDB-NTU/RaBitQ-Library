#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/core/metric.hpp"  // IWYU pragma: export

#define RABITQLIB_QUANT_API_VERSION_MAJOR 1
#define RABITQLIB_QUANT_API_VERSION_MINOR 0

namespace rabitqlib::quant::v1 {

// Stable C++ source API. This header exposes no index, Eigen, or SIMD implementation.
// See docs/docs/rabitq/core_api.md for the complete ownership/numerical contract.
enum class CodeLayout : std::uint8_t { UnpackedU16 = 0, SplitPacked = 1 };

struct Parameters {
    // Working dimension >= 2; SplitPacked additionally requires dim % 32 == 0.
    size_t dim = 0;
    size_t total_bits = 1;  // One sign bit plus 0..8 magnitude bits.
    MetricType metric = METRIC_L2;
    double t_const = -1;  // <= 0: existing exact search; > 0: existing fast quantizer.
};

struct Buffer {
    void* data = nullptr;
    size_t bytes = 0;
};

struct ConstBuffer {
    const void* data = nullptr;
    size_t bytes = 0;
};

struct CodeOutput {
    CodeLayout layout = CodeLayout::UnpackedU16;
    Buffer codes;
    Buffer extra;  // Independent storage; used only by SplitPacked with total_bits > 1.
};

struct CodeView {
    CodeLayout layout = CodeLayout::UnpackedU16;
    ConstBuffer codes;
    ConstBuffer extra;
};

// Values, not a record format. Callers may store these in separate columns/records.
struct Factors {
    float add = 0;
    float rescale = 0;
    float error = 0;
};

struct Requirements {
    size_t code_bytes;
    size_t extra_bytes;
    size_t workspace_bytes;
    size_t workspace_alignment;
};

class QueryState;

Requirements requirements(const Parameters& parameters, CodeLayout layout);

// Input/centroid contain dim float32 values. nullptr centroid means zero.
// Output and workspace must be disjoint from inputs and each other.
Factors encode(
    const Parameters& parameters,
    const float* input,
    const float* centroid,
    CodeOutput output,
    Buffer workspace
);

// Lossless code conversion; factors are unchanged. In-place conversion is not supported.
void convert_codes(
    const Parameters& parameters, CodeView source, CodeOutput destination, Buffer workspace
);

// Borrows the transformed query without copying it. It must remain alive and unchanged
// while the returned/copied state is used. The centroid is not retained.
QueryState prepare_query(
    const Parameters& parameters, const float* query, const float* centroid
);

// Full float-query estimation, including at one bit; no quantized-query filter is implied.
inline float compute(const QueryState& query, CodeView code, const Factors& factors);

class QueryState {
   public:
    QueryState() = default;
    [[nodiscard]] size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] size_t total_bits() const noexcept { return total_bits_; }
    [[nodiscard]] MetricType metric_type() const noexcept { return metric_; }

   private:
    const float* query_ = nullptr;
    size_t dim_ = 0;
    size_t total_bits_ = 0;
    MetricType metric_ = METRIC_L2;
    float g_add_ = 0;
    float k1xsumq_ = 0;
    float (*extra_ip_)(const float*, const uint8_t*, size_t) = nullptr;

    static float compute_checked(
        const QueryState& query, const CodeView& code, const Factors& factors
    );

    friend QueryState prepare_query(const Parameters&, const float*, const float*);
    friend float compute(const QueryState&, CodeView, const Factors&);
};

inline float compute(const QueryState& query, CodeView code, const Factors& factors) {
    // Keep the public value semantics; the compiled implementation borrows this
    // local view instead of passing the entire aggregate through its call ABI.
    return QueryState::compute_checked(query, code, factors);
}

}  // namespace rabitqlib::quant::v1
