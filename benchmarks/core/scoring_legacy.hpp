#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/utils/space.hpp"

namespace scoring_benchmark {
namespace api = rabitqlib::quant::v1;

struct Query {
    const float* values;
    size_t dim;
    size_t bits;
    float add;
    float half_sum;
    float (*extra_ip)(const float*, const uint8_t*, size_t);
};

// Same full-distance expression and existing kernels, with no public-API checks.
// The benchmark always supplies validated buffers; this is not a library entry point.
inline float unchecked_inline(
    const Query& query, api::CodeView code, const api::Factors& f
) {
    const auto* bytes = static_cast<const uint8_t*>(code.codes.data);
    float ip = 0;
    if (code.layout == api::CodeLayout::UnpackedU16) {
        for (size_t i = 0; i < query.dim; ++i) {
            const uint16_t value =
                static_cast<uint16_t>(bytes[2 * i] | (uint16_t{bytes[2 * i + 1]} << 8U));
            ip += query.values[i] * static_cast<float>(value);
        }
    } else {
        const size_t extra_bits = query.bits - 1;
        ip = static_cast<float>(1U << extra_bits) *
             rabitqlib::mask_ip_x0_q(query.values, bytes, query.dim);
        if (extra_bits > 0) {
            ip += query.extra_ip(
                query.values, static_cast<const uint8_t*>(code.extra.data), query.dim
            );
        }
    }
    return rabitqlib::quant::detail::full_est_dist_from_ip(
        ip, query.bits, f.add, f.rescale, query.add, query.half_sum
    );
}

// Compiled in a separate translation unit, without LTO, to retain a call boundary.
float unchecked_call(const Query& query, api::CodeView code, const api::Factors& f);
float unchecked_ref(const Query& query, const api::CodeView& code, const api::Factors& f);
}  // namespace scoring_benchmark
