#pragma once

#include "rabitqlib/core/detail/simd/space_dispatch.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/detail/simd.hpp"

namespace rabitqlib::fastscan::detail {

// Standard LUTs use consecutive 16-entry tables on every backend. HACC stores
// low/high byte planes for two tables at a time (32 bytes) or four (64 bytes).
// This describes transient query scratch, never persisted codes.
enum class HaccLutLayout { Pair32, Quad64 };

// One immutable selection owns the producer/consumer pairing. Keep this private:
// no backend pointers or ISA-specific layouts become public query state.
struct Backend {
    const char* name;
    HaccLutLayout hacc_layout;
    decltype(&simd::pack_lut_generic) pack_lut;
    decltype(&simd::pack_lut_range_generic) pack_lut_range;
    decltype(&rabitqlib::simd::scalar_quantize_uint8_generic) quantize_lut;
    decltype(&simd::transfer_lut_hacc_generic) transfer_hacc;
    decltype(&simd::accumulate_generic) accumulate;
    decltype(&simd::accumulate_hacc_generic) accumulate_hacc;
    decltype(&rabitqlib::simd::qg_batch_estdist_generic) qg;
    decltype(&rabitqlib::simd::qg_batch_estdist_mask_generic) qg_mask;
    decltype(&rabitqlib::simd::split_batch_estdist_generic) split;
};

// Call only during wrapper initialization. Warm wrappers cache the member pointer
// locally, avoiding an extra getter or table lookup on every scored batch.
const Backend& selected_backend();

}  // namespace rabitqlib::fastscan::detail
