#pragma once

#include <cstddef>
#include <optional>

namespace rabitqlib::quant::rabitq_impl::ex_bits {

// Frozen fast-mode scales, so faster_config gives the same t_const on every platform,
// standard library, compiler and CPU. The table covers padded_dim in multiples of
// kFastScaleDimStep up to kFastScaleMaxDim and ex_bits in [1, kFastScaleMaxExBits].
// Values are const_scaling_factor(padded_dim, ex_bits) as computed once by
// scripts/generate_fast_scale_table.cpp; see src/core/fast_scale_table.cpp. The
// table is frozen data: never regenerate or edit it.
inline constexpr size_t kFastScaleDimStep = 32;
inline constexpr size_t kFastScaleMaxDim = 65536;
inline constexpr size_t kFastScaleMaxExBits = 8;

// Returns the frozen scale, or no value when (padded_dim, ex_bits) is outside the table.
std::optional<double> frozen_const_scaling_factor(
    size_t padded_dim, size_t ex_bits
) noexcept;

}  // namespace rabitqlib::quant::rabitq_impl::ex_bits
