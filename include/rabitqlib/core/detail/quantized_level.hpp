#pragma once

#include <algorithm>

namespace rabitqlib::quant::detail {

// Preconditions: magnitude >= 0 and finite; scale >= 0 and finite; max_code >= 0.
// Keep this helper local to each translation unit: callers use different ISA flags.
// Search evaluation and final encoding must use the same division-based thresholds.
static inline int quantized_level_at_scale(double magnitude, double scale, int max_code) {
    if (magnitude == 0)
        return 0;
    int code = static_cast<int>(std::min(scale * magnitude, static_cast<double>(max_code)));
    // Multiplication can round to either side of an event; correct against the
    // exact thresholds rather than adding an epsilon that could cross other events.
    if (code < max_code && (code + 1.0) / magnitude <= scale)
        ++code;
    else if (code > 0 && static_cast<double>(code) / magnitude > scale)
        --code;
    return code;
}

}  // namespace rabitqlib::quant::detail
