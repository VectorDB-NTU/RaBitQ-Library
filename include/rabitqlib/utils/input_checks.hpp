#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace rabitqlib {

// True when every float32 value is finite. The integer reduction lets portable
// builds check several values per instruction without relaxing floating-point
// compiler options, and never allocates. Internal linkage keeps every ISA-flagged
// translation unit that includes this header on its own copy.
static inline bool all_finite(const float* values, size_t count) {
    static_assert(
        sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559
    );
    uint32_t nonfinite = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, values + i, sizeof(bits));
        nonfinite |= static_cast<uint32_t>((bits & 0x7f800000U) == 0x7f800000U);
    }
    return nonfinite == 0;
}

template <typename T>
static inline bool all_finite(const T* values, size_t count) {
    size_t nonfinite = 0;
    for (size_t i = 0; i < count; ++i) {
        nonfinite += static_cast<size_t>(!std::isfinite(values[i]));
    }
    return nonfinite == 0;
}

// Largest coordinate magnitude whose squared distances, their products in graph
// pruning, and rotated residuals stay inside float32; shared with clustering.
static inline float max_coordinate_magnitude(size_t dim) {
    return static_cast<float>(std::sqrt(
        std::sqrt(static_cast<double>(std::numeric_limits<float>::max())) /
        (64.0 * static_cast<double>(dim))
    ));
}

// Reject vectors the indexes cannot quantize. `what` names the input in the message.
static inline void check_coordinates(
    const float* values, size_t count, size_t dim, const char* what
) {
    if (!all_finite(values, count)) {
        throw std::invalid_argument(std::string(what) + " must contain only finite values");
    }
    const float bound = max_coordinate_magnitude(dim);
    size_t out_of_range = 0;
    for (size_t i = 0; i < count; ++i) {
        out_of_range += static_cast<size_t>(!(std::fabs(values[i]) <= bound));
    }
    if (out_of_range != 0) {
        throw std::invalid_argument(
            std::string(what) + " exceeds the safe float32 coordinate range"
        );
    }
}

}  // namespace rabitqlib
