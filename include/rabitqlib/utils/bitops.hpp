#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace rabitqlib::bitops {

inline unsigned popcount32(uint32_t value) {
#if defined(_MSC_VER)
    value -= (value >> 1) & 0x55555555U;
    value = (value & 0x33333333U) + ((value >> 2) & 0x33333333U);
    return ((value + (value >> 4)) & 0x0F0F0F0FU) * 0x01010101U >> 24;
#else
    return static_cast<unsigned>(__builtin_popcount(value));
#endif
}

inline unsigned popcount64(uint64_t value) {
#if defined(_MSC_VER)
    value -= (value >> 1) & 0x5555555555555555ULL;
    value = (value & 0x3333333333333333ULL) + ((value >> 2) & 0x3333333333333333ULL);
    return static_cast<unsigned>(
        (((value + (value >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56
    );
#else
    return static_cast<unsigned>(__builtin_popcountll(value));
#endif
}

// Callers pass nonzero masks.
inline unsigned countr_zero32(uint32_t value) {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward(&index, value);
    return static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_ctz(value));
#endif
}

}  // namespace rabitqlib::bitops
