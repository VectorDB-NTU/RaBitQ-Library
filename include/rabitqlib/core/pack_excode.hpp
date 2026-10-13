#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "rabitqlib/core/detail/simd/pack_excode_dispatch.hpp"

namespace rabitqlib::quant::rabitq_impl::ex_bits {
inline void packing_1bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 16 == 0
    for (size_t j = 0; j < dim; j += 16) {
        uint16_t code = 0;
        for (size_t i = 0; i < 16; ++i) {
            code |= static_cast<uint16_t>(o_raw[i]) << i;
        }
        std::memcpy(o_compact, &code, sizeof(uint16_t));

        o_raw += 16;
        o_compact += 2;
    }
}

inline void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_2bit_excode(o_raw, o_compact, dim);
}

inline void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_3bit_excode(o_raw, o_compact, dim);
}

inline void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_4bit_excode(o_raw, o_compact, dim);
}

inline void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_5bit_excode(o_raw, o_compact, dim);
}

inline void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_6bit_excode(o_raw, o_compact, dim);
}

inline void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_7bit_excode(o_raw, o_compact, dim);
}

inline void packing_8bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    std::memcpy(o_compact, o_raw, sizeof(uint8_t) * dim);
}

/**
 * @brief Pack one vector's extra-bit codes into dim * ex_bits / 8 bytes.
 * Layouts depend on bit width, not the selected SIMD backend. See
 * docs/docs/compact_code.md, "How to read the layouts" and "Final 32-coordinate
 * tail", for coordinate-to-bit mappings. The scalar mapping is implemented by
 * packed_bit and tail_code in src/simd/portable_generic.cpp.
 *
 * @param o_raw unpacked code, code for each dim is represented by uint8
 * @param o_compact compact format of code
 * @param dim dimension of code, a multiple of 32 for every supported bit width
 * @param ex_bits number of bits used for code
 */
inline void packing_rabitqplus_code(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim, size_t ex_bits
) {
    if (ex_bits == 1) {
        packing_1bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 2) {
        packing_2bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 3) {
        packing_3bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 4) {
        packing_4bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 5) {
        packing_5bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 6) {
        packing_6bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 7) {
        packing_7bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 8) {
        packing_8bit_excode(o_raw, o_compact, dim);
    } else {
        throw std::invalid_argument("ex_bits must be in [1, 8]");
    }
}

/**
 * @brief Inverse of packing_rabitqplus_code. Scalar for every width; the packed
 * layout does not depend on which backend wrote it.
 */
inline void unpacking_rabitqplus_code(
    const uint8_t* o_compact, uint8_t* o_raw, size_t dim, size_t ex_bits
) {
    ::rabitqlib::simd::unpacking_excode_generic(o_compact, o_raw, dim, ex_bits);
}
}  // namespace rabitqlib::quant::rabitq_impl::ex_bits
