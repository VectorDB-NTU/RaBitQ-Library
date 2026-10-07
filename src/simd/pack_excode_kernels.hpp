#pragma once

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rabitqlib::simd::detail {

template <size_t Bits>
inline void pack_excode_tail_intrinsics(const uint8_t* raw, uint8_t* compact, size_t dim) {
    static_assert(Bits == 2 || Bits == 3 || Bits == 5 || Bits == 6 || Bits == 7);
    if (dim == 0)
        return;
    // A partial block is always 32 coordinates.
    const __m128i pair_weights =
        _mm_set1_epi16(static_cast<int16_t>(1U | (1U << (Bits + 8))));
    const __m128i group_weights =
        _mm_set1_epi32(static_cast<int32_t>(1U | (1U << (2 * Bits + 16))));
    alignas(16) static constexpr auto kPositions = [] {
        std::array<uint8_t, 16> positions{};
        for (size_t i = 0; i < 16; ++i) {
            positions[i] = static_cast<uint8_t>(
                Bits % 2 == 0 ? (i < 2 * Bits ? (i / (Bits / 2)) * 4 + i % (Bits / 2) : 128)
                              : (i < Bits ? i : (i < 2 * Bits ? 8 + i - Bits : 128))
            );
        }
        return positions;
    }();
    const __m128i shuffle =
        _mm_load_si128(reinterpret_cast<const __m128i*>(kPositions.data()));
    for (size_t i = 0; i < 32; i += 16) {
        const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + i));
        // Codes are at most 127: signed byte operands are nonnegative, and
        // even a pair of 7-bit codes fits without saturating the 16-bit sum.
        const __m128i pairs = _mm_maddubs_epi16(pair_weights, values);
        const __m128i groups = _mm_madd_epi16(pairs, group_weights);
        __m128i packed = groups;
        // Even widths already form whole-byte groups of four codes; the shuffle
        // compacts those bytes directly. Odd widths need bitwise concatenation.
        if constexpr (Bits % 2 != 0) {
            packed = _mm_or_si128(
                _mm_and_si128(groups, _mm_set1_epi64x(0xFFFFFFFFLL)),
                _mm_slli_epi64(_mm_srli_epi64(groups, 32), 4 * Bits)
            );
        }
        packed = _mm_shuffle_epi8(packed, shuffle);
        // Write the packed bytes in exact 8/4/2-byte pieces.
        if constexpr (Bits >= 4) {
            _mm_storel_epi64(reinterpret_cast<__m128i*>(compact), packed);
        }
        if constexpr (Bits == 2 || Bits == 3) {
            _mm_storeu_si32(compact, packed);
        } else if constexpr (Bits == 6 || Bits == 7) {
            _mm_storeu_si32(compact + 8, _mm_srli_si128(packed, 8));
        }
        if constexpr (Bits % 2 != 0) {
            const auto last = static_cast<uint16_t>(_mm_extract_epi16(packed, Bits - 1));
            std::memcpy(compact + 2 * Bits - 2, &last, sizeof(last));
        }
        compact += 2 * Bits;
    }
}

// Pack the selected bit from eight rows of eight bytes, preserving column order.
template <size_t Bit>
inline uint64_t pack_excode_high_bit(const uint8_t* raw) {
    static_assert(Bit == 2 || Bit == 4 || Bit == 6);
    const __m256i mask = _mm256_set1_epi64x(0x0101010101010101LL);
    const __m256i low = _mm256_and_si256(
        _mm256_srli_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(raw)), Bit),
        mask
    );
    const __m256i high = _mm256_and_si256(
        _mm256_srli_epi64(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(raw + 32)), Bit
        ),
        mask
    );
    const __m256i pairs = _mm256_or_si256(low, _mm256_slli_epi64(high, 4));
    const __m256i placed = _mm256_sllv_epi64(pairs, _mm256_setr_epi64x(0, 1, 2, 3));
    const __m128i folded =
        _mm_or_si128(_mm256_castsi256_si128(placed), _mm256_extracti128_si256(placed, 1));
    return static_cast<uint64_t>(
        _mm_cvtsi128_si64(_mm_or_si128(folded, _mm_srli_si128(folded, 8)))
    );
}

template <auto pack_tail = pack_excode_tail_intrinsics<2>>
inline void packing_2bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
    // Full blocks preserve the existing byte layout.
    for (size_t j = 0; j < dim - dim % 64; j += 64) {
        // pack 64 2-bit codes into 128 bits (16 bytes)
        // the lower 2 bits of each byte represent vec00 to vec04...
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, _mm_slli_epi16(vec_16_to_31, 2)),
            _mm_or_si128(_mm_slli_epi16(vec_32_to_47, 4), _mm_slli_epi16(vec_48_to_63, 6))
        );

        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

        o_raw += 64;
        o_compact += 16;
    }
    pack_tail(o_raw, o_compact, dim % 64);
}

inline void packing_3bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
    // Full blocks preserve the existing byte layout.
    const __m128i mask = _mm_set1_epi8(0b11);
    for (size_t d = 0; d < dim - dim % 64; d += 64) {
        // split 3-bit codes into 2 bits and 1 bit
        // for 2-bit part, compact it like 2-bit code
        // for 1-bit part, compact 64 1-bit code into a int64
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));
        const uint64_t top_bit = pack_excode_high_bit<2>(o_raw);

        vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
        vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2);
        vec_32_to_47 = _mm_slli_epi16(_mm_and_si128(vec_32_to_47, mask), 4);
        vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 6);

        __m128i compact2 = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, vec_16_to_31),
            _mm_or_si128(vec_32_to_47, vec_48_to_63)
        );

        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact2);
        o_compact += 16;

        // from lower to upper, each bit in each byte represents vec00 to vec07,
        // ..., vec56 to vec63
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_raw += 64;
        o_compact += 8;
    }
    pack_excode_tail_intrinsics<3>(o_raw, o_compact, dim % 64);
}

// Each 16-coordinate group stores the first eight codes in the low nibbles
// and the next eight in the high nibbles. Load whole groups before packing.
template <size_t Dim>
inline void pack_4bit_block(const uint8_t* raw, uint8_t* compact) {
    static_assert(Dim == 16 || Dim == 32 || Dim == 64 || Dim == 128);
    if constexpr (Dim == 128) {
        const __m512i lo = _mm512_loadu_si512(raw);
        const __m512i hi = _mm512_loadu_si512(raw + 64);
        const __m512i even =
            _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), hi);
        const __m512i odd =
            _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), hi);
        _mm512_storeu_si512(compact, _mm512_or_si512(even, _mm512_slli_epi64(odd, 4)));
    } else if constexpr (Dim == 64) {
        const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(raw));
        const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(raw + 32));
        const __m256i even = _mm256_permute4x64_epi64(_mm256_unpacklo_epi64(lo, hi), 0xD8);
        const __m256i odd = _mm256_permute4x64_epi64(_mm256_unpackhi_epi64(lo, hi), 0xD8);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(compact),
            _mm256_or_si256(even, _mm256_slli_epi64(odd, 4))
        );
    } else if constexpr (Dim == 32) {
        const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw));
        const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + 16));
        const __m128i even = _mm_unpacklo_epi64(lo, hi);
        const __m128i odd = _mm_unpackhi_epi64(lo, hi);
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(compact), _mm_or_si128(even, _mm_slli_epi64(odd, 4))
        );
    } else {
        uint64_t lo, hi;
        std::memcpy(&lo, raw, sizeof(lo));
        std::memcpy(&hi, raw + 8, sizeof(hi));
        const uint64_t packed = lo | (hi << 4);
        std::memcpy(compact, &packed, sizeof(packed));
    }
}

template <size_t BlockDim>
inline void packing_4bit_excode_intrinsics(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    static_assert(BlockDim == 64 || BlockDim == 128);
    for (; dim >= BlockDim; dim -= BlockDim) {
        pack_4bit_block<BlockDim>(raw, compact);
        raw += BlockDim;
        compact += BlockDim / 2;
    }
    if constexpr (BlockDim == 128) {
        if (dim >= 64) {
            pack_4bit_block<64>(raw, compact);
            raw += 64;
            compact += 32;
            dim -= 64;
        }
    }
    if (dim >= 32) {
        pack_4bit_block<32>(raw, compact);
        raw += 32;
        compact += 16;
        dim -= 32;
    }
    if (dim != 0) {
        pack_4bit_block<16>(raw, compact);
    }
}

inline void packing_5bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
    // Full blocks preserve the existing byte layout.
    const __m128i mask = _mm_set1_epi8(0b1111);
    for (size_t j = 0; j < dim - dim % 64; j += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));
        const uint64_t top_bit = pack_excode_high_bit<4>(o_raw);

        vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
        vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 4);
        vec_32_to_47 = _mm_and_si128(vec_32_to_47, mask);
        vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 4);

        __m128i compact4_1 = _mm_or_si128(vec_00_to_15, vec_16_to_31);
        __m128i compact4_2 = _mm_or_si128(vec_32_to_47, vec_48_to_63);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact4_1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 16), compact4_2);

        o_compact += 32;

        // from lower to upper, each bit in each byte represents vec00 to vec07,
        // ..., vec56 to vec63
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_raw += 64;
        o_compact += 8;
    }
    pack_excode_tail_intrinsics<5>(o_raw, o_compact, dim % 64);
}

template <auto pack_tail = pack_excode_tail_intrinsics<6>>
inline void packing_6bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
    // for vec00 to vec47, split code into 6
    // for vec48 to vec63, split code into 2 + 2 + 2
    // Complementary masks let AVX512 fuse each bit selection into one instruction.
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    for (size_t d = 0; d < dim - dim % 64; d += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            _mm_and_si128(vec_00_to_15, mask6),
            _mm_andnot_si128(mask6, _mm_slli_epi16(vec_48_to_63, 6))
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_16_to_31, mask6),
            _mm_andnot_si128(mask6, _mm_slli_epi16(vec_48_to_63, 4))
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 16), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_32_to_47, mask6),
            _mm_andnot_si128(mask6, _mm_slli_epi16(vec_48_to_63, 2))
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 32), compact);
        o_compact += 48;
        o_raw += 64;
    }
    pack_tail(o_raw, o_compact, dim % 64);
}

inline void packing_7bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
    // for vec00 to vec47, split code into 6 + 1
    // for vec48 to vec63, split code into 2 + 2 + 2 + 1
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    for (size_t d = 0; d < dim - dim % 64; d += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));
        const uint64_t top_bit = pack_excode_high_bit<6>(o_raw);

        __m128i compact = _mm_or_si128(
            _mm_and_si128(vec_00_to_15, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 6), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_16_to_31, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 4), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 16), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_32_to_47, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 32), compact);
        o_compact += 48;

        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_compact += 8;
        o_raw += 64;
    }
    pack_excode_tail_intrinsics<7>(o_raw, o_compact, dim % 64);
}

}  // namespace rabitqlib::simd::detail
