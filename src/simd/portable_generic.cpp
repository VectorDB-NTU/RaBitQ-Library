#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/rotator_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/bitops.hpp"

namespace rabitqlib::simd {
namespace {
// Coordinate-to-bit mapping of the existing x86 packed format. Odd widths
// store the high bit in eight bytes, with coordinate i at byte i%8, bit i/8.
template <size_t Bits>
size_t packed_bit(size_t i, size_t bit) {
    if constexpr (Bits == 1) {
        return i;
    } else if constexpr (Bits == 8) {
        return i * 8 + bit;
    } else if constexpr (Bits == 4) {
        return (i / 16) * 64 + (i % 8) * 8 + ((i % 16) / 8) * 4 + bit;
    } else {
        const size_t block = (i / 64) * Bits * 64;
        i %= 64;
        if constexpr (Bits % 2 == 1) {
            if (bit == Bits - 1) {
                return block + (Bits - 1) * 64 + (i % 8) * 8 + i / 8;
            }
        }
        if constexpr (Bits <= 3) {
            return block + (i % 16) * 8 + (i / 16) * 2 + bit;
        } else if constexpr (Bits == 5) {
            return block + (i / 32) * 128 + (i % 16) * 8 + ((i / 16) % 2) * 4 + bit;
        } else {
            return block +
                   (i < 48 ? i * 8 + bit : (bit / 2) * 128 + (i % 16) * 8 + 6 + bit % 2);
        }
    }
}

template <size_t Bits>
void pack(const uint8_t* raw, uint8_t* compact, size_t dim) {
    std::fill_n(compact, dim * Bits / 8, uint8_t{0});
    for (size_t i = 0; i < dim; ++i) {
        for (size_t b = 0; b < Bits; ++b) {
            const size_t pos = packed_bit<Bits>(i, b);
            compact[pos / 8] |= ((raw[i] >> b) & 1U) << (pos % 8);
        }
    }
}

template <size_t Bits>
float excode_ip(const float* query, const uint8_t* compact, size_t dim) {
    double sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        unsigned code = 0;
        for (size_t b = 0; b < Bits; ++b) {
            const size_t pos = packed_bit<Bits>(i, b);
            code |= ((compact[pos / 8] >> (pos % 8)) & 1U) << b;
        }
        sum += static_cast<double>(query[i]) * code;
    }
    return static_cast<float>(sum);
}

template <typename T>
void quantize(T* result, const float* data, size_t dim, float lo, float delta) {
    const float reciprocal = 1.0F / delta;
    for (size_t i = 0; i < dim; ++i) {
        result[i] = static_cast<T>(std::round((data[i] - lo) * reciprocal));
    }
}

void fht(float* data, size_t dim) {
    for (size_t width = 1; width < dim; width *= 2) {
        for (size_t block = 0; block < dim; block += width * 2) {
            for (size_t j = 0; j < width; ++j) {
                const float a = data[block + j], b = data[block + j + width];
                data[block + j] = a + b;
                data[block + j + width] = a - b;
            }
        }
    }
}
}  // namespace

#define RABITQ_PACK(Bits)                                \
    void packing_##Bits##bit_excode_generic(             \
        const uint8_t* raw, uint8_t* compact, size_t dim \
    ) {                                                  \
        pack<Bits>(raw, compact, dim);                   \
    }
RABITQ_PACK(2)
RABITQ_PACK(3)
RABITQ_PACK(4)
RABITQ_PACK(5)
RABITQ_PACK(6)
RABITQ_PACK(7)
#undef RABITQ_PACK

namespace excode_ipimpl {
#define RABITQ_IP(Block, Bits)                              \
    float ip##Block##_fxu##Bits##_generic(                  \
        const float* query, const uint8_t* code, size_t dim \
    ) {                                                     \
        return excode_ip<Bits>(query, code, dim);           \
    }
RABITQ_IP(16, 1)
RABITQ_IP(64, 2)
RABITQ_IP(64, 3)
RABITQ_IP(16, 4)
RABITQ_IP(64, 5)
RABITQ_IP(64, 6)
RABITQ_IP(64, 7)
RABITQ_IP(16, 8)
#undef RABITQ_IP
}  // namespace excode_ipimpl

void scalar_quantize_uint8_generic(
    uint8_t* result, const float* data, size_t dim, float lo, float delta
) {
    quantize(result, data, dim, lo, delta);
}
void scalar_quantize_uint16_generic(
    uint16_t* result, const float* data, size_t dim, float lo, float delta
) {
    quantize(result, data, dim, lo, delta);
}

void new_transpose_bin_generic(
    const uint16_t* query, uint64_t* transposed, size_t dim, size_t bits
) {
    for (size_t block = 0; block < dim; block += 64) {
        for (size_t b = 0; b < bits; ++b) {
            uint64_t word = 0;
            for (size_t i = 0; i < 64; ++i) {
                word |= uint64_t{(query[block + i] >> b) & 1U} << (63 - i);
            }
            *transposed++ = word;
        }
    }
}

void new_transpose_bin_512_generic(
    const uint8_t* query, uint64_t* transposed, size_t dim, size_t bits
) {
    for (size_t block = 0; block < dim; block += 512) {
        const size_t chunks = std::min(size_t{512}, dim - block) / 64;
        for (size_t b = 0; b < bits; ++b) {
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                uint64_t word = 0;
                for (size_t i = 0; i < 64; ++i) {
                    word |= uint64_t{(query[block + chunk * 64 + i] >> b) & 1U} << (63 - i);
                }
                *transposed++ = word;
            }
        }
    }
}

float mask_ip_x0_q_generic(const float* query, const uint8_t* data, size_t dim) {
    double sum = 0;
    for (size_t block = 0; block < dim; block += 64) {
        uint64_t word;
        std::memcpy(&word, data + block / 8, sizeof(word));
        for (size_t i = 0; i < 64; ++i) {
            if ((word >> (63 - i)) & 1U)
                sum += query[block + i];
        }
    }
    return static_cast<float>(sum);
}
float mask_ip_x0_q_generic(const float* query, const uint64_t* data, size_t dim) {
    return mask_ip_x0_q_generic(query, reinterpret_cast<const uint8_t*>(data), dim);
}

float warmup_ip_x0_q_512_generic(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t dim,
    size_t bits
) {
    if (bits > 8)
        throw std::invalid_argument("warmup_ip_x0_q_512 requires at most 8 query bits");
    uint64_t ip = 0, count = 0;
    for (size_t block = 0; block < dim; block += 512) {
        const size_t chunks = std::min(size_t{512}, dim - block) / 64;
        for (size_t chunk = 0; chunk < chunks; ++chunk) {
            uint64_t word;
            std::memcpy(&word, data + block / 8 + chunk * 8, sizeof(word));
            count += bitops::popcount64(word);
            for (size_t b = 0; b < bits; ++b) {
                ip += uint64_t{bitops::popcount64(word & query[b * chunks + chunk])} << b;
            }
        }
        query += chunks * bits;
    }
    return delta * static_cast<float>(ip) + vl * static_cast<float>(count);
}
float warmup_ip_x0_q_512_generic(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t dim,
    size_t bits
) {
    return warmup_ip_x0_q_512_generic(
        reinterpret_cast<const uint8_t*>(data), query, delta, vl, dim, bits
    );
}

void flip_sign_generic(const uint8_t* flip, float* data, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        if ((flip[i / 8] >> (i % 8)) & 1U)
            data[i] = -data[i];
    }
}
void kacs_walk_generic(float* data, size_t len) {
    for (size_t i = 0; i < len / 2; ++i) {
        const float a = data[i], b = data[i + len / 2];
        data[i] = a + b;
        data[i + len / 2] = a - b;
    }
}
void fht_rotate_generic(
    const float* data,
    float* output,
    size_t dim,
    size_t padded,
    size_t trunc,
    float factor,
    const uint8_t* flip
) {
    if (trunc < 64 || trunc > 65536 || (trunc & (trunc - 1)) != 0) {
        throw std::invalid_argument("Unsupported dimension for FhtKacRotator");
    }
    std::memmove(output, data, dim * sizeof(float));
    std::fill(output + dim, output + padded, 0.0F);
    for (size_t pass = 0; pass < 4; ++pass) {
        flip_sign_generic(flip + pass * padded / 8, output, padded);
        const size_t start = pass % 2 == 0 ? 0 : padded - trunc;
        fht(output + start, trunc);
        for (size_t i = start; i < start + trunc; ++i)
            output[i] *= factor;
        if (padded != trunc)
            kacs_walk_generic(output, padded);
    }
    if (padded != trunc) {
        for (size_t i = 0; i < padded; ++i)
            output[i] *= 0.25F;
    }
}
}  // namespace rabitqlib::simd
