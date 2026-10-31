// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace rabitqlib::fastscan {

inline constexpr size_t kBatchSize = 32;  // number of vectors in each batch

// Interleave lanes 0..7 and 8..15 for the 16-bit SIMD accumulators. Internal linkage
// lets hot kernels address the table directly instead of through the GOT.
static constexpr std::array<int, 16> kPerm0 = {
    0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

template <typename TP, typename Column>
static inline void get_column(
    const TP* src, size_t rows, size_t cols, size_t row, size_t col, Column& dest
) {
    size_t k = 0;
    size_t max_k = std::min(rows - row, dest.size());
    for (; k < max_k; ++k) {
        dest[k] = src[((k + row) * cols) + col];
    }
    if (k < dest.size()) {
        std::fill(dest.begin() + k, dest.end(), 0);
    }
}

/**
 * @brief Pack sign codes into 32-vector blocks, zero-filling missing tail lanes.
 * For each input byte column, output bytes j and j + 16 pair vectors kPerm0[j]
 * and kPerm0[j] + 16. The first output holds both inputs' high nibbles, the second
 * their low nibbles; the first vector always occupies the output's low nibble.
 *
 * @param padded_dim dimension of quantized data (i.e., quantization code)
 * @param quantization_code num_points rows of padded_dim / 8 compact sign-code bytes
 * @param num_points number of logical vectors
 * @param blocks output, ceil(num_points / 32) * 32 * padded_dim / 8 bytes
 */
inline void pack_codes(
    size_t padded_dim, const uint8_t* quantization_code, size_t num_points, uint8_t* blocks
) {
    // Treat the codes as a matrix: one row per vector, one column per code byte.
    size_t cols = padded_dim / 8;

    // Full batches can gather directly in output order, without a temporary column.
    size_t row = 0;
    for (; num_points - row >= kBatchSize; row += kBatchSize) {
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                const size_t first = row + kPerm0[j];
                const uint8_t a = quantization_code[first * cols + i];
                const uint8_t b = quantization_code[(first + 16) * cols + i];
                blocks[j] = (a >> 4) | (b & 0xF0);
                blocks[j + 16] = (a & 0x0F) | (b << 4);
            }
            blocks += 32;
        }
    }

    // Only the final partial batch needs a zero-padded column.
    if (row < num_points) {
        std::array<uint8_t, kBatchSize> col;
        for (size_t i = 0; i < cols; ++i) {
            get_column(quantization_code, num_points, cols, row, i, col);
            for (size_t j = 0; j < 16; ++j) {
                // the lower 4 bits represent vector 0 to 15
                // the upper 4 bits represent vector 16 to 31
                const uint8_t a = col[kPerm0[j]];
                const uint8_t b = col[kPerm0[j] + 16];
                blocks[j] = (a >> 4) | (b & 0xF0);
                blocks[j + 16] = (a & 0x0F) | (b << 4);
            }
            blocks += 32;
        }
    }
}

/**
 * @brief Inverse of pack_codes: recover the compact quantization codes of the first
 * `num_points` vectors from their packed blocks. Padding lanes of a partial final batch
 * are ignored.
 *
 * @param padded_dim dimension of quantized data (i.e., quantization code)
 * @param blocks packed quantization code, as produced by pack_codes
 * @param num_points number of quantization codes to recover
 * @param quantization_code recovered codes, num_points * padded_dim / 8 bytes
 */
inline void unpack_codes(
    size_t padded_dim, const uint8_t* blocks, size_t num_points, uint8_t* quantization_code
) {
    const size_t cols = padded_dim / 8;

    for (size_t row = 0; row < num_points; row += kBatchSize) {
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                // blocks[j] holds both input vectors' high nibbles; blocks[j + 16]
                // holds both low nibbles. Undo the pairing documented in pack_codes.
                const size_t first = row + kPerm0[j];
                const auto a = static_cast<uint8_t>(
                    ((blocks[j] & 0x0F) << 4) | (blocks[j + 16] & 0x0F)
                );
                const auto b =
                    static_cast<uint8_t>((blocks[j] & 0xF0) | (blocks[j + 16] >> 4));
                if (first < num_points) {
                    quantization_code[first * cols + i] = a;
                }
                if (first + 16 < num_points) {
                    quantization_code[(first + 16) * cols + i] = b;
                }
            }
            blocks += 32;
        }
    }
}

}  // namespace rabitqlib::fastscan
