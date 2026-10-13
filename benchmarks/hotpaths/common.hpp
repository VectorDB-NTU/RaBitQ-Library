#pragma once

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "rabitqlib/utils/cpu_features.hpp"

namespace bench {
using Clock = std::chrono::steady_clock;
inline volatile uint64_t sink = 0;

inline size_t number(const char* text) {
    size_t value = 0;
    const char* end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value == 0) {
        throw std::invalid_argument("Arguments must be positive decimal integers");
    }
    return value;
}

inline size_t product(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() / b) {
        throw std::length_error("Benchmark size overflow");
    }
    return a * b;
}

inline std::vector<float> values(size_t count, uint32_t seed = 42) {
    std::vector<float> result(count);
    for (auto& value : result) {
        seed = seed * 1664525U + 1013904223U;
        value = static_cast<float>(seed >> 8) / 8388608.0F - 1.0F;
    }
    return result;
}

inline uint64_t hash(
    const void* data, size_t bytes, uint64_t seed = 14695981039346656037ULL
) {
    const auto* source = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        seed = (seed ^ source[i]) * 1099511628211ULL;
    }
    return seed;
}

inline void metadata(size_t dim, size_t rows, size_t threads) {
    std::cout << std::setprecision(12) << "# dim=" << dim << " rows=" << rows
              << " threads=" << threads << " seed=42 metric=L2 openmp=" << RABITQ_USE_OPENMP
              << " avx2=" << rabitqlib::cpu::has_avx2()
              << " avx512=" << rabitqlib::cpu::has_avx512_core()
              << " neon=" << rabitqlib::cpu::has_neon() << '\n';
    std::cout << "operation,repetition,operations,seconds,ns_per_operation,hash\n";
}

inline void row(
    const char* name, size_t rep, size_t operations, double seconds, uint64_t h
) {
    sink = h;
    std::cout << name << ',' << rep << ',' << operations << ',' << seconds << ','
              << seconds * 1e9 / static_cast<double>(operations) << ',' << h << '\n';
}

// Hashing is deliberately outside the timed region. Each work invocation must
// leave observable results, and the benchmark targets are built without IPO.
template <typename Work, typename Checksum>
void measure(
    const char* name, size_t operations, size_t reps, Work work, Checksum checksum
) {
    work();
    const uint64_t expected = checksum();
    for (size_t rep = 0; rep < reps; ++rep) {
        const auto start = Clock::now();
        work();
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        const uint64_t actual = checksum();
        if (actual != expected) {
            throw std::runtime_error("Benchmark output changed between repetitions");
        }
        row(name, rep, operations, seconds, actual);
    }
}
}  // namespace bench
