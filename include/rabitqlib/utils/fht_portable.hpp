#pragma once
#include <cstddef>

// Same unnormalized Sylvester Hadamard transform as upstream's x86 assembly.
template <std::size_t LogN>
inline void portable_hadamard(float* data) {
    constexpr std::size_t n = 1UL << LogN;
    for (std::size_t half = 1; half < n; half *= 2) {
        for (std::size_t base = 0; base < n; base += 2 * half) {
            for (std::size_t i = 0; i < half; ++i) {
                const float a = data[base + i], b = data[base + half + i];
                data[base + i] = a + b;
                data[base + half + i] = a - b;
            }
        }
    }
}
inline void helper_float_6(float* p) { portable_hadamard<6>(p); }
inline void helper_float_7(float* p) { portable_hadamard<7>(p); }
inline void helper_float_8(float* p) { portable_hadamard<8>(p); }
inline void helper_float_9(float* p) { portable_hadamard<9>(p); }
inline void helper_float_10(float* p) { portable_hadamard<10>(p); }
inline void helper_float_11(float* p) { portable_hadamard<11>(p); }
