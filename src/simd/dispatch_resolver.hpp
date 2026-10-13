#pragma once

#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::simd {
namespace {
// Resolve once on first use; function-local caches are safe before main().
// The override is used only by kernels needing a stricter AVX-512 subset.
template <typename Function>
Function resolve_kernel(
    Function avx512,
    Function avx2,
    Function fallback,
    bool avx512_supported = cpu::has_avx512_core()
) {
    if (avx512_supported) {
        return avx512;
    }
    if (cpu::has_avx2()) {
        return avx2;
    }
    return fallback;
}

template <typename Function>
Function resolve_kernel(Function avx2, Function fallback) {
    if (cpu::has_avx2()) {
        return avx2;
    }
    return fallback;
}

// Use a distinct name: a fallback function pointer can implicitly convert to bool.
// Overloading the x86 resolver would let MSVC select this overload when __restrict
// makes the SIMD and scalar function-pointer types differ.
template <typename Function>
Function resolve_optional_kernel(Function preferred, Function fallback, bool supported) {
    return supported ? preferred : fallback;
}

}  // namespace
}  // namespace rabitqlib::simd

#if defined(__GNUC__) || defined(__clang__)
#define RABITQ_COLD_INIT __attribute__((cold, noinline))
#else
#define RABITQ_COLD_INIT
#endif

// Non-x86 builds must not reference symbols from excluded AVX objects.
#if defined(__x86_64__) || defined(_M_X64)
// Keep feature probes out of the hot wrapper's register-allocation scope.
// The calling function's local static still provides thread-safe initialization.
#define RABITQ_RESOLVE(...) \
    [&]() RABITQ_COLD_INIT { return rabitqlib::simd::resolve_kernel(__VA_ARGS__); }()
#define RABITQ_RESOLVE2(...) RABITQ_RESOLVE(__VA_ARGS__)
#else
#define RABITQ_RESOLVE(avx512, avx2, fallback, ...) fallback
#define RABITQ_RESOLVE2(avx2, fallback) fallback
#endif
