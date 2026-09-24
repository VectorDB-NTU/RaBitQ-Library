#pragma once

#include <cstdint>

namespace rabitqlib::cpu {

struct Features {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512dq = false;
    bool avx512vpopcntdq = false;
    bool avx512vl = false;
    bool avx512cd = false;
    bool neon = false;
};

const Features& features();
bool has_avx2();
bool has_neon();
bool has_avx512_core();
bool has_avx512_popcnt();

namespace detail {

bool supports_avx512_core(const Features& detected);

Features filter_usable_features(
    const Features& hardware, bool avx, bool osxsave, uint64_t xcr0
);

}  // namespace detail

}  // namespace rabitqlib::cpu
