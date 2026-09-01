#include "rabitqlib/utils/cpu_features.hpp"

#include <gtest/gtest.h>

TEST(CpuFeatures, returns_stable_result) {
    const auto& first = rabitqlib::cpu::features();
    const auto& second = rabitqlib::cpu::features();

    ASSERT_EQ(&first, &second);
    ASSERT_EQ(
        rabitqlib::cpu::has_avx512_popcnt(),
        rabitqlib::cpu::has_avx512_core() && first.avx512vpopcntdq
    );
}
