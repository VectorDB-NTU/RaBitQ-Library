#include "rabitqlib/utils/cpu_features.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

rabitqlib::cpu::Features all_hardware_features() {
    return {
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
    };
}

TEST(CpuFeatures, ReturnsStableResult) {
    const auto& first = rabitqlib::cpu::features();
    const auto& second = rabitqlib::cpu::features();

    ASSERT_EQ(&first, &second);
    ASSERT_EQ(
        rabitqlib::cpu::has_avx512_popcnt(),
        rabitqlib::cpu::has_avx512_core() && first.avx512vpopcntdq
    );
}

TEST(CpuFeatures, RequiresOsManagedAvxState) {
    constexpr uint64_t kAvxState = 0x6;
    const auto hardware = all_hardware_features();

    auto usable =
        rabitqlib::cpu::detail::filter_usable_features(hardware, true, false, kAvxState);
    EXPECT_FALSE(usable.avx2);
    EXPECT_FALSE(usable.fma);
    EXPECT_FALSE(usable.avx512f);

    usable =
        rabitqlib::cpu::detail::filter_usable_features(hardware, false, true, kAvxState);
    EXPECT_FALSE(usable.avx2);
    EXPECT_FALSE(usable.fma);
    EXPECT_FALSE(usable.avx512f);

    usable = rabitqlib::cpu::detail::filter_usable_features(hardware, true, true, 0);
    EXPECT_FALSE(usable.avx2);
    EXPECT_FALSE(usable.fma);
    EXPECT_FALSE(usable.avx512f);
}

TEST(CpuFeatures, EnablesAvx2WithoutAvx512State) {
    constexpr uint64_t kAvxState = 0x6;
    const auto usable = rabitqlib::cpu::detail::filter_usable_features(
        all_hardware_features(), true, true, kAvxState
    );

    EXPECT_TRUE(usable.avx2);
    EXPECT_TRUE(usable.fma);
    EXPECT_FALSE(usable.avx512f);
    EXPECT_FALSE(usable.avx512bw);
    EXPECT_FALSE(usable.avx512dq);
    EXPECT_FALSE(usable.avx512vpopcntdq);
    EXPECT_FALSE(usable.avx512vl);
    EXPECT_FALSE(usable.avx512cd);
}

TEST(CpuFeatures, EnablesAvx512OnlyWithCompleteAvx512State) {
    constexpr uint64_t kAvx512State = 0xe6;
    const auto usable = rabitqlib::cpu::detail::filter_usable_features(
        all_hardware_features(), true, true, kAvx512State
    );

    EXPECT_TRUE(usable.avx2);
    EXPECT_TRUE(usable.fma);
    EXPECT_TRUE(usable.avx512f);
    EXPECT_TRUE(usable.avx512bw);
    EXPECT_TRUE(usable.avx512dq);
    EXPECT_TRUE(usable.avx512vpopcntdq);
    EXPECT_TRUE(usable.avx512vl);
    EXPECT_TRUE(usable.avx512cd);

    for (const uint64_t state_bit :
         {uint64_t{1} << 5, uint64_t{1} << 6, uint64_t{1} << 7}) {
        const auto incomplete = rabitqlib::cpu::detail::filter_usable_features(
            all_hardware_features(), true, true, kAvx512State & ~state_bit
        );
        EXPECT_TRUE(incomplete.avx2);
        EXPECT_TRUE(incomplete.fma);
        EXPECT_FALSE(incomplete.avx512f);
        EXPECT_FALSE(incomplete.avx512bw);
        EXPECT_FALSE(incomplete.avx512dq);
        EXPECT_FALSE(incomplete.avx512vpopcntdq);
        EXPECT_FALSE(incomplete.avx512vl);
        EXPECT_FALSE(incomplete.avx512cd);
    }
}

TEST(CpuFeatures, Avx512SelectionRequiresEveryCompilerEnabledFeature) {
    using rabitqlib::cpu::Features;
    using rabitqlib::cpu::detail::supports_avx512_core;
    EXPECT_TRUE(supports_avx512_core(all_hardware_features()));
    for (auto feature :
         {&Features::avx2,
          &Features::fma,
          &Features::avx512f,
          &Features::avx512bw,
          &Features::avx512dq}) {
        auto missing = all_hardware_features();
        missing.*feature = false;
        EXPECT_FALSE(supports_avx512_core(missing));
    }
    for (auto feature : {&Features::avx512vl, &Features::avx512cd}) {
        auto missing = all_hardware_features();
        missing.*feature = false;
#if defined(_MSC_VER)
        EXPECT_FALSE(supports_avx512_core(missing));
#else
        EXPECT_TRUE(supports_avx512_core(missing));
#endif
    }
}

}  // namespace
