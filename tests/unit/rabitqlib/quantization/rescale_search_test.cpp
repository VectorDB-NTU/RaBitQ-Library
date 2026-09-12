#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "rabitqlib/quantization/rabitq_impl.hpp"
#include "rabitqlib/simd/quantization_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::quant {
namespace {

using RescaleSearch = double (*)(const float*, size_t, int, double, double);

std::vector<int> levels_at_threshold(
    const std::vector<float>& magnitudes, double scale, int max_code
) {
    std::vector<int> levels(magnitudes.size());
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        if (magnitudes[i] == 0)
            continue;
        for (int level = 1; level <= max_code; ++level) {
            if (static_cast<double>(level) / magnitudes[i] <= scale)
                levels[i] = level;
        }
    }
    return levels;
}

long double squared_cosine(
    const std::vector<float>& magnitudes, const std::vector<int>& levels
) {
    long double numerator = 0;
    long double denominator = 0;
    long double input_norm = 0;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        const long double magnitude = magnitudes[i];
        const long double value = levels[i] + 0.5L;
        numerator += value * magnitude;
        denominator += value * value;
        input_norm += magnitude * magnitude;
    }
    return input_norm == 0 ? 0 : numerator * numerator / (denominator * input_norm);
}

long double exhaustive_best_squared_cosine(
    const std::vector<float>& magnitudes, int max_code, double start, double end
) {
    std::vector<int> levels = levels_at_threshold(magnitudes, start, max_code);
    std::vector<std::pair<double, size_t>> events;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        if (magnitudes[i] == 0)
            continue;
        for (int level = levels[i] + 1; level <= max_code; ++level) {
            const double threshold = static_cast<double>(level) / magnitudes[i];
            if (threshold < end)
                events.emplace_back(threshold, i);
        }
    }
    std::sort(events.begin(), events.end());
    long double best = squared_cosine(magnitudes, levels);
    for (size_t event = 0; event < events.size();) {
        const double threshold = events[event].first;
        do {
            ++levels[events[event++].second];
        } while (event < events.size() && events[event].first == threshold);
        // Recompute in long double rather than sharing the search's reductions
        // or accumulating floating-point error across thousands of events.
        best = std::max(best, squared_cosine(magnitudes, levels));
    }
    return best;
}

void normalize_magnitudes(std::vector<float>& magnitudes) {
    long double norm_squared = 0;
    for (float magnitude : magnitudes)
        norm_squared += static_cast<long double>(magnitude) * magnitude;
    if (norm_squared == 0)
        return;
    const long double norm = std::sqrt(norm_squared);
    for (float& magnitude : magnitudes)
        magnitude = static_cast<float>(magnitude / norm);
}

std::pair<double, double> search_interval(
    const std::vector<float>& magnitudes, size_t bits
) {
    const double maximum = *std::max_element(magnitudes.begin(), magnitudes.end());
    if (maximum == 0)
        return {0, 1};
    const double end = static_cast<double>((1 << bits) + 9) / maximum;
    return {end * rabitq_impl::ex_bits::kTightStart[bits], end};
}

void expect_optimal_canonical_scale(
    const std::vector<float>& magnitudes, int max_code, double start, double end, double t
) {
    ASSERT_TRUE(std::isfinite(t));
    ASSERT_GE(t, start);
    ASSERT_LT(t, end);
    const auto levels = levels_at_threshold(magnitudes, t, max_code);
    double first = start;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        if (levels[i] > 0)
            first = std::max(first, static_cast<double>(levels[i]) / magnitudes[i]);
    }
    EXPECT_EQ(t, first);
    const long double actual = squared_cosine(magnitudes, levels);
    const long double optimum =
        exhaustive_best_squared_cosine(magnitudes, max_code, start, end);
    // Float magnitudes times half-integer levels fit exactly in double. The
    // tolerance covers reduction/comparison rounding, without permitting a
    // meaningful loss in the optimized cosine objective.
    constexpr long double kTolerance = 8 * std::numeric_limits<double>::epsilon();
    EXPECT_GE(actual + kTolerance, optimum);
}

class RabitqRescaleBackendTest : public ::testing::TestWithParam<bool> {
   protected:
    void SetUp() override {
        if (GetParam() ? !cpu::has_avx512_core() : !cpu::has_avx2())
            GTEST_SKIP() << "Requested SIMD backend is unavailable";
    }

    RescaleSearch search() const {
        return GetParam() ? simd::best_rescale_factor_avx512
                          : simd::best_rescale_factor_avx2;
    }
};

TEST_P(RabitqRescaleBackendTest, MatchesExhaustiveObjectiveAndCanonicalThreshold) {
    for (size_t dim :
         {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 33U, 65U, 129U}) {
        for (int pattern = 0; pattern < 7; ++pattern) {
            std::vector<float> magnitudes(dim);
            std::mt19937 random(8143);
            std::normal_distribution<float> gaussian;
            for (size_t i = 0; i < dim; ++i) {
                if (pattern == 1)
                    magnitudes[i] = i == dim / 2 ? 1.0F : 0;
                else if (pattern == 2)
                    magnitudes[i] = static_cast<float>(i % 4);
                else if (pattern == 3)
                    magnitudes[i] = 1.0F + static_cast<float>(i % 7) *
                                               std::numeric_limits<float>::epsilon();
                else if (pattern == 4)
                    magnitudes[i] = i % 5 == 0 ? 0 : std::abs(gaussian(random));
                else if (pattern == 5)
                    magnitudes[i] = i + 1 == dim
                                        ? std::numeric_limits<float>::denorm_min()
                                        : std::ldexp(1.0F, -static_cast<int>(i % 150));
                else if (pattern == 6)
                    magnitudes[i] = std::abs(gaussian(random));
            }
            normalize_magnitudes(magnitudes);
            for (size_t bits = 1; bits <= 8; ++bits) {
                SCOPED_TRACE(
                    ::testing::Message()
                    << "dim=" << dim << " pattern=" << pattern << " bits=" << bits
                );
                const auto [start, end] = search_interval(magnitudes, bits);
                const int max_code = (1 << bits) - 1;
                const double t = search()(magnitudes.data(), dim, max_code, start, end);
                ASSERT_TRUE(std::isfinite(t));
                if (t >= 0)
                    expect_optimal_canonical_scale(magnitudes, max_code, start, end, t);
            }
        }
    }
}

TEST_P(RabitqRescaleBackendTest, CompletesGaussianSearchWithoutFallback) {
    std::vector<float> magnitudes(128);
    std::mt19937 random(42);
    std::normal_distribution<float> gaussian;
    for (float& magnitude : magnitudes)
        magnitude = std::abs(gaussian(random));
    normalize_magnitudes(magnitudes);
    const auto [start, end] = search_interval(magnitudes, 7);
    const double t = search()(magnitudes.data(), magnitudes.size(), 127, start, end);
    ASSERT_GE(t, 0);
    expect_optimal_canonical_scale(magnitudes, 127, start, end, t);
}

TEST_P(RabitqRescaleBackendTest, ReusesScratchAcrossDimensionsAndThreads) {
    std::vector<std::vector<float>> inputs;
    std::vector<std::pair<double, double>> intervals;
    std::vector<double> expected;
    const auto backend = search();
    std::mt19937 random(71);
    std::normal_distribution<float> gaussian;
    for (size_t dim : {129U, 7U, 257U, 1U, 65U, 513U, 3U, 128U, 0U, 17U}) {
        std::vector<float> magnitudes(dim);
        for (float& magnitude : magnitudes)
            magnitude = dim == 17 ? 0 : std::abs(gaussian(random));
        normalize_magnitudes(magnitudes);
        const auto interval =
            dim == 0 ? std::pair<double, double>{0, 1} : search_interval(magnitudes, 7);
        expected.push_back(
            backend(magnitudes.data(), dim, 127, interval.first, interval.second)
        );
        intervals.push_back(interval);
        inputs.push_back(std::move(magnitudes));
    }

    size_t mismatches = 0;
#pragma omp parallel for num_threads(8) schedule(dynamic, 1) reduction(+ : mismatches)
    for (size_t call = 0; call < 160; ++call) {
        const size_t index = call % inputs.size();
        const auto& magnitudes = inputs[index];
        const auto [start, end] = intervals[index];
        const double actual =
            backend(magnitudes.data(), magnitudes.size(), 127, start, end);
        mismatches += actual != expected[index];
    }
    EXPECT_EQ(mismatches, 0U);
}

TEST_P(RabitqRescaleBackendTest, FallsBackWhenNearlyCollinearWinnersAreAmbiguous) {
    std::vector<float> magnitudes(129);
    for (size_t i = 0; i < magnitudes.size(); ++i)
        magnitudes[i] =
            1.0F + static_cast<float>(i) * std::numeric_limits<float>::epsilon();
    normalize_magnitudes(magnitudes);
    const auto [start, end] = search_interval(magnitudes, 8);
    const double t = search()(magnitudes.data(), magnitudes.size(), 255, start, end);
    ASSERT_TRUE(std::isfinite(t));
    ASSERT_LT(t, 0);

    std::vector<uint8_t> code(magnitudes.size());
    const float factor = rabitq_impl::ex_bits::quantize_ex(
        magnitudes.data(), code.data(), magnitudes.size(), 8
    );
    EXPECT_TRUE(std::isnormal(factor));
    EXPECT_GT(factor, 0);
    const std::vector<int> levels(code.begin(), code.end());
    const long double actual = squared_cosine(magnitudes, levels);
    const long double optimum = exhaustive_best_squared_cosine(magnitudes, 255, start, end);
    EXPECT_GE(actual + 8 * std::numeric_limits<double>::epsilon(), optimum);
}

INSTANTIATE_TEST_SUITE_P(
    ExplicitSimd,
    RabitqRescaleBackendTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "Avx512" : "Avx2";
    }
);

}  // namespace
}  // namespace rabitqlib::quant
