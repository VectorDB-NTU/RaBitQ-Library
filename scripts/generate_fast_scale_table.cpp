// Historical record of how src/core/fast_scale_table.cpp, the frozen fast-mode
// scales read by quant::faster_config, was generated. The table is frozen data and
// is never regenerated: rerunning this after a change to the computation would
// silently change every fast-mode code. A different table would be a new, separately
// named table. The original run, from the repository root:
//
//   g++ -std=c++17 -O3 -ffp-contract=off -pthread -DEIGEN_MAX_ALIGN_BYTES=64 \
//       -DEIGEN_MAX_STATIC_ALIGN_BYTES=64 -Iinclude \
//       scripts/generate_fast_scale_table.cpp -o /tmp/generate_fast_scale_table
//   /tmp/generate_fast_scale_table > src/core/fast_scale_table.cpp
//   clang-format-23 -i src/core/fast_scale_table.cpp
//
// The frozen values are the ones portable x86-64 Linux builds computed before the
// table existed: libstdc++'s std::normal_distribution, the scalar event sweep (the
// double-precision search never uses the SIMD kernel) and no fused multiply-add.
// Other standard libraries draw different samples and change the values by several
// percent; FMA contraction or AVX vectorization (a different Eigen reduction order)
// changes the last bits. The guards below refuse to build in those configurations.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <thread>
#include <vector>

#include "rabitqlib/core/detail/fast_scale_table.hpp"
#include "rabitqlib/core/detail/rabitq_impl.hpp"

#if !defined(__GLIBCXX__)
#error "Generate the table with libstdc++; its std::normal_distribution defines the values"
#endif
#if !defined(__x86_64__) || defined(__AVX__) || defined(__FMA__)
#error "Generate the table on baseline x86-64 (SSE2, no -m/-march flags), -ffp-contract=off"
#endif

namespace ex_bits = rabitqlib::quant::rabitq_impl::ex_bits;

int main() {
    constexpr size_t kRows = ex_bits::kFastScaleMaxDim / ex_bits::kFastScaleDimStep;
    constexpr size_t kCols = ex_bits::kFastScaleMaxExBits;
    std::vector<double> table(kRows * kCols);

    // Largest dimensions first, so the slowest jobs do not finish last.
    std::atomic<size_t> next{0};
    auto worker = [&] {
        for (size_t job = next++; job < table.size(); job = next++) {
            const size_t index = table.size() - 1 - job;
            const size_t padded_dim = (index / kCols + 1) * ex_bits::kFastScaleDimStep;
            const size_t bits = index % kCols + 1;
            table[index] = ex_bits::const_scaling_factor(padded_dim, bits);
        }
    };
    std::vector<std::thread> threads(std::max(1U, std::thread::hardware_concurrency()));
    for (auto& thread : threads) {
        thread = std::thread(worker);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    std::printf(
        "// Generated once by scripts/generate_fast_scale_table.cpp. Frozen data: never\n"
        "// regenerate or edit it; a different table would be a new, separately named "
        "table.\n"
        "// Each value is const_scaling_factor(padded_dim, ex_bits) from libstdc++ on\n"
        "// baseline x86-64 (SSE2, no FMA), written as an exact hexadecimal "
        "floating-point\n"
        "// literal.\n"
        "\n"
        "#include \"rabitqlib/core/detail/fast_scale_table.hpp\"\n"
        "\n"
        "#include <array>\n"
        "#include <cstddef>\n"
        "#include <optional>\n"
        "\n"
        "namespace rabitqlib::quant::rabitq_impl::ex_bits {\n"
        "namespace {\n"
        "\n"
        "constexpr size_t kFastScaleRows = kFastScaleMaxDim / kFastScaleDimStep;\n"
        "\n"
        "// Entry (row, col) holds padded_dim = (row + 1) * kFastScaleDimStep and\n"
        "// ex_bits = col + 1, at index row * kFastScaleMaxExBits + col.\n"
        "constexpr std::array<double, kFastScaleRows * kFastScaleMaxExBits> kFastScales = "
        "{\n"
    );
    for (size_t row = 0; row < kRows; ++row) {
        std::printf("    ");
        for (size_t col = 0; col < kCols; ++col) {
            std::printf("%a,%s", table[row * kCols + col], col + 1 < kCols ? " " : "\n");
        }
    }
    std::printf(
        "};\n"
        "\n"
        "}  // namespace\n"
        "\n"
        "std::optional<double> frozen_const_scaling_factor(\n"
        "    size_t padded_dim, size_t ex_bits\n"
        ") noexcept {\n"
        "    if (padded_dim == 0 || padded_dim %% kFastScaleDimStep != 0 ||\n"
        "        padded_dim > kFastScaleMaxDim || ex_bits == 0 ||\n"
        "        ex_bits > kFastScaleMaxExBits) {\n"
        "        return std::nullopt;\n"
        "    }\n"
        "    const size_t row = padded_dim / kFastScaleDimStep - 1;\n"
        "    return kFastScales[(row * kFastScaleMaxExBits) + ex_bits - 1];\n"
        "}\n"
        "\n"
        "}  // namespace rabitqlib::quant::rabitq_impl::ex_bits\n"
    );
    return 0;
}
