#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"
#include "scoring_legacy.hpp"

namespace {
namespace api = rabitqlib::quant::v1;
volatile double sink = 0;

size_t number(const char* text) {
    size_t value = 0;
    const char* end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value == 0) {
        throw std::invalid_argument("Expected a positive decimal integer");
    }
    return value;
}

size_t product(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        throw std::length_error("Benchmark buffer size overflows size_t");
    }
    return a * b;
}

template <class Score>
double score_loop(
    Score score,
    size_t iterations,
    const std::vector<size_t>& order,
    const std::vector<uint8_t>& codes,
    const std::vector<uint8_t>& extras,
    const std::vector<api::Factors>& factors,
    const api::Requirements& sizes,
    api::CodeLayout layout
) {
    double sum = 0;
    const size_t extra_stride = std::max(size_t{1}, sizes.extra_bytes);
    for (size_t i = 0; i < iterations; ++i) {
        const size_t row = order[i & (order.size() - 1)];
        sum += score(
            {layout,
             {codes.data() + row * sizes.code_bytes, sizes.code_bytes},
             {extras.data() + row * extra_stride, sizes.extra_bytes}},
            factors[row]
        );
    }
    return sum;
}

int run(int argc, char** argv) {
    if (argc != 1 && argc != 7 && argc != 8 && argc != 9) {
        throw std::invalid_argument(
            "Usage: rabitq_core_scoring_benchmark [dim bits packed|unpacked "
            "hot|sequential|random iterations repetitions [working_set_mib [all|v1]]]"
        );
    }
    const size_t dim = argc == 1 ? 128 : number(argv[1]);
    const size_t bits = argc == 1 ? 4 : number(argv[2]);
    const std::string layout_name = argc == 1 ? "packed" : argv[3];
    const std::string pattern = argc == 1 ? "hot" : argv[4];
    const size_t requested_iterations = argc == 1 ? 1000000 : number(argv[5]);
    const size_t repeats = argc == 1 ? 5 : number(argv[6]);
    const size_t mib = argc >= 8 ? number(argv[7]) : 64;
    const std::string paths = argc == 9 ? argv[8] : "all";
    if (paths != "all" && paths != "v1") {
        throw std::invalid_argument("Paths must be all or v1");
    }
    if (layout_name != "packed" && layout_name != "unpacked") {
        throw std::invalid_argument("Layout must be packed or unpacked");
    }
    if (pattern != "hot" && pattern != "sequential" && pattern != "random") {
        throw std::invalid_argument("Pattern must be hot, sequential or random");
    }
    const auto layout = layout_name == "packed" ? api::CodeLayout::SplitPacked
                                                : api::CodeLayout::UnpackedU16;
    const api::Parameters parameters{dim, bits};
    const auto sizes = api::requirements(parameters, layout);
    const size_t extra_stride = std::max(size_t{1}, sizes.extra_bytes);
    const size_t row_bytes = sizes.code_bytes + extra_stride + sizeof(api::Factors);
    const size_t desired_rows = product(mib, size_t{1} << 20U) / row_bytes;
    size_t rows = 1;
    if (pattern != "hot") {
        while (rows <= desired_rows / 2) {
            rows *= 2;
        }
    }
    const size_t iterations = std::max(requested_iterations, rows);
    std::vector<uint8_t> codes(product(rows, sizes.code_bytes));
    std::vector<uint8_t> extras(product(rows, extra_stride));
    std::vector<api::Factors> factors(rows);
    std::vector<size_t> order(rows);
    std::iota(order.begin(), order.end(), size_t{0});
    if (pattern == "random") {
        std::mt19937 random(91);
        std::shuffle(order.begin(), order.end(), random);
    }
    std::vector<std::max_align_t> memory(
        (sizes.workspace_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)
    );
    const api::Buffer work{memory.data(), sizes.workspace_bytes};
    std::vector<float> input(dim), query(dim);
    uint32_t random = 42;
    const auto next = [&] {
        random = random * 1664525U + 1013904223U;
        return static_cast<float>(random >> 8U) / 8388608.0F - 1.0F;
    };
    for (auto& value : query) {
        value = next();
    }
    const size_t prototypes = std::min(rows, size_t{32});
    for (size_t row = 0; row < prototypes; ++row) {
        for (auto& value : input) {
            value = next();
        }
        factors[row] = api::encode(
            parameters,
            input.data(),
            nullptr,
            {layout,
             {codes.data() + row * sizes.code_bytes, sizes.code_bytes},
             {extras.data() + row * extra_stride, sizes.extra_bytes}},
            work
        );
    }
    for (size_t row = prototypes; row < rows; ++row) {
        const size_t source = row % prototypes;
        std::memcpy(
            codes.data() + row * sizes.code_bytes,
            codes.data() + source * sizes.code_bytes,
            sizes.code_bytes
        );
        std::memcpy(
            extras.data() + row * extra_stride,
            extras.data() + source * extra_stride,
            sizes.extra_bytes
        );
        factors[row] = factors[source];
    }
    const auto prepared = api::prepare_query(parameters, query.data(), nullptr);
    const scoring_benchmark::Query legacy{
        query.data(),
        dim,
        bits,
        rabitqlib::l2norm_sqr(query.data(), dim),
        std::accumulate(query.begin(), query.end(), 0.0F) * -0.5F,
        rabitqlib::select_excode_ipfunc(bits - 1)};
    const double eps = std::numeric_limits<float>::epsilon();
    const double gamma = (dim + 8) * eps / (1 - (dim + 8) * eps);
    const double max_code = static_cast<double>((1U << bits) - 1);
    const double absolute_query =
        std::accumulate(query.begin(), query.end(), 0.0, [](double sum, float value) {
            return sum + std::abs(value);
        });
    for (size_t row = 0; row < prototypes; ++row) {
        const api::CodeView code{
            layout,
            {codes.data() + row * sizes.code_bytes, sizes.code_bytes},
            {extras.data() + row * extra_stride, sizes.extra_bytes}};
        const float checked = api::compute(prepared, code, factors[row]);
        // The installed library and this consumer can use different reduction
        // orders. Bound their float32 rounding error before timing either path.
        const double tolerance =
            2 * gamma * std::abs(factors[row].rescale) * max_code * absolute_query +
            8 * eps *
                (1 + std::abs(factors[row].add) + std::abs(legacy.add) + std::abs(checked));
        if (std::abs(
                checked - scoring_benchmark::unchecked_inline(legacy, code, factors[row])
            ) > tolerance ||
            std::abs(
                checked - scoring_benchmark::unchecked_call(legacy, code, factors[row])
            ) > tolerance ||
            std::abs(
                checked - scoring_benchmark::unchecked_ref(legacy, code, factors[row])
            ) > tolerance) {
            throw std::runtime_error("Checked and unchecked score results differ");
        }
    }
    std::cout
        << std::setprecision(17)
        << "# seed=42 metric=L2 centroid=zero threads=1 setup/encoding excluded\n"
        << "# rows=" << rows << " code_factor_bytes=" << product(rows, row_bytes)
        << " order_bytes=" << product(rows, sizeof(size_t)) << " prototypes=" << prototypes
        << '\n'
        << "# avx2=" << rabitqlib::cpu::has_avx2()
        << " avx512_core=" << rabitqlib::cpu::has_avx512_core()
        << " neon=" << rabitqlib::cpu::has_neon() << '\n'
        << "dim,bits,layout,pattern,path,repeat,operations,seconds,ns_per_score,checksum\n";
    const std::array<const char*, 4> names{
        "v1", "unchecked_inline", "unchecked_call", "unchecked_ref"};
    const auto execute = [&](size_t path, size_t count) {
        if (path == 0) {
            return score_loop(
                [&](api::CodeView c, const api::Factors& f) {
                    return api::compute(prepared, c, f);
                },
                count,
                order,
                codes,
                extras,
                factors,
                sizes,
                layout
            );
        }
        if (path == 1) {
            return score_loop(
                [&](api::CodeView c, const api::Factors& f) {
                    return scoring_benchmark::unchecked_inline(legacy, c, f);
                },
                count,
                order,
                codes,
                extras,
                factors,
                sizes,
                layout
            );
        }
        if (path == 2) {
            return score_loop(
                [&](api::CodeView c, const api::Factors& f) {
                    return scoring_benchmark::unchecked_call(legacy, c, f);
                },
                count,
                order,
                codes,
                extras,
                factors,
                sizes,
                layout
            );
        }
        return score_loop(
            [&](api::CodeView c, const api::Factors& f) {
                return scoring_benchmark::unchecked_ref(legacy, c, f);
            },
            count,
            order,
            codes,
            extras,
            factors,
            sizes,
            layout
        );
    };
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        const size_t path_count = paths == "all" ? names.size() : 1;
        for (size_t offset = 0; offset < path_count; ++offset) {
            const size_t path = (repeat + offset) % path_count;
            sink = execute(path, std::max(rows, iterations));
            const auto start = std::chrono::steady_clock::now();
            const double checksum = execute(path, iterations);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                    .count();
            sink = checksum;
            std::cout << dim << ',' << bits << ',' << layout_name << ',' << pattern << ','
                      << names[path] << ',' << repeat << ',' << iterations << ',' << seconds
                      << ',' << seconds * 1e9 / static_cast<double>(iterations) << ','
                      << checksum << '\n';
        }
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
