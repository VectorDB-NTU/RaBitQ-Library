#include <algorithm>
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
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace {
namespace api = rabitqlib::quant::v1;
volatile double sink = 0;

size_t number(const char* text) {
    size_t result = 0;
    const char* end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end || result == 0) {
        throw std::invalid_argument("Arguments must be positive decimal integers");
    }
    return result;
}

size_t product(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() / b) {
        throw std::length_error("Benchmark dimensions overflow size_t");
    }
    return a * b;
}

template <class Function>
void measure(const char* operation, size_t operations, size_t repeats, Function work) {
    sink = work();  // One untimed warmup; setup and allocation are outside work.
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        const double checksum = work();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        sink = checksum;
        std::cout << operation << ',' << repeat << ',' << operations << ',' << seconds
                  << ',' << static_cast<double>(operations) / seconds << ','
                  << seconds * 1e9 / static_cast<double>(operations) << ',' << checksum
                  << '\n';
    }
}

int run(int argc, char** argv) {
    if (argc != 1 && argc != 6) {
        throw std::invalid_argument(
            "Usage: rabitq_core_benchmark [dim bits vectors queries repetitions]"
        );
    }
    const size_t dim = argc == 1 ? 128 : number(argv[1]);
    const size_t bits = argc == 1 ? 4 : number(argv[2]);
    const size_t count = argc == 1 ? 4096 : number(argv[3]);
    const size_t queries = argc == 1 ? 64 : number(argv[4]);
    const size_t repeats = argc == 1 ? 5 : number(argv[5]);
    constexpr size_t kTopK = 10;
    if (dim < 64 || dim % 32 != 0) {
        throw std::invalid_argument(
            "This FHT/Kac benchmark requires dim >= 64 and divisible by 32"
        );
    }
    if (count < kTopK) {
        throw std::invalid_argument("At least ten vectors are needed for recall@10");
    }
    const api::Parameters parameters{dim, bits, rabitqlib::METRIC_L2};
    const auto sizes = api::requirements(parameters, api::CodeLayout::SplitPacked);
    const size_t comparisons = product(count, queries);
    std::vector<float> data(product(count, dim)), query(product(queries, dim)), input(dim);
    std::unique_ptr<rabitqlib::Rotator<float>> rotation(
        rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator, 0, 42)
    );
    uint32_t seed = 42;
    for (auto* matrix : {&data, &query}) {
        for (size_t row = 0; row < matrix->size() / dim; ++row) {
            for (float& value : input) {
                seed = seed * 1664525U + 1013904223U;
                value = static_cast<float>(seed >> 8U) / 8388608.0F - 1.0F;
            }
            rotation->rotate(input.data(), matrix->data() + row * dim);
        }
    }
    std::vector<std::max_align_t> scratch(
        (sizes.workspace_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)
    );
    const api::Buffer workspace{scratch.data(), sizes.workspace_bytes};
    std::vector<uint8_t> signs(product(count, sizes.code_bytes));
    // At one bit the unused extra buffer has a harmless one-byte stride.
    const size_t extra_stride = std::max(size_t{1}, sizes.extra_bytes);
    std::vector<uint8_t> extra(product(count, extra_stride));
    std::vector<api::Factors> factors(count);
    std::vector<api::QueryState> states(queries);
    std::vector<uint8_t> unpacked(product(dim, sizeof(uint16_t)));
    const auto code = [&](size_t row) {
        return api::CodeView{
            api::CodeLayout::SplitPacked,
            {signs.data() + row * sizes.code_bytes, sizes.code_bytes},
            {extra.data() + row * extra_stride, sizes.extra_bytes}};
    };
    const auto encode = [&](size_t row) {
        return api::encode(
            parameters,
            data.data() + row * dim,
            nullptr,
            {api::CodeLayout::SplitPacked,
             {signs.data() + row * sizes.code_bytes, sizes.code_bytes},
             {extra.data() + row * extra_stride, sizes.extra_bytes}},
            workspace
        );
    };
    for (size_t row = 0; row < count; ++row) {
        factors[row] = encode(row);
    }
    for (size_t row = 0; row < queries; ++row) {
        states[row] = api::prepare_query(parameters, query.data() + row * dim, nullptr);
    }

    double squared_error = 0, max_error = 0;
    size_t hits = 0;
    std::vector<std::pair<double, size_t>> exact(count), estimated(count);
    for (size_t q = 0; q < queries; ++q) {
        for (size_t row = 0; row < count; ++row) {
            double distance = 0;
            for (size_t j = 0; j < dim; ++j) {
                const double delta =
                    static_cast<double>(data[row * dim + j]) - query[q * dim + j];
                distance += delta * delta;
            }
            const double approximation = api::compute(states[q], code(row), factors[row]);
            exact[row] = {distance, row};
            estimated[row] = {approximation, row};
            const double error = approximation - distance;
            squared_error += error * error;
            max_error = std::max(max_error, std::abs(error));
        }
        std::partial_sort(exact.begin(), exact.begin() + kTopK, exact.end());
        std::partial_sort(estimated.begin(), estimated.begin() + kTopK, estimated.end());
        for (size_t i = 0; i < kTopK; ++i) {
            for (size_t j = 0; j < kTopK; ++j) {
                hits += estimated[i].second == exact[j].second ? 1 : 0;
            }
        }
    }
    std::cout
        << std::setprecision(12) << "# dim=" << dim << " bits=" << bits
        << " vectors=" << count << " queries=" << queries << " repetitions=" << repeats
        << '\n'
        << "# metric=L2 layout=SplitPacked threads=1 seed=42 rotation=FHT/Kac\n"
        << "# rotation/setup/allocations excluded; full scan has no reranking\n"
        << "# avx2=" << rabitqlib::cpu::has_avx2()
        << " avx512_core=" << rabitqlib::cpu::has_avx512_core()
        << " neon=" << rabitqlib::cpu::has_neon() << " openmp_build=" << RABITQ_USE_OPENMP
        << '\n'
        << "# rmse=" << std::sqrt(squared_error / static_cast<double>(comparisons))
        << " max_abs_error=" << max_error << " flat_recall_at_10="
        << static_cast<double>(hits) / static_cast<double>(product(queries, kTopK)) << '\n'
        << "operation,repeat,operations,seconds,ops_per_second,ns_per_operation,checksum\n";
    measure("encode_packed", count, repeats, [&] {
        double checksum = 0;
        for (size_t row = 0; row < count; ++row) {
            factors[row] = encode(row);
            checksum += factors[row].add;
        }
        return checksum;
    });
    const size_t preparations = product(queries, size_t{256});
    measure("prepare_query", preparations, repeats, [&] {
        double checksum = 0;
        for (size_t row = 0; row < preparations; ++row) {
            const size_t q = row % queries;
            states[q] = api::prepare_query(parameters, query.data() + q * dim, nullptr);
            checksum += static_cast<double>(states[q].dimension());
        }
        return checksum;
    });
    measure("compute_packed", comparisons, repeats, [&] {
        double checksum = 0;
        for (size_t q = 0; q < queries; ++q) {
            for (size_t row = 0; row < count; ++row) {
                checksum += api::compute(states[q], code(row), factors[row]);
            }
        }
        return checksum;
    });
    measure("convert_to_unpacked", count, repeats, [&] {
        double checksum = 0;
        for (size_t row = 0; row < count; ++row) {
            api::convert_codes(
                parameters,
                code(row),
                {api::CodeLayout::UnpackedU16, {unpacked.data(), unpacked.size()}, {}},
                workspace
            );
            checksum += unpacked.front();
        }
        return checksum;
    });
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
