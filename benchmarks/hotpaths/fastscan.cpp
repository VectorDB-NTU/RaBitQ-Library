#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/utils/space.hpp"

namespace {
void run(int argc, char** argv) {
    if (argc != 5) {
        throw std::invalid_argument(
            "Usage: rabitq_fastscan_hotpaths dim batches repetitions mode(standard|hacc)"
        );
    }
    const size_t dim = bench::number(argv[1]), batches = bench::number(argv[2]);
    const size_t reps = bench::number(argv[3]);
    const std::string mode(argv[4]);
    if (dim % 32 != 0 || (mode != "standard" && mode != "hacc")) {
        throw std::invalid_argument("Require dim divisible by 32 and standard or hacc");
    }
    constexpr size_t kRows = 32;
    const bool hacc = mode == "hacc";
    auto data = bench::values(bench::product(kRows, dim));
    const auto query = bench::values(dim, 91);
    const std::vector<float> centroid(dim, 0);
    std::vector<char> storage(rabitqlib::BatchDataMap<float>::data_bytes(dim));
    rabitqlib::quant::quantize_one_batch(
        data.data(), centroid.data(), kRows, dim, storage.data()
    );
    rabitqlib::SplitBatchQuery<float> prepared(
        query.data(), dim, 3, rabitqlib::METRIC_L2, hacc
    );
    prepared.set_g_add(std::sqrt(rabitqlib::l2norm_sqr(query.data(), dim)));
    std::array<float, kRows> distance{}, lower{}, ip{}, ref_distance{}, ref_lower{},
        ref_ip{};
    rabitqlib::simd::split_batch_estdist_generic(
        storage.data(),
        prepared,
        dim,
        ref_distance.data(),
        ref_lower.data(),
        ref_ip.data(),
        hacc
    );
    const auto score = [&] {
        rabitqlib::split_batch_estdist(
            storage.data(), prepared, dim, distance.data(), lower.data(), ip.data(), hacc
        );
    };
    score();
    for (size_t row = 0; row < kRows; ++row) {
        const double tolerance = 2e-6 * static_cast<double>(dim);
        if (!(std::abs(distance[row] - ref_distance[row]) <= tolerance) ||
            !(std::abs(lower[row] - ref_lower[row]) <= tolerance) ||
            !(std::abs(ip[row] - ref_ip[row]) <= tolerance)) {
            throw std::runtime_error("Batch estimator differs from generic correction");
        }
    }
    // Numerical quality is separate from timing. This is one-bit filtering of
    // 32 synthetic vectors, not complete multi-bit ANN search.
    std::array<double, kRows> exact{};
    double squared_error = 0;
    for (size_t row = 0; row < kRows; ++row) {
        for (size_t d = 0; d < dim; ++d) {
            const double delta = static_cast<double>(data[row * dim + d]) - query[d];
            exact[row] += delta * delta;
        }
        squared_error += std::pow(exact[row] - distance[row], 2);
    }
    std::array<size_t, kRows> expected_order{}, actual_order{};
    std::iota(expected_order.begin(), expected_order.end(), 0);
    actual_order = expected_order;
    std::sort(expected_order.begin(), expected_order.end(), [&](size_t a, size_t b) {
        return exact[a] < exact[b];
    });
    std::sort(actual_order.begin(), actual_order.end(), [&](size_t a, size_t b) {
        return distance[a] < distance[b];
    });
    size_t hits = 0;
    for (size_t i = 0; i < 10; ++i) {
        hits += std::find(
                    expected_order.begin(), expected_order.begin() + 10, actual_order[i]
                ) != expected_order.begin() + 10;
    }
    bench::metadata(dim, kRows, 1);
    std::cout << "# one_bit_filter_rmse=" << std::sqrt(squared_error / kRows)
              << " recall10=" << static_cast<double>(hits) / 10 << '\n';
    bench::measure(
        mode.c_str(),
        bench::product(batches, kRows),
        reps,
        [&] {
            for (size_t batch = 0; batch < batches; ++batch)
                score();
        },
        [&] {
            uint64_t h = bench::hash(distance.data(), sizeof(distance));
            h = bench::hash(lower.data(), sizeof(lower), h);
            return bench::hash(ip.data(), sizeof(ip), h);
        }
    );
}
}  // namespace
int main(int argc, char** argv) {
    try {
        run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
