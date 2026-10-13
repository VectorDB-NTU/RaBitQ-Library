#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "rabitqlib/core/detail/rabitq_impl.hpp"
#include "rabitqlib/utils/openmp.hpp"
#include "rabitqlib/utils/space.hpp"

namespace {
void run(int argc, char** argv) {
    if (argc != 7) {
        throw std::invalid_argument(
            "Usage: rabitq_core_hotpaths dim rows threads iterations repetitions "
            "mode(encode|l2)"
        );
    }
    const size_t dim = bench::number(argv[1]), rows = bench::number(argv[2]);
    const size_t threads = bench::number(argv[3]), iterations = bench::number(argv[4]);
    const size_t reps = bench::number(argv[5]);
    const std::string mode(argv[6]);
    if (dim % 32 != 0 || threads > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (mode != "encode" && mode != "l2")) {
        throw std::invalid_argument(
            "Require dim divisible by 32, valid thread count and mode"
        );
    }
#if RABITQ_USE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(threads));
#else
    if (threads != 1) {
        throw std::invalid_argument("OpenMP OFF requires threads=1");
    }
#endif
    auto data = bench::values(bench::product(rows, dim));
    const auto centroid = bench::values(dim, 91);
    bench::metadata(dim, rows, mode == "l2" ? 1 : threads);
    if (mode == "l2") {
        std::vector<float> distances(rows);
        const auto work = [&] {
            for (size_t i = 0; i < iterations; ++i) {
                for (size_t row = 0; row < rows; ++row) {
                    distances[row] = rabitqlib::euclidean_sqr(
                        data.data() + row * dim, centroid.data(), dim
                    );
                }
            }
        };
        work();
        for (size_t row = 0; row < rows; ++row) {
            double reference = 0;
            for (size_t d = 0; d < dim; ++d) {
                const double delta = static_cast<double>(data[row * dim + d]) - centroid[d];
                reference += delta * delta;
            }
            if (!(std::abs(distances[row] - reference) <= 2e-6 * static_cast<double>(dim)
                )) {
                throw std::runtime_error("L2 differs from float64 reference");
            }
        }
        bench::measure("l2", bench::product(rows, iterations), reps, work, [&] {
            return bench::hash(distances.data(), distances.size() * sizeof(float));
        });
        return;
    }
    std::vector<uint8_t> codes(bench::product(rows, dim / 8));
    std::vector<float> add(rows), rescale(rows), error(rows);
    const auto encode = [&](bool parallel) {
        using namespace rabitqlib::quant::rabitq_impl::one_bit;
        if (parallel) {
            one_bit_compact_codes<float, uint8_t, true>(
                data.data(),
                centroid.data(),
                rows,
                dim,
                codes.data(),
                add.data(),
                rescale.data(),
                error.data()
            );
        } else {
            one_bit_compact_codes<float, uint8_t, false>(
                data.data(),
                centroid.data(),
                rows,
                dim,
                codes.data(),
                add.data(),
                rescale.data(),
                error.data()
            );
        }
    };
    const auto checksum = [&] {
        uint64_t h = bench::hash(codes.data(), codes.size());
        for (const auto* factors : {&add, &rescale, &error}) {
            h = bench::hash(factors->data(), factors->size() * sizeof(float), h);
        }
        return h;
    };
    encode(false);
    const uint64_t serial_hash = checksum();
    encode(threads > 1);
    if (checksum() != serial_hash) {
        throw std::runtime_error("Serial and parallel encodings differ");
    }
    bench::measure(
        "encode",
        bench::product(rows, iterations),
        reps,
        [&] {
            for (size_t i = 0; i < iterations; ++i)
                encode(threads > 1);
        },
        checksum
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
