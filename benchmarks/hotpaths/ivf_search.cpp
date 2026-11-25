// Load one fixed IVF index; time only search_batch and verify Recall@10 outside timing.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include "common.hpp"
#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"

namespace {
const char* metric_name(rabitqlib::MetricType metric) {
    switch (metric) {
        case rabitqlib::METRIC_L2:
            return "L2";
        case rabitqlib::METRIC_IP:
            return "IP";
        case rabitqlib::METRIC_COSINE:
            return "cosine";
    }
    throw std::invalid_argument("Unknown index metric");
}

void run(int argc, char** argv) {
    if (argc != 9) {
        throw std::invalid_argument(
            "Usage: rabitq_ivf_search_hotpaths index queries.fvecs truth.ivecs "
            "batch nprobe threads reps max_queries"
        );
    }
    const size_t batch = bench::number(argv[4]);
    const size_t nprobe = bench::number(argv[5]);
    const size_t threads = bench::number(argv[6]);
    const size_t reps = bench::number(argv[7]);
    // Last argument bounds queries without changing the input files; useful for
    // expensive all-cluster scans. A larger limit uses every available query.
    const size_t limit = bench::number(argv[8]);
    rabitqlib::ivf::IVF index;
    index.load(argv[1]);
    rabitqlib::RowMajorMatrix<float> queries;
    rabitqlib::RowMajorMatrix<rabitqlib::PID> truth;
    rabitqlib::load_vecs<float>(argv[2], queries);
    rabitqlib::load_vecs<rabitqlib::PID>(argv[3], truth);
    constexpr size_t k = 10;
    if (static_cast<size_t>(queries.cols()) != index.dimension() ||
        queries.rows() != truth.rows() || truth.cols() < static_cast<int>(k)) {
        throw std::invalid_argument("Query dimensions or ground-truth shape do not match");
    }
    const size_t rows = std::min(limit, static_cast<size_t>(queries.rows()));
    std::vector<rabitqlib::PID> ids(bench::product(rows, k));
    std::vector<float> distances(ids.size());
    std::cout << "# index=" << argv[1] << " queries=" << argv[2]
              << " clusters=" << index.num_clusters() << " batch=" << batch
              << " nprobe=" << nprobe << '\n';
    bench::metadata(
        index.dimension(), rows, threads, index.seed(), metric_name(index.metric_type())
    );
    const auto work = [&] {
        for (size_t first = 0; first < rows; first += batch) {
            index.search_batch(
                queries.data() + first * index.dimension(),
                std::min(batch, rows - first),
                k,
                nprobe,
                ids.data() + first * k,
                distances.data() + first * k,
                std::nullopt,
                threads
            );
        }
    };
    bench::measure("ivf_search", rows, reps, work, [&] {
        return bench::hash(
            distances.data(),
            distances.size() * sizeof(float),
            bench::hash(ids.data(), ids.size() * sizeof(rabitqlib::PID))
        );
    });
    double recall_sum = 0.0;
    for (size_t row = 0; row < rows; ++row) {
        const auto* expected = truth.data() + row * static_cast<size_t>(truth.cols());
        const auto* found = ids.data() + row * k;
        size_t hits = 0, valid = 0;
        for (size_t i = 0; i < k; ++i) {
            if (expected[i] != rabitqlib::kPidMax) {
                ++valid;
                hits += std::find(found, found + k, expected[i]) != found + k;
            }
        }
        if (valid == 0) {
            throw std::invalid_argument(
                "Recall requires a valid ground-truth ID for every query"
            );
        }
        recall_sum += static_cast<double>(hits) / static_cast<double>(valid);
    }
    std::cout << "# recall_at_10=" << recall_sum / static_cast<double>(rows) << '\n';
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
