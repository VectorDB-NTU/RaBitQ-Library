// Times one IVF::add() per repetition, each on a freshly loaded copy of the base index.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"

namespace {
void run(int argc, char** argv) {
    if (argc != 8) {
        throw std::invalid_argument(
            "Usage: rabitq_ivf_add_hotpaths path dim clusters base_n add_n reps "
            "single|routed"
        );
    }
    const std::string path(argv[1]);
    const size_t dim = bench::number(argv[2]);
    const size_t clusters = bench::number(argv[3]);
    const size_t base_n = bench::number(argv[4]);
    const size_t add_n = bench::number(argv[5]);
    const size_t reps = bench::number(argv[6]);
    const std::string mode(argv[7]);
    if (mode != "single" && mode != "routed") {
        throw std::invalid_argument("Mode must be single or routed");
    }
    if (clusters < 1 || base_n < clusters) {
        throw std::invalid_argument("base_n must be at least clusters");
    }

    const auto base_data = bench::values(bench::product(base_n, dim), 42);
    const auto add_data = bench::values(bench::product(add_n, dim), 91);
    const auto centroids = bench::values(bench::product(clusters, dim), 7);
    // Round-robin assignment; clustering quality is irrelevant to storage cost.
    std::vector<rabitqlib::PID> base_assigned(base_n);
    for (size_t i = 0; i < base_n; ++i) {
        base_assigned[i] = static_cast<rabitqlib::PID>(i % clusters);
    }
    // "single" sends every new point to cluster 0; "routed" lets add() pick the nearest.
    const std::vector<rabitqlib::PID> single_ids(add_n, rabitqlib::PID{0});
    const rabitqlib::PID* add_cluster_ids = mode == "single" ? single_ids.data() : nullptr;

    {
        rabitqlib::ivf::IVF base(base_n, dim, clusters, 4, rabitqlib::METRIC_L2);
        base.construct(base_data.data(), centroids.data(), base_assigned.data(), false, 1);
        base.save(path.c_str());
    }

    bench::metadata(dim, base_n, 1);
    const char* name = mode == "single" ? "add_single_cluster" : "add_routed";
    uint64_t expected = 0;
    for (size_t rep = 0; rep <= reps; ++rep) {
        rabitqlib::ivf::IVF index;
        index.load(path.c_str());

        const auto start = bench::Clock::now();
        index.add(add_data.data(), add_n, add_cluster_ids, false, 1);
        const double seconds =
            std::chrono::duration<double>(bench::Clock::now() - start).count();

        uint64_t h = 14695981039346656037ULL;
        const size_t total = index.max_elements();
        h = bench::hash(&total, sizeof(total), h);
        std::vector<rabitqlib::PID> ids(10);
        std::vector<float> distances(10);
        index.search(base_data.data(), ids.size(), clusters, ids.data(), distances.data());
        h = bench::hash(ids.data(), ids.size() * sizeof(rabitqlib::PID), h);
        h = bench::hash(distances.data(), distances.size() * sizeof(float), h);

        if (rep == 0) {
            expected = h;
        } else {
            if (h != expected) {
                throw std::runtime_error("add() result changed between repetitions");
            }
            bench::row(name, rep - 1, add_n, seconds, h);
        }
    }
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
