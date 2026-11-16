// Times one IvfIndex::add() per repetition, each on a freshly loaded copy of the base
// index.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/ivf_index.hpp"
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

    rabitqlib::AddOptions add_options;
    add_options.num_threads = 1;
    {
        rabitqlib::IvfOptions options;
        options.nbits = 4;
        options.seed = 42;  // a fixed rotation keeps hashes comparable across processes
        rabitqlib::IvfIndex base(dim, clusters, options);
        base.set_centroids(centroids.data(), 1);
        add_options.cluster_ids = base_assigned.data();
        base.add(base_data.data(), base_n, add_options);
        base.save(path.c_str());
    }
    add_options.cluster_ids = add_cluster_ids;
    rabitqlib::IvfSearchOptions search_options;
    search_options.nprobe = clusters;

    bench::metadata(dim, base_n, 1);
    const char* name = mode == "single" ? "add_single_cluster" : "add_routed";
    uint64_t expected = 0;
    for (size_t rep = 0; rep <= reps; ++rep) {
        auto index = rabitqlib::IvfIndex::load(path.c_str());

        const auto start = bench::Clock::now();
        index.add(add_data.data(), add_n, add_options);
        const double seconds =
            std::chrono::duration<double>(bench::Clock::now() - start).count();

        if (index.num_stored_points() != base_n + add_n ||
            index.num_points() != base_n + add_n) {
            throw std::runtime_error("add() did not append the requested points");
        }
        // Validate known self-neighbors outside timing, independently of prior runs.
        for (size_t row = 0; row < std::min(add_n, size_t{3}); ++row) {
            rabitqlib::Id id = rabitqlib::kNoId;
            float distance = 0;
            index.search(add_data.data() + row * dim, 1, 1, &id, &distance, search_options);
            if (id != static_cast<rabitqlib::Id>(base_n + row) ||
                !std::isfinite(distance)) {
                throw std::runtime_error("Added point is not searchable with its new ID");
            }
        }

        uint64_t h = 14695981039346656037ULL;
        const size_t total = index.num_stored_points();
        h = bench::hash(&total, sizeof(total), h);
        const size_t k = std::min(size_t{10}, index.num_points());
        std::vector<rabitqlib::Id> ids(k);
        std::vector<float> distances(k);
        index.search(
            base_data.data(), 1, ids.size(), ids.data(), distances.data(), search_options
        );
        h = bench::hash(ids.data(), ids.size() * sizeof(rabitqlib::Id), h);
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
