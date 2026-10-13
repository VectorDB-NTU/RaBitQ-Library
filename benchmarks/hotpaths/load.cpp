#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/path.hpp"

namespace {
void run(int argc, char** argv) {
    if (argc != 6) {
        throw std::invalid_argument(
            "Usage: rabitq_load_hotpaths prepare|load path dim centroids repetitions"
        );
    }
    const std::string mode(argv[1]), path(argv[2]);
    const size_t dim = bench::number(argv[3]), count = bench::number(argv[4]);
    const size_t reps = bench::number(argv[5]);
    if (mode != "prepare" && mode != "load") {
        throw std::invalid_argument("Mode must be prepare or load");
    }
    if (count < 8)
        throw std::invalid_argument("At least eight centroids are required");
    const auto data = bench::values(bench::product(count, dim));
    if (mode == "prepare") {
        // Repeated smoke tests reuse their fixture; never replace an existing file.
        if (!std::filesystem::exists(rabitqlib::io_impl::filesystem_path(path + ".hnsw"))) {
            rabitqlib::ivf::HNSWInitializer index(dim, count);
            index.add_vectors(data.data(), 1);
            std::ofstream unused;
            index.save(unused, path.c_str());
            return;
        }
    }
    bench::metadata(dim, count, 1);
    uint64_t expected = 0;
    for (size_t rep = 0; rep <= reps; ++rep) {
        rabitqlib::ivf::HNSWInitializer index(dim, count);
        std::ifstream unused;
        const auto start = bench::Clock::now();
        index.load(unused, path.c_str());
        const double seconds =
            std::chrono::duration<double>(bench::Clock::now() - start).count();
        uint64_t h = 14695981039346656037ULL;
        // Validate every stored vector outside timing, including reused fixtures.
        for (size_t row = 0; row < count; ++row) {
            const float* stored = index.centroid(static_cast<rabitqlib::PID>(row));
            if (std::memcmp(stored, data.data() + row * dim, dim * sizeof(float)) != 0) {
                throw std::runtime_error("Fixture differs from deterministic input");
            }
            h = bench::hash(stored, dim * sizeof(float), h);
        }
        std::vector<rabitqlib::AnnCandidate<float>> candidates(8);
        index.centroids_distances(data.data(), candidates.size(), candidates);
        for (const auto& candidate : candidates) {
            h = bench::hash(&candidate.id, sizeof(candidate.id), h);
            h = bench::hash(&candidate.distance, sizeof(candidate.distance), h);
        }
        if (rep == 0)
            expected = h;
        else {
            if (h != expected)
                throw std::runtime_error("Loaded search results changed");
            bench::row("sidecar_load", rep - 1, 1, seconds, h);
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
