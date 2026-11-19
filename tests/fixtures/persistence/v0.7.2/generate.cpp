// Build only against the pinned v0.7.2 source, never the current checkout. Writes an
// HNSW-RaBitQ index that was constructed and saved without points, which the 0.7.x
// C++ API allowed (0.8 and later refuse to save an empty index).
#include <filesystem>
#include <stdexcept>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument("Pass the fixture output directory");
    }
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    // Capacity 130, 65 dimensions, 4 bits, M 8, ef_construction 64, seed 42.
    rabitqlib::hnsw::HierarchicalNSW index(130, 65, 4, 8, 64, 42, rabitqlib::METRIC_L2);
    index.save((directory / "hnsw4_empty_l2.index").string().c_str());
    return 0;
}
