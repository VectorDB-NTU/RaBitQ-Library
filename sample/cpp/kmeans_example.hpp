#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/io.hpp"

namespace kmeans_example {

template <typename T>
void write_vecs(const char* path, const T* data, size_t rows, size_t cols) {
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(rabitqlib::io_impl::filesystem_path(path), std::ios::binary);
    const auto dimension = static_cast<uint32_t>(cols);
    for (size_t row = 0; row < rows; ++row) {
        output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        output.write(
            reinterpret_cast<const char*>(data + row * cols),
            static_cast<std::streamsize>(cols * sizeof(T))
        );
    }
    output.close();
}

template <typename KMeans, typename Parameters>
int cluster_file(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr
            << "Usage: " << argv[0]
            << " <data.fvecs> <num_clusters> <centroids.fvecs> <labels.ivecs> [l2|ip]\n"
            << "Without arguments, run a synthetic clustering demo.\n";
        return 1;
    }
    const std::string count(argv[2]);
    size_t num_clusters = 0;
    const auto parsed =
        std::from_chars(count.data(), count.data() + count.size(), num_clusters);
    if (parsed.ec != std::errc{} || parsed.ptr != count.data() + count.size() ||
        num_clusters == 0) {
        throw std::invalid_argument("num_clusters must be a positive integer");
    }
    const std::string metric = argc == 6 ? argv[5] : "l2";
    if (metric != "l2" && metric != "ip") {
        throw std::invalid_argument("metric must be l2 or ip; ip requires normalized input"
        );
    }

    rabitqlib::RowMajorArray<float> data;
    rabitqlib::load_vecs<float>(argv[1], data);
    Parameters parameters;
    parameters.spherical = metric == "ip";
    parameters.final_assignment = decltype(parameters.final_assignment)::Exact;
    parameters.verbose = true;
    KMeans clustering(data.cols(), num_clusters, parameters);
    clustering.train(data.rows(), data.data());

    write_vecs(argv[3], clustering.centroids.data(), num_clusters, data.cols());
    write_vecs(argv[4], clustering.assignments.data(), data.rows(), 1);
    std::cout << "Completed " << clustering.iteration_stats.size()
              << " iterations; final objective=" << clustering.final_obj << '\n';
    return 0;
}

}  // namespace kmeans_example
