#include "rabitqlib/clustering/qgemeans.hpp"

#include <charconv>
#include <cstddef>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "kmeans_example.hpp"
#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/utils/io.hpp"

namespace {

int run_demo() {
    constexpr size_t kNumClusters = 33;
    constexpr size_t kPointsPerCluster = 40;
    constexpr size_t kDimension = 64;
    constexpr size_t kNumPoints = kNumClusters * kPointsPerCluster;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0F, 0.05F);
    std::vector<float> data(kNumPoints * kDimension);
    for (size_t point = 0; point < kNumPoints; ++point) {
        const float center = static_cast<float>(point / kPointsPerCluster);
        for (size_t dim = 0; dim < kDimension; ++dim) {
            data[point * kDimension + dim] = center + noise(rng);
        }
    }

    rabitqlib::qgemeans::QGEMeansParameters parameters;
    parameters.epochs = 8;
    parameters.ef_build = 64;
    parameters.ef_search = 32;
    parameters.verbose = true;

    rabitqlib::qgemeans::QGEMeans clustering(kDimension, kNumClusters, parameters);
    clustering.train(kNumPoints, data.data());

    std::cout << "Completed " << clustering.iteration_stats.size()
              << " epochs; final objective=" << clustering.final_obj << '\n';
    return 0;
}

// Usage: qgemeans <data.fvecs> <num_clusters> <centroids.fvecs> <labels.ivecs>
int cluster_file(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.fvecs> <num_clusters> <centroids.fvecs> <labels.ivecs>\n"
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

    rabitqlib::RowMajorArray<float> data;
    rabitqlib::load_vecs<float>(argv[1], data);
    rabitqlib::qgemeans::QGEMeansParameters parameters;
    parameters.verbose = true;
    rabitqlib::qgemeans::QGEMeans clustering(data.cols(), num_clusters, parameters);
    clustering.train(data.rows(), data.data());

    kmeans_example::write_vecs(
        argv[3], clustering.centroids.data(), num_clusters, data.cols()
    );
    kmeans_example::write_vecs(argv[4], clustering.assignments.data(), data.rows(), 1);
    std::cout << "Completed " << clustering.iteration_stats.size()
              << " epochs; final objective=" << clustering.final_obj << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            return run_demo();
        }
        return cluster_file(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
