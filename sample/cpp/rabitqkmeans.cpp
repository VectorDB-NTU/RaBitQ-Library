#include "rabitqlib/clustering/rabitqkmeans.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

#include "kmeans_example.hpp"

int run_demo() {
    constexpr size_t kNumClusters = 16;
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

    rabitqlib::rabitqkmeans::RaBitQKMeansParameters parameters;
    parameters.niter = 5;
    parameters.verbose = true;

    rabitqlib::rabitqkmeans::RaBitQKMeans clustering(kDimension, kNumClusters, parameters);
    clustering.train(kNumPoints, data.data());

    std::cout << "Completed " << clustering.iteration_stats.size()
              << " iterations; final assignment count=" << clustering.assignments.size()
              << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            return run_demo();
        }
        return kmeans_example::cluster_file<
            rabitqlib::rabitqkmeans::RaBitQKMeans,
            rabitqlib::rabitqkmeans::RaBitQKMeansParameters>(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
