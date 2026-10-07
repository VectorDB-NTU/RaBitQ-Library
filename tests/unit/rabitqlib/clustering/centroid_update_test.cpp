#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/clustering/detail/kmeans.hpp"
#include "rabitqlib/clustering/qgkmeans.hpp"

namespace rabitqlib::clustering::detail {
namespace {
struct RecordedParameters : qgkmeans::QGKMeansParameters {
    // Owned by the test for the duration of synchronous training.
    std::vector<float>* initial_centroids = nullptr;
    size_t occupied_clusters = 0;
};

class RecordedAssignment {
    size_t dim_;
    size_t clusters_;
    size_t points_;
    RecordedParameters parameters_;

   public:
    static constexpr const char* kName = "RecordedAssignment";
    static void validate_parameters(size_t, const RecordedParameters&) {}

    RecordedAssignment(
        size_t d,
        size_t k,
        const float*,
        size_t n,
        uint32_t,
        const RecordedParameters& parameters
    )
        : dim_(d), clusters_(k), points_(n), parameters_(parameters) {}

    void assign(const float* centroids, PID* labels, float* distances, bool final = false) {
        if (!final) {
            parameters_.initial_centroids->assign(centroids, centroids + clusters_ * dim_);
        }
        for (size_t point = 0; point < points_; ++point) {
            labels[point] = static_cast<PID>(point % parameters_.occupied_clusters);
            distances[point] = 0;
        }
    }
};

TEST(CentroidUpdateTest, ShiftIncludesNormalizedAndRefilledCentroids) {
    constexpr size_t kDim = 64, kClusters = 3, kPoints = 12;
    std::vector<float> data(kPoints * kDim, 0);
    for (size_t point = 0; point < kPoints; ++point) {
        data[point * kDim + point] = 1;
    }
    for (bool spherical : {false, true}) {
        for (size_t empty : {0U, 1U, 2U}) {
            for (uint32_t threads : {1U, 4U}) {
                SCOPED_TRACE(
                    ::testing::Message() << spherical << '/' << empty << '/' << threads
                );
                std::vector<float> initial;
                RecordedParameters parameters;
                parameters.niter = 1;
                parameters.spherical = spherical;
                parameters.num_threads = threads;
                parameters.initial_centroids = &initial;
                parameters.occupied_clusters = kClusters - empty;
                LloydKMeans<RecordedParameters, RecordedAssignment> model(
                    kDim, kClusters, parameters
                );
                model.train(kPoints, data.data());
                ASSERT_EQ(initial.size(), model.centroids.size());
                double expected = 0;
                for (size_t i = 0; i < initial.size(); ++i) {
                    const double delta =
                        static_cast<double>(model.centroids[i]) - initial[i];
                    expected += delta * delta;
                }
                ASSERT_EQ(model.iteration_stats.size(), 1U);
                EXPECT_EQ(model.iteration_stats.front().nsplit, empty);
                EXPECT_NEAR(model.iteration_stats.front().shift, expected, 1e-12);
            }
        }
    }
}
}  // namespace
}  // namespace rabitqlib::clustering::detail
