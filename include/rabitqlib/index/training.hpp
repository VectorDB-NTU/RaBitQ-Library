#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/clustering/rabitqkmeans.hpp"  // IWYU pragma: export
#include "rabitqlib/core/metric.hpp"

namespace rabitqlib {

/**
 * Centroid training parameters for IVF::train and HierarchicalNSW::train: 25
 * iterations on a sample of at most 256 points per centroid, spherical k-means for
 * inner product, and the given seed. The k-means classes keep their own defaults,
 * which train on all points.
 */
inline rabitqkmeans::RaBitQKMeansParameters default_training_parameters(
    MetricType metric_type, uint32_t seed
) {
    validate_metric_type(metric_type);
    rabitqkmeans::RaBitQKMeansParameters cp;
    cp.niter = 25;
    cp.max_points_per_centroid = 256;
    cp.spherical = metric_type == METRIC_IP;
    cp.seed = seed;
    return cp;
}

namespace ivf {
// IVF::train uses RaBitQKMeans up to this many clusters and QGKMeans above it.
// Tentative: this threshold has not been measured.
inline constexpr size_t kMaxRaBitQKMeansClusters = 2000;
}  // namespace ivf

}  // namespace rabitqlib
