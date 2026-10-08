#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

#include "rabitqlib/clustering/qgkmeans.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace {
constexpr size_t kDimension = 65;
constexpr size_t kPoints = 67;

// Use public results and an independent scalar reference. Checks must also run
// in Release builds, where assert() would be disabled.
template <typename Model>
bool check_clustering(Model& model, const std::vector<float>& data) {
    model.train(kPoints, data.data());
    if (model.centroids.size() != model.k * kDimension ||
        model.assignments.size() != kPoints || model.distances.size() != kPoints ||
        model.iteration_stats.empty()) {
        return false;
    }
    for (float value : model.centroids) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    double objective = 0;
    for (size_t point = 0; point < kPoints; ++point) {
        if (model.assignments[point] >= model.k) {
            return false;
        }
        double nearest = std::numeric_limits<double>::infinity();
        double selected = 0;
        for (size_t cluster = 0; cluster < model.k; ++cluster) {
            double distance = 0;
            for (size_t dim = 0; dim < kDimension; ++dim) {
                const double delta = static_cast<double>(data[point * kDimension + dim]) -
                                     model.centroids[cluster * kDimension + dim];
                distance += delta * delta;
            }
            nearest = std::min(nearest, distance);
            if (cluster == model.assignments[point]) {
                selected = distance;
            }
        }
        const double tolerance = 1e-4 * std::max(1.0, nearest);
        if (!(std::abs(selected - nearest) <= tolerance) ||
            !(std::abs(model.distances[point] - selected) <= tolerance)) {
            return false;
        }
        objective += selected;
    }
    return std::abs(model.final_obj - objective) <= 1e-4 * std::max(1.0, objective);
}
}  // namespace

int main() {
    const auto& features = rabitqlib::cpu::features();
    if (features.avx512vpopcntdq && !features.avx512f) {
        return 1;
    }
    std::vector<float> data(kPoints * kDimension);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.17F);
    }

    rabitqlib::qgkmeans::QGKMeansParameters graph_parameters;
    graph_parameters.niter = 2;
    graph_parameters.num_threads = 2;
    graph_parameters.ef_build = 33;
    graph_parameters.ef_search = 33;
    graph_parameters.final_assignment = rabitqlib::clustering::FinalAssignmentMode::Exact;
    rabitqlib::qgkmeans::QGKMeans graph(kDimension, 33, graph_parameters);
    if (!check_clustering(graph, data)) {
        std::cerr << "Installed QGKMeans returned invalid clustering results\n";
        return 1;
    }

    rabitqlib::rabitqkmeans::RaBitQKMeansParameters flat_parameters;
    flat_parameters.niter = 2;
    flat_parameters.num_threads = 2;
    flat_parameters.final_assignment = rabitqlib::clustering::FinalAssignmentMode::Exact;
    rabitqlib::rabitqkmeans::RaBitQKMeans flat(kDimension, 4, flat_parameters);
    if (!check_clustering(flat, data)) {
        std::cerr << "Installed RaBitQKMeans returned invalid clustering results\n";
        return 1;
    }

    rabitqlib::ivf::IVF index(
        kPoints,
        kDimension,
        flat.k,
        4,
        rabitqlib::METRIC_L2,
        rabitqlib::RotatorType::FhtKacRotator,
        rabitqlib::ivf::InitializerType::FlatRaBitQ
    );
    if (index.initializer_type() != rabitqlib::ivf::InitializerType::FlatRaBitQ) {
        std::cerr << "Installed IVF did not preserve explicit centroid routing\n";
        return 1;
    }
    index.construct(data.data(), flat.centroids.data(), flat.assignments.data(), false, 2);
    std::vector<rabitqlib::PID> ids(kPoints);
    std::vector<float> distances(kPoints);
    index.search_batch(data.data(), kPoints, 1, flat.k, ids.data(), distances.data());
    for (size_t i = 0; i < kPoints; ++i) {
        rabitqlib::PID id = 0;
        float distance = 0;
        index.search(data.data() + i * kDimension, 1, flat.k, &id, &distance);
        if (ids[i] != id || distances[i] != distance) {
            std::cerr << "Installed IVF batch and single-query results differ\n";
            return 1;
        }
    }

    rabitqlib::symqg::QuantizedGraph<float> symqg(kPoints, kDimension, 32);
    rabitqlib::symqg::QGBuilder builder(symqg, 64, data.data(), 2);
    builder.build();
    symqg.set_ef(64);
    symqg.search_batch(data.data(), kPoints, 1, ids.data(), distances.data(), 2);
    for (size_t i = 0; i < kPoints; ++i) {
        rabitqlib::PID id = 0;
        float distance = 0;
        symqg.search(data.data() + i * kDimension, 1, &id, &distance);
        if (ids[i] != id || distances[i] != distance) {
            std::cerr << "Installed SymphonyQG batch and single-query results differ\n";
            return 1;
        }
    }
    return 0;
}
