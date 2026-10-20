#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/utils/space.hpp"

int main() {
    try {
        constexpr size_t kDim = 65, kCount = 49, kExBits = 3;
        std::unique_ptr<rabitqlib::Rotator<float>> rotator(
            rabitqlib::choose_rotator<float>(
                kDim, rabitqlib::RotatorType::FhtKacRotator, 0, 42
            )
        );
        const size_t padded_dim = rotator->size();
        std::vector<float> vector(kDim), query(kDim), original_centroid(kDim);
        std::vector<float> rotated_data(kCount * padded_dim), rotated_query(padded_dim);
        std::vector<float> centroid(padded_dim, 0.0F);
        for (size_t i = 0; i < kDim; ++i) {
            query[i] = std::cos(static_cast<float>(i) * 0.11F);
            original_centroid[i] = 0.1F * std::sin(static_cast<float>(i) * 0.23F);
        }
        rotator->rotate(query.data(), rotated_query.data());
        rotator->rotate(original_centroid.data(), centroid.data());
        for (size_t row = 0; row < kCount; ++row) {
            for (size_t i = 0; i < kDim; ++i) {
                vector[i] = std::sin(static_cast<float>(i + row) * 0.17F);
            }
            rotator->rotate(vector.data(), rotated_data.data() + row * padded_dim);
        }

        // A partial batch still owns storage and score outputs for all 32 lanes.
        std::vector<char> batch(rabitqlib::BatchDataMap<float>::data_bytes(padded_dim));
        const size_t extra_stride =
            rabitqlib::ExDataMap<float>::data_bytes(padded_dim, kExBits);
        constexpr size_t kBatchSize = rabitqlib::fastscan::kBatchSize;
        std::vector<char> extra(kBatchSize * extra_stride);
        std::array<float, kBatchSize> estimates{}, bounds{}, products{};
        const auto extra_ip = rabitqlib::select_excode_ipfunc(kExBits);
        const float residual_norm = std::sqrt(
            rabitqlib::euclidean_sqr(rotated_query.data(), centroid.data(), padded_dim)
        );
        const float centroid_ip =
            rabitqlib::dot_product(rotated_query.data(), centroid.data(), padded_dim);
        for (auto metric : {rabitqlib::METRIC_L2, rabitqlib::METRIC_IP}) {
            // This object borrows rotated_query for refinement; keep it alive.
            rabitqlib::SplitBatchQuery<float> prepared_query(
                rotated_query.data(), padded_dim, kExBits, metric, true
            );
            prepared_query.set_g_add(residual_norm, centroid_ip);
            for (size_t first = 0; first < kCount; first += kBatchSize) {
                const size_t count = std::min(kBatchSize, kCount - first);
                rabitqlib::quant::quantize_split_batch(
                    rotated_data.data() + first * padded_dim,
                    centroid.data(),
                    count,
                    padded_dim,
                    kExBits,
                    batch.data(),
                    extra.data(),
                    metric
                );
                rabitqlib::split_batch_estdist(
                    batch.data(),
                    prepared_query,
                    prepared_query.padded_dim(),
                    estimates.data(),
                    bounds.data(),
                    products.data(),
                    prepared_query.use_hacc()
                );
                // Ignore physical padding lanes in the final, partial batch.
                for (size_t row = 0; row < count; ++row) {
                    const float estimate = rabitqlib::split_distance_boosting(
                        extra.data() + row * extra_stride,
                        extra_ip,
                        prepared_query,
                        padded_dim,
                        kExBits,
                        products[row]
                    );
                    if (!std::isfinite(estimate)) {
                        return 1;
                    }
                    std::cout << (metric == rabitqlib::METRIC_L2 ? "L2 " : "IP ")
                              << first + row << ": " << estimate << '\n';
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
