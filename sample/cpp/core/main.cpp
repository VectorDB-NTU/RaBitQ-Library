#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/query.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

int main() {
    try {
        constexpr size_t kDim = 65;
        constexpr size_t kTotalBits = 4;
        constexpr size_t kExBits = kTotalBits - 1;
        std::vector<float> vector(kDim), query(kDim);
        for (size_t i = 0; i < kDim; ++i) {
            vector[i] = std::sin(static_cast<float>(i) * 0.17F);
            query[i] = std::cos(static_cast<float>(i) * 0.11F);
        }

        // Data, centroid and query must use the same rotation and padded domain.
        std::unique_ptr<rabitqlib::Rotator<float>> rotator(
            rabitqlib::choose_rotator<float>(
                kDim, rabitqlib::RotatorType::FhtKacRotator, 0, 42
            )
        );
        const size_t padded_dim = rotator->size();
        std::vector<float> rotated_vector(padded_dim), rotated_query(padded_dim);
        std::vector<float> centroid(padded_dim, 0.0F);
        rotator->rotate(vector.data(), rotated_vector.data());
        rotator->rotate(query.data(), rotated_query.data());

        std::vector<char> bin(rabitqlib::BinDataMap<float>::data_bytes(padded_dim));
        std::vector<char> extra(
            rabitqlib::ExDataMap<float>::data_bytes(padded_dim, kExBits)
        );
        const rabitqlib::quant::RabitqConfig config;
        rabitqlib::quant::quantize_split_single(
            rotated_vector.data(),
            centroid.data(),
            padded_dim,
            kExBits,
            bin.data(),
            extra.data(),
            rabitqlib::METRIC_L2,
            config
        );

        // The query object borrows rotated_query, which stays alive through estimation.
        rabitqlib::SplitSingleQuery<float> prepared_query(
            rotated_query.data(), padded_dim, kExBits, config
        );
        prepared_query.set_g_add(
            std::sqrt(rabitqlib::l2norm_sqr(rotated_query.data(), padded_dim))
        );
        const auto estimate = rabitqlib::estimate_split_single_full(
            bin.data(),
            extra.data(),
            rabitqlib::select_excode_ipfunc(kExBits),
            prepared_query,
            padded_dim,
            kExBits,
            prepared_query.g_add(),
            prepared_query.g_error()
        );
        std::cout << "Approximate squared L2: " << estimate.distance << '\n'
                  << "Exact squared L2: "
                  << rabitqlib::euclidean_sqr(vector.data(), query.data(), kDim) << '\n';
        return std::isfinite(estimate.distance) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
